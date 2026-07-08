import os
os.environ["PANDAS_NO_PYARROW"] = "1"
import torch
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("TMDB_PATH")
MODEL_PATH = os.getenv("FINETUNED_MODEL_PATH")

df = pd.read_csv(f"{DATA_PATH}/movies_processed.csv")
embeddings_base = np.load(f"{DATA_PATH}/embeddings.npy")

client = chromadb.HttpClient(host="localhost", port=8000)
base_col = client.get_collection("movies_base")

model_base = SentenceTransformer("all-MiniLM-L6-v2")

# --- Reranker (QLoRA cross-encoder on Llama 3.2 3B) ---
# The trained cross-encoder needs ~6GB free to memory-map its weights. On a
# 16GB machine with the app stack (ChromaDB client, sentence-transformers,
# embeddings, FastAPI) already resident, that load can hard-fail. We guard it:
# a RAM pre-check catches the common case before the (uncatchable) mmap crash,
# and a try/except catches the recoverable load errors. If the reranker can't
# load, rerank() returns None so the agent falls back to Ollama reranking.

_reranker = None
_reranker_tokenizer = None
_reranker_device = None
_reranker_failed = False          # set True once we know the load can't succeed
_MIN_FREE_GB = 6.0                # headroom needed to mmap the 3B weights


def _enough_free_ram() -> bool:
    """Best-effort check for enough free RAM to load the reranker.
    Returns True if we can't determine it (don't block on missing psutil)."""
    try:
        import psutil
        free_gb = psutil.virtual_memory().available / (1024 ** 3)
        return free_gb >= _MIN_FREE_GB
    except Exception:
        return True


def _load_reranker() -> bool:
    """Attempt to load the cross-encoder. Returns True on success, False if the
    reranker is unavailable (caller should fall back to Ollama)."""
    global _reranker, _reranker_tokenizer, _reranker_device, _reranker_failed
    if _reranker is not None:
        return True
    if _reranker_failed:
        return False

    # If there is a GPU, use it; otherwise the 4-bit bitsandbytes path won't
    # work on CPU, so we don't even attempt it and fall back cleanly.
    if not torch.cuda.is_available():
        print("[reranker] no CUDA device — falling back to Ollama reranking.")
        _reranker_failed = True
        return False

    if not _enough_free_ram():
        print(f"[reranker] <{_MIN_FREE_GB}GB free RAM — skipping load, "
              f"falling back to Ollama reranking.")
        _reranker_failed = True
        return False

    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForSequenceClassification.from_pretrained(
            "unsloth/Llama-3.2-3B",
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={0: "6GB", "cpu": "12GB"},
            num_labels=1,
        )
        tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B")
        tok.pad_token = tok.eos_token
        base.config.pad_token_id = tok.pad_token_id
        model = PeftModel.from_pretrained(base, MODEL_PATH)
        model.eval()

        _reranker = model
        _reranker_tokenizer = tok
        _reranker_device = "cuda"
        print("[reranker] cross-encoder loaded on GPU.")
        return True
    except Exception as e:
        # Recoverable failures (OOM raised as exception, missing files, etc.).
        # A hard mmap/access-violation crash can't be caught here — the RAM
        # pre-check above is what protects against that case.
        print(f"[reranker] load failed ({type(e).__name__}: {e}) — "
              f"falling back to Ollama reranking.")
        _reranker_failed = True
        _reranker = None
        return False


def rerank(query_text: str, candidates: list, top_n: int = 10):
    """Rerank candidates with the trained cross-encoder.

    candidates: list of dicts with a 'rich_text' key.
    Returns the reordered list on success, or None if the reranker is
    unavailable (caller should fall back to another strategy).
    """
    if not _load_reranker():
        return None
    scores = []
    for cand in candidates:
        enc = _reranker_tokenizer(
            query_text, cand["rich_text"],
            truncation=True, max_length=256,
            padding="max_length", return_tensors="pt"
        ).to(_reranker_device)
        with torch.no_grad():
            with torch.amp.autocast(_reranker_device):
                logit = _reranker(**enc).logits.squeeze()
        scores.append(logit.item())
    # High logit = more similar (verified against full-list ranking); keep [::-1].
    order = np.argsort(scores)[::-1][:top_n]
    return [candidates[i] for i in order]


def recommend_by_title(title: str, n: int = 10) -> pd.DataFrame:
    matches = df[df["title"].str.lower() == title.lower()]
    if matches.empty:
        return pd.DataFrame()
    idx = matches.index[0]
    results = base_col.query(
        query_embeddings=embeddings_base[idx].tolist(),
        n_results=n + 1,
        include=["metadatas", "distances"]
    )
    rows = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        if meta["title"].lower() == title.lower():
            continue
        rows.append({
            "title": meta["title"],
            "director": meta["director"],
            "vote_average": meta["vote_average"],
            "similarity": round(1 - dist, 4)
        })
    return pd.DataFrame(rows[:n])


def recommend_by_query(query: str, n: int = 10) -> pd.DataFrame:
    qe = model_base.encode([query])[0].tolist()
    results = base_col.query(
        query_embeddings=qe,
        n_results=n,
        include=["metadatas", "distances"]
    )
    rows = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        rows.append({
            "title": meta["title"],
            "director": meta["director"],
            "vote_average": meta["vote_average"],
            "similarity": round(1 - dist, 4)
        })
    return pd.DataFrame(rows)


def get_rich_text(title: str) -> str:
    m = df[df["title"] == title]
    return m.iloc[0]["rich_text"] if not m.empty else title