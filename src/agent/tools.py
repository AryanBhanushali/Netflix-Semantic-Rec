import os
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("TMDB_PATH")

df = pd.read_csv(f"{DATA_PATH}/movies_processed.csv")
embeddings_base = np.load(f"{DATA_PATH}/embeddings.npy")

client = chromadb.HttpClient(host="localhost", port=8000)
base_col = client.get_collection("movies_base")

model_base = SentenceTransformer("all-MiniLM-L6-v2")


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