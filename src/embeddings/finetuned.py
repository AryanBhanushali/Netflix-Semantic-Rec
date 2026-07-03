import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("TMDB_PATH")
MODEL_PATH = os.getenv("FINETUNED_MODEL_PATH")

df = pd.read_csv(f"{DATA_PATH}/movies_processed.csv")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModel.from_pretrained(
    "NousResearch/Meta-Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model.eval()

def get_embedding(text):
    enc = tokenizer(text, truncation=True, max_length=128, padding='max_length', return_tensors='pt').to('cuda')
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            output = model(**enc)
    mask = enc['attention_mask'].unsqueeze(-1).expand(output.last_hidden_state.size()).to(output.last_hidden_state.dtype)
    return ((output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).squeeze().cpu().numpy()

texts = df["rich_text"].tolist()
embeddings = []
for i, text in enumerate(texts):
    embeddings.append(get_embedding(text))
    if (i + 1) % 500 == 0:
        print(f"{i+1}/{len(texts)}")

embeddings = np.array(embeddings)
np.save(f"{DATA_PATH}/embeddings_ft.npy", embeddings)
print(embeddings.shape)