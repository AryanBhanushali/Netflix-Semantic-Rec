import os
import numpy as np
import pandas as pd
import chromadb
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("TMDB_PATH")

df = pd.read_csv(f"{DATA_PATH}/movies_processed.csv")
embeddings_base = np.load(f"{DATA_PATH}/embeddings.npy")
embeddings_ft = np.load(f"{DATA_PATH}/embeddings_ft.npy")

client = chromadb.HttpClient(host="localhost", port=8000)

base_col = client.get_or_create_collection(name="movies_base", metadata={"hnsw:space": "cosine"})
ft_col = client.get_or_create_collection(name="movies_ft", metadata={"hnsw:space": "cosine"})

ids = df["id"].astype(str).tolist()
documents = df["rich_text"].tolist()
metadatas = df[["title", "vote_average", "release_date", "director"]].to_dict(orient="records")

BATCH = 100
for i in range(0, len(ids), BATCH):
    end = min(i + BATCH, len(ids))
    base_col.upsert(
        ids=ids[i:end],
        embeddings=embeddings_base[i:end].tolist(),
        documents=documents[i:end],
        metadatas=metadatas[i:end]
    )
    ft_col.upsert(
        ids=ids[i:end],
        embeddings=embeddings_ft[i:end].tolist(),
        documents=documents[i:end],
        metadatas=metadatas[i:end]
    )
    print(f"{end}/{len(ids)}")

print(base_col.count(), ft_col.count())