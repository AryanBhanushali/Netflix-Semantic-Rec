import os
import numpy as np
import pandas as pd
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("TMDB_PATH")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

df = pd.read_csv(f"{DATA_PATH}/movies_processed.csv")
embeddings_base = np.load(f"{DATA_PATH}/embeddings.npy")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("movies-base")

BATCH = 100
for i in range(0, len(df), BATCH):
    end = min(i + BATCH, len(df))
    vectors = []
    for j in range(i, end):
        vectors.append({
            "id": str(df.iloc[j]["id"]),
            "values": embeddings_base[j].tolist(),
            "metadata": {
                "title": str(df.iloc[j]["title"]),
                "director": str(df.iloc[j]["director"]) if pd.notna(df.iloc[j]["director"]) else "Unknown",
                "vote_average": float(df.iloc[j]["vote_average"]),
            }
        })
    index.upsert(vectors=vectors)
    print(f"{end}/{len(df)}")

print(index.describe_index_stats())