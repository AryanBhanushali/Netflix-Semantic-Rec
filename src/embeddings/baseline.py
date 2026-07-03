import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("TMDB_PATH")

df = pd.read_csv(f"{DATA_PATH}/movies_processed.csv")
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = df["rich_text"].tolist()
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

np.save(f"{DATA_PATH}/embeddings.npy", embeddings)
print(embeddings.shape)