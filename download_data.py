import os
import kagglehub
from dotenv import load_dotenv

load_dotenv()

os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")
os.environ["TMDB_API_KEY"] = os.getenv("TMDB_API_KEY")

tmdb_path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
movielens_path = kagglehub.dataset_download("sriharshabsprasad/movielens-dataset-100k-ratings")

print(tmdb_path)
print(movielens_path)