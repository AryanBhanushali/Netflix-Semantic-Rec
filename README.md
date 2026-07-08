# Netflix Semantic Recommendation Engine

A semantic movie recommendation system that uses sentence-transformer embeddings for retrieval, a QLoRA fine-tuned Llama 3.2 3B cross-encoder for reranking, dual vector databases (ChromaDB + Pinecone), and a LangGraph agent pipeline to deliver personalized recommendations, even for cold-start users with zero viewing history.

The whole system runs with `docker compose up`.

## The problem

Traditional collaborative filtering recommends movies based on what similar users watched. But for new users with no viewing history (the cold-start problem), CF has nothing to work with and degrades to a static popularity list, the same handful of movies for everyone. This system uses semantic understanding of movie content to provide personalized recommendations from the first interaction, using only a natural language description of what the user likes.

## Architecture

```
User Query
    |
    v
+------------------------------------------------------+
|                  LangGraph Agent                     |
|                                                      |
|  +-----------+   +-----------+   +-----------+       |
|  |  Query    |-->|  Vector   |-->| Re-Ranker |       |
|  |  Parser   |   |  Search   |   | (Llama    |       |
|  |(Llama 3.1)|   | (MiniLM)  |   |  3.2 3B   |       |
|  +-----------+   +-----+-----+   |  cross-   |       |
|                        |         |  encoder) |       |
|                        v         +-----+-----+       |
|                 +------------+         |             |
|                 | ChromaDB   |         v             |
|                 | (Docker)   |   +-----------+       |
|                 +------------+   | Explainer |       |
|                 | Pinecone   |   | (Llama    |       |
|                 | (Cloud)    |   |  3.1 8B)  |       |
|                 +------------+   +-----------+       |
+------------------------------------------------------+
    |
    v
FastAPI + Frontend (port 8001)
```

## How it works

**Retrieval embeddings:** Movie plots, genres, cast, keywords, and director are combined into a rich text field per movie. Sentence-transformers (all-MiniLM-L6-v2) encodes these into 384-dimensional vectors for semantic search. Retrieval uses these MiniLM embeddings throughout.

**QLoRA fine-tuned reranker:** A Llama 3.2 3B model was fine-tuned with QLoRA (4-bit NF4 quantization, LoRA r=16, alpha=32) as a cross-encoder for sequence classification. It takes a (query, candidate) text pair and outputs a single relevance score, then reorders the retrieved candidates. It was trained on roughly 42K positive and negative movie-similarity pairs sourced from TMDB's similar movies API. This is a reranking stage on top of MiniLM retrieval, not a replacement for it.

**Why a cross-encoder and not fine-tuned embeddings:** An earlier experiment fine-tuned an 8B model as a bi-encoder to produce similarity embeddings, but on our evaluation it scored below the much smaller base MiniLM model, so it was dropped. Decoder-only models with mean-pooled hidden states are poorly suited to producing similarity embeddings off the shelf. A cross-encoder, where the query and candidate attend to each other in a single pass, is the architecture that actually adds value on top of MiniLM retrieval.

**Vector databases:** Embeddings are indexed into both ChromaDB (self-hosted Docker container) and Pinecone (cloud free tier) for dual-store retrieval and benchmarking.

**LangGraph agent:** A 4-node stateful agent pipeline:

1. **Query Parser** - classifies intent (by-title vs free-text) and extracts movie titles (Llama 3.1 8B via Ollama)
2. **Vector Search** - queries ChromaDB for top-20 candidates using MiniLM embeddings
3. **Re-Ranker** - the QLoRA Llama 3.2 3B cross-encoder rescores and reorders candidates
4. **Explainer** - generates a natural language explanation of recommendations (Llama 3.1 8B via Ollama)

## Reranker availability and fallback

The cross-encoder needs a GPU and roughly 6GB of free memory to load its 4-bit weights. The application detects this at runtime:

- With a GPU and sufficient memory, the agent uses the QLoRA cross-encoder.
- Without a GPU, or when free memory is too low to load the model safely, the agent automatically falls back to reranking the candidate list with the stock Llama 3.1 8B model through Ollama.

This means the agent endpoint runs on any machine. It uses the trained reranker where the hardware allows and degrades gracefully to LLM reranking otherwise, rather than failing.

## Dashboard

Three tabs, built for a non-technical user:

- **Find similar** enters a movie title and returns the closest matches by vector similarity.
- **Describe what you want** takes a free-text description of mood, genre, or theme and finds matching movies via semantic search.
- **Ask the agent** runs the full LangGraph pipeline: parses intent, searches, reranks, and explains why each movie was recommended.

## Cold-start benchmark

The key experiment: simulate users with no viewing history. The user provides only a text description of what they like. Compare semantic search against a standard CF baseline (SVD on MovieLens 100K).

| Metric                  | Semantic | CF (SVD) |
| ----------------------- | -------- | -------- |
| NDCG@10                 | 0.2391   | 0.3964   |
| Hit-Rate@10             | 0.3724   | 0.5918   |
| Unique recs (196 users) | 782      | 10       |
| Personalization         | Yes      | None     |

How to read this: CF scores higher on NDCG@10 and Hit-Rate@10, because SVD is evaluated on MovieLens users who do have rating history, which is the setting CF is built for. The point of the comparison is what happens under cold-start. CF has no per-user signal to work with, so it collapses to a static popularity list, the same 10 movies (Shawshank Redemption, Godfather, Fight Club, and so on) for every one of the 196 users regardless of taste. Semantic search returns 782 unique recommendations across those users using only a text description of preferences. The takeaway is not that semantic retrieval beats CF on accuracy where history exists; it is that semantic retrieval provides personalized, diverse recommendations in the cold-start setting that CF structurally cannot.

## Reranker evaluation

The QLoRA cross-encoder reaches NDCG@10 of approximately 0.20 on a held-out set of TMDB similar-movie ground truth. This is a real signal, meaningfully better than random ordering, but a reranker improves the ordering of already-retrieved candidates rather than working miracles. TMDB's similar-movies lists are sparse ground truth, so absolute NDCG values are low for all methods on this evaluation.

## Training details

| Parameter        | Value                                                         |
| ---------------- | ------------------------------------------------------------- |
| Base Model       | Llama 3.2 3B                                                  |
| Task             | Cross-encoder, sequence classification (1 label)              |
| Quantization     | 4-bit NF4 (QLoRA)                                             |
| LoRA Rank        | 16                                                            |
| LoRA Alpha       | 32                                                            |
| Target Modules   | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Saved Modules    | classifier / score head                                       |
| Training Pairs   | ~21K positive + ~21K negative                                 |
| Source           | TMDB similar movies API                                       |
| Best Checkpoint  | Epoch 2 (selected by NDCG@10)                                 |
| Reranker NDCG@10 | ~0.20                                                         |
| GPU              | NVIDIA H200 (Northeastern HPC)                                |

## Stack

| Layer         | Technology                               |
| ------------- | ---------------------------------------- |
| Fine-Tuning   | QLoRA (HuggingFace PEFT + bitsandbytes)  |
| Retrieval     | sentence-transformers (all-MiniLM-L6-v2) |
| Reranker      | Llama 3.2 3B cross-encoder (QLoRA)       |
| Agent LLM     | Llama 3.1 8B via Ollama                  |
| Vector DB     | ChromaDB (Docker), Pinecone (Cloud)      |
| Agent         | LangGraph + LangChain                    |
| API           | FastAPI                                  |
| Frontend      | Vanilla HTML/CSS/JS                      |
| Orchestration | Docker Compose                           |
| Training GPU  | NVIDIA H200 (Northeastern HPC)           |

## Running it

### Prerequisites

- Docker and Docker Compose
- Python 3.12
- Pinecone API key (free tier)
- TMDB API key (free)
- A GPU is optional. Without one, the agent uses the Ollama reranking fallback.

### Setup

1. Clone the repo:

   ```bash
   git clone https://github.com/AryanBhanushali/Netflix-Semantic-Rec.git
   cd Netflix-Semantic-Rec
   ```

2. Create a `.env` file in the project root:

   ```
   KAGGLE_USERNAME=your_username
   KAGGLE_KEY=your_kaggle_key
   TMDB_API_KEY=your_tmdb_key
   PINECONE_API_KEY=your_pinecone_key
   TMDB_PATH=data/processed
   MOVIELENS_PATH=data/processed
   FINETUNED_MODEL_PATH=models/finetuned
   ```

3. Download and process the data:

   ```bash
   pip install -r requirements.txt
   python download_data.py
   python src/embeddings/baseline.py
   ```

   Note: `torch` is pinned to 2.5.1. For GPU support, install the CUDA build from the PyTorch index before installing the rest:

   ```bash
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
   ```

4. Start the services:

   ```bash
   docker compose up -d
   ```

5. Pull the LLM model:

   ```bash
   docker compose exec ollama ollama pull llama3.1:8b
   ```

6. Index into vector databases:

   ```bash
   python src/vectordb/chroma_client.py
   python src/vectordb/pinecone_client.py
   ```

7. Open the dashboard at `http://localhost:8001`.

### Running without Docker

If you prefer to run the API directly:

```bash
docker compose up -d chromadb
ollama run llama3.1:8b
uvicorn src.api.main:app --port 8001
```

Then open `http://localhost:8001`.

### Services

| Service   | URL                        |
| --------- | -------------------------- |
| Dashboard | http://localhost:8001      |
| API docs  | http://localhost:8001/docs |
| ChromaDB  | http://localhost:8000      |

## API

| Endpoint           | Method | Description                                          |
| ------------------ | ------ | ---------------------------------------------------- |
| `/health`          | GET    | Health check                                         |
| `/recommend/title` | POST   | Recommend by movie title (vector similarity)         |
| `/recommend/query` | POST   | Recommend by free-text query (semantic search)       |
| `/recommend/agent` | POST   | Full agent pipeline (parse, search, rerank, explain) |

### Example

```bash
POST /recommend/agent
{"query": "movies similar to Inception", "n": 5}

-> Inception           (similarity: 0.62)
-> Trance              (similarity: 0.49)
-> Cube                (similarity: 0.49)
-> Vanilla Sky         (similarity: 0.47)
-> Primer              (similarity: 0.47)

Explanation: "Recommended movies share themes of mind-bending
psychological tension, reality manipulation, and layered
narratives similar to Inception..."
```

## Data

- **TMDB 5000 Movies** - plots, genres, cast, keywords ([Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata))
- **MovieLens 100K** - user ratings for CF baseline comparison ([Kaggle](https://www.kaggle.com/datasets/sriharshabsprasad/movielens-dataset-100k-ratings))
- **TMDB API** - similar movies endpoint (ground truth for training and evaluation)

## Project structure

```
netflix-semantic-rec/
├── static/
│   └── index.html              dashboard UI
├── notebooks/
│   ├── 01_data_prep.ipynb      TMDB cleaning, pair generation
│   ├── 02_lora_finetune.ipynb  QLoRA cross-encoder training on H200
│   └── 03_benchmark.ipynb      semantic vs CF comparison
├── src/
│   ├── embeddings/
│   │   ├── baseline.py         sentence-transformers embeddings
│   │   └── finetuned.py        fine-tuned model embeddings
│   ├── vectordb/
│   │   ├── chroma_client.py    ChromaDB Docker indexing
│   │   └── pinecone_client.py  Pinecone cloud indexing
│   ├── agent/
│   │   ├── graph.py            LangGraph 4-node agent
│   │   └── tools.py            search + rerank tools
│   └── api/
│       └── main.py             FastAPI + frontend serving
├── data/
│   └── processed/              processed CSVs (gitignored)
├── models/
│   └── finetuned/              QLoRA adapter weights
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Performance notes

The Ollama container runs CPU-only by default, so agent responses that use the Ollama fallback take 2 to 5 minutes. The `/recommend/title` and `/recommend/query` endpoints are instant since they only use vector similarity. To speed up agent responses, configure GPU passthrough via the NVIDIA Container Toolkit, which also lets the QLoRA cross-encoder load instead of the fallback.
