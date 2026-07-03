# Netflix Semantic Recommendation Engine

A semantic movie recommendation system that uses fine-tuned LLM embeddings, dual vector databases (ChromaDB + Pinecone), and a LangGraph multi-agent pipeline to deliver personalized recommendations, even for cold-start users with zero viewing history.

The whole system runs with `docker compose up`.

## The problem

Traditional collaborative filtering recommends movies based on what similar users watched. But for new users with no viewing history (the cold-start problem), CF has nothing to work with and degrades to a static popularity list (the same 10 movies for everyone). This system uses semantic understanding of movie content to provide personalized recommendations from the first interaction, using only a natural language description of what the user likes.

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────┐
│                  LangGraph Agent                     │
│                                                      │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐       │
│  │  Query    │──▶│  Vector   │──▶│ Re-Ranker│       │
│  │  Parser   │   │  Search   │   │ (Llama    │       │
│  │(Llama 3.1)│   │           │   │  3.1 8B)  │       │
│  └───────────┘   └─────┬─────┘   └─────┬─────┘       │
│                        │               │             │
│                        ▼               ▼             │
│                 ┌────────────┐   ┌───────────┐       │
│                 │ ChromaDB   │   │ Explainer │       │
│                 │ (Docker)   │   │ (Llama    │       │
│                 ├────────────┤   │  3.1 8B)  │       │
│                 │ Pinecone   │   └───────────┘       │
│                 │ (Cloud)    │                       │
│                 └────────────┘                       │
└──────────────────────────────────────────────────────┘
    │
    ▼
FastAPI (port 8001)
```

## How it works

**Embeddings:** Movie plots, genres, cast, keywords, and director are combined into a rich text field per movie. Sentence-transformers (all-MiniLM-L6-v2) encodes these into 384-dimensional vectors for semantic search.

**QLoRA Fine-Tuning:** Llama 3.1 8B was fine-tuned with QLoRA (4-bit NF4 quantization, LoRA r=16, alpha=32) on 25K positive/negative movie-similarity pairs sourced from TMDB's similar movies API. Training ran for 3 epochs on an NVIDIA H200 GPU. The fine-tuned model serves as the re-ranker in the agent pipeline.

**Vector Databases:** Embeddings are indexed into both ChromaDB (self-hosted Docker container) and Pinecone (cloud free tier) for dual-store retrieval and benchmarking.

**LangGraph Agent:** A 4-node stateful agent pipeline:

1. **Query Parser** — classifies intent (by-title vs free-text) and extracts movie titles
2. **Vector Search** — queries ChromaDB for top-20 candidates
3. **Re-Ranker** — Llama 3.1 8B re-ranks candidates based on query context
4. **Explainer** — generates a natural language explanation of recommendations

## Cold-start benchmark

The key experiment: simulate users with no viewing history. The user provides only a text description of what they like. Compare semantic search against a standard CF baseline (SVD on MovieLens 100K).

| Metric                  | Semantic | CF (SVD) |
| ----------------------- | -------- | -------- |
| NDCG@10                 | 0.2391   | 0.3964   |
| Hit-Rate@10             | 0.3724   | 0.5918   |
| Unique recs (196 users) | 782      | 10       |
| Personalization         | Yes      | None     |

CF degrades to a static popularity list for cold-start users — the same 10 movies (Shawshank Redemption, Godfather, Fight Club, etc.) for every user regardless of taste. Semantic search provides 78x more diverse, personalized recommendations using only a text description of preferences.

## Training details

| Parameter        | Value                             |
| ---------------- | --------------------------------- |
| Base Model       | Llama 3.1 8B                      |
| Quantization     | 4-bit NF4 (QLoRA)                 |
| LoRA Rank        | 16                                |
| LoRA Alpha       | 32                                |
| Target Modules   | q_proj, v_proj                    |
| Training Pairs   | 25,796 positive + 25,796 negative |
| Source           | TMDB similar movies API           |
| Epochs           | 3                                 |
| Final Train Loss | 0.1638                            |
| Final Val Loss   | 0.1728                            |
| GPU              | NVIDIA H200 (Northeastern HPC)    |

## Stack

| Layer         | Technology                               |
| ------------- | ---------------------------------------- |
| Fine-Tuning   | QLoRA (HuggingFace PEFT + bitsandbytes)  |
| LLM           | Llama 3.1 8B via Ollama                  |
| Embeddings    | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB     | ChromaDB (Docker), Pinecone (Cloud)      |
| Agent         | LangGraph + LangChain                    |
| API           | FastAPI                                  |
| Orchestration | Docker Compose                           |
| Training GPU  | NVIDIA H200 (Northeastern HPC)           |

## Running it

### Prerequisites

- Docker and Docker Compose
- Python 3.12
- Pinecone API key (free tier)
- TMDB API key (free)

### Setup

1. Create a `.env` file in the project root:

   ```
   KAGGLE_USERNAME=your_username
   KAGGLE_KEY=your_kaggle_key
   TMDB_API_KEY=your_tmdb_key
   PINECONE_API_KEY=your_pinecone_key
   TMDB_PATH=data/processed
   MOVIELENS_PATH=data/processed
   FINETUNED_MODEL_PATH=models/finetuned
   ```

2. Download and process the data:

   ```bash
   pip install -r requirements.txt
   python download_data.py
   python src/embeddings/baseline.py
   ```

3. Start the services:

   ```bash
   docker compose up -d
   ```

4. Pull the LLM model:

   ```bash
   docker compose exec ollama ollama pull llama3.1:8b
   ```

5. Index into vector databases:

   ```bash
   python src/vectordb/chroma_client.py
   python src/vectordb/pinecone_client.py
   ```

6. Open the API docs at `http://localhost:8001/docs`.

### Services

| Service  | URL                        |
| -------- | -------------------------- |
| API docs | http://localhost:8001/docs |
| ChromaDB | http://localhost:8000      |

## API

| Endpoint           | Method | Description                                              |
| ------------------ | ------ | -------------------------------------------------------- |
| `/health`          | GET    | Health check                                             |
| `/recommend/title` | POST   | Recommend by movie title (vector similarity)             |
| `/recommend/query` | POST   | Recommend by free-text query (semantic search)           |
| `/recommend/agent` | POST   | Full agent pipeline (parse → search → re-rank → explain) |

### Example

```bash
POST /recommend/agent
{"query": "movies similar to Inception", "n": 5}

→ Inception           (similarity: 0.62)
→ Trance              (similarity: 0.49)
→ Cube                (similarity: 0.49)
→ Vanilla Sky         (similarity: 0.47)
→ Primer              (similarity: 0.47)

Explanation: "Recommended movies share themes of mind-bending
psychological tension, reality manipulation, and layered
narratives similar to Inception..."
```

## Data

- **TMDB 5000 Movies** — plots, genres, cast, keywords ([Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata))
- **MovieLens 100K** — user ratings for CF baseline comparison ([Kaggle](https://www.kaggle.com/datasets/sriharshabsprasad/movielens-dataset-100k-ratings))
- **TMDB API** — similar movies endpoint (ground truth for evaluation)

## Project structure

```
netflix-semantic-rec/
├── notebooks/
│   ├── 01_data_prep.ipynb          TMDB cleaning, pair generation
│   ├── 02_lora_finetune.ipynb      QLoRA training on H200
│   └── 03_benchmark.ipynb          semantic vs CF comparison
├── src/
│   ├── embeddings/
│   │   ├── baseline.py             sentence-transformers embeddings
│   │   └── finetuned.py            fine-tuned model embeddings
│   ├── vectordb/
│   │   ├── chroma_client.py        ChromaDB Docker indexing
│   │   └── pinecone_client.py      Pinecone cloud indexing
│   ├── agent/
│   │   ├── graph.py                LangGraph 4-node agent
│   │   └── tools.py                search + rerank tools
│   └── api/
│       └── main.py                 FastAPI endpoints
├── data/
│   └── processed/                  processed CSVs (gitignored)
├── models/
│   └── finetuned/                  QLoRA adapter weights
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Performance notes

The Ollama container runs CPU-only by default, so agent responses take 2 to 5 minutes. The `/recommend/title` and `/recommend/query` endpoints are instant since they only use vector similarity. To speed up agent responses, configure GPU passthrough for the Ollama service via the NVIDIA Container Toolkit.
