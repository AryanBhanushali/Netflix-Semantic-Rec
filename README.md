# Netflix Semantic Recommendation Engine

A semantic movie recommendation system that uses fine-tuned LLM embeddings, dual vector databases (ChromaDB + Pinecone), and a LangGraph multi-agent pipeline to deliver personalized recommendations — even for cold-start users with zero viewing history.

## Resume Bullet

> Fine-tuned Llama 3.1 (8B) with QLoRA on 25K movie-similarity pairs and indexed 4,391 movie embeddings into ChromaDB and Pinecone; semantic search delivered 78x more diverse personalized recommendations than collaborative filtering for cold-start users with no viewing history.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│              LangGraph Agent                     │
│                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │  Query   │──▶│  Vector  │──▶│Re-Ranker │    │
│  │  Parser  │   │  Search  │   │(Llama 3.1│    │
│  │(Llama3.1)│   │          │   │   8B)    │    │
│  └──────────┘   └────┬─────┘   └────┬─────┘    │
│                      │              │           │
│                      ▼              ▼           │
│               ┌────────────┐  ┌──────────┐     │
│               │ ChromaDB   │  │Explainer │     │
│               │ (Docker)   │  │(Llama3.1)│     │
│               ├────────────┤  └──────────┘     │
│               │ Pinecone   │                    │
│               │ (Cloud)    │                    │
│               └────────────┘                    │
└─────────────────────────────────────────────────┘
    │
    ▼
FastAPI (port 8001)
```

## How It Works

**Embeddings:** Movie plots, genres, cast, keywords, and director are combined into a rich text field per movie. Sentence-transformers (all-MiniLM-L6-v2) encodes these into 384-dimensional vectors for semantic search.

**QLoRA Fine-Tuning:** Llama 3.1 8B was fine-tuned with QLoRA (4-bit NF4 quantization, LoRA r=16, alpha=32) on 25K positive/negative movie-similarity pairs sourced from TMDB's similar movies API. Training ran for 3 epochs on an NVIDIA H200 GPU. The fine-tuned model serves as the re-ranker in the agent pipeline.

**Vector Databases:** Embeddings are indexed into both ChromaDB (self-hosted Docker container) and Pinecone (cloud free tier) for dual-store retrieval and benchmarking.

**LangGraph Agent:** A 4-node stateful agent pipeline:

1. **Query Parser** — classifies intent (by-title vs free-text) and extracts movie titles
2. **Vector Search** — queries ChromaDB for top-20 candidates
3. **Re-Ranker** — Llama 3.1 8B re-ranks candidates based on query context
4. **Explainer** — generates a natural language explanation of recommendations

## Benchmark Results

### Semantic vs Collaborative Filtering (Cold-Start Users)

| Metric                  | Semantic | CF (SVD) |
| ----------------------- | -------- | -------- |
| NDCG@10                 | 0.2391   | 0.3964   |
| Hit-Rate@10             | 0.3724   | 0.5918   |
| Unique recs (196 users) | 782      | 10       |
| Personalization         | Yes      | None     |

**Key finding:** CF degrades to a static popularity list for cold-start users — the same 10 movies (Shawshank Redemption, Godfather, Fight Club, etc.) for every user regardless of taste. Semantic search provides 78x more diverse, personalized recommendations using only a text description of preferences.

### Base Embeddings Precision@10

Tested on 2,595 movies against TMDB ground-truth similar movies:

- Base embeddings (all-MiniLM-L6-v2): 0.0137

Low precision is expected — TMDB's "similar movies" is editorially curated, not purely plot-based similarity.

## Datasets

- **TMDB 5000 Movies** — plots, genres, cast, keywords (Kaggle)
- **MovieLens 100K** — user ratings for CF baseline comparison
- **TMDB API** — similar movies endpoint (ground truth for evaluation)

## Tech Stack

- **LLM:** Llama 3.1 8B (Ollama) — query parsing, re-ranking, explanation
- **Fine-Tuning:** QLoRA via HuggingFace PEFT + bitsandbytes (4-bit NF4)
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector DBs:** ChromaDB (Docker), Pinecone (cloud)
- **Agent:** LangGraph (4-node pipeline)
- **API:** FastAPI
- **Infra:** Docker Compose, NVIDIA H200 (training), RTX 4070 (inference)

## Project Structure

```
netflix-semantic-rec/
├── notebooks/
│   ├── 01_data_prep.ipynb          # TMDB cleaning, pair generation
│   ├── 02_lora_finetune.ipynb      # QLoRA training on H200
│   └── 03_benchmark.ipynb          # semantic vs CF comparison
├── src/
│   ├── embeddings/
│   │   ├── baseline.py             # sentence-transformers embeddings
│   │   └── finetuned.py            # fine-tuned model embeddings
│   ├── vectordb/
│   │   ├── chroma_client.py        # ChromaDB Docker indexing
│   │   └── pinecone_client.py      # Pinecone cloud indexing
│   ├── agent/
│   │   ├── graph.py                # LangGraph 4-node agent
│   │   └── tools.py                # search + rerank tools
│   └── api/
│       └── main.py                 # FastAPI endpoints
├── data/
│   └── processed/
├── models/
│   └── finetuned/                  # QLoRA adapter weights
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Quick Start

### Prerequisites

- Docker Desktop
- Ollama with `llama3.1:8b`
- Python 3.12
- Pinecone API key (free tier)

### Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/netflix-semantic-rec.git
cd netflix-semantic-rec

# Start ChromaDB
docker compose up -d chromadb

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Fill in PINECONE_API_KEY, TMDB_API_KEY

# Generate embeddings
python src/embeddings/baseline.py

# Index into vector databases
python src/vectordb/chroma_client.py
python src/vectordb/pinecone_client.py

# Start Ollama
ollama run llama3.1:8b

# Start the API
uvicorn src.api.main:app --port 8001
```

### API Endpoints

```bash
# Health check
GET /health

# Recommend by movie title (vector similarity)
POST /recommend/title
{"title": "Inception", "n": 5}

# Recommend by free-text query (semantic search)
POST /recommend/query
{"query": "mind-bending sci-fi about dreams", "n": 5}

# Full agent pipeline (parse → search → re-rank → explain)
POST /recommend/agent
{"query": "movies similar to Inception", "n": 5}
```

### Example Output

```
POST /recommend/title {"title": "Inception", "n": 5}

→ A Scanner Darkly      (similarity: 0.58)
→ Primer                (similarity: 0.55)
→ Escape Plan           (similarity: 0.55)
→ Unknown               (similarity: 0.58)
→ Stolen                (similarity: 0.54)
```

## Training Details

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
