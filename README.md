# Introduction to Vector Databases

Hands-on demos that explain vector database fundamentals through a bookstore storyline: semantic search, RAG, recommendations, and performance tuning.

## ✅ Preferred: Run in a VS Code DevContainer or GitHub Codespace
1. Install [VS Code](https://code.visualstudio.com/) + [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), or open this repo in **GitHub Codespaces**.
2. Open the repo and choose **“Reopen in Container”**.
3. Wait for Docker services to start (Postgres, Pinecone Local, Ollama, Qdrant). The devcontainer handles networking and env vars for you.
4. Initialize data once:
   ```bash
   cd setup
   ./init_all.sh   # or run the individual scripts listed below
   ```
5. Run any demo from inside the container (see table below). URLs are printed to the console (e.g., `http://localhost:8081`).

> ℹ️ DevContainer/Codespaces is the fastest path. Networking and default hosts (`pinecone`, `qdrant`, `ollama`) are already wired. Use local/Ollama on your host if you prefer; the container is preconfigured to talk to it.

## 🧠 Samples Overview
| Module | Path | What It Shows | Default Port | Run Command |
|--------|------|---------------|--------------|-------------|
| 1 | [`module_1/finding_meaning_enhanced`](module_1/finding_meaning_enhanced) | SQL keyword search vs semantic vector search across **descriptions + full book text** | 8081 | `cd module_1/finding_meaning_enhanced && python app.py` |
| 3 | [`module_3/faceted_search`](module_3/faceted_search) | Semantic search with **metadata filters** (genre, rating, price, stock) | 8001 | `cd module_3/faceted_search && python app.py` |
| 4 | [`module_4/rag_webapp`](module_4/rag_webapp) | **RAG** (retrieval-augmented generation) with per-book scoping | 5001 | `cd module_4/rag_webapp && python app.py` |
| 4 | [`module_4/recommendation_webapp`](module_4/recommendation_webapp) | **Content-based + collaborative** recommendations side-by-side | 5001 (shared—run one at a time) | `cd module_4/recommendation_webapp && python app.py` |
| 5 | [`module_5/recall_vs_latency`](module_5/recall_vs_latency) | **Recall vs latency** trade-offs in ANN search (Qdrant + HNSW) | 8080 | `cd module_5/recall_vs_latency && python app.py` |

## 🚀 One-time Setup (inside DevContainer)
```bash
cd setup
# Initialize Postgres data
python init_postgres.py
# Initialize Pinecone Local index + embeddings
python init_pinecone.py
# (Optional) Enhanced Module 1 SQL init
cd ../module_1/finding_meaning_enhanced && python init_enhanced_postgres.py
```

Short on time? Run the helper script:
```bash
cd setup && ./init_all.sh
```

## 🤖 Embeddings & LLMs
- **Easiest path:** run **local Ollama** (the container is configured to talk to `http://host.docker.internal:11434`). Install [Ollama](https://ollama.com/download) on your machine and pull an embedding model, e.g.:
  ```bash
  ollama pull mxbai-embed-large   # embeddings
  ollama pull llama3.1            # optional for RAG answers
  ```
- If Ollama is unavailable, the code falls back to **`sentence-transformers`** automatically.
- Env vars used by the demos:
  - `OLLAMA_HOST` (defaults to `http://ollama:11434` in container)
  - `PINECONE_HOST` (defaults to `http://pinecone:5081`)
  - `PINECONE_INDEX_HOST` (auto-detected; override if running locally)
  - `QDRANT_HOST` / `QDRANT_PORT`

## 🔗 Want to run outside the DevContainer?
That path is more involved (installing Postgres, Pinecone Local/Qdrant, configuring env vars). See **[RUNNING_LOCALLY.md](RUNNING_LOCALLY.md)** for a step-by-step guide.

## 🧪 Resetting Data
```bash
cd setup
python init_postgres.py --reset
python init_pinecone.py --reset
# Module 1 enhanced tables
cd ../module_1/finding_meaning_enhanced && python init_enhanced_postgres.py --reset
# Qdrant recall vs latency demo
cd ../module_5/recall_vs_latency && python init_qdrant.py --reset
```

## 🐛 Troubleshooting
- **Services healthy?**
  ```bash
  docker compose -f .devcontainer/docker-compose.yml ps
  curl http://pinecone:5081 || curl http://localhost:5081
  curl http://qdrant:6333/healthz || curl http://localhost:6333/healthz
  ```
- **Embeddings failing?** Ensure Ollama is running. The fallback model (`sentence-transformers`) will kick in automatically but may be slower.
- **Port already in use?** Override: `PORT=9000 python app.py`

## License
See [LICENSE](LICENSE) for details.
