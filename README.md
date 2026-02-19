# Introduction to Vector Databases

Hands-on demos that explain vector database fundamentals through a bookstore storyline: semantic search, RAG, recommendations, and performance tuning.

## ✅ Preferred: Run in a VS Code DevContainer or GitHub Codespace
1. Install [VS Code](https://code.visualstudio.com/) + [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), or open this repo in **GitHub Codespaces**.
2. Open the repo and choose **“Reopen in Container”**.
3. Wait for Docker services to start (Postgres, Pinecone Local, Ollama, Qdrant). The DevContainer handles networking and environment variables for you.
4. **Data is initialized automatically.** The DevContainer's `postStartCommand` runs `setup/init_all.sh`, which populates Postgres and Pinecone. You only need to re-run it if you want to reset the data (see [Resetting Data](#-resetting-data)).
5. Run any demo from inside the container (see table below). URLs are printed to the console (e.g., `http://localhost:8081`).

> ℹ️ DevContainer/Codespaces is the fastest path. Networking and default hosts (`pinecone`, `qdrant`, `ollama`) are already wired. Use local/Ollama on your host if you prefer; the container is preconfigured to talk to it.

## 🧠 Samples Overview
| Module | Path | What It Shows | Default Port | Run Command |
|--------|------|---------------|--------------|-------------|
| 1 | [`module_1/finding_meaning_enhanced`](module_1/finding_meaning_enhanced) | SQL keyword search vs semantic vector search across **descriptions + full book text** | 8081 | `cd module_1/finding_meaning_enhanced && python app.py` |
| 3 | [`module_3/faceted_search`](module_3/faceted_search) | Semantic search with **metadata filters** (genre, rating, price, stock) | 8001 | `cd module_3/faceted_search && python app.py` |
| 4 | [`module_4/rag_webapp`](module_4/rag_webapp) | **RAG** (retrieval-augmented generation) with per-book scoping | 8008 | `cd module_4/rag_webapp && python app.py` |
| 4 | [`module_4/recommendation_webapp`](module_4/recommendation_webapp) | **Content-based + collaborative** recommendations side-by-side | 5001 | `cd module_4/recommendation_webapp && python app.py` |
| 5 | [`module_5/recall_vs_latency`](module_5/recall_vs_latency) | **Recall vs latency** trade-offs in ANN search (Qdrant + HNSW) | 8080 | `cd module_5/recall_vs_latency && python app.py` |

## 🚀 Additional Setup (only needed for Module 5)
> **Note:** The DevContainer automatically runs `setup/init_all.sh` on start, which initializes Postgres (including the enhanced full-text data for Module 1) and Pinecone. The only module that requires manual setup is Module 5.

```bash
# Module 5 (Recall vs Latency): creates a Qdrant collection with 100K random vectors
cd module_5/recall_vs_latency && python init_qdrant.py --reset
```

## 🤖 Embeddings & LLMs
- **Easiest path:** install [Ollama](https://ollama.com/download) on your **host machine** and pull an embedding model:
  ```bash
  ollama pull all-minilm            # embeddings (used by most demos)
  ollama pull llama3.2:3b           # LLM for RAG synthesized answers
  ```
  The DevContainer is configured to reach your host Ollama via `http://host.docker.internal:11434`.
- If Ollama is unavailable, the code falls back to **`sentence-transformers`** automatically (slower but works without any extra setup).
- Environment variables used by the demos:
  - `OLLAMA_HOST` (defaults to `http://host.docker.internal:11434` in the DevContainer)
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

# Module 5 Qdrant recall vs latency demo
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
