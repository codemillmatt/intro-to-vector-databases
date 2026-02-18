# Running Locally (Outside the DevContainer)

> ⚠️ Recommended path is **DevContainer / Codespaces**. Use this guide only if you must run on your host. You’ll need to manage Docker services, env vars, and networking yourself.

## Prerequisites
- **Python** 3.10+
- **Docker** + **Docker Compose**
- **Ollama** (optional but recommended for embeddings/LLM answers)
- `git`, `curl`

## 1) Start the backing services
Use the repo’s compose file (ships with the devcontainer) to launch databases locally:

```bash
docker compose -f .devcontainer/docker-compose.yml up -d db pinecone qdrant ollama
```

Service ports (host):
- **Postgres**: `localhost:5432` (container exposes 8000/8080 internally; psql connects on 5432)
- **Pinecone Local**: `localhost:5081` (control-plane) and `localhost:5082` (data-plane)
- **Qdrant**: `localhost:6333` (HTTP), `6334` (gRPC)
- **Ollama**: run it on your host (`ollama serve` / app). If you want the containerized Ollama, uncomment the port mapping in `.devcontainer/docker-compose.yml` (11434).

### Env vars to export
```bash
export POSTGRES_USER=bookstore
export POSTGRES_PASSWORD=bookstore
export POSTGRES_DB=bookstore
export POSTGRES_HOST=localhost

export PINECONE_HOST=http://localhost:5081
export PINECONE_INDEX_HOST=http://localhost:5082

export QDRANT_HOST=localhost
export QDRANT_PORT=6333

# If running Ollama locally
export OLLAMA_HOST=http://localhost:11434
```

> Note: The Pinecone SDK requires a dot in the hostname for custom hosts. `pinecone_utils.get_pinecone_index_host()` auto-adds `pinecone.` when inside Docker. For local use, `localhost` is fine.

## 2) Python environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Initialize data
```bash
# Postgres tables + seed data
python setup/init_postgres.py

# Pinecone vectors (descriptions + book content chunks)
python setup/init_pinecone.py

# Optional: enhanced Module 1 SQL tables
python module_1/finding_meaning_enhanced/init_enhanced_postgres.py

# Qdrant recall vs latency demo (100k vectors)
python module_5/recall_vs_latency/init_qdrant.py --reset

# Optional: Qdrant books variant
python module_5/recall_vs_latency/init_qdrant_books.py --reset
```

## 4) Pull embedding / LLM models (Ollama)
```bash
ollama pull mxbai-embed-large   # embeddings used across demos
ollama pull llama3.1            # optional LLM for RAG synthesized answers
```
> If Ollama isn’t available, the code falls back to `sentence-transformers` automatically—slower but works.

## 5) Run the demos
| Module | Path | Command | Default URL |
|--------|------|---------|-------------|
| 1 | `module_1/finding_meaning_enhanced` | `python app.py` | http://localhost:8081 |
| 3 | `module_3/faceted_search` | `python app.py` | http://localhost:8001 |
| 4 | `module_4/rag_webapp` | `python app.py` | http://localhost:5001 |
| 4 | `module_4/recommendation_webapp` | `python app.py` | http://localhost:5001 (stop RAG first) |
| 5 | `module_5/recall_vs_latency` | `python app.py` | http://localhost:8080 |

> Some modules assume Docker DNS names (e.g., `pinecone`). Override with env vars above when running locally.

## Health checks
```bash
curl http://localhost:5081   # Pinecone control
curl http://localhost:5082/vectors   # Pinecone data plane (if exposed)
curl http://localhost:6333/healthz   # Qdrant
psql -h localhost -U bookstore -d bookstore -c "select count(*) from books;"
```

## Common issues
- **Connection refused (pinecone/qdrant)**: Ensure the ports are exposed; re-run `docker compose ps`.
- **Ollama not found**: Set `OLLAMA_HOST` to your host; confirm `ollama list` works.
- **SSL/hostname errors**: Double-check `PINECONE_INDEX_HOST`—use `http://localhost:5082`.
- **Port already in use**: Override with `PORT=9000 python app.py`.

## Cleanup
```bash
docker compose -f .devcontainer/docker-compose.yml down
rm -rf .venv
```

Happy hacking! 🚀