# Introduction to Vector Databases

Demo code for the "Introduction to Vector Databases" course. This repository contains hands-on demonstrations that teach the fundamentals of vector databases through a bookstore/library storyline.

## Overview

This course demonstrates the value of vector databases and their features, compares them against traditional databases, shows how to tune for recall and latency, and provides real-world applications showing where vector databases fit in intelligent applications.

## Getting Started

### Prerequisites

This project uses DevContainers to provide a consistent development environment. You will need:

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Quick Start

1. Clone this repository
2. Open in VS Code
3. When prompted, click "Reopen in Container" (or use Command Palette: "Dev Containers: Reopen in Container")
4. Wait for the container to build and dependencies to install
5. Initialize the databases:
   ```bash
   cd setup
   python init_postgres.py
   python init_pinecone.py
   ```

Known-good init command (DevContainer):

```bash
export PINECONE_HOST=http://pinecone:5081
export PINECONE_INDEX_HOST=http://pinecone.:5082
export OLLAMA_HOST=http://ollama:11434

cd setup
python init_postgres.py
python init_pinecone.py
```

### Services & Environment Variables

This repo is designed to run inside the DevContainer, where Docker Compose starts these services:

- **Postgres** (database)
- **Pinecone Local** (vector DB)
- **Ollama** (optional embedding server)

The Python code reads configuration from environment variables. Defaults are chosen for the DevContainer.

- `PINECONE_HOST`
   - DevContainer default: `http://pinecone:5081`
   - If running from your host machine (outside the container): `http://localhost:5081`
- `PINECONE_INDEX_HOST` (optional override for Pinecone “data plane”)
   - DevContainer recommended override (only if needed): `http://pinecone.:5082`
   - Host-machine override (only if needed): `http://localhost:5082`
- `OLLAMA_HOST`
   - DevContainer default: `http://ollama:11434`
   - Host-machine default: `http://localhost:11434`

Notes:

- Pinecone Local typically uses **5081** for control-plane operations and **5082** for vector upserts/queries.
- The Pinecone SDK validates custom hosts by requiring a dot (`.`) in the hostname. In Docker DNS, `pinecone.` (with a trailing dot) resolves correctly and satisfies this check.

## Project Structure

```
├── .devcontainer/          # DevContainer configuration
├── setup/                  # Database setup scripts and mock data
│   ├── data/              # Book catalog, user data, and book texts
│   ├── embeddings.py      # Shared embedding utilities
│   ├── init_postgres.py   # PostgreSQL initialization
│   └── init_pinecone.py   # Pinecone initialization
├── module_1/              # Finding Meaning - SQL vs Vector Search
├── module_3/              # Metadata Filtering
├── module_4/              # RAG and Product Recommendations
│   ├── rag/
│   └── product_recommendation/
└── module_5/              # Tuning, Recall, and Latency
```

## Modules

### Module 1: Finding Meaning

**Demo: SQL vs Semantic Search**

Compares traditional SQL keyword-based search against semantic vector similarity search, demonstrating the immediate value proposition of vector databases.

```bash
cd module_1/finding_meaning
python main.py
```

### Module 2: Embeddings (No demos)

This module covers the theory of how embedding models transform text into numerical vectors.

### Module 3: Vector Database Architecture

**Demo: Metadata Filtering**

Extends the semantic search from Module 1 with metadata filters, showing how to narrow results by genre, author, price range, etc.

```bash
cd module_3/metadata_filtering
python main.py
```

### Module 4: Real-World Use Cases

**Demo: RAG (Retrieval Augmented Generation)**

Ask questions about books using the RAG pattern. The demo loads full book texts, chunks them, and uses semantic search to find relevant passages for answering questions.

```bash
cd module_4/rag
python main.py
```

**Demo: Product Recommendations**

A basic recommendation engine that suggests books based on user reading preferences and habits of similar users.

```bash
cd module_4/product_recommendation
python main.py
```

### Module 5: Engineering Strategy

**Demo: Tuning, Recall, and Latency**

A web interface with sliders to control vector database parameters, showing how tuning affects query latency and result quality.

```bash
cd module_5/tuning
python app.py
# Open http://localhost:8000 in your browser
```

## Technologies

- **Python** - Primary programming language
- **Pinecone Local** - Vector database (containerized for development)
- **PostgreSQL** - Traditional database for comparison
- **Ollama** - Local embedding model server
- **sentence-transformers** - Fallback embedding models
- **Flask** - Web framework for the tuning demo
- **DevContainers** - Consistent development environment

## Resetting Databases

To reset the databases and start fresh:

```bash
cd setup
python init_postgres.py --reset
python init_pinecone.py --reset
```

## Troubleshooting

### Error: package renamed from `pinecone-client` to `pinecone`

If you see an exception like:

> The official Pinecone python package has been renamed from `pinecone-client` to `pinecone`...

It means your active Python environment has `pinecone-client` installed.

Fix (run inside the DevContainer):

```bash
python -m pip uninstall -y pinecone-client
python -m pip install -U pinecone
```

Also confirm your dependencies do not list `pinecone-client` (this repo uses `pinecone`).

### Error: failed to connect to Ollama

If embedding generation fails with an Ollama connection error:

- Make sure the `ollama` service is running in Docker Compose (DevContainer handles this by default).
- If you’re running outside the container, set:

```bash
export OLLAMA_HOST=http://localhost:11434
```

This repo also supports a fallback to `sentence-transformers` (it may download a model the first time it runs).

### Error: Pinecone upsert/query tries `localhost:5082`

Inside containers, `localhost` points at the container itself, not the Pinecone Local service. If you see retries against `localhost:5082` or connection refused, set:

```bash
export PINECONE_HOST=http://pinecone:5081
export PINECONE_INDEX_HOST=http://pinecone.:5082
```

Then rerun:

```bash
cd setup
python init_pinecone.py
```

## License

See the [LICENSE](LICENSE) file for details.
