# Copilot Instructions for Introduction to Vector Databases

## Project Overview

This is an educational repository demonstrating vector database fundamentals through a bookstore/library storyline. The project compares vector databases against traditional databases and shows real-world applications like semantic search, RAG (Retrieval Augmented Generation), and product recommendations.

## Tech Stack

- **Python 3.x** - Primary programming language
- **Pinecone Local** - Vector database (containerized for development)
- **PostgreSQL** - Traditional database for comparison demonstrations
- **Ollama** - Local embedding model server
- **sentence-transformers** - Fallback embedding models
- **Flask** - Web framework for interactive demos
- **DevContainers** - Consistent development environment

## Development Environment

### DevContainer Requirement

**IMPORTANT**: This project MUST run inside the DevContainer. All code assumes DevContainer networking and service availability.

- Docker Compose automatically starts: Postgres, Pinecone Local, and Ollama services
- Environment variables are pre-configured for container networking
- Do NOT assume code can run directly on the host machine

### Environment Variables

Default values are for DevContainer environment:

- `PINECONE_HOST=http://pinecone:5081` (control plane)
- `PINECONE_INDEX_HOST=http://pinecone.:5082` (data plane, note the trailing dot for SDK validation)
- `OLLAMA_HOST=http://ollama:11434`

**Never use `localhost` in container code** - it refers to the container itself, not host services.

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

## Coding Conventions

### Python Style

- Follow PEP 8 style guidelines
- Use meaningful variable names that reflect the educational context
- Add docstrings to functions that explain concepts clearly for learners
- Keep code simple and readable - this is teaching material
- Use type hints where they improve clarity

### Dependencies

- Use the existing dependencies in `requirements.txt`
- Prefer `sentence-transformers` as fallback when Ollama is unavailable
- Always handle connection failures gracefully with informative error messages

### Error Handling

- Provide clear, educational error messages that help learners troubleshoot
- Include suggestions for common fixes (e.g., "Make sure you're running in DevContainer")
- Handle missing environment variables with helpful defaults or clear instructions

## Building and Testing

### Database Initialization

Before running any demos, databases must be initialized:

```bash
cd setup
python init_postgres.py
python init_pinecone.py
```

Reset databases:

```bash
cd setup
python init_postgres.py --reset
python init_pinecone.py --reset
```

### Running Demos

Each module has its own executable demo:

- **Module 1**: `cd module_1/finding_meaning && python main.py`
- **Module 3**: `cd module_3/metadata_filtering && python main.py`
- **Module 4 RAG**: `cd module_4/rag && python main.py`
- **Module 4 Recommendations**: `cd module_4/product_recommendation && python main.py`
- **Module 5**: `cd module_5/tuning && python app.py` (Flask web app on port 8000)

### Testing

- No formal test suite exists - this is demo/teaching code
- Manual verification by running demos is the validation approach
- When making changes, verify by running the affected module's demo

## Common Patterns

### Embedding Generation

Shared utility in `setup/embeddings.py`:

```python
from setup.embeddings import get_embeddings

# Generate embeddings for text
embeddings = get_embeddings(["text to embed"])
```

Falls back from Ollama to sentence-transformers automatically.

### Database Connections

- PostgreSQL: Use `psycopg2` with environment-based config
- Pinecone: Use official `pinecone` SDK (not `pinecone-client`)

## What to Focus On

### Good Tasks for This Repository

- Fixing bugs in demo scripts
- Improving error messages and user feedback
- Adding new educational examples
- Updating documentation for clarity
- Refactoring for better code readability

### Avoid

- Breaking DevContainer compatibility
- Changing the core educational flow/storyline
- Adding complex production-level features (this is teaching code)
- Requiring additional external services
- Making demos non-interactive or harder to understand

## Troubleshooting Tips

### Common Issues

1. **Pinecone errors**: Check `PINECONE_HOST` and `PINECONE_INDEX_HOST` are set correctly
2. **Ollama connection fails**: Service might not be running; code should fall back to sentence-transformers
3. **PostgreSQL connection fails**: Ensure `init_postgres.py` ran successfully
4. **Module import errors**: Scripts expect to be run from their own directory

### Debug Approach

- Run commands from inside DevContainer
- Check Docker Compose logs: `docker compose logs <service>`
- Verify services are running: `docker compose ps`
- Test connectivity: `curl http://pinecone:5081` or similar

## Educational Context

This is teaching material for a course on vector databases. When making changes:

- Keep explanations clear and beginner-friendly
- Preserve the narrative flow (bookstore/library theme)
- Ensure demos remain self-contained and runnable
- Comments should teach concepts, not just describe code
- Error messages should be learning opportunities
