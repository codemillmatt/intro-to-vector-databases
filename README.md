# Introduction to Vector Databases

Hands-on demos for learning vector database fundamentals through a bookstore example.

## Quick Start

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Clone this repository and open in VS Code
3. Click "Reopen in Container" when prompted
4. Wait for setup to complete (databases initialize automatically)

That's it! You're ready to run the demos.

## Demos

| Module | Demo | What It Shows | Run Command |
|--------|------|---------------|-------------|
| 1 | [SQL vs Semantic Search](module_1/finding_meaning_webapp/) | Why vector search beats keyword search | `cd module_1/finding_meaning_webapp && python app.py` |
| 3 | [Faceted Search](module_3/faceted_search/) | Combining filters with semantic search | `cd module_3/faceted_search && python app.py` |
| 4 | [RAG](module_4/rag_webapp/) | Question answering over book content | `cd module_4/rag_webapp && python app.py` |
| 4 | [Recommendations](module_4/recommendation_webapp/) | Content + collaborative filtering | `cd module_4/recommendation_webapp && python app.py` |
| 5 | [Performance Tuning](module_5/tuning/) | How parameters affect latency & recall | `cd module_5/tuning && python app.py` |

After running a command, open the URL shown in your browser (usually `http://localhost:8080` or similar).

## What You'll Learn

- **Module 1**: Vector databases understand *meaning*, not just keywords
- **Module 3**: Combine semantic search with metadata filters (genre, price, rating)
- **Module 4**: Build real applications - RAG chatbots and recommendation engines
- **Module 5**: Tune performance for your specific use case

## Technologies

- **Pinecone** - Vector database
- **PostgreSQL** - Traditional database (for comparison)
- **Python/Flask** - Web applications
- **Ollama** - Local embedding model

## Troubleshooting

**Databases not initialized?**
```bash
cd setup && python init_postgres.py && python init_pinecone.py
```

**Need to reset everything?**
```bash
cd setup
python init_postgres.py --reset
python init_pinecone.py --reset
```

**Port already in use?**
Set a different port: `PORT=9000 python app.py`

## License

See [LICENSE](LICENSE) for details.
