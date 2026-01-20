# RAG Web Application

A web-based Retrieval Augmented Generation (RAG) demo that provides a focused, book-centric question-answering experience.

## What This Sample Does

This demo implements a chat-style interface where users:

1. **Select a book** from available options to use as their knowledge base
2. **Ask questions** about that specific book
3. **View retrieved passages** that are semantically relevant to their question
4. **Receive synthesized answers** (when an LLM is available) or browse the passages directly

Unlike the CLI-based RAG demo, this web application:
- Provides a visual, user-friendly interface
- Scopes all questions to a single selected book
- Shows the retrieval process transparently
- Gracefully handles the absence of an LLM

## Why This Matters for Vector Databases

This sample demonstrates several key RAG concepts that are critical in production systems:

### Scoped Retrieval

Real-world RAG applications often need to limit search scope:
- **Multi-tenant systems**: Users should only query their own documents
- **Access control**: Sensitive documents need permission-based retrieval  
- **Context focus**: Better answers come from focused, relevant sources

This demo uses metadata filtering to scope queries to the selected book:

```python
filters = {
    "type": "chunk",
    "book_id": book_id  # Only retrieve from selected book
}
```

### Transparency in AI

The UI shows users exactly which passages informed the answer:
- Builds trust by showing source material
- Allows users to verify information
- Demonstrates the "R" in RAG isn't a black box

### Graceful Degradation

When no LLM is available, the system still provides value:
- Users see relevant passages directly
- The retrieval step works independently
- Applications remain functional without expensive LLM infrastructure

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Web Browser                          │
│  ┌─────────────┐    ┌─────────────────────────────────────┐ │
│  │ Book Select │    │         Chat Interface              │ │
│  └─────────────┘    │  ┌─────────────────────────────┐    │ │
│                     │  │ Question → Passages → Answer │    │ │
│                     │  └─────────────────────────────┘    │ │
│                     └─────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      Flask Backend                          │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │ /api/books     │  │ /api/ask       │  │ /api/status   │  │
│  └────────────────┘  └────────┬───────┘  └───────────────┘  │
│                               │                              │
│         ┌─────────────────────┼─────────────────────┐        │
│         │                     │                     │        │
│         ▼                     ▼                     ▼        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  Embeddings  │    │   Pinecone   │    │    Ollama    │   │
│  │  (Query)     │───▶│  (Retrieve)  │    │  (Generate)  │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Code Highlights

### Scoped Vector Search

The retrieval function filters by book ID to ensure focused results:

```python
def retrieve_passages(query: str, book_id: str, top_k: int = 3) -> list[dict]:
    embedding_client = get_embedding_client()
    query_embedding = embedding_client.embed(query)
    
    # Metadata filter ensures we only search within the selected book
    filters = {
        "type": "chunk",
        "book_id": book_id
    }
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filters
    )
```

### RAG Prompt Engineering

The prompt explicitly grounds the LLM in retrieved content:

```python
prompt = f"""You are a helpful assistant answering questions about the book "{book_title}".

Based ONLY on the following passages from the book, answer the user's question.
If the answer cannot be found in the passages, say "I couldn't find the answer in the provided passages."
Do not make up information that isn't in the passages.

PASSAGES:
{context}

QUESTION: {query}

ANSWER:"""
```

Key elements:
- **Role definition**: Focuses the model on the specific book
- **Grounding instruction**: "Based ONLY on the following passages"
- **Honesty instruction**: Admit when information isn't available
- **Anti-hallucination**: "Do not make up information"

### LLM Availability Detection

The app gracefully handles missing LLM infrastructure:

```python
OLLAMA_AVAILABLE = False
try:
    import ollama
    ollama.list()  # Test if Ollama is actually running
    OLLAMA_AVAILABLE = True
except Exception:
    pass
```

This pattern allows the demo to work in any environment.

## Running the Demo

### Basic (Without LLM)

```bash
# From the project root
cd module_4/rag_webapp
python app.py
```

Open http://localhost:5001 in your browser. The app will run in fallback mode, showing retrieved passages without LLM-generated answers.

### With LLM (Ollama)

```bash
# Install Ollama from https://ollama.ai
# Then pull a model:
ollama pull llama3.2:1b

# Run the app
cd module_4/rag_webapp
python app.py
```

The status badge will show the active model, and you'll receive synthesized answers.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5001` | Web server port |
| `FLASK_DEBUG` | `true` | Enable debug mode |
| `FLASK_SECRET_KEY` | (dev key) | Session secret |
| `OLLAMA_MODEL` | `llama3.2:1b` | Ollama model to use |

## Prerequisites

Ensure the Pinecone index has been initialized with book text chunks:

```bash
cd setup
python init_pinecone.py
```

This loads both book metadata and text chunks needed for RAG.

## User Flow

1. **Select a Book**: Click one of the available books to set it as your knowledge base
2. **Ask Questions**: Type questions in the chat input
3. **View Results**: See the answer and expand to view the source passages
4. **Change Book**: Click "Change Book" to select a different knowledge base

## Key Takeaways

1. **Scoped retrieval improves accuracy**: Limiting search to relevant documents produces better results
2. **Metadata filters are powerful**: Vector databases enable precise filtering alongside semantic search
3. **Show your sources**: Transparency builds trust in AI-generated answers
4. **Design for degradation**: Applications should work even when components are unavailable
5. **Prompt engineering matters**: Clear instructions prevent hallucination
6. **Web UIs democratize access**: Non-technical users can benefit from RAG without CLI knowledge
