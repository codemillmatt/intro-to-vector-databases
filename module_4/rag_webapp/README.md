# RAG (Retrieval Augmented Generation)

Ask questions about books and get answers based on their actual content.

## Run

```bash
cd module_4/rag_webapp
python app.py
```

Open http://localhost:5001

## What It Shows

1. Select a book as your knowledge base
2. Ask questions about it
3. See the retrieved passages that answer your question
4. Get a synthesized answer (if Ollama is running)

## How RAG Works

```
Question: "What challenges does the protagonist face?"
    ↓
Convert to embedding
    ↓
Find similar text chunks in Pinecone (filtered by book_id)
    ↓
Pass chunks + question to LLM
    ↓
Answer grounded in actual book content
```

## With vs Without LLM

| With Ollama | Without Ollama |
|-------------|----------------|
| Get synthesized answers | See relevant passages directly |
| LLM summarizes the chunks | You read the source material |

Both modes demonstrate the retrieval step - the "R" in RAG.

## Key Concept: Scoped Retrieval

The demo filters by `book_id` so questions only search within the selected book:

```python
filter={"type": "chunk", "book_id": selected_book}
```

This pattern is essential for multi-tenant apps, access control, and focused answers.
