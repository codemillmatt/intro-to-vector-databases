# RAG (Retrieval Augmented Generation)
Ask questions about books and get answers grounded in the **actual text**, not just a generic LLM reply.

## 🧠 What this app demonstrates
- **Retrieval**: Embed the question, pull top-matching chunks from Pinecone.
- **Augmentation**: Provide retrieved passages to the LLM for context.
- **Generation**: Synthesize an answer using the provided evidence.
- **Scoped retrieval**: filter by `book_id` to keep results tenant/book-specific.

## 🚀 Run (DevContainer/Codespaces preferred)
```bash
# One-time data init (if not already done)
cd setup
python init_pinecone.py

# Run the app
cd ../module_4/rag_webapp
python app.py   # http://localhost:5001
```
> 🤖 **LLM/embeddings**: Easiest path is local **Ollama** (`OLLAMA_HOST=http://localhost:11434`). Pull models:
> ```bash
> ollama pull mxbai-embed-large   # embeddings
> ollama pull llama3.1           # optional for synthesized answers
> ```
> If Ollama isn’t available, retrieval still works; the app falls back to `sentence-transformers` for embeddings and shows the retrieved chunks so you can read the source directly.

## 🔍 How RAG flows
```
Question → embed → Pinecone query (filter by book_id) → top-k chunks
                                 ↓
                         LLM (optional)
                                 ↓
                    Grounded answer + citations
```

## 🧪 Try this
1. Pick a book from the dropdown.
2. Ask: `What challenge does the protagonist face?`
3. Observe retrieved passages and the synthesized answer (if LLM enabled).
4. Toggle to a different book and note how filtering avoids cross-book leakage.

## 📦 Key pattern: scoped retrieval
```python
filter={"type": "chunk", "book_id": selected_book}
```
Use this for multi-tenancy, access control, or user-specific partitions.

## Reset / re-init
```bash
cd setup && python init_pinecone.py --reset
```
