# Retrieval Augmented Generation (RAG)

A demonstration of the RAG pattern—one of the most powerful applications of vector databases in the age of Large Language Models (LLMs).

## What This Sample Does

This demo implements a question-answering system over a collection of books. Instead of relying solely on an LLM's training data, it:

1. **Retrieves** relevant passages from the book collection using vector similarity search
2. **Augments** the LLM prompt with this retrieved context
3. **Generates** an answer grounded in the actual source material

When you run the demo, you can:
- Ask natural language questions about the books
- See which passages were retrieved as relevant
- Receive answers based on the retrieved content
- Understand the three-step RAG process in action

## Why This Matters for Vector Databases

RAG has become the go-to pattern for building AI applications that need accurate, up-to-date, and verifiable responses. Vector databases are the critical infrastructure that makes RAG possible.

### The Problem RAG Solves

LLMs have significant limitations:
- **Knowledge cutoff**: Training data becomes stale
- **Hallucination**: Models confidently generate false information
- **No source attribution**: Difficult to verify claims
- **Context limits**: Can't process entire document collections

### How Vector Databases Enable RAG

| RAG Stage | Vector Database Role |
|-----------|---------------------|
| **Indexing** | Store document chunks as embeddings with metadata |
| **Retrieval** | Find semantically relevant passages in milliseconds |
| **Filtering** | Scope searches by document, date, category, etc. |
| **Ranking** | Return results ordered by similarity score |

### The Power of Semantic Search

Traditional keyword search fails when:
- Users phrase questions differently than source text
- Concepts are expressed with synonyms
- Understanding requires inference

Vector search succeeds because embeddings capture *meaning*, not just words. A question like "What happens at night in the garden?" will match passages about "midnight in the courtyard" because the concepts are semantically similar.

### Real-World RAG Applications
- **Enterprise knowledge bases**: Answer questions from internal documents
- **Customer support**: Ground responses in product documentation
- **Legal research**: Find relevant case law and statutes
- **Medical information**: Surface evidence from clinical literature
- **Code assistants**: Retrieve relevant documentation and examples

## Code Highlights

### Semantic Retrieval

The `retrieve_relevant_chunks()` function demonstrates the core RAG retrieval pattern:

```python
def retrieve_relevant_chunks(query: str, book_id: str | None = None, top_k: int = 3):
    # Convert question to embedding vector
    embedding_client = get_embedding_client()
    query_embedding = embedding_client.embed(query)
    
    # Build filter for text chunks
    filters = {"type": "chunk"}
    if book_id:
        filters["book_id"] = book_id
    
    # Vector similarity search
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filters
    )
```

**Key insight**: The same embedding model used to index documents must be used to embed queries. This ensures questions and answers exist in the same semantic space.

### Metadata Filtering for Scoped Search

Notice how filters enable targeted retrieval:

```python
filters = {"type": "chunk"}  # Only search text chunks, not book summaries
if book_id:
    filters["book_id"] = book_id  # Optionally limit to a specific book
```

This is crucial for:
- **Multi-tenant systems**: Ensure users only see their own data
- **Access control**: Respect permissions at retrieval time
- **Precision**: Narrow searches to relevant document sets

### Building Context for the LLM

Retrieved chunks are assembled into a context string:

```python
context = "\n\n---\n\n".join([
    f"From '{chunk['title']}' by {chunk['author']}:\n{chunk['text']}"
    for chunk in chunks
])
```

**Best practice**: Include source attribution in the context so the LLM can cite its sources in the response.

### Grounded Generation

The generation step uses a structured prompt:

```python
prompt = f"""Based on the following context from a book, answer the question.
If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
```

**Critical instruction**: Telling the model to admit when it can't find an answer prevents hallucination—a key benefit of RAG.

### Graceful Fallback

The demo handles missing LLM gracefully:

```python
if OLLAMA_AVAILABLE:
    response = ollama.chat(model="llama3.2:1b", messages=[...])
    return response["message"]["content"]

# Fallback: show retrieved context
return f"[LLM not available - showing retrieved context]\n\n{context}"
```

This demonstrates that retrieval alone provides value—users can read the relevant passages even without generation.

## Running the Demo

```bash
# From the project root
python module_4/rag/main.py
```

### With Ollama (Recommended)
For actual LLM-generated answers:
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2:1b
python module_4/rag/main.py
```

### Example Questions to Try
- "What happens in the garden at midnight?"
- "How does the physicist save reality?"
- "What is the lighthouse keeper's name?"
- "How does the AI become conscious?"

## Key Takeaways

1. **Retrieval before generation**: Always fetch relevant context first
2. **Semantic beats keyword**: Vector search finds conceptually relevant content
3. **Metadata enables precision**: Filters scope searches appropriately
4. **Ground truth matters**: Retrieved passages provide verifiable sources
5. **Chunking is critical**: Document segmentation affects retrieval quality
6. **Embedding consistency**: Use the same model for indexing and querying
7. **Instruct the LLM**: Tell it to stay grounded in the provided context

## The RAG Pipeline Visualized

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   User      │     │  Vector         │     │   LLM            │
│   Question  │────▶│  Database       │────▶│   (Generation)   │
└─────────────┘     │  (Retrieval)    │     └────────┬─────────┘
                    └─────────────────┘              │
                            │                        │
                            ▼                        ▼
                    ┌─────────────────┐     ┌──────────────────┐
                    │  Relevant       │     │  Grounded        │
                    │  Passages       │────▶│  Answer          │
                    └─────────────────┘     └──────────────────┘
```

RAG transforms LLMs from general-purpose text generators into knowledgeable assistants with access to your specific data—and vector databases make the retrieval fast, accurate, and scalable.
