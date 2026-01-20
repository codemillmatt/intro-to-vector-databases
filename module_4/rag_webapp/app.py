"""
Module 4: RAG Web Application

A web-based Retrieval Augmented Generation demo that allows users to:
1. Select a book to use as the knowledge base
2. Ask questions grounded in that book's content
3. View retrieved passages and (optionally) LLM-generated answers

This demonstrates a focused RAG pattern where the retrieval scope
is explicitly controlled by the user's book selection.
"""

import os
import sys

# Add setup directory to path for shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "setup"))

# Set Ollama host for devcontainer environment
os.environ.setdefault("OLLAMA_HOST", "http://ollama:11434")

from flask import Flask, render_template, request, jsonify, session

from embeddings import get_embedding_client
from pinecone_utils import get_pinecone_index

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")

# Check for Ollama availability
OLLAMA_AVAILABLE = False
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

try:
    import ollama
    # Test if Ollama is actually running
    ollama.list()
    OLLAMA_AVAILABLE = True
except Exception:
    pass


def get_available_books() -> list[dict]:
    """
    Get list of books that have full text chunks available for RAG.
    
    Returns books with text chunks indexed in Pinecone.
    """
    index = get_pinecone_index()
    
    # Query for chunk type to find books with text
    results = index.query(
        vector=[0.0] * 384,  # Dummy vector for metadata scan
        top_k=100,
        include_metadata=True,
        filter={"type": "chunk"}
    )
    
    # Get unique books
    books = {}
    for match in results.matches:
        book_id = match.metadata.get("book_id", "")
        if book_id and book_id not in books:
            books[book_id] = {
                "id": book_id,
                "title": match.metadata.get("title", "Unknown Title"),
                "author": match.metadata.get("author", "Unknown Author")
            }
    
    return list(books.values())


def retrieve_passages(query: str, book_id: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve relevant text passages from a specific book.
    
    This is the core retrieval step in RAG - finding semantically
    similar passages to the user's question.
    """
    embedding_client = get_embedding_client()
    query_embedding = embedding_client.embed(query)
    
    # Filter to only chunks from the selected book
    filters = {
        "type": "chunk",
        "book_id": book_id
    }
    
    index = get_pinecone_index()
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filters
    )
    
    passages = []
    for match in results.matches:
        passages.append({
            "text": match.metadata.get("text", ""),
            "chunk_index": match.metadata.get("chunk_index", 0),
            "score": round(match.score, 3),
            "title": match.metadata.get("title", ""),
            "author": match.metadata.get("author", "")
        })
    
    return passages


def generate_answer(query: str, passages: list[dict], book_title: str) -> dict:
    """
    Generate an answer using retrieved passages.
    
    If Ollama is available, uses an LLM to synthesize an answer.
    Otherwise, returns the passages as a fallback.
    """
    # Build context from passages
    context_parts = []
    for i, passage in enumerate(passages, 1):
        context_parts.append(f"[Passage {i}]\n{passage['text']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    if not OLLAMA_AVAILABLE:
        return {
            "answer": None,
            "llm_available": False,
            "message": "No LLM available. Showing retrieved passages that may answer your question."
        }
    
    # Build the RAG prompt
    prompt = f"""You are a helpful assistant answering questions about the book "{book_title}".

Based ONLY on the following passages from the book, answer the user's question.
If the answer cannot be found in the passages, say "I couldn't find the answer in the provided passages."
Do not make up information that isn't in the passages.

PASSAGES:
{context}

QUESTION: {query}

ANSWER:"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            "answer": response["message"]["content"],
            "llm_available": True,
            "model": OLLAMA_MODEL
        }
    except Exception as e:
        return {
            "answer": None,
            "llm_available": False,
            "message": f"LLM error: {str(e)}. Showing retrieved passages instead."
        }


@app.route("/")
def index():
    """Render the main RAG interface."""
    books = get_available_books()
    return render_template(
        "index.html",
        books=books,
        llm_available=OLLAMA_AVAILABLE,
        llm_model=OLLAMA_MODEL if OLLAMA_AVAILABLE else None
    )


@app.route("/api/books")
def api_get_books():
    """Get available books for RAG."""
    books = get_available_books()
    return jsonify({
        "books": books,
        "llm_available": OLLAMA_AVAILABLE
    })


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """
    Process a question using RAG.
    
    Expects JSON: { "question": "...", "book_id": "..." }
    """
    data = request.get_json()
    
    question = data.get("question", "").strip()
    book_id = data.get("book_id", "").strip()
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    if not book_id:
        return jsonify({"error": "Please select a book first"}), 400
    
    # Step 1: Retrieve relevant passages
    passages = retrieve_passages(question, book_id, top_k=3)
    
    if not passages:
        return jsonify({
            "question": question,
            "passages": [],
            "answer": None,
            "message": "No relevant passages found in the selected book."
        })
    
    # Get book title for the response
    book_title = passages[0]["title"] if passages else "Unknown"
    
    # Step 2: Generate answer (or fallback)
    generation_result = generate_answer(question, passages, book_title)
    
    return jsonify({
        "question": question,
        "book_id": book_id,
        "book_title": book_title,
        "passages": passages,
        "answer": generation_result.get("answer"),
        "llm_available": generation_result.get("llm_available", False),
        "llm_model": generation_result.get("model"),
        "message": generation_result.get("message")
    })


@app.route("/api/status")
def api_status():
    """Check system status including LLM availability."""
    return jsonify({
        "llm_available": OLLAMA_AVAILABLE,
        "llm_model": OLLAMA_MODEL if OLLAMA_AVAILABLE else None,
        "embedding_model": "all-MiniLM-L6-v2"
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    
    print(f"\n🔍 RAG Web Application")
    print(f"   LLM Status: {'✅ ' + OLLAMA_MODEL if OLLAMA_AVAILABLE else '❌ Not available (fallback mode)'}")
    print(f"   Running on: http://localhost:{port}\n")
    
    app.run(host="0.0.0.0", port=port, debug=debug)
