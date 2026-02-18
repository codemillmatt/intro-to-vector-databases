"""
Module 4: RAG Web Application

A web-based Retrieval Augmented Generation demo that allows users to:
1. Select a book to use as the knowledge base
2. Ask questions grounded in that book's content
3. View retrieved passages and (optionally) LLM-generated answers

This demonstrates a focused RAG pattern where the retrieval scope
is explicitly controlled by the user's book selection.
"""

import json
import os
import re
import sys
from glob import glob

# ---------------------------------------------------------------------------
# Allow Python to find the shared utilities in the setup/ directory.
# This adds setup/ to the module search path so we can write:
#     from embeddings import get_embedding_client
# instead of dealing with complex relative imports.
# ---------------------------------------------------------------------------
SETUP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "setup")
sys.path.insert(0, SETUP_DIR)

# In the DevContainer, OLLAMA_HOST is already set by docker-compose.yml
# (http://host.docker.internal:11434). This fallback is only used when
# running outside the DevContainer without setting the variable.
os.environ.setdefault("OLLAMA_HOST", "http://ollama:11434")

from flask import Flask, render_template, request, jsonify, Response

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

# ---------------------------------------------------------------------------
# Cached singletons — initialized once at startup, reused on every request
# ---------------------------------------------------------------------------
print("⏳ Initializing embedding client (singleton)...")
_embedding_client = get_embedding_client()

# Pre-warm the embedding model so the first real query is fast
print("⏳ Pre-warming embedding model...")
_embedding_client.embed("warmup")
print("✅ Embedding model ready")

print("⏳ Connecting to Pinecone (singleton)...")
_pinecone_index = get_pinecone_index()
print("✅ Pinecone index ready")

# ---------------------------------------------------------------------------
# Preload book catalog and full texts from disk (small files, ~5 KB each)
# ---------------------------------------------------------------------------
BOOK_TEXTS_DIR = os.path.join(SETUP_DIR, "data", "book_texts")
BOOKS_JSON = os.path.join(SETUP_DIR, "data", "books.json")


def _md_to_html(md_text: str) -> str:
    """Minimal Markdown-to-HTML for the book texts (headings, bold, italic, paragraphs)."""
    lines = md_text.split("\n")
    html_parts: list[str] = []
    for line in lines:
        stripped = line.rstrip()
        # Headings
        if stripped.startswith("### "):
            html_parts.append(f"<h3>{stripped[4:]}</h3>")
        elif stripped.startswith("## "):
            html_parts.append(f"<h2>{stripped[3:]}</h2>")
        elif stripped.startswith("# "):
            html_parts.append(f"<h1>{stripped[2:]}</h1>")
        elif stripped == "---":
            html_parts.append("<hr>")
        elif stripped == "":
            html_parts.append("")
        else:
            html_parts.append(f"<p>{stripped}</p>")
    html = "\n".join(html_parts)
    # Inline formatting: bold then italic
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
    return html


def _load_books_and_texts() -> tuple[list[dict], dict[str, str]]:
    """Load the book catalog (filtered to those with full text) and their HTML content."""
    # Discover which book IDs have full-text files
    text_files = glob(os.path.join(BOOK_TEXTS_DIR, "book_*.md"))
    available_ids: set[str] = set()
    texts: dict[str, str] = {}
    for path in text_files:
        book_id = os.path.splitext(os.path.basename(path))[0]  # e.g. "book_001"
        with open(path, "r") as f:
            texts[book_id] = _md_to_html(f.read())
        available_ids.add(book_id)

    # Load catalog metadata, keep only books with full text
    with open(BOOKS_JSON, "r") as f:
        all_books = json.load(f)
    books = [
        {"id": b["id"], "title": b["title"], "author": b["author"]}
        for b in all_books
        if b["id"] in available_ids
    ]
    return books, texts


print("⏳ Loading book catalog and full texts...")
CACHED_BOOKS, CACHED_BOOK_TEXTS = _load_books_and_texts()
print(f"✅ {len(CACHED_BOOKS)} books loaded with full text")


def retrieve_passages(query: str, book_id: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve relevant text passages from a specific book.
    
    This is the core retrieval step in RAG - finding semantically
    similar passages to the user's question.
    Uses the cached singleton embedding client and Pinecone index.
    """
    query_embedding = _embedding_client.embed(query)
    
    # Filter to only chunks from the selected book
    filters = {
        "type": "chunk",
        "book_id": book_id
    }
    
    results = _pinecone_index.query(
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


def _build_rag_prompt(query: str, passages: list[dict], book_title: str) -> str:
    """Build the RAG prompt from retrieved passages."""
    context_parts = []
    for i, passage in enumerate(passages, 1):
        context_parts.append(f"[Passage {i}]\n{passage['text']}")
    context = "\n\n---\n\n".join(context_parts)

    return f"""You are a helpful assistant answering questions about the book "{book_title}".

Based ONLY on the following passages from the book, answer the user's question.
If the answer cannot be found in the passages, say "I couldn't find the answer in the provided passages."
Do not make up information that isn't in the passages.
Keep your answer concise — ideally 2-4 sentences.

PASSAGES:
{context}

QUESTION: {query}

ANSWER:"""


def _build_direct_prompt(query: str, book_title: str) -> str:
    """Build a prompt that sends the question directly to the LLM without any
    retrieved context.  This lets learners compare RAG-grounded answers against
    the LLM's general knowledge to see how retrieval improves accuracy."""
    return f"""You are a helpful assistant. Answer the following question using only your general knowledge. Keep your answer concise — ideally 2-4 sentences.

QUESTION: {query}

ANSWER:"""


@app.route("/")
def index():
    """Render the main RAG interface."""
    return render_template(
        "index.html",
        books=CACHED_BOOKS,
        llm_available=OLLAMA_AVAILABLE,
        llm_model=OLLAMA_MODEL if OLLAMA_AVAILABLE else None
    )


@app.route("/api/books")
def api_get_books():
    """Get available books for RAG."""
    return jsonify({
        "books": CACHED_BOOKS,
        "llm_available": OLLAMA_AVAILABLE
    })


@app.route("/api/book-text/<book_id>")
def api_book_text(book_id):
    """Return the full text of a book as pre-rendered HTML."""
    html = CACHED_BOOK_TEXTS.get(book_id)
    if html is None:
        return jsonify({"error": "Book text not found"}), 404
    return jsonify({"book_id": book_id, "html": html})


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """
    Process a question using RAG with Server-Sent Events.

    Streams the response in stages:
      1. 'passages' event — retrieved passages (arrives fast)
      2. 'token' events — LLM answer streamed token-by-token
      3. 'done' event — signals the stream is complete
    """
    data = request.get_json()
    question = data.get("question", "").strip()
    book_id = data.get("book_id", "").strip()
    use_rag = data.get("use_rag", True)

    if not question:
        return jsonify({"error": "Question is required"}), 400
    if not book_id:
        return jsonify({"error": "Please select a book first"}), 400

    # Look up book title from the catalog so both modes can reference it
    book_title = "Unknown"
    for b in CACHED_BOOKS:
        if b["id"] == book_id:
            book_title = b["title"]
            break

    def generate():
        nonlocal book_title
        import json as _json

        # ----- Direct LLM mode (no RAG) -----
        if not use_rag:
            yield f"data: {_json.dumps({'type': 'mode', 'mode': 'direct'})}\n\n"

            if not OLLAMA_AVAILABLE:
                yield f"data: {_json.dumps({'type': 'done', 'llm_available': False, 'message': 'No LLM available. Direct mode requires an LLM.'})}\n\n"
                return

            prompt = _build_direct_prompt(question, book_title)
            try:
                stream = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )
                for chunk in stream:
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield f"data: {_json.dumps({'type': 'token', 'token': token})}\n\n"
            except Exception as e:
                yield f"data: {_json.dumps({'type': 'error', 'message': str(e)})}\n\n"

            yield f"data: {_json.dumps({'type': 'done', 'llm_available': True, 'model': OLLAMA_MODEL})}\n\n"
            return

        # ----- RAG mode (retrieve then generate) -----
        # Step 1: Retrieve passages (fast — cached singletons)
        passages = retrieve_passages(question, book_id, top_k=3)
        if passages:
            book_title = passages[0]["title"]

        yield f"data: {_json.dumps({'type': 'passages', 'passages': passages, 'book_title': book_title})}\n\n"

        if not passages:
            yield f"data: {_json.dumps({'type': 'done', 'message': 'No relevant passages found in the selected book.'})}\n\n"
            return

        # Step 2: Stream LLM answer token-by-token
        if not OLLAMA_AVAILABLE:
            yield f"data: {_json.dumps({'type': 'done', 'llm_available': False, 'message': 'No LLM available. Showing retrieved passages that may answer your question.'})}\n\n"
            return

        prompt = _build_rag_prompt(question, passages, book_title)
        try:
            stream = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in stream:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield f"data: {_json.dumps({'type': 'token', 'token': token})}\n\n"
        except Exception as e:
            yield f"data: {_json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        yield f"data: {_json.dumps({'type': 'done', 'llm_available': True, 'model': OLLAMA_MODEL})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/status")
def api_status():
    """Check system status including LLM availability."""
    return jsonify({
        "llm_available": OLLAMA_AVAILABLE,
        "llm_model": OLLAMA_MODEL if OLLAMA_AVAILABLE else None,
        "embedding_model": "all-MiniLM-L6-v2"
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8008))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    
    print(f"\n🔍 RAG Web Application")
    print(f"   LLM Status: {'✅ ' + OLLAMA_MODEL if OLLAMA_AVAILABLE else '❌ Not available (fallback mode)'}")
    print(f"   Running on: http://localhost:{port}\n")
    
    app.run(host="0.0.0.0", port=port, debug=debug)
