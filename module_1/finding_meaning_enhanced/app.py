"""
Module 1: Finding Meaning (Enhanced) - Web Application

A web-based demo comparing PostgreSQL full-text search (tsvector/tsquery)
against semantic vector similarity search. Unlike the standard demo, this
version:
  - Removes the "Browse All Books" tab — focuses purely on search comparison.
  - Uses PostgreSQL full-text search (ts_rank) instead of simple ILIKE.
  - Queries both book-description AND book-text-chunk vectors in Pinecone,
    aggregating the best match per book for richer semantic results.

Prerequisites:
    1. Run  setup/init_postgres.py       (creates the books table)
    2. Run  setup/init_pinecone.py       (creates description + chunk vectors)
    3. Run  module_1/finding_meaning_enhanced/init_enhanced_postgres.py
       (adds full_text column and GIN index)
"""

import os
import sys

# Add setup directory to path for shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "setup"))

import psycopg2
from flask import Flask, render_template, request, jsonify

from embeddings import get_embedding_client
from pinecone_utils import get_pinecone_index

app = Flask(__name__)

# ── Singletons ──────────────────────────────────────────────────────────
_embedding_client = get_embedding_client()
_pinecone_index = get_pinecone_index()

# Pre-warm the embedding model so the first user query isn't slow
_embedding_client.embed("warmup")

# ── PostgreSQL config ───────────────────────────────────────────────────
POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "database": os.getenv("POSTGRES_DB", "bookstore"),
    "user": os.getenv("POSTGRES_USER", "bookstore"),
    "password": os.getenv("POSTGRES_PASSWORD", "bookstore"),
}


# ── Search functions ────────────────────────────────────────────────────

def sql_search(query: str) -> list[dict]:
    """
    Search books using plain SQL ILIKE substring matching.

    Searches across title, description, genre, and full book text for
    an exact (case-insensitive) substring match. No stemming, no
    tokenization — "foxes" will NOT match "fox". This deliberately
    shows the limitations of keyword-based search compared to semantic
    vector search which understands meaning.
    """
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    try:
        with conn.cursor() as cur:
            sql = """
                SELECT id, title, author, description, genre, rating
                FROM books
                WHERE title ILIKE %s
                   OR description ILIKE %s
                   OR genre ILIKE %s
                   OR full_text ILIKE %s
                ORDER BY rating DESC
                LIMIT 5
            """
            pattern = f"%{query}%"
            cur.execute(sql, (pattern, pattern, pattern, pattern))

            columns = ["id", "title", "author", "description", "genre", "rating"]
            results = []
            for row in cur.fetchall():
                d = dict(zip(columns, row))
                d["rating"] = float(d["rating"]) if d["rating"] else 0.0
                results.append(d)
            return results
    finally:
        conn.close()


def semantic_search(query: str) -> list[dict]:
    """
    Search books using semantic vector similarity across descriptions
    AND full book text chunks.

    Queries Pinecone with no type filter (top_k=15) to capture matches
    from both book-description vectors and book-text-chunk vectors.
    Results are aggregated per book, keeping the highest scoring match
    and tracking whether it came from the description or text content.
    """
    query_embedding = _embedding_client.embed(query)
    index = _pinecone_index

    # Query without type filter to search both descriptions and text chunks.
    # top_k=15 gives headroom for deduplication across chunks of the same book.
    results = index.query(
        vector=query_embedding,
        top_k=15,
        include_metadata=True,
    )

    # Separate book-level and chunk-level matches so we can build
    # a complete metadata picture for every book.
    book_meta: dict[str, dict] = {}   # book_id → full metadata from type="book" vectors
    best_by_book: dict[str, dict] = {}  # book_id → best match record

    for match in results.matches:
        meta = match.metadata or {}
        match_type = meta.get("type", "book")

        if match_type == "chunk":
            book_id = meta.get("book_id", match.id)
        else:
            book_id = match.id
            # Cache full book metadata (genre, rating, description, etc.)
            book_meta[book_id] = meta

        score = match.score
        existing = best_by_book.get(book_id)

        if existing is None or score > existing["score"]:
            best_by_book[book_id] = {
                "id": book_id,
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "matched_text": meta.get("text", ""),
                "score": round(score, 3),
                "match_source": "content" if match_type == "chunk" else "description",
            }

    # For any book where the best match was a chunk but we also have the
    # book-level vector in results, enrich with full metadata. If the
    # book-level vector wasn't in the top-15 results, fetch it directly.
    books_needing_meta = [
        bid for bid, rec in best_by_book.items()
        if bid not in book_meta
    ]
    if books_needing_meta:
        try:
            fetched = index.fetch(ids=books_needing_meta)
            for bid, vec in (fetched.vectors or {}).items():
                book_meta[bid] = vec.metadata or {}
        except Exception:
            pass  # If fetch fails, we'll use whatever metadata we have

    # Merge full book metadata into each result
    for bid, rec in best_by_book.items():
        meta = book_meta.get(bid, {})
        rec["genre"] = meta.get("genre", rec.get("genre", ""))
        rec["rating"] = meta.get("rating", rec.get("rating", 0))
        rec["description"] = meta.get("text", "")  # Book description
        # matched_text holds the actual text that scored highest
        # (could be the description itself or a content chunk)

    # Sort by score descending, return top 5
    ranked = sorted(best_by_book.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:5]


# ── Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the search comparison interface."""
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def api_search():
    """
    Execute both SQL full-text and semantic searches and return results.

    Expects JSON: { "query": "..." }
    Returns JSON: { "query": "...", "sql_results": [...], "semantic_results": [...] }
    """
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        sql_results = sql_search(query)
        semantic_results = semantic_search(query)

        return jsonify({
            "query": query,
            "sql_results": sql_results,
            "semantic_results": semantic_results,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8081))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"

    print(f"\n📚 Finding Meaning (Enhanced) — Full-Text vs Semantic Search")
    print(f"   Running on: http://localhost:{port}\n")

    app.run(host="0.0.0.0", port=port, debug=debug)
