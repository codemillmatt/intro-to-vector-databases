"""
Module 3: Faceted Search UI Demo

This demo shows a production-style faceted search interface where users:
1. Enter a semantic search query in the search box
2. Use UI controls (checkboxes, sliders, dropdowns) to filter results

This is how most e-commerce sites (Amazon, Netflix, Airbnb) implement
combined semantic + metadata search.
"""

import os
import sys
import json

# ---------------------------------------------------------------------------
# Allow Python to find the shared utilities in the setup/ directory.
# This adds setup/ to the module search path so we can write:
#     from embeddings import get_embedding_client
# instead of dealing with complex relative imports.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "setup"))

from flask import Flask, render_template, request, jsonify

from embeddings import get_embedding_client
from pinecone_utils import get_pinecone_index

app = Flask(__name__)

# Cache embedding client and Pinecone index as singletons to avoid
# re-initializing on every request (the main source of latency).
_embedding_client = get_embedding_client()
_pinecone_index = get_pinecone_index()

# Pre-warm the embedding model so the first user query isn't slow
_embedding_client.embed("warmup")


def get_available_genres() -> list[str]:
    """Get all genres from the books catalog, sorted alphabetically."""
    books_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "setup", "data", "books.json"
    )
    try:
        with open(books_path) as f:
            books = json.load(f)
        genres = sorted({book["genre"] for book in books if "genre" in book})
        return genres
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback if the file can't be read
        return ["Fantasy", "Science Fiction", "Mystery", "Literary Fiction"]


def search_books(
    query: str = "",
    genres: list[str] = None,
    min_rating: float = None,
    max_price: float = None,
    in_stock_only: bool = False,
    top_k: int = 20
) -> list[dict]:
    """
    Search books with semantic query and metadata filters.
    
    This is the same search pattern as the other demos, but driven
    by structured UI inputs rather than parsed natural language.
    """
    has_query = bool(query.strip())

    if has_query:
        query_embedding = _embedding_client.embed(query)
    else:
        # No search text — generate a random unit vector so Pinecone
        # returns an essentially random sample of books.
        import numpy as np
        dim = len(_embedding_client.embed("test"))
        vec = np.random.randn(dim).astype(np.float32)
        query_embedding = (vec / np.linalg.norm(vec)).tolist()
    
    # Build metadata filters
    filters = {"type": "book"}
    
    if genres and len(genres) > 0:
        if len(genres) == 1:
            filters["genre"] = genres[0]
        else:
            # Multiple genres: use $in operator
            filters["genre"] = {"$in": genres}
    
    if min_rating is not None and min_rating > 0:
        filters["rating"] = {"$gte": min_rating}
    
    if max_price is not None and max_price < 50:  # 50 is our max
        filters["price"] = {"$lte": max_price}
    
    if in_stock_only:
        filters["in_stock"] = True
    
    # Query Pinecone (using cached singleton)
    results = _pinecone_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filters
    )
    
    # When the user typed a query, drop results below the similarity
    # floor (0.3) to avoid clearly irrelevant matches.
    MIN_SCORE = 0.3

    books = []
    for match in results.matches:
        if has_query and match.score < MIN_SCORE:
            continue
        books.append({
            "id": match.id,
            "title": match.metadata.get("title", ""),
            "author": match.metadata.get("author", ""),
            "genre": match.metadata.get("genre", ""),
            "rating": match.metadata.get("rating", 0),
            "price": match.metadata.get("price", 0),
            "in_stock": match.metadata.get("in_stock", False),
            "description": match.metadata.get("text", "")[:150] + "...",
            "score": round(match.score, 3)
        })
    
    return books


@app.route("/")
def index():
    """Render the faceted search page."""
    genres = get_available_genres()
    return render_template("index.html", genres=genres)


@app.route("/api/search", methods=["POST"])
def api_search():
    """Handle search requests from the UI."""
    data = request.get_json()
    
    query = data.get("query", "")
    genres = data.get("genres", [])
    min_rating = data.get("min_rating")
    max_price = data.get("max_price")
    in_stock_only = data.get("in_stock_only", False)
    
    # Convert types
    if min_rating is not None:
        min_rating = float(min_rating)
    if max_price is not None:
        max_price = float(max_price)
    
    try:
        results = search_books(
            query=query,
            genres=genres,
            min_rating=min_rating,
            max_price=max_price,
            in_stock_only=in_stock_only
        )
        
        # Build filter summary for display
        active_filters = []
        if genres:
            active_filters.append(f"Genre: {', '.join(genres)}")
        if min_rating and min_rating > 0:
            active_filters.append(f"Rating ≥ {min_rating}★")
        if max_price and max_price < 50:
            active_filters.append(f"Price ≤ ${max_price}")
        if in_stock_only:
            active_filters.append("In Stock")
        
        return jsonify({
            "results": results,
            "count": len(results),
            "query": query,
            "active_filters": active_filters
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Create templates directory if needed
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    os.makedirs(templates_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("  Faceted Search Demo")
    print("  Open http://localhost:8001 in your browser")
    print("="*60 + "\n")
    
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=8001, debug=debug_mode)
