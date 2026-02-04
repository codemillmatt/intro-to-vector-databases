"""
Module 1: Finding Meaning - Web Application

A web-based demo comparing traditional SQL keyword search against 
semantic vector similarity search. This demonstrates the immediate 
value proposition of vector databases.
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

# PostgreSQL Configuration
POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "database": os.getenv("POSTGRES_DB", "bookstore"),
    "user": os.getenv("POSTGRES_USER", "bookstore"),
    "password": os.getenv("POSTGRES_PASSWORD", "bookstore"),
}


def sql_search(query: str) -> list[dict]:
    """
    Search books using traditional SQL LIKE pattern matching.
    
    This represents how traditional databases handle text search -
    keyword matching that misses semantic meaning.
    """
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    try:
        with conn.cursor() as cur:
            # Use ILIKE for case-insensitive search
            # Search in title, description, and genre
            sql = """
                SELECT id, title, author, description, genre, rating
                FROM books
                WHERE title ILIKE %s 
                   OR description ILIKE %s
                   OR genre ILIKE %s
                ORDER BY rating DESC
                LIMIT 5
            """
            search_pattern = f"%{query}%"
            cur.execute(sql, (search_pattern, search_pattern, search_pattern))
            
            columns = ["id", "title", "author", "description", "genre", "rating"]
            results = [dict(zip(columns, row)) for row in cur.fetchall()]
            return results
    finally:
        conn.close()


def semantic_search(query: str) -> list[dict]:
    """
    Search books using semantic vector similarity.
    
    This demonstrates how vector databases understand meaning -
    finding conceptually similar content even without keyword matches.
    """
    # Get embedding for the query
    embedding_client = get_embedding_client()
    query_embedding = embedding_client.embed(query)
    
    # Search in Pinecone
    index = get_pinecone_index()
    
    # Only search book descriptions, not text chunks
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True,
        filter={"type": "book"}
    )
    
    # Format results
    formatted = []
    for match in results.matches:
        formatted.append({
            "id": match.id,
            "title": match.metadata.get("title", ""),
            "author": match.metadata.get("author", ""),
            "description": match.metadata.get("text", ""),
            "genre": match.metadata.get("genre", ""),
            "rating": match.metadata.get("rating", 0),
            "score": round(match.score, 3)
        })
    
    return formatted


@app.route("/")
def index():
    """Render the main comparison interface."""
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def api_search():
    """
    Execute both SQL and semantic searches and return results.
    
    Expects JSON: { "query": "..." }
    """
    data = request.get_json()
    query = data.get("query", "").strip()
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    try:
        # Execute both searches
        sql_results = sql_search(query)
        semantic_results = semantic_search(query)
        
        return jsonify({
            "query": query,
            "sql_results": sql_results,
            "semantic_results": semantic_results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    
    print(f"\n📚 Finding Meaning - SQL vs Semantic Search Demo")
    print(f"   Running on: http://localhost:{port}\n")
    
    app.run(host="0.0.0.0", port=port, debug=debug)
