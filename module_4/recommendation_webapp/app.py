"""
Module 4: Product Recommendation Web Application

A web-based demonstration of vector database-powered recommendations that shows:
1. Users and their liked books (the foundation for recommendations)
2. Content-based filtering (semantic similarity)
3. Collaborative filtering (users with similar tastes)

This visual interface makes it easy to understand how both
recommendation approaches work and what influences the results.
"""

import os
import sys
import json

# Add setup directory to path for shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "setup"))

from flask import Flask, render_template, request, jsonify

from embeddings import get_embedding_client
from pinecone_utils import get_pinecone_index

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")


def load_books() -> dict:
    """Load books data as a dictionary keyed by book ID."""
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "setup", "data", "books.json"
    )
    with open(data_path, "r") as f:
        return {book["id"]: book for book in json.load(f)}


def load_users() -> list[dict]:
    """Load user reading habits data."""
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "setup", "data", "users.json"
    )
    with open(data_path, "r") as f:
        return json.load(f)


def get_book_embeddings(book_ids: list[str]) -> list[list[float]]:
    """Get embeddings for a list of books from Pinecone."""
    index = get_pinecone_index()
    
    # Fetch book vectors
    results = index.fetch(ids=book_ids)
    
    embeddings = []
    for book_id in book_ids:
        if book_id in results.vectors:
            embeddings.append(results.vectors[book_id].values)
    
    return embeddings


def average_embeddings(embeddings: list[list[float]]) -> list[float]:
    """Calculate the average of multiple embedding vectors."""
    if not embeddings:
        # Get dimension from embedding client for consistency
        embedding_client = get_embedding_client()
        return [0.0] * embedding_client.dimension
    
    num_dims = len(embeddings[0])
    avg = []
    for i in range(num_dims):
        total = sum(emb[i] for emb in embeddings)
        avg.append(total / len(embeddings))
    
    return avg


def recommend_similar_books(
    liked_books: list[str],
    exclude_books: list[str] | None = None,
    top_k: int = 5
) -> list[dict]:
    """
    Recommend books similar to the user's liked books.
    
    This is content-based filtering - finding books with similar
    semantic content to what the user already enjoys.
    """
    books_data = load_books()
    
    # Get embeddings for liked books
    embeddings = get_book_embeddings(liked_books)
    
    if not embeddings:
        return []
    
    # Create a "preference vector" by averaging liked book embeddings
    preference_vector = average_embeddings(embeddings)
    
    # Search for similar books
    index = get_pinecone_index()
    
    results = index.query(
        vector=preference_vector,
        top_k=top_k + len(liked_books or []) + len(exclude_books or []),
        include_metadata=True,
        filter={"type": "book"}
    )
    
    # Filter out already-read books
    exclude_set = set(liked_books or []) | set(exclude_books or [])
    
    recommendations = []
    for match in results.matches:
        if match.id not in exclude_set:
            book = books_data.get(match.id, {})
            recommendations.append({
                "id": match.id,
                "title": match.metadata.get("title", ""),
                "author": match.metadata.get("author", ""),
                "genre": match.metadata.get("genre", ""),
                "description": book.get("description", ""),
                "rating": match.metadata.get("rating", 0),
                "score": round(match.score, 3)
            })
            
            if len(recommendations) >= top_k:
                break
    
    return recommendations


def recommend_collaborative(
    user_liked_books: list[str],
    all_users: list[dict],
    exclude_books: list[str] | None = None,
    top_k: int = 5
) -> dict:
    """
    Recommend books based on what similar users liked.
    
    Returns both the recommendations and information about
    which users influenced the recommendations.
    """
    books = load_books()
    exclude_set = set(user_liked_books or []) | set(exclude_books or [])
    
    # Find users with overlapping tastes
    similar_users = []
    for user in all_users:
        overlap = set(user["liked_books"]) & set(user_liked_books)
        if overlap:
            similarity = len(overlap) / max(
                len(user["liked_books"]), len(user_liked_books)
            )
            # Get the books they liked that you haven't read
            unique_books = [b for b in user["liked_books"] if b not in exclude_set]
            similar_users.append({
                "user_id": user["user_id"],
                "name": user["name"],
                "similarity": round(similarity, 2),
                "overlap_count": len(overlap),
                "overlap_books": list(overlap),
                "unique_recommendations": unique_books
            })
    
    # Sort by similarity
    similar_users.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Collect book recommendations from similar users
    book_scores = {}
    book_sources = {}  # Track which users recommended each book
    
    for similar_user in similar_users:
        for book_id in similar_user["unique_recommendations"]:
            if book_id not in book_scores:
                book_scores[book_id] = 0
                book_sources[book_id] = []
            book_scores[book_id] += similar_user["similarity"]
            book_sources[book_id].append(similar_user["name"])
    
    # Sort by score and return top k
    sorted_books = sorted(
        book_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:top_k]
    
    recommendations = []
    for book_id, score in sorted_books:
        if book_id in books:
            book = books[book_id]
            recommendations.append({
                "id": book_id,
                "title": book["title"],
                "author": book["author"],
                "genre": book["genre"],
                "description": book.get("description", ""),
                "rating": book["rating"],
                "score": round(score, 2),
                "recommended_by": book_sources.get(book_id, [])
            })
    
    return {
        "recommendations": recommendations,
        "similar_users": similar_users[:5]  # Top 5 similar users
    }


@app.route("/")
def index():
    """Main page with the recommendation interface."""
    books = load_books()
    users = load_users()
    
    # Enrich user data with book details
    enriched_users = []
    for user in users:
        liked_books_details = []
        for book_id in user["liked_books"]:
            if book_id in books:
                book = books[book_id]
                liked_books_details.append({
                    "id": book_id,
                    "title": book["title"],
                    "author": book["author"],
                    "genre": book["genre"],
                    "rating": book["rating"]
                })
        enriched_users.append({
            "user_id": user["user_id"],
            "name": user["name"],
            "liked_books": liked_books_details
        })
    
    return render_template(
        "index.html",
        books=list(books.values()),
        users=enriched_users
    )


@app.route("/api/recommend", methods=["POST"])
def get_recommendations():
    """
    API endpoint to get recommendations for selected books.
    
    Expects JSON: {"liked_books": ["book_001", "book_002", ...]}
    Returns both content-based and collaborative recommendations.
    """
    data = request.get_json()
    liked_books = data.get("liked_books", [])
    
    if not liked_books:
        return jsonify({"error": "No books selected"}), 400
    
    users = load_users()
    books = load_books()
    
    # Get content-based recommendations
    content_recommendations = recommend_similar_books(liked_books)
    
    # Get collaborative recommendations
    collab_result = recommend_collaborative(liked_books, users)
    
    # Get details of liked books for display
    liked_books_details = []
    for book_id in liked_books:
        if book_id in books:
            book = books[book_id]
            liked_books_details.append({
                "id": book_id,
                "title": book["title"],
                "author": book["author"],
                "genre": book["genre"],
                "description": book.get("description", ""),
                "rating": book["rating"]
            })
    
    return jsonify({
        "liked_books": liked_books_details,
        "content_based": content_recommendations,
        "collaborative": collab_result["recommendations"],
        "similar_users": collab_result["similar_users"]
    })


@app.route("/api/user/<user_id>/recommendations")
def get_user_recommendations(user_id: str):
    """
    Get recommendations for a specific user based on their liked books.
    
    This endpoint makes it easy to explore how recommendations
    differ for different user profiles.
    """
    users = load_users()
    books = load_books()
    
    # Find the user
    user = next((u for u in users if u["user_id"] == user_id), None)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    liked_books = user["liked_books"]
    
    # Get content-based recommendations
    content_recommendations = recommend_similar_books(liked_books)
    
    # Get collaborative recommendations (exclude self)
    other_users = [u for u in users if u["user_id"] != user_id]
    collab_result = recommend_collaborative(liked_books, other_users)
    
    # Get details of liked books
    liked_books_details = []
    for book_id in liked_books:
        if book_id in books:
            book = books[book_id]
            liked_books_details.append({
                "id": book_id,
                "title": book["title"],
                "author": book["author"],
                "genre": book["genre"],
                "description": book.get("description", ""),
                "rating": book["rating"]
            })
    
    return jsonify({
        "user": {
            "user_id": user["user_id"],
            "name": user["name"]
        },
        "liked_books": liked_books_details,
        "content_based": content_recommendations,
        "collaborative": collab_result["recommendations"],
        "similar_users": collab_result["similar_users"]
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)
