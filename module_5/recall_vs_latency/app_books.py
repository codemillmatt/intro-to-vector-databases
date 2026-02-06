"""
Module 5: Recall vs Latency Demo

This Flask application demonstrates the fundamental tradeoff between
recall (search accuracy) and latency (search speed) in vector databases.

The key teaching concept:
- HNSW index uses an 'ef' (exploration factor) parameter
- Higher ef = more nodes explored = better recall, but slower
- Lower ef = fewer nodes explored = faster, but may miss results

Learners can interactively adjust ef and see real-time impact on both
recall percentage and query latency.
"""

import os
import sys
import time
import json

# Add setup directory to path for shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "setup"))

from flask import Flask, render_template, request, jsonify

from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams, HnswConfigDiff

from embeddings import get_embedding_client

app = Flask(__name__)

# Qdrant connection settings
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "books_large"

# Cache for ground truth (computed on first request)
_ground_truth_cache = {}


def get_qdrant_client():
    """Create Qdrant client."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def compute_ground_truth(query_embedding, top_k=100):
    """
    Compute exact (brute-force) search results for a query.
    
    This uses exact=True which triggers exact search in Qdrant,
    giving us the "perfect" results to compare against.
    """
    client = get_qdrant_client()
    
    # Exact search (brute force)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        search_params=SearchParams(
            exact=True  # Force exact/brute-force search
        )
    )
    
    return [r.id for r in results.points]


def search_with_ef(query_embedding, ef, top_k=20):
    """
    Search with a specific ef parameter and measure latency.
    
    Returns results and timing information.
    """
    client = get_qdrant_client()
    
    start_time = time.perf_counter()
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        search_params=SearchParams(
            hnsw_ef=ef  # The key parameter we're teaching about
        )
    )
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return results.points, latency_ms


def calculate_recall(retrieved_ids, ground_truth_ids):
    """
    Calculate recall: what fraction of ground truth results did we find?
    
    Recall = |retrieved ∩ ground_truth| / |ground_truth|
    
    This is the core metric showing the accuracy tradeoff.
    """
    if not ground_truth_ids:
        return 0.0
    
    retrieved_set = set(retrieved_ids)
    ground_truth_set = set(ground_truth_ids[:len(retrieved_ids)])  # Fair comparison
    
    overlap = len(retrieved_set & ground_truth_set)
    recall = overlap / len(ground_truth_set)
    
    return recall


@app.route("/")
def index():
    """Render the main demo page."""
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def api_search():
    """
    Execute a search with specified ef parameter.
    
    Returns results, latency, and recall percentage.
    """
    data = request.get_json()
    query = data.get("query", "")
    ef = int(data.get("ef", 64))
    top_k = int(data.get("top_k", 20))
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    try:
        # Generate query embedding
        embedding_client = get_embedding_client()
        embed_start = time.perf_counter()
        query_embedding = embedding_client.embed(query)
        embed_time = (time.perf_counter() - embed_start) * 1000
        
        # Get ground truth (cached per query)
        cache_key = query
        if cache_key not in _ground_truth_cache:
            _ground_truth_cache[cache_key] = compute_ground_truth(
                query_embedding, 
                top_k=100  # Get more for fair recall calculation
            )
        ground_truth = _ground_truth_cache[cache_key]
        
        # Search with specified ef - get more results to allow deduplication
        results, search_latency = search_with_ef(query_embedding, ef, top_k * 3)
        
        # Calculate recall using raw results (before deduplication)
        retrieved_ids = [r.id for r in results]
        recall = calculate_recall(retrieved_ids, ground_truth)
        
        # Deduplicate by book_id, keeping only best match per book
        seen_books = set()
        formatted_results = []
        for r in results:
            book_id = r.payload.get("book_id", r.id)
            if book_id in seen_books:
                continue
            seen_books.add(book_id)
            
            formatted_results.append({
                "id": r.id,
                "book_id": book_id,
                "title": r.payload.get("title", ""),
                "author": r.payload.get("author", ""),
                "genre": r.payload.get("genre", ""),
                "score": round(r.score, 4),
                "in_ground_truth": r.id in ground_truth[:top_k]
            })
            
            if len(formatted_results) >= top_k:
                break
        
        return jsonify({
            "results": formatted_results,
            "metrics": {
                "recall_percent": round(recall * 100, 1),
                "search_latency_ms": round(search_latency, 2),
                "embed_latency_ms": round(embed_time, 2),
                "total_latency_ms": round(search_latency + embed_time, 2)
            },
            "parameters": {
                "ef": ef,
                "top_k": top_k
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/recall_curve", methods=["POST"])
def api_recall_curve():
    """
    Generate recall vs latency curve for multiple ef values.
    
    This is the key visualization showing the tradeoff.
    """
    data = request.get_json()
    query = data.get("query", "")
    top_k = int(data.get("top_k", 20))
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    # ef values to test (exponential scale for good visualization)
    ef_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    try:
        embedding_client = get_embedding_client()
        query_embedding = embedding_client.embed(query)
        
        # Get ground truth
        ground_truth = compute_ground_truth(query_embedding, top_k=100)
        
        # Test each ef value
        curve_data = []
        for ef in ef_values:
            results, latency = search_with_ef(query_embedding, ef, top_k)
            retrieved_ids = [r.id for r in results]
            recall = calculate_recall(retrieved_ids, ground_truth)
            
            curve_data.append({
                "ef": ef,
                "recall_percent": round(recall * 100, 1),
                "latency_ms": round(latency, 2)
            })
        
        return jsonify({
            "curve": curve_data,
            "top_k": top_k,
            "ground_truth_size": len(ground_truth)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/collection_info")
def api_collection_info():
    """Get information about the vector collection."""
    try:
        client = get_qdrant_client()
        info = client.get_collection(COLLECTION_NAME)
        
        return jsonify({
            "collection": COLLECTION_NAME,
            "vector_count": info.points_count,
            "dimension": info.config.params.vectors.size,
            "status": "ready"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "not_initialized",
            "hint": "Run: python init_qdrant.py"
        }), 500


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=8080, debug=debug_mode)
