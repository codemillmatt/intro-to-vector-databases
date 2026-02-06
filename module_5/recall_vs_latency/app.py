"""
Module 5: Recall vs Latency Demo

This Flask application demonstrates the fundamental tradeoff between
recall (search accuracy) and latency (search speed) in vector databases.

The key teaching concept:
- HNSW index uses an 'ef' (exploration factor) parameter
- Higher ef = more nodes explored = better recall, but slower
- Lower ef = fewer nodes explored = faster, but may miss results

Uses 100K random vectors to clearly show how approximate search
degrades at low ef values.
"""

import os
import time
import numpy as np

from flask import Flask, render_template, request, jsonify

from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

app = Flask(__name__)

# Qdrant connection settings
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "recall_demo"
VECTOR_DIM = 128

# Cache for ground truth and query vectors
_cache = {}


def get_qdrant_client():
    """Create Qdrant client."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def get_random_query_vector(seed=None):
    """Generate a random unit query vector."""
    if seed is not None:
        np.random.seed(seed)
    vec = np.random.randn(VECTOR_DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def compute_ground_truth(query_vector, top_k=100):
    """
    Compute exact (brute-force) search results for a query.
    This gives us the "perfect" results to compare against.
    """
    client = get_qdrant_client()
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        search_params=SearchParams(exact=True)
    )
    
    return [r.id for r in results.points]


def search_with_ef(query_vector, ef, top_k=20):
    """
    Search with a specific ef parameter and measure latency.
    """
    client = get_qdrant_client()
    
    start_time = time.perf_counter()
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        search_params=SearchParams(hnsw_ef=ef)
    )
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return results.points, latency_ms


def calculate_recall(retrieved_ids, ground_truth_ids):
    """
    Calculate recall: what fraction of ground truth results did we find?
    """
    if not ground_truth_ids:
        return 0.0
    
    retrieved_set = set(retrieved_ids)
    ground_truth_set = set(ground_truth_ids[:len(retrieved_ids)])
    
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
    """
    data = request.get_json()
    query_seed = data.get("query_seed", 42)
    ef = int(data.get("ef", 64))
    top_k = int(data.get("top_k", 20))
    
    try:
        # Generate query vector from seed (deterministic)
        query_vector = get_random_query_vector(seed=query_seed)
        
        # Get ground truth (cached per seed)
        cache_key = f"gt_{query_seed}"
        if cache_key not in _cache:
            _cache[cache_key] = compute_ground_truth(query_vector, top_k=100)
        ground_truth = _cache[cache_key]
        
        # Search with specified ef
        results, search_latency = search_with_ef(query_vector, ef, top_k)
        
        # Calculate recall
        retrieved_ids = [r.id for r in results]
        recall = calculate_recall(retrieved_ids, ground_truth)
        
        # Format results
        formatted_results = []
        for r in results:
            formatted_results.append({
                "id": r.id,
                "score": round(r.score, 4),
                "in_ground_truth": r.id in ground_truth[:top_k]
            })
        
        return jsonify({
            "results": formatted_results,
            "metrics": {
                "recall_percent": round(recall * 100, 1),
                "search_latency_ms": round(search_latency, 2),
            },
            "parameters": {
                "ef": ef,
                "top_k": top_k,
                "query_seed": query_seed
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
    query_seed = data.get("query_seed", 42)
    top_k = int(data.get("top_k", 20))
    
    # ef values to test (log scale for better visualization)
    ef_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    
    try:
        query_vector = get_random_query_vector(seed=query_seed)
        ground_truth = compute_ground_truth(query_vector, top_k=100)
        
        curve_data = []
        for ef in ef_values:
            # Warm-up run (discard - avoids cold cache effects)
            search_with_ef(query_vector, ef, top_k)
            
            # Run multiple iterations for stable timing (median is more robust)
            latencies = []
            for _ in range(10):
                results, latency = search_with_ef(query_vector, ef, top_k)
                latencies.append(latency)
            
            retrieved_ids = [r.id for r in results]
            recall = calculate_recall(retrieved_ids, ground_truth)
            
            # Use median to reduce outlier impact
            latencies.sort()
            median_latency = latencies[len(latencies) // 2]
            
            curve_data.append({
                "ef": ef,
                "recall_percent": round(recall * 100, 1),
                "latency_ms": round(median_latency, 2)
            })
        
        return jsonify({
            "curve": curve_data,
            "top_k": top_k
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/new_query", methods=["POST"])
def api_new_query():
    """Generate a new random query seed."""
    import random
    new_seed = random.randint(1, 100000)
    return jsonify({"query_seed": new_seed})


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
            "hint": "Run: python init_qdrant_new.py --reset"
        }), 500


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=8080, debug=debug_mode)
