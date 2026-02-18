"""
Module 5: Recall vs Latency Demo

This Flask application demonstrates the fundamental tradeoff between
recall (search accuracy) and latency (search speed) in vector databases.

Uses 100K random vectors (128D, HNSW with m=8) to clearly show how
approximate search degrades at low ef values. Random vectors spread
uniformly in high-dimensional space, making HNSW approximation errors
much more visible than with clustered real-world data.

Two demo modes:
1. Exact vs Approximate Comparison — side-by-side for a single query
2. Batch Benchmark — many queries aggregated for a noise-free curve
"""

import os
import time
import random

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

# Number of random queries for batch benchmark
NUM_BATCH_QUERIES = 50


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


def search_exact(query_vector, top_k=20):
    """
    Brute-force exact search. Checks every vector.
    Always returns the true best results (100% recall).
    """
    client = get_qdrant_client()

    start_time = time.perf_counter()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        search_params=SearchParams(exact=True),
    )
    latency_ms = (time.perf_counter() - start_time) * 1000

    return results.points, latency_ms


def search_approximate(query_vector, ef, top_k=20):
    """
    HNSW approximate search with specified ef.
    Faster than exact, but may miss some results.
    """
    client = get_qdrant_client()

    start_time = time.perf_counter()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        search_params=SearchParams(hnsw_ef=ef),
    )
    latency_ms = (time.perf_counter() - start_time) * 1000

    return results.points, latency_ms


def calculate_recall(retrieved_ids, ground_truth_ids):
    """
    Calculate recall: what fraction of ground truth results did we find?

    Recall = |retrieved ∩ ground_truth| / |ground_truth|
    """
    if not ground_truth_ids:
        return 0.0

    retrieved_set = set(retrieved_ids)
    ground_truth_set = set(ground_truth_ids[: len(retrieved_ids)])

    overlap = len(retrieved_set & ground_truth_set)
    return overlap / len(ground_truth_set)


@app.route("/")
def index():
    """Render the main demo page."""
    return render_template("index.html")


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """
    Compare exact vs approximate search side-by-side for a single query.

    Uses random vectors, so results are identified by vector ID and score.
    """
    data = request.get_json()
    query_seed = data.get("query_seed", 42)
    ef = int(data.get("ef", 64))
    top_k = int(data.get("top_k", 20))

    try:
        query_vector = get_random_query_vector(seed=query_seed)

        # Run exact search (ground truth)
        exact_results, exact_latency = search_exact(query_vector, top_k)
        exact_ids = set(r.id for r in exact_results)

        # Run approximate search
        approx_results, approx_latency = search_approximate(query_vector, ef, top_k)
        approx_ids = set(r.id for r in approx_results)

        # Calculate recall
        recall = calculate_recall(
            [r.id for r in approx_results],
            [r.id for r in exact_results],
        )

        # Format exact results
        exact_formatted = []
        for r in exact_results:
            exact_formatted.append({
                "id": r.id,
                "score": round(r.score, 6),
                "found_by_approx": r.id in approx_ids,
            })

        # Format approx results
        approx_formatted = []
        for r in approx_results:
            approx_formatted.append({
                "id": r.id,
                "score": round(r.score, 6),
                "in_ground_truth": r.id in exact_ids,
            })

        return jsonify({
            "exact": {
                "results": exact_formatted,
                "latency_ms": round(exact_latency, 2),
            },
            "approximate": {
                "results": approx_formatted,
                "latency_ms": round(approx_latency, 2),
            },
            "recall_percent": round(recall * 100, 1),
            "speedup": round(exact_latency / max(approx_latency, 0.01), 1),
            "parameters": {
                "ef": ef,
                "top_k": top_k,
                "query_seed": query_seed,
            },
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    """
    Run a batch benchmark across multiple ef values.

    Generates many random query vectors and measures aggregate latency
    and average recall at each ef value. This eliminates per-query noise
    and shows the true recall/latency tradeoff clearly.
    """
    data = request.get_json()
    top_k = int(data.get("top_k", 20))
    num_queries = int(data.get("num_queries", NUM_BATCH_QUERIES))
    num_queries = min(num_queries, 200)  # Safety cap

    # ef values to benchmark (all >= top_k to be meaningful)
    all_ef_values = [1, 2, 4, 8, 16, 20, 32, 48, 64, 96, 128, 192, 256, 512]
    ef_values = [ef for ef in all_ef_values if ef >= top_k]
    if not ef_values:
        ef_values = [top_k]
    if ef_values[0] != top_k and top_k not in ef_values:
        ef_values.insert(0, top_k)

    try:
        # Generate deterministic random query vectors
        query_vectors = []
        for i in range(num_queries):
            query_vectors.append(get_random_query_vector(seed=10000 + i))

        # Pre-compute ground truth for all queries (exact search)
        ground_truths = []
        exact_total_time = 0.0
        for qv in query_vectors:
            results, latency = search_exact(qv, top_k)
            ground_truths.append([r.id for r in results])
            exact_total_time += latency

        # Benchmark each ef value
        curve_data = []
        for ef in ef_values:
            total_latency = 0.0
            total_recall = 0.0

            for i, qv in enumerate(query_vectors):
                results, latency = search_approximate(qv, ef, top_k)
                retrieved_ids = [r.id for r in results]
                recall = calculate_recall(retrieved_ids, ground_truths[i])

                total_latency += latency
                total_recall += recall

            avg_recall = (total_recall / num_queries) * 100
            avg_latency = total_latency / num_queries

            curve_data.append({
                "ef": ef,
                "recall_percent": round(avg_recall, 1),
                "avg_latency_ms": round(avg_latency, 2),
                "total_latency_ms": round(total_latency, 1),
            })

        return jsonify({
            "curve": curve_data,
            "exact_total_ms": round(exact_total_time, 1),
            "exact_avg_ms": round(exact_total_time / num_queries, 2),
            "num_queries": num_queries,
            "top_k": top_k,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/new_query", methods=["POST"])
def api_new_query():
    """Generate a new random query seed."""
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
            "status": "ready",
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "not_initialized",
            "hint": "Run: python init_qdrant.py --reset",
        }), 500


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=8080, debug=debug_mode)
