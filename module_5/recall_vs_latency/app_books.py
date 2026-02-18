"""
Module 5: Recall vs Latency Demo (Books Edition)

This Flask application demonstrates the fundamental tradeoff between
recall (search accuracy) and latency (search speed) in vector databases.

Instead of trying to show per-query latency curves (where noise dominates
with small datasets), this demo uses two clear teaching approaches:

1. **Exact vs Approximate Comparison**: Side-by-side view of brute-force
   search (100% recall, slower) vs HNSW approximate search (fast, may
   miss results). This always produces a visible, reliable difference.

2. **Batch Benchmark**: Runs many queries at each ef value and reports
   *aggregate* timings. By summing across 50+ queries, sub-millisecond
   per-query differences become seconds-level differences — clearly
   visible and noise-free.
"""

import os
import sys
import time

# Add setup directory to path for shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "setup"))

from flask import Flask, render_template, request, jsonify

from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

from embeddings import get_embedding_client

app = Flask(__name__)

# Qdrant connection settings
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "books_large"

# Sample queries for batch benchmarking (diverse genres and intents)
SAMPLE_QUERIES = [
    "epic fantasy adventure with dragons and magic",
    "mystery novel set in a small town",
    "science fiction space exploration story",
    "romance novel with a happy ending",
    "historical fiction set during world war two",
    "thriller with a detective solving crimes",
    "coming of age story about a young person",
    "horror story set in a haunted house",
    "literary fiction exploring family dynamics",
    "adventure novel set in the wilderness",
    "dystopian future society and rebellion",
    "cozy mystery with an amateur detective",
    "psychological thriller with unreliable narrator",
    "fantasy quest to save the world",
    "heartwarming story about friendship and love",
    "dark and suspenseful crime novel",
    "journey through unknown lands and discovery",
    "story about overcoming personal challenges",
    "tale of political intrigue and betrayal",
    "philosophical novel about the meaning of life",
    "lighthearted comedy with quirky characters",
    "gripping survival story in harsh conditions",
    "magical realism blending everyday and wonder",
    "war story told from a soldier's perspective",
    "story about artificial intelligence becoming sentient",
    "detective solving impossible locked room mystery",
    "young wizard learning to control their powers",
    "underwater adventure exploring the deep ocean",
    "time travel story with paradoxes",
    "story about a haunted library full of secrets",
    "pirates searching for legendary treasure",
    "robot uprising against human creators",
    "love story across different time periods",
    "spy thriller with double agents and deception",
    "quiet story about life in a rural village",
    "post-apocalyptic survival and rebuilding society",
    "courtroom drama with a shocking verdict",
    "fairy tale retelling with a dark twist",
    "story about musicians and the power of music",
    "mountain climbing expedition gone wrong",
    "alien first contact and communication challenges",
    "boarding school mystery with hidden passages",
    "generational saga spanning a century",
    "heist story with an elaborate plan",
    "story about a bookshop and its eccentric owner",
    "wilderness survival after a plane crash",
    "gothic romance in a crumbling mansion",
    "cyberpunk hacker fighting a corporation",
    "story about cooking and cultural heritage",
    "space colony struggling to survive on Mars",
]


def get_qdrant_client():
    """Create Qdrant client."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def search_exact(query_embedding, top_k=20):
    """
    Brute-force exact search. Checks every vector.
    Always returns the true best results (100% recall).
    """
    client = get_qdrant_client()

    start_time = time.perf_counter()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        search_params=SearchParams(exact=True),
    )
    latency_ms = (time.perf_counter() - start_time) * 1000

    return results.points, latency_ms


def search_approximate(query_embedding, ef, top_k=20):
    """
    HNSW approximate search with specified ef.
    Faster than exact, but may miss some results.
    """
    client = get_qdrant_client()

    start_time = time.perf_counter()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
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


def deduplicate_results(results, top_k):
    """Deduplicate results by book_id, keeping best match per book."""
    seen_books = set()
    deduped = []
    for r in results:
        book_id = r.payload.get("book_id", r.id)
        if book_id in seen_books:
            continue
        seen_books.add(book_id)
        deduped.append(r)
        if len(deduped) >= top_k:
            break
    return deduped


def format_result(r, ground_truth_ids=None):
    """Format a single search result for the frontend."""
    return {
        "id": r.id,
        "book_id": r.payload.get("book_id", r.id),
        "title": r.payload.get("title", ""),
        "author": r.payload.get("author", ""),
        "genre": r.payload.get("genre", ""),
        "score": round(r.score, 4),
        "in_ground_truth": r.id in ground_truth_ids if ground_truth_ids else True,
    }


@app.route("/")
def index():
    """Render the main demo page."""
    return render_template("index_books.html")


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """
    Compare exact vs approximate search side-by-side.

    This is the core teaching endpoint: it shows learners exactly what
    approximate search trades away for speed.
    """
    data = request.get_json()
    query = data.get("query", "")
    ef = int(data.get("ef", 64))
    top_k = int(data.get("top_k", 10))

    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Generate query embedding
        embedding_client = get_embedding_client()
        query_embedding = embedding_client.embed(query)

        # Run exact search (ground truth)
        exact_results_raw, exact_latency = search_exact(
            query_embedding, top_k=top_k * 3
        )
        exact_ids = [r.id for r in exact_results_raw]
        exact_results = deduplicate_results(exact_results_raw, top_k)

        # Run approximate search
        approx_results_raw, approx_latency = search_approximate(
            query_embedding, ef, top_k=top_k * 3
        )
        approx_ids = [r.id for r in approx_results_raw]
        approx_results = deduplicate_results(approx_results_raw, top_k)

        # Calculate recall (on raw results before dedup)
        recall = calculate_recall(approx_ids, exact_ids)

        # Format results
        exact_formatted = [format_result(r) for r in exact_results]
        approx_formatted = [
            format_result(r, set(exact_ids[:top_k])) for r in approx_results
        ]

        # Identify which exact results the approximate search missed
        approx_book_ids = {r["book_id"] for r in approx_formatted}
        for r in exact_formatted:
            r["found_by_approx"] = r["book_id"] in approx_book_ids

        return jsonify(
            {
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
                "parameters": {"ef": ef, "top_k": top_k},
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    """
    Run a batch benchmark across multiple ef values.

    Instead of timing a single query (where noise dominates), we run
    many queries at each ef value. The *aggregate* time is large enough
    to measure reliably, producing a clean recall-vs-latency curve.
    """
    data = request.get_json()
    top_k = int(data.get("top_k", 10))
    num_queries = int(data.get("num_queries", 50))

    # Clamp to available sample queries
    num_queries = min(num_queries, len(SAMPLE_QUERIES))

    # ef values to benchmark (all >= top_k to be meaningful)
    all_ef_values = [8, 16, 20, 32, 48, 64, 96, 128, 192, 256]
    ef_values = [ef for ef in all_ef_values if ef >= top_k]
    if not ef_values:
        ef_values = [top_k]
    if ef_values[0] != top_k and top_k not in ef_values:
        ef_values.insert(0, top_k)

    try:
        embedding_client = get_embedding_client()

        # Pre-compute all query embeddings
        queries = SAMPLE_QUERIES[:num_queries]
        query_embeddings = []
        for q in queries:
            query_embeddings.append(embedding_client.embed(q))

        # Pre-compute ground truth for all queries (exact search)
        ground_truths = []
        exact_total_time = 0.0
        for emb in query_embeddings:
            results, latency = search_exact(emb, top_k=top_k)
            ground_truths.append([r.id for r in results])
            exact_total_time += latency

        # Benchmark each ef value
        curve_data = []
        for ef in ef_values:
            total_latency = 0.0
            total_recall = 0.0

            for i, emb in enumerate(query_embeddings):
                results, latency = search_approximate(emb, ef, top_k)
                retrieved_ids = [r.id for r in results]
                recall = calculate_recall(retrieved_ids, ground_truths[i])

                total_latency += latency
                total_recall += recall

            avg_recall = (total_recall / num_queries) * 100
            avg_latency = total_latency / num_queries

            curve_data.append(
                {
                    "ef": ef,
                    "recall_percent": round(avg_recall, 1),
                    "avg_latency_ms": round(avg_latency, 2),
                    "total_latency_ms": round(total_latency, 1),
                }
            )

        return jsonify(
            {
                "curve": curve_data,
                "exact_total_ms": round(exact_total_time, 1),
                "exact_avg_ms": round(exact_total_time / num_queries, 2),
                "num_queries": num_queries,
                "top_k": top_k,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/collection_info")
def api_collection_info():
    """Get information about the vector collection."""
    try:
        client = get_qdrant_client()
        info = client.get_collection(COLLECTION_NAME)

        return jsonify(
            {
                "collection": COLLECTION_NAME,
                "vector_count": info.points_count,
                "dimension": info.config.params.vectors.size,
                "status": "ready",
            }
        )
    except Exception as e:
        return jsonify(
            {
                "error": str(e),
                "status": "not_initialized",
                "hint": "Run: python init_qdrant_books.py",
            }
        ), 500


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=8080, debug=debug_mode)
