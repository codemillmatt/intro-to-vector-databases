"""
Module 5: Tuning, Recall, and Latency Demo

This Flask web application demonstrates how tuning parameters
affect vector database query performance and result quality.
"""

import os
import sys
import time

# Add setup directory to path for shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "setup"))

from flask import Flask, render_template, request, jsonify

from embeddings import get_embedding_client
from pinecone_utils import get_pinecone_index

app = Flask(__name__)


def search_with_timing(
    query: str,
    top_k: int = 10,
    include_metadata: bool = True
) -> dict:
    """
    Execute a vector search and measure performance.
    
    Returns results along with timing information to demonstrate
    the latency/recall tradeoff.
    """
    embedding_client = get_embedding_client()
    
    # Time the embedding generation
    embed_start = time.perf_counter()
    query_embedding = embedding_client.embed(query)
    embed_time = (time.perf_counter() - embed_start) * 1000  # ms
    
    # Connect to Pinecone
    index = get_pinecone_index()
    
    # Time the vector search
    search_start = time.perf_counter()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=include_metadata,
        filter={"type": "book"}
    )
    search_time = (time.perf_counter() - search_start) * 1000  # ms
    
    # Calculate total time
    total_time = embed_time + search_time
    
    # Format results
    formatted_results = []
    for match in results.matches:
        formatted_results.append({
            "id": match.id,
            "title": match.metadata.get("title", "") if include_metadata else "",
            "author": match.metadata.get("author", "") if include_metadata else "",
            "genre": match.metadata.get("genre", "") if include_metadata else "",
            "score": round(match.score, 4)
        })
    
    return {
        "results": formatted_results,
        "timing": {
            "embedding_ms": round(embed_time, 2),
            "search_ms": round(search_time, 2),
            "total_ms": round(total_time, 2)
        },
        "parameters": {
            "top_k": top_k,
            "include_metadata": include_metadata
        }
    }


@app.route("/")
def index():
    """Render the main tuning demo page."""
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def api_search():
    """Handle search requests with tuning parameters."""
    data = request.get_json()
    
    query = data.get("query", "")
    top_k = data.get("top_k", 10)
    include_metadata = data.get("include_metadata", True)
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    try:
        result = search_with_timing(
            query=query,
            top_k=int(top_k),
            include_metadata=bool(include_metadata)
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    """Run multiple searches to benchmark performance."""
    data = request.get_json()
    
    query = data.get("query", "")
    iterations = min(data.get("iterations", 5), 20)  # Cap at 20
    top_k = data.get("top_k", 10)
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    try:
        embedding_times = []
        search_times = []
        
        embedding_client = get_embedding_client()
        index = get_pinecone_index()
        
        for _ in range(iterations):
            # Time embedding
            embed_start = time.perf_counter()
            query_embedding = embedding_client.embed(query)
            embedding_times.append((time.perf_counter() - embed_start) * 1000)
            
            # Time search
            search_start = time.perf_counter()
            index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter={"type": "book"}
            )
            search_times.append((time.perf_counter() - search_start) * 1000)
        
        return jsonify({
            "iterations": iterations,
            "embedding_times": {
                "min": round(min(embedding_times), 2),
                "max": round(max(embedding_times), 2),
                "avg": round(sum(embedding_times) / len(embedding_times), 2)
            },
            "search_times": {
                "min": round(min(search_times), 2),
                "max": round(max(search_times), 2),
                "avg": round(sum(search_times) / len(search_times), 2)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    os.makedirs(templates_dir, exist_ok=True)
    
    # Debug mode should only be enabled in development
    # Set FLASK_DEBUG=1 environment variable to enable debug mode
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=8000, debug=debug_mode)
