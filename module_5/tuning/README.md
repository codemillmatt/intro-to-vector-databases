# Module 5: Tuning, Recall, and Latency Demo

**Interactive Performance Dashboard for Vector Database Optimization**

This web application provides a hands-on way to explore how tuning parameters affect vector database query performance and result quality. Experiment with different settings and see real-time metrics to understand the tradeoffs involved in production vector search systems.

---

## What This Demo Shows

Vector databases require careful tuning to balance competing concerns:

```
                    ┌─────────────────────────────────────┐
                    │         THE TUNING TRADEOFF         │
                    ├─────────────────────────────────────┤
                    │                                     │
  More Results ◄────│───── top_k parameter ─────────►    │ Fewer Results
  (Higher Recall)   │                                     │ (Lower Recall)
  (More Latency)    │                                     │ (Less Latency)
                    │                                     │
  Richer Data ◄─────│──── include_metadata ─────────►    │ IDs Only
  (Larger Payload)  │                                     │ (Minimal Payload)
  (More Latency)    │                                     │ (Less Latency)
                    │                                     │
                    └─────────────────────────────────────┘
```

This interactive dashboard lets you:
- Adjust `top_k` to control how many results are returned
- Toggle metadata inclusion on/off
- See real-time latency measurements (embedding + search)
- Run benchmarks to measure performance consistency
- Visualize how parameter changes affect both speed and result quality

---

## Why This Matters for Vector Databases

Understanding performance characteristics is critical for production vector search systems. Every millisecond counts when serving millions of queries.

### The Performance Equation

| Factor | Impact on Latency | Impact on Quality |
|--------|-------------------|-------------------|
| **Embedding Generation** | Fixed cost per query (~50-200ms) | Determines semantic accuracy |
| **top_k (result count)** | More results = more processing | Higher recall, more options to rank |
| **Metadata Inclusion** | More data to transfer | Richer results for display/filtering |
| **Index Type** | HNSW vs IVF tradeoffs | Exact vs approximate neighbors |

### Real-World Tuning Scenarios

**High-Throughput API** (e.g., autocomplete)
- Low `top_k` (3-5)
- Minimal metadata
- Target: <50ms latency

**Comprehensive Search** (e.g., research tool)
- Higher `top_k` (20-50)
- Full metadata
- Target: <500ms latency

**RAG Applications** (e.g., chatbots)
- Medium `top_k` (5-10)
- Content metadata required
- Target: <200ms for retrieval step

---

## Running the Demo

```bash
cd module_5/tuning
python app.py
```

Then open **http://localhost:8000** in your browser.

---

## Features

### 1. Interactive Search Panel

| Control | Description | Effect |
|---------|-------------|--------|
| **Search Query** | Natural language input | Converted to embedding vector |
| **top_k Slider** | Number of results (1-50) | More results = higher recall, more latency |
| **Include Metadata** | Toggle metadata retrieval | Affects payload size and response time |
| **Search Button** | Execute query | Displays results with timing |

### 2. Real-Time Metrics

The dashboard displays three key timing metrics:

- **Embedding Time**: How long to convert your query to a vector
- **Search Time**: How long Pinecone took to find similar vectors
- **Total Time**: End-to-end latency for the complete operation

### 3. Benchmark Mode

Run multiple iterations of the same query to measure:
- **Min/Max/Average** embedding times
- **Min/Max/Average** search times
- Consistency and variability in performance

---

## How It Works (Code Walkthrough)

### 1. Search with Timing Measurement

The core function wraps the standard search with precise timing:

```python
def search_with_timing(
    query: str,
    top_k: int = 10,
    include_metadata: bool = True
) -> dict:
    # Time the embedding generation
    embed_start = time.perf_counter()
    query_embedding = embedding_client.embed(query)
    embed_time = (time.perf_counter() - embed_start) * 1000  # ms
    
    # Time the vector search
    search_start = time.perf_counter()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=include_metadata,
        filter={"type": "book"}
    )
    search_time = (time.perf_counter() - search_start) * 1000  # ms
```

**Key insight**: Using `time.perf_counter()` provides high-resolution timing accurate to microseconds.

### 2. Benchmark Endpoint

The benchmark runs multiple iterations to account for variance:

```python
@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    iterations = min(data.get("iterations", 5), 20)  # Cap at 20
    
    embedding_times = []
    search_times = []
    
    for _ in range(iterations):
        # Time embedding
        embed_start = time.perf_counter()
        query_embedding = embedding_client.embed(query)
        embedding_times.append((time.perf_counter() - embed_start) * 1000)
        
        # Time search
        search_start = time.perf_counter()
        index.query(vector=query_embedding, top_k=top_k, ...)
        search_times.append((time.perf_counter() - search_start) * 1000)
    
    return jsonify({
        "embedding_times": {
            "min": min(embedding_times),
            "max": max(embedding_times),
            "avg": sum(embedding_times) / len(embedding_times)
        },
        # ... search_times similarly
    })
```

### 3. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Render the interactive dashboard |
| `/api/search` | POST | Execute a search with timing |
| `/api/benchmark` | POST | Run performance benchmarks |

---

## Understanding the Results

### Timing Breakdown

```
┌────────────────────────────────────────────────────────┐
│                    Query: "magical adventure"          │
├────────────────────────────────────────────────────────┤
│  Embedding:  85.32 ms   ████████░░░░░░░░░░░░ (54%)    │
│  Search:     72.45 ms   ██████░░░░░░░░░░░░░░ (46%)    │
│  ─────────────────────────────────────────────────     │
│  Total:     157.77 ms                                  │
└────────────────────────────────────────────────────────┘
```

### What Affects Each Component

**Embedding Time** (usually the larger factor):
- Embedding model size and complexity
- Local vs. remote embedding service
- GPU vs. CPU inference
- Query length (longer = slightly more time)

**Search Time** (typically very fast with managed services):
- Index size (number of vectors)
- `top_k` value
- Metadata payload size
- Network latency to vector database

---

## Experiments to Try

### Experiment 1: top_k Impact
1. Search for "exciting adventure story"
2. Set top_k to 5, note the search time
3. Increase to 25, compare
4. Increase to 50, observe the difference

*Expected*: Search time increases slightly with higher top_k

### Experiment 2: Metadata Overhead
1. Run a search with metadata enabled
2. Disable metadata, run the same search
3. Compare the search times

*Expected*: Searches without metadata are slightly faster

### Experiment 3: Benchmark Consistency
1. Enter a query and run a 10-iteration benchmark
2. Observe the min/max spread
3. Note: First query is often slower (cold start)

*Expected*: ~10-30% variance is normal; large variance suggests infrastructure issues

---

## Production Tuning Tips

### 1. Right-Size Your top_k

```
Don't: Always use top_k=100 "just in case"
Do:    Use the minimum top_k that meets your recall requirements
```

### 2. Fetch Metadata Strategically

```
Don't: Always include all metadata fields
Do:    Request only the metadata fields you need for display/ranking
```

### 3. Cache Embeddings When Possible

```
Don't: Re-embed the same queries repeatedly
Do:    Cache embeddings for common/repeated queries
```

### 4. Monitor and Measure

```
Don't: Assume performance is constant
Do:    Log timing metrics and set up alerting for latency spikes
```

---

## Prerequisites

Before running this demo, ensure you have:

1. **Completed database initialization** (see main repository README)
2. **Pinecone index created** with book embeddings loaded
3. **Embedding service available** (Ollama or sentence-transformers)
4. **Dependencies installed**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Connection refused" error | Ensure Pinecone API key is set in environment |
| Very slow embedding times | Check if Ollama is running; consider GPU acceleration |
| No results returned | Verify the Pinecone index has data loaded |
| High variance in benchmarks | Network instability or cold starts; run more iterations |

---

## Related Modules

- **Module 1**: [Finding Meaning](../../module_1/finding_meaning/) — SQL vs. semantic search comparison
- **Module 3**: [Metadata Filtering](../../module_3/metadata_filtering/) — Combining filters with vector search
- **Module 4**: [RAG](../../module_4/rag/) — Retrieval-augmented generation pattern

---

## Key Takeaways

1. **Embedding time often dominates** — optimizing the embedding pipeline matters
2. **top_k affects both recall and latency** — tune based on your specific use case
3. **Metadata has a cost** — only request what you need
4. **Measure in production** — performance characteristics vary by workload
5. **Benchmarking reveals variance** — don't rely on single-query measurements
