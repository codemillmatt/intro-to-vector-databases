# Recall vs Latency Demo

This demo teaches the fundamental tradeoff between **recall** (search accuracy) and **latency** (search speed) in vector databases.

## The Concept

Vector databases use approximate nearest neighbor (ANN) algorithms like HNSW to search quickly. The key insight:

- **Exact search** scans every vector → 100% recall, but slow
- **Approximate search** explores a subset → faster, but may miss results

The **ef** (exploration factor) parameter controls this tradeoff:

| ef Value | Behavior |
|----------|----------|
| Low (1-8) | Very fast, but misses many relevant results |
| Medium (32-64) | Good balance for most use cases |
| High (128-512) | Near-perfect recall, but slower |

## Why 100K Random Vectors?

Real data (like book embeddings) tends to cluster in semantic space, making ANN search easy — even low ef finds everything. To show the actual tradeoff, we use **100,000 uniformly random vectors** which spread across the space and make approximate search genuinely challenging.

## Prerequisites

- **DevContainer** — This project must run inside the DevContainer. Qdrant runs as a container service and is automatically started by Docker Compose.
- **Python dependencies** — Install from the project root before running any demos:

```bash
pip install -r requirements.txt
```

This installs `qdrant-client`, `flask`, `numpy`, and other required packages.

## Running the Demo

### 1. Start the DevContainer

Open this repository in VS Code and reopen in the DevContainer. Docker Compose will start Qdrant automatically. You can verify it's running:

```bash
curl http://qdrant:6333/healthz
```

### 2. Initialize the Database

```bash
cd module_5/recall_vs_latency
python init_qdrant.py --reset
```

This generates and inserts **100,000 random vectors** (128 dimensions, cosine distance) with an HNSW index configured at `m=8` and `ef_construct=64`. Takes 1–2 minutes.

To check the collection status at any time:

```bash
python init_qdrant.py --info
```

### 3. Run the Flask App

```bash
python app.py
```

Open http://localhost:8080 in your browser.

## Using the Demo

The demo has two tabs:

### Tab 1: Exact vs Approximate

Compare brute-force (exact) search against HNSW approximate search side-by-side for a single query.

1. **Adjust the ef slider** to control how thoroughly HNSW explores the graph
2. **Click "Compare"** to see both result sets side-by-side
3. **Click "New Random Query"** to try a different query vector

Results are color-coded:
- **Green** — Found by both exact and HNSW
- **Red (MISSED)** — In exact results but HNSW missed it
- **Yellow (WRONG)** — HNSW returned it but it's not in the true top-k

### Tab 2: Batch Benchmark

Runs many random queries at each ef value and measures aggregate timing to produce a clean recall-vs-latency curve.

1. **Choose number of queries** (50 recommended) and **top_k**
2. **Click "Run Benchmark"** — takes about a minute
3. View the chart and raw data table

### What to Look For

- At **ef=1-4**: Fast but recall drops to 40-70% — HNSW is cutting too many corners
- At **ef=64**: Good balance with 90-95% recall
- At **ef=256+**: Near-perfect recall but noticeably slower

## Why This Matters

In production systems, you must choose where on this curve to operate:

- **User-facing search**: Lower ef for fast response (<50ms), accept some missed results
- **Recommendation systems**: Higher ef for quality, batch processing OK
- **Hybrid approach**: Start with low ef, increase if results seem poor

## Files

- `app.py` - Flask web application
- `init_qdrant.py` - Database initialization (100K random vectors)
- `templates/index.html` - Interactive UI

## Technical Details

- **Vector DB**: Qdrant with HNSW index (m=8, ef_construct=64)
- **Vectors**: 100,000 random unit vectors (128 dimensions)
- **Ground Truth**: Computed via exact search for each query
- **Recall**: % of true top-k results found by approximate search
