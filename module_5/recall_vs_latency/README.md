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

## Running the Demo

### 1. Start the DevContainer

The DevContainer includes Qdrant automatically.

### 2. Initialize the Database

```bash
cd module_5/recall_vs_latency
python init_qdrant.py --reset
```

This generates and inserts 100K random vectors. Takes 1-2 minutes.

### 3. Run the Flask App

```bash
python app.py
```

Open http://localhost:8080 in your browser.

## Using the Demo

1. **Adjust the ef slider** and click "Search" to see recall/latency for one ef value
2. **Click "Generate Full Curve"** to see the complete tradeoff visualization
3. **Click "New Random Query"** to try different query vectors

### What to Look For

- At **ef=1-4**: Fast (~1-3ms) but recall drops to 40-70%
- At **ef=64**: Good balance (~5-10ms) with 90-95% recall  
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
