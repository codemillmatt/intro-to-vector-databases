# Performance Tuning

Experiment with vector database parameters and see how they affect speed and results.

## Run

```bash
cd module_5/tuning
python app.py
```

Open http://localhost:8000

## What It Shows

Adjust parameters and watch the metrics change:

| Parameter | Effect |
|-----------|--------|
| **top_k** | More results = higher recall, but slower |
| **Include Metadata** | Richer results, but larger payload |

## Metrics Displayed

- **Embedding Time**: Converting your query to a vector
- **Search Time**: Finding similar vectors in Pinecone
- **Total Time**: End-to-end latency

## Experiments to Try

### 1. top_k Impact
Search with top_k=5, then 25, then 50. Watch search time increase.

### 2. Metadata Overhead
Toggle metadata on/off for the same query. Compare times.

### 3. Benchmark Mode
Run multiple iterations to see min/max/average times and measure consistency.

## Real-World Tuning

| Use Case | Recommended Settings |
|----------|---------------------|
| Autocomplete | Low top_k (3-5), minimal metadata |
| Research tool | Higher top_k (20-50), full metadata |
| RAG chatbot | Medium top_k (5-10), content metadata |
