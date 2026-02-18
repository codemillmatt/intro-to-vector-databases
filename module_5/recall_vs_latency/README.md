# Recall vs Latency Demo
Explore the classic trade-off between **recall** (accuracy) and **latency** (speed) in ANN search using Qdrant + HNSW.

## 🚀 Run (DevContainer/Codespaces preferred)
```bash
# One-time init (not done automatically by the DevContainer)
cd module_5/recall_vs_latency
python init_qdrant.py --reset   # generates and inserts 100K random vectors

# Start the UI
python app.py                    # http://localhost:8080
```
> This demo doesn’t depend on embeddings or Ollama — everything is synthetic. The Qdrant service is started automatically by Docker Compose in the DevContainer, but you must run `init_qdrant.py` yourself to populate it. For running outside the container, see [RUNNING_LOCALLY.md](../../RUNNING_LOCALLY.md).

## 🧠 Concept recap
- **Exact search** scans every vector → 100% recall, slower.
- **Approximate search** (HNSW) explores a subset → fast, may miss some neighbors.
- **`ef` (exploration factor)** controls the trade-off:

| ef Value | Behavior |
|----------|----------|
| Low (1–8) | Very fast, low recall |
| Medium (32–64) | Good balance |
| High (128–512) | Near-perfect recall, slower |

## Why 100K random vectors?
Real embeddings cluster; even small `ef` can look “too good”. Using **100,000 uniformly random vectors (128-D)** makes the ANN trade-off visible and measurable.

## 🔍 Using the app
Two tabs:
1. **Exact vs Approximate**
   - Slider for `ef`
   - “Compare” shows true top-k vs HNSW results (color-coded: ✅ both, ❌ missed, ⚠️ wrong)
   - “New Random Query” generates a fresh query vector
2. **Batch Benchmark**
   - Runs many queries per `ef`
   - Plots recall vs latency
   - Table shows per-ef metrics

### What to look for
- `ef=1–4`: blazing fast, recall can drop to 40–70%
- `ef≈64`: 90–95% recall, good default
- `ef≥256`: near-perfect recall, but latency increases

## Key files
- `init_qdrant.py` — sets up the collection (`m=8`, `ef_construct=64`)
- `app.py` — Flask UI (port 8080)
- `templates/index.html` — single-page UI

## Bonus: Book embeddings variant
There’s also a smaller book-focused variant:
```bash
python init_qdrant_books.py --reset
python app_books.py   # uses book embeddings instead of random vectors
```

## Health checks
```bash
curl http://qdrant:6333/healthz   # inside devcontainer
curl http://localhost:6333/healthz  # running locally
```

## Reset / info
```bash
python init_qdrant.py --reset   # wipe & rebuild
python init_qdrant.py --info    # show collection stats
```
