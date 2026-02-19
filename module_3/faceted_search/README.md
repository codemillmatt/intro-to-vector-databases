# Faceted Search
Semantic search + metadata filters (like Amazon/Netflix). Natural language finds the *right concepts*, filters narrow to the right *inventory*.

## 🧠 What this app does
- Embed the user’s query (semantic search) and filter results with Pinecone’s metadata filter.
- Demonstrates combining **meaning** (vectors) with **facets** (genre, rating, price, stock).
- Shows how to structure filters for exact matches, ranges, and booleans.

## 🚀 Run (DevContainer/Codespaces preferred)
```bash
# The DevContainer automatically initializes Pinecone on start.
# Just run the app:
cd module_3/faceted_search
python app.py   # http://localhost:8001
```
> 🤖 Embeddings: the DevContainer connects to Ollama on your host machine automatically. If Ollama isn’t available, the app falls back to `sentence-transformers`.

## 🧪 Try this flow
1. Search: `magical adventure`
2. Filters: check **Fantasy**, rating ≥ 4, **In Stock only**, price ≤ $15
3. Notice how semantic search finds “magical journey” even if those exact words aren’t in metadata, while filters enforce the numeric/boolean constraints.

## 🔧 How it works (code sketch)
```python
results = index.query(
    vector=query_embedding,           # semantic similarity
    top_k=10,
    filter={                          # metadata facets
        "genre": {"$in": ["Fantasy"]},
        "rating": {"$gte": 4.0},
        "price": {"$lte": 15.0},
        "in_stock": True,
    },
    include_metadata=True,
)
```
> One Pinecone query handles both the vector search **and** the structured filters. No need for a separate SQL filter step.

## Reset / re-init
```bash
cd setup && python init_pinecone.py --reset
```
