# Faceted Search

Combine semantic search with metadata filters - like Amazon or Netflix search.

## Run

```bash
cd module_3/faceted_search
python app.py
```

Open http://localhost:8001

## What It Shows

Search with natural language **plus** apply filters:

- **Genre** checkboxes (Fantasy, Sci-Fi, Mystery, etc.)
- **Rating** slider (minimum stars)
- **Price** slider (maximum price)
- **In Stock** toggle

## Example

1. Type "magical adventure" in the search box
2. Check "Fantasy" genre
3. Set rating to 4+ stars
4. Enable "In Stock Only"

Results are books that match the *meaning* of your query AND meet all filter criteria.

## How It Works

```python
results = index.query(
    vector=query_embedding,           # Semantic search
    filter={
        "genre": {"$in": ["Fantasy"]},  # Exact match
        "rating": {"$gte": 4.0},        # Numeric range
        "in_stock": True                # Boolean
    }
)
```

One Pinecone query handles both vector similarity and metadata filtering.
