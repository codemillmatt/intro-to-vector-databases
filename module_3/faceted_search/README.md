# Module 3: Faceted Search UI

**Production-Style Search Interface with Semantic + Metadata Filters**

This demo shows how most real-world search systems work: a search box for semantic queries combined with UI controls (checkboxes, sliders, toggles) for metadata filtering.

---

## What This Demo Shows

This is how sites like Amazon, Netflix, and Airbnb implement search:

```
┌─────────────────────────────────────────────────────────────┐
│  🔍 [magical adventure                    ] [Search]        │
├─────────────────────────────────────────────────────────────┤
│  FILTERS           │  RESULTS                               │
│                    │                                        │
│  Genre:            │  ┌─────────────────────────────────┐  │
│  ☑ Fantasy         │  │ The Dragon's Apprentice         │  │
│  ☐ Sci-Fi          │  │ by Sarah Windholm               │  │
│  ☐ Mystery         │  │ Fantasy │ ★4.5 │ $14.99         │  │
│                    │  └─────────────────────────────────┘  │
│  Min Rating:       │                                        │
│  ★★★★☆ (4+)        │  ┌─────────────────────────────────┐  │
│                    │  │ The Midnight Garden              │  │
│  Max Price:        │  │ by Elena Vasquez                │  │
│  [$0]━━━●━━[$50]   │  │ Fantasy │ ★4.2 │ $12.99         │  │
│       $20          │  └─────────────────────────────────┘  │
│                    │                                        │
│  ☑ In Stock Only   │                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Running the Demo

```bash
cd module_3/faceted_search
python app.py
```

Then open **http://localhost:8001** in your browser.

---

## Features

| UI Element | Filter Type | Pinecone Query |
|------------|-------------|----------------|
| Search box | Semantic (vector) | `vector=embed(query)` |
| Genre checkboxes | Exact match | `{"genre": {"$in": [...]}}` |
| Rating slider | Numeric range | `{"rating": {"$gte": 4.0}}` |
| Price slider | Numeric range | `{"price": {"$lte": 20}}` |
| Stock toggle | Boolean | `{"in_stock": true}` |

---

## How It Works

### 1. User interacts with UI

- Types "magical adventure" in search box
- Checks "Fantasy" genre
- Sets rating slider to 4+
- Enables "In Stock Only"

### 2. Frontend sends structured request

```javascript
{
    "query": "magical adventure",
    "genres": ["Fantasy"],
    "min_rating": 4.0,
    "max_price": null,
    "in_stock_only": true
}
```

### 3. Backend builds Pinecone query

```python
# Semantic part
query_embedding = embedding_client.embed("magical adventure")

# Metadata filters
filters = {
    "type": "book",
    "genre": {"$in": ["Fantasy"]},
    "rating": {"$gte": 4.0},
    "in_stock": True
}

# Combined query
results = index.query(
    vector=query_embedding,
    filter=filters,
    top_k=20
)
```

### 4. Results returned

Books that are:
- Semantically similar to "magical adventure"
- In the Fantasy genre
- Rated 4+ stars
- Currently in stock

---

## Why This Pattern?

### Compared to natural language parsing:

| Aspect | Natural Language | Faceted UI |
|--------|-----------------|------------|
| User effort | Low (just type) | Medium (use controls) |
| Precision | Depends on parsing | Exact (user selects) |
| Discoverability | Low | High (see all options) |
| Implementation | Complex (LLM/NLP) | Simple (form data) |
| Edge cases | Many | Few |

### Faceted UI wins for e-commerce because:

1. **Users see available options** — No guessing what filters exist
2. **No parsing errors** — User explicitly selects values
3. **Faster to implement** — No NLP/LLM required
4. **Works in all languages** — Just translate labels

---

## Code Structure

```
faceted_search/
├── app.py              # Flask backend with search logic
├── templates/
│   └── index.html      # Frontend with filters and results
└── README.md           # This file
```

---

## Key Takeaways

1. **Separation of concerns** — UI handles structured input, backend handles queries
2. **Real-time filtering** — Filters auto-trigger search on change
3. **Combined query** — One Pinecone call handles both vector + metadata
4. **Best of both worlds** — Semantic understanding + precise filtering

---

## Extending This Demo

Ideas for enhancement:

- **Sort options** — By relevance, price, rating
- **Pagination** — Handle large result sets
- **Filter counts** — Show "(12)" next to each genre
- **Price histogram** — Show price distribution
- **Saved searches** — Remember user preferences
