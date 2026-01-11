# Module 3: Metadata Filtering

**Combining Semantic Search with Structured Filters**

This demo shows how vector databases go beyond simple similarity search by letting you filter results using metadata—just like a traditional database, but applied to semantically-matched results.

---

## What This Demo Does

1. **Semantic search** — You type a natural language query (e.g., "books about adventure")
2. **Vector similarity** — The query is converted to an embedding and matched against book embeddings
3. **Metadata filtering** — You can narrow results by genre, rating, price, or stock status
4. **Side-by-side comparison** — See results with and without filters to understand the difference

---

## Why Metadata Filtering Matters

Traditional databases use **keyword matching**:
```sql
SELECT * FROM books WHERE description LIKE '%adventure%' AND genre = 'Fantasy';
```
This misses books about "quests" or "journeys" that don't contain the word "adventure."

Vector databases use **semantic matching + filtering**:
```
Query: "adventure"  →  embedding  →  find similar books  →  filter by genre = "Fantasy"
```
This finds books that are *conceptually* about adventure, then narrows by your business rules.

### The Key Insight

Metadata filtering happens **after** vector similarity:

1. Your query becomes a vector
2. The database finds the most similar book vectors
3. Filters are applied to those matches

This means you're filtering **conceptually similar items**, not just keyword matches.

---

## Running the Demo

```bash
cd module_3/metadata_filtering
python main.py
```

You'll see an interactive prompt:

1. Enter a search query (e.g., "exciting stories with magic")
2. View unfiltered results ranked by semantic similarity
3. Add optional filters:
   - **Genre** — Fantasy, Science Fiction, Mystery, etc.
   - **Minimum rating** — 1 to 5 stars
   - **Maximum price** — e.g., 15 for books under $15
   - **In stock only** — yes/no
4. View filtered results and compare

Type `quit` to exit.

---

## Available Filters

| Filter | Description | Example |
|--------|-------------|---------|
| Genre | Exact match on book genre | `Fantasy`, `Mystery` |
| Min Rating | Books rated at or above this value | `4.0` (4+ stars) |
| Max Price | Books priced at or below this value | `20` (under $20) |
| In Stock | Only show books currently available | `yes` |

---

## How It Works (Code Walkthrough)

### 1. Generate Query Embedding

```python
embedding_client = get_embedding_client()
query_embedding = embedding_client.embed(query)
```

Your natural language query is converted into a 384-dimensional vector that captures its semantic meaning.

### 2. Build Metadata Filter

```python
filters = {"type": "book"}

if genre:
    filters["genre"] = genre

if min_rating is not None:
    filters["rating"] = {"$gte": min_rating}  # greater than or equal

if max_price is not None:
    filters["price"] = {"$lte": max_price}    # less than or equal

if in_stock_only:
    filters["in_stock"] = True
```

Pinecone uses a JSON-style filter syntax with operators like `$gte` (≥) and `$lte` (≤).

### 3. Query the Vector Database

```python
results = index.query(
    vector=query_embedding,
    top_k=10,
    include_metadata=True,
    filter=filters
)
```

This single call:
- Finds the 10 most similar vectors to your query
- Filters out any that don't match your metadata criteria
- Returns results with their similarity scores

---

## Example Session

```
Enter your search query: magical adventure

Search WITHOUT filters:
┌─────────────────────────┬──────────────────┬────────┬────────┬───────┬───────┐
│ Title                   │ Author           │ Genre  │ Rating │ Price │ Score │
├─────────────────────────┼──────────────────┼────────┼────────┼───────┼───────┤
│ The Dragon's Apprentice │ Sarah Windholm   │ Fantasy│ 4.5★   │ $14.99│ 0.847 │
│ Starship Legacy         │ Marcus Chen      │ Sci-Fi │ 4.2★   │ $16.99│ 0.812 │
│ The Hidden Valley       │ Emma Blackwood   │ Fantasy│ 4.0★   │ $12.99│ 0.798 │
└─────────────────────────┴──────────────────┴────────┴────────┴───────┴───────┘

Filter by genre: Fantasy
Minimum rating: 4
Maximum price: 15
Only show in-stock items? yes

Search WITH filters:
┌─────────────────────────┬──────────────────┬────────┬────────┬───────┬───────┐
│ Title                   │ Author           │ Genre  │ Rating │ Price │ Score │
├─────────────────────────┼──────────────────┼────────┼────────┼───────┼───────┤
│ The Dragon's Apprentice │ Sarah Windholm   │ Fantasy│ 4.5★   │ $14.99│ 0.847 │
│ The Hidden Valley       │ Emma Blackwood   │ Fantasy│ 4.0★   │ $12.99│ 0.798 │
└─────────────────────────┴──────────────────┴────────┴────────┴───────┴───────┘
```

Notice how the Sci-Fi book was filtered out (wrong genre), and results still maintain their semantic relevance order.

---

## Real-World Use Cases

| Scenario | Semantic Query | Metadata Filters |
|----------|---------------|------------------|
| E-commerce | "comfortable running shoes" | brand, price range, size, in stock |
| Job search | "remote data science role" | salary range, experience level, location |
| Content platform | "funny cat videos" | upload date, duration, view count |
| Recipe finder | "quick healthy dinner" | prep time < 30min, vegetarian, cuisine |

---

## Key Takeaways

1. **Best of both worlds** — Semantic understanding + structured filtering
2. **Filter operators** — Pinecone supports `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`
3. **Order matters** — Filters narrow vector matches, they don't replace semantic search
4. **Performance** — Metadata indexes in Pinecone are optimized for fast filtering at scale

---

## Next Steps

- **Module 4**: See how this applies to RAG (Retrieval Augmented Generation) and product recommendations
- **Module 5**: Learn how to tune vector search for latency vs. recall tradeoffs
