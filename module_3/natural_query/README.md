# Module 3: Natural Language Query

**Parse a Single Query into Semantic Search + Metadata Filters**

This demo shows how to take a natural language query like **"fantasy under $15"** and automatically extract both the semantic intent and structured filters—then combine them in one vector database query.

---

## What This Demo Does

1. **Parse** your query for metadata patterns (price, rating, genre, stock)
2. **Extract** structured filters from the patterns found
3. **Use remaining text** as the semantic search query
4. **Execute** a combined vector + metadata search

---

## Running the Demo

```bash
cd module_3/natural_query
python main.py
```

---

## Example Queries

| Query | Semantic Search | Extracted Filters |
|-------|-----------------|-------------------|
| `fantasy under $15` | "fantasy" | price ≤ $15.00 |
| `mystery books rated 4+ stars` | "mystery" | rating ≥ 4.0★ |
| `science fiction in stock` | "science fiction" | in stock only, genre = Science Fiction |
| `adventure stories under $20 with 4 stars or better` | "adventure stories" | price ≤ $20.00, rating ≥ 4.0★ |
| `romance available` | "romance" | in stock only, genre = Romance |
| `thriller below $18` | "thriller" | price ≤ $18.00, genre = Thriller |

---

## Supported Patterns

### Price Filters

| Pattern | Example | Filter |
|---------|---------|--------|
| `under $X` | "under $15" | price ≤ 15 |
| `below $X` | "below $20" | price ≤ 20 |
| `less than $X` | "less than $12" | price ≤ 12 |
| `over $X` | "over $10" | price ≥ 10 |
| `above $X` | "above $25" | price ≥ 25 |
| `max $X` | "max $15" | price ≤ 15 |

### Rating Filters

| Pattern | Example | Filter |
|---------|---------|--------|
| `rated X+` | "rated 4+" | rating ≥ 4 |
| `X+ stars` | "4+ stars" | rating ≥ 4 |
| `above X stars` | "above 4 stars" | rating ≥ 4 |
| `X stars or better` | "4 stars or better" | rating ≥ 4 |
| `at least X stars` | "at least 3.5 stars" | rating ≥ 3.5 |

### Stock Filters

| Pattern | Example | Filter |
|---------|---------|--------|
| `in stock` | "fantasy in stock" | in_stock = true |
| `available` | "mystery available" | in_stock = true |

### Genre Filters

Recognized genres: Fantasy, Science Fiction (sci-fi, scifi), Mystery, Thriller, Romance, Historical Fiction, Literary Fiction, Memoir

---

## Example Session

```
Enter your query: fantasy under $15

┌─────────────────────────────────────────────────────────────┐
│ Query Parsing                                               │
├─────────────────────────────────────────────────────────────┤
│ Original query: fantasy under $15                           │
│                                                             │
│ Semantic search: "fantasy"                                  │
│                                                             │
│ Extracted filters: price ≤ $15.00, genre = Fantasy          │
└─────────────────────────────────────────────────────────────┘

                        Search Results
┌────────────────────┬──────────────────┬─────────┬────────┬────────┬───────┐
│ Title              │ Author           │ Genre   │ Rating │ Price  │ Score │
├────────────────────┼──────────────────┼─────────┼────────┼────────┼───────┤
│ The Dragon's App...│ Sarah Windholm   │ Fantasy │ 4.5★   │ $14.99 │ 0.823 │
│ The Hidden Valley  │ Emma Blackwood   │ Fantasy │ 4.0★   │ $12.99 │ 0.756 │
└────────────────────┴──────────────────┴─────────┴────────┴────────┴───────┘
```

---

## How It Works

```
User query: "adventure stories under $20 with 4 stars or better"
                              │
                              ▼
                    ┌─────────────────┐
                    │  Query Parser   │
                    └─────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Semantic Text │    │ Price Filter  │    │ Rating Filter │
│  "adventure   │    │ price ≤ $20   │    │ rating ≥ 4.0  │
│   stories"    │    │               │    │               │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        ▼                     └──────────┬──────────┘
┌───────────────┐                        ▼
│   Embedding   │              ┌───────────────────┐
│   (vector)    │              │ Metadata Filters  │
└───────────────┘              └───────────────────┘
        │                                │
        └────────────────┬───────────────┘
                         ▼
              ┌─────────────────────┐
              │   Pinecone Query    │
              │  vector + filters   │
              └─────────────────────┘
                         │
                         ▼
                    Results!
```

---

## Key Differences from `metadata_filtering`

| Aspect | metadata_filtering | natural_query |
|--------|-------------------|---------------|
| Input | Separate prompts for query and each filter | Single natural language query |
| UX | Step-by-step, explicit | Conversational, intuitive |
| Parsing | User provides structured input | Automatic extraction via regex |
| Best for | Teaching the concepts | Real-world user experience |

---

## Limitations

This demo uses simple regex pattern matching. In production, you might use:

- **LLM-based parsing** — Use GPT/Claude to extract structured data from queries
- **NER (Named Entity Recognition)** — Identify entities like prices, ratings, genres
- **Query understanding models** — Purpose-built models for e-commerce search

---

## Real-World Applications

This pattern is used in:

- **E-commerce search**: "red Nike shoes under $100 size 10"
- **Job boards**: "remote senior engineer $150k+ python"
- **Real estate**: "3 bedroom house under $500k with pool"
- **Travel**: "flights to Paris under $500 direct"

The key insight: **users think in natural language, but databases need structured queries**. This bridge makes search feel magical.
