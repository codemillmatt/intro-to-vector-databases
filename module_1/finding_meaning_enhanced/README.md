# Enhanced SQL vs Semantic Search

An improved version of the Finding Meaning demo that searches both book descriptions **and full book text content** on both the SQL and semantic sides. SQL uses plain `ILIKE` substring matching (no stemming, no fuzzy logic), making the contrast with semantic search especially clear.

## What's Different from the Standard Demo

| Feature | Standard Demo | Enhanced Demo |
|---------|---------------|---------------|
| SQL search method | `ILIKE` keyword matching | `ILIKE` keyword matching (same, but wider scope) |
| SQL search scope | Title, description, genre | Title, description, genre, **full book text** |
| Semantic search scope | Book descriptions only | Book descriptions **+ text content chunks** |
| Browse tab | All Books listing | Removed — focused on search comparison |
| Match source | Not shown | Label: "Description match" vs "Content match" |
| Port | 8080 | 8081 |

## Setup

**Prerequisites**: The base databases must be initialized first.

```bash
# 1. Initialize PostgreSQL (if not already done)
cd setup
python init_postgres.py

# 2. Initialize Pinecone vectors (if not already done)
python init_pinecone.py

# 3. Add full-text search capabilities
cd ../module_1/finding_meaning_enhanced
python init_enhanced_postgres.py
```

## Run

```bash
cd module_1/finding_meaning_enhanced
python app.py
```

Open http://localhost:8081

## Try These Queries

| Query | What to notice |
|-------|----------------|
| `dragon` | Both methods find keyword matches; semantic also finds thematically similar books |
| `stories about loss and healing` | SQL may find "loss" or "healing" in book text; semantic understands the concept |
| `quantum physics` | Finds Book 002 via book text content — not just the description |
| `a hero's journey through darkness` | SQL struggles with abstract concepts; semantic excels |
| `mystery` | Both find genre matches, but semantic also finds books with mysterious themes |

## How It Works

### SQL Keyword Search
```sql
-- Plain ILIKE substring matching — no stemming, no tokenization
-- "foxes" will NOT match "fox", "healing" will NOT match "heal"
WHERE title ILIKE '%query%'
   OR description ILIKE '%query%'
   OR genre ILIKE '%query%'
   OR full_text ILIKE '%query%'
ORDER BY rating DESC
```
Even with full book text available, ILIKE only finds exact substring matches.

### Semantic Vector Search
1. Embeds the query using the shared embedding model (Ollama or sentence-transformers)
2. Searches Pinecone with **no type filter** — matches both book-description vectors and book-text-chunk vectors
3. Aggregates results per book, keeping the highest similarity score
4. Labels each result as "Description match" or "Content match"

## Reset

To remove the enhanced PostgreSQL changes:

```bash
cd module_1/finding_meaning_enhanced
python init_enhanced_postgres.py --reset
```
