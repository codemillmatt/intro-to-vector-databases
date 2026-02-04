# SQL vs Semantic Search

Compare traditional keyword search against vector similarity search side-by-side.

## Run

```bash
cd module_1/finding_meaning_webapp
python app.py
```

Open http://localhost:8080

## What It Shows

| Left Panel (SQL) | Right Panel (Vector) |
|------------------|----------------------|
| Finds exact keyword matches | Finds similar *meaning* |
| "space" only finds books with "space" in them | "space exploration" finds sci-fi even without the word "space" |

## Try These Queries

- "books about space exploration and adventure"
- "stories dealing with loss and healing"  
- "thriller with technology and hacking"
- "fantasy with magical creatures"

Notice how SQL often returns nothing while vector search finds relevant results.

## How It Works

1. **SQL Search**: `WHERE title ILIKE '%query%' OR description ILIKE '%query%'`
2. **Vector Search**: Convert query to embedding, find nearest neighbors in Pinecone

The vector search includes a similarity score (0-1) showing how close each result is to your query.
