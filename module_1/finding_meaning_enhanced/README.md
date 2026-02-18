# Enhanced SQL vs Semantic Search
Search both **book descriptions** and **full book text** using two approaches side-by-side:
- **SQL keyword search** (plain `ILIKE` — exact substring matching)
- **Semantic vector search** (Pinecone + embeddings)

> 💡 You’ll see which results came from the description vs a full-text chunk. This makes the limitations of keyword search obvious when the concept doesn’t literally appear in the text.

## 🧠 What this demo teaches
- Keyword search ≠ meaning; `ILIKE` won’t match “heal” when you search for “healing”.
- Semantic search can surface *conceptual matches* in full book content, not just metadata.
- How to merge hits from descriptions and content chunks into one per-book score.

## ⚙️ What’s different vs the standard Module 1
| Feature | Standard Demo | Enhanced Demo |
|---------|---------------|---------------|
| SQL search scope | title, description, genre | + **full book text** |
| Semantic scope | description vectors | + **chunked book content vectors** |
| UI | includes “Browse” tab | focused on search comparison |
| Result labels | none | "Description match" vs "Content match" |
| Default port | 8080 | **8081** |

## 🚀 Run (DevContainer/Codespaces preferred)
```bash
# One-time init (if you haven’t run setup yet)
cd setup
python init_postgres.py
python init_pinecone.py

# Add enhanced SQL tables/indexes
cd ../module_1/finding_meaning_enhanced
python init_enhanced_postgres.py

# Start the app
python app.py   # http://localhost:8081
```
> 🤖 Embeddings: easiest path is a local **Ollama** server (`OLLAMA_HOST=http://localhost:11434`). If Ollama isn’t available, the app automatically falls back to `sentence-transformers`.

## 🧪 Try these queries
| Query | What to notice |
|-------|----------------|
| `dragon` | Both methods find literal matches; semantic also surfaces thematic matches |
| `stories about loss and healing` | SQL might catch “loss” in text; semantic handles the concept |
| `quantum physics` | Found via book text chunk; not obvious in the description |
| `a hero's journey through darkness` | Abstract phrasing—semantic wins |
| `mystery` | Keyword matches genre, but semantic finds “mysterious” themes too |

## 🔍 How it works
**SQL keyword search** (exact substring matching):
```sql
WHERE title ILIKE '%query%'
   OR description ILIKE '%query%'
   OR genre ILIKE '%query%'
   OR full_text ILIKE '%query%'
ORDER BY rating DESC
```

**Semantic vector search**:
1. Embed the query (Ollama ➜ fallback: sentence-transformers)
2. Query Pinecone without a type filter → can return description or chunk vectors
3. Aggregate per book and label the source

## 🔁 Reset / cleanup
```bash
python init_enhanced_postgres.py --reset
```
This drops the extra full-text column/indexes added for the enhanced demo.
