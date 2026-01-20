# Finding Meaning Web Application

A web-based demonstration comparing traditional SQL keyword search against semantic vector similarity search. This interactive demo makes it easy to see the immediate value of vector databases.

## 🎯 What This Demo Demonstrates

This web application shows the core concept of **Module 1: Finding Meaning** - the difference between:

1. **SQL Keyword Search (Traditional)**: Matches exact words in book titles, descriptions, and genres
2. **Semantic Vector Search (Modern)**: Understands meaning and finds conceptually related content

### Key Learning Points

- **Keyword Limitations**: SQL search only finds exact word matches, missing books that are conceptually relevant
- **Semantic Understanding**: Vector search finds books based on meaning, even without keyword matches
- **Real-World Value**: See immediately how vector databases solve the "semantic gap" problem
- **Side-by-Side Comparison**: Direct visual comparison helps understand the differences

## 📋 Prerequisites

Before running this demo, ensure you have:

1. **Completed the setup** from the main repository README:
   - DevContainer running with Docker services (PostgreSQL, Pinecone Local, Ollama)
   - Databases initialized with book data and embeddings

2. **Required Services Running**:
   - PostgreSQL (port 5432) - for SQL search
   - Pinecone Local (ports 5081/5082) - for vector search
   - Ollama (port 11434) - for generating embeddings

## 🚀 How to Run

### From DevContainer (Recommended)

1. **Open in DevContainer**:
   ```bash
   # VS Code should prompt you to reopen in container
   # Or use Command Palette: "Dev Containers: Reopen in Container"
   ```

2. **Navigate to the web app**:
   ```bash
   cd module_1/finding_meaning_webapp
   ```

3. **Run the Flask application**:
   ```bash
   python app.py
   ```

4. **Open in your browser**:
   ```
   http://localhost:5000
   ```

### From Host Machine (Alternative)

If running outside the DevContainer, set these environment variables first:

```bash
export PINECONE_HOST=http://localhost:5081
export PINECONE_INDEX_HOST=http://localhost:5082
export OLLAMA_HOST=http://localhost:11434
export POSTGRES_HOST=localhost

cd module_1/finding_meaning_webapp
python app.py
```

Then open `http://localhost:5000` in your browser.

## 🎮 Using the Demo

1. **Enter a search query** in the search box. Try natural language descriptions like:
   - "books about space exploration and adventure"
   - "stories dealing with loss and healing"
   - "thriller with technology and hacking"

2. **Click Search** or press Enter

3. **Compare the results**:
   - **Left panel (SQL)**: Shows books that contain your search keywords
   - **Right panel (Semantic)**: Shows books that are conceptually similar to your query

4. **Try the example queries** by clicking the suggestion buttons

### 💡 Example Queries to Try

These queries demonstrate semantic search particularly well:

- **"books about space exploration and adventure"** - Finds sci-fi books without requiring the word "space"
- **"stories dealing with loss and healing"** - Finds emotional, introspective books based on themes
- **"thriller with technology and hacking"** - Finds techno-thrillers using concept matching
- **"fantasy with magical creatures"** - Finds fantasy books by understanding the genre concepts

## 🔍 Understanding the Code

### Backend (app.py)

**SQL Search Function** (`sql_search`):
```python
# Uses PostgreSQL ILIKE for case-insensitive pattern matching
# Searches title, description, and genre fields
# Simple but limited to exact keyword matches
WHERE title ILIKE %s OR description ILIKE %s OR genre ILIKE %s
```

**Semantic Search Function** (`semantic_search`):
```python
# 1. Convert query to embedding vector
query_embedding = embedding_client.embed(query)

# 2. Search Pinecone for similar vectors
results = index.query(
    vector=query_embedding,
    top_k=5,
    filter={"type": "book"}  # Only search book descriptions
)

# 3. Returns books with similarity scores
```

**Key Differences**:
- SQL search is fast but literal - finds text patterns
- Semantic search is intelligent - finds similar meanings
- Semantic search includes a similarity score (0-1)

### Frontend (templates/index.html)

**Simple Architecture**:
- Clean, minimal design focused on comparison
- Side-by-side results panels for easy comparison
- Color-coded: SQL (cyan) vs Semantic (red)
- Shows similarity scores for semantic results
- Responsive design works on mobile

**User Experience**:
- Example queries guide users to good test cases
- Real-time search with loading states
- Clear visual distinction between search types
- Easy to understand which approach worked better

## 🎓 Learning Outcomes

After using this demo, you'll understand:

1. **Why keyword search fails**: It can't understand meaning or context
2. **How vector search helps**: It finds conceptually related content
3. **When to use vector databases**: Any time semantic understanding matters
4. **The "semantic gap"**: The difference between words and meaning

## 🛠 Technical Details

### Dependencies
- **Flask**: Lightweight web framework
- **psycopg2**: PostgreSQL database adapter
- **pinecone**: Vector database client
- **embeddings.py**: Shared embedding utilities (from `/setup`)
- **pinecone_utils.py**: Shared Pinecone connection utilities (from `/setup`)

### Architecture
```
User Input (Web Browser)
    ↓
Flask Web Server (app.py)
    ↓
    ├── SQL Search → PostgreSQL (keyword matching)
    ↓
    └── Semantic Search → Pinecone (vector similarity)
         ↓
         Embeddings (via Ollama/sentence-transformers)
```

### Embedding Model
- Uses **all-MiniLM-L6-v2** (384 dimensions)
- Fast, efficient, good quality for general text
- Same model used across all modules for consistency

## 🔧 Customization

### Change the Port
```bash
PORT=8080 python app.py
```

### Adjust Results Count
Edit `app.py` and change `top_k=5` to return more/fewer results.

### Modify Search Fields
Edit the SQL query in `sql_search()` to include/exclude fields:
```python
WHERE title ILIKE %s 
   OR description ILIKE %s
   OR author ILIKE %s  # Add author search
```

## ❓ Troubleshooting

### Error: "Connection refused" (Pinecone or PostgreSQL)

Make sure services are running:
```bash
docker ps
# Should show: postgres, pinecone, ollama containers
```

### Error: "No results found"

The databases may not be initialized:
```bash
cd setup
python init_postgres.py
python init_pinecone.py
```

### Error: "Module not found"

Install dependencies:
```bash
pip install -r ../../requirements.txt
```

### Port 5000 Already in Use

Change the port:
```bash
PORT=8080 python app.py
```

## 📚 Related Demos

- **CLI Version**: `/module_1/finding_meaning/main.py` - Command-line interface for the same demo
- **Module 3**: Metadata filtering - extends semantic search with filters
- **Module 4**: RAG applications - uses semantic search for question answering
- **Module 5**: Tuning demo - another web interface showing performance tuning

## 📖 Further Reading

- [Module 1 Learning Content](https://learn.microsoft.com/training/modules/intro-to-vector-databases/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Sentence Transformers](https://www.sbert.net/)

## 📝 License

See the main repository [LICENSE](../../LICENSE) file for details.
