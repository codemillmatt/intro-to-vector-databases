# Product Recommendations
Content-based vs collaborative recommendations side-by-side, using the same bookstore dataset.

## 🧠 What this app demonstrates
- **Content-based filtering**: average embeddings of liked books → find nearest neighbors (via Pinecone).
- **Collaborative filtering**: find users with similar history → recommend what they liked that you haven’t read.
- Simple UI to compare both approaches for the same user.

## 🚀 Run (DevContainer/Codespaces preferred)
```bash
# One-time init (if not already done)
cd setup && python init_pinecone.py

# Run the app
cd ../module_4/recommendation_webapp
python app.py   # http://localhost:5001
```
> ⚠️ Port clash: shares port **5001** with the RAG demo. Stop one before starting the other.
> 🤖 Embeddings: local **Ollama** is easiest (`OLLAMA_HOST=http://localhost:11434`). Fallback to `sentence-transformers` is automatic if Ollama is unavailable.

## 🔍 How it works
**Content-based**:
```python
user_vector = np.mean(book_embeddings)   # books the user liked
neighbors = pinecone_index.query(vector=user_vector, top_k=5)
```
**Collaborative**:
```python
similar_users = find_similar_users(history)
ranked_recs = score_books_from(similar_users, exclude=already_read)
```

## 🧪 Try it
1. Browse user profiles in the UI.
2. Pick a user; observe the two recommendation lists.
3. Add/remove liked books and see content-based recs update immediately.
4. Compare overlap/divergence between the two methods—hybrid strategies often work best.

## Reset / re-init
```bash
cd setup && python init_pinecone.py --reset
```
