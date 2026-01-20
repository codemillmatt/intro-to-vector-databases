# Product Recommendation Web Application

A visual demonstration of how vector databases power modern recommendation systems, combining **content-based filtering** and **collaborative filtering** techniques.

## What This Sample Does

This web application showcases recommendation patterns commonly used in e-commerce, streaming services, and content platforms:

1. **Browse User Profiles**: See what books different users have liked, providing transparency into the data that drives collaborative recommendations
2. **Custom Selection**: Choose your own books to see personalized recommendations
3. **Side-by-Side Comparison**: View content-based and collaborative recommendations together to understand how each approach surfaces different results

## Key Features

### 👥 User Profile Browser
See all users and their reading preferences at a glance. Click any user to:
- View the books they've liked
- See which other users share similar tastes
- Get recommendations based on their profile

This transparency helps illustrate **why** certain books are recommended—you can trace the path from liked books to recommendations.

### 🧠 Content-Based Filtering
Uses vector similarity to find books with similar themes and content:
- Averages embeddings of liked books to create a "preference vector"
- Queries the vector database for nearest neighbors
- Returns books with semantically similar content

### 👥 Collaborative Filtering  
Recommends books based on what similar users enjoyed:
- Finds users with overlapping book preferences
- Weights recommendations by similarity (more shared likes = higher influence)
- Shows which users influenced each recommendation

## Running the Application

```bash
cd module_4/recommendation_webapp
python app.py
```

The application will start on `http://localhost:5001`.

## Why This Matters for Vector Databases

### The Problem
Traditional recommendation systems struggle with:
- **Cold start**: New items with no interaction history
- **Content understanding**: Relying only on keywords misses semantic similarity
- **Scalability**: Finding similar items in millions of records

### The Vector Database Solution
By storing item embeddings in a vector database:

| Challenge | How Vector DBs Help |
|-----------|---------------------|
| Cold start | New items can be recommended immediately based on content similarity |
| Semantic understanding | Embeddings capture meaning, not just keywords |
| Scale | Approximate nearest neighbor search finds similar items in milliseconds |
| Hybrid approaches | Combine content and behavioral signals seamlessly |

## Code Highlights

### Preference Vector Creation
```python
def average_embeddings(embeddings):
    """Create a 'taste profile' by averaging liked item embeddings."""
    num_dims = len(embeddings[0])
    avg = []
    for i in range(num_dims):
        total = sum(emb[i] for emb in embeddings)
        avg.append(total / len(embeddings))
    return avg
```

### Vector Similarity Search
```python
# Query Pinecone for similar books
results = index.query(
    vector=preference_vector,  # User's taste profile
    top_k=5,
    include_metadata=True,
    filter={"type": "book"}
)
```

### Collaborative Filtering with Transparency
```python
# Track which users influenced each recommendation
for similar_user in similar_users:
    for book_id in similar_user["unique_recommendations"]:
        book_sources[book_id].append(similar_user["name"])
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/api/recommend` | POST | Get recommendations for selected books |
| `/api/user/<id>/recommendations` | GET | Get recommendations for a specific user |

## Educational Value

This demo is designed to make recommendation systems understandable:

1. **Transparency**: See the users and books that drive recommendations
2. **Comparison**: Content-based vs collaborative side-by-side
3. **Explainability**: "Recommended because users X, Y, Z also liked it"
4. **Interactivity**: Experiment with different book combinations
