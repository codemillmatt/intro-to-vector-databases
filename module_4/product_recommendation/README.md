# Product Recommendation System

A demonstration of how vector databases power modern recommendation engines, combining **content-based filtering** and **collaborative filtering** techniques.

## What This Sample Does

This demo simulates a book recommendation system that suggests titles based on your reading preferences. It showcases two fundamental recommendation approaches:

1. **Content-Based Filtering**: Finds books semantically similar to ones you've enjoyed
2. **Collaborative Filtering**: Recommends books liked by readers with similar tastes

When you run the demo, you can:
- Browse available books in the catalog
- Select books you've enjoyed
- Receive personalized recommendations from both methods
- Compare how each approach surfaces different suggestions

## Why This Matters for Vector Databases

Vector databases are revolutionizing recommendation systems. Here's why:

### The Traditional Problem
Classic recommendation systems relied on:
- **Keyword matching**: Limited to exact term matches
- **Category-based filtering**: Crude groupings that miss nuance
- **Purchase history correlation**: Only works with large datasets

### The Vector Database Solution
By converting items into high-dimensional vectors (embeddings), vector databases enable:

| Capability | Benefit |
|------------|---------|
| **Semantic Understanding** | Recommend a mystery novel to someone who liked thrillers—even if they share no common keywords |
| **Preference Modeling** | Average multiple item vectors to create a "taste profile" |
| **Real-Time Similarity Search** | Find the nearest neighbors to any vector in milliseconds |
| **Hybrid Approaches** | Combine content signals with behavioral data seamlessly |

### Real-World Applications
This same pattern powers:
- **E-commerce**: "Customers who viewed this also liked..."
- **Streaming services**: Movie and music recommendations
- **Job boards**: Matching candidates to opportunities
- **Content platforms**: Article and video suggestions

## Code Highlights

### Creating a User Preference Vector

The system creates a "preference profile" by averaging the embeddings of items a user liked:

```python
def average_embeddings(embeddings: list[list[float]]) -> list[float]:
    """Calculate the average of multiple embedding vectors."""
    if not embeddings:
        # Get dimension from embedding client for consistency
        embedding_client = get_embedding_client()
        return [0.0] * embedding_client.dimension
    
    num_dims = len(embeddings[0])
    avg = []
    for i in range(num_dims):
        total = sum(emb[i] for emb in embeddings)
        avg.append(total / len(embeddings))
    
    return avg
```

This averaged vector represents the user's overall taste, positioned in the semantic space where similar content clusters together.

### Content-Based Recommendation

The `recommend_similar_books()` function demonstrates the core vector search pattern:

```python
def recommend_similar_books(liked_books: list[str], ...) -> list[dict]:
    # Get embeddings for liked books
    embeddings = get_book_embeddings(liked_books)
    
    # Create a "preference vector" by averaging liked book embeddings
    preference_vector = average_embeddings(embeddings)
    
    # Search for similar books using vector similarity
    results = index.query(
        vector=preference_vector,
        top_k=top_k,
        include_metadata=True,
        filter={"type": "book"}
    )
```

**Key insight**: The preference vector acts as a query point in semantic space. Books near this point share thematic elements with the user's favorites.

### Collaborative Filtering Enhancement

While content-based filtering uses vectors for item similarity, collaborative filtering adds the social dimension:

```python
def recommend_collaborative(user_liked_books, all_users, ...):
    # Find users with overlapping tastes
    for user in all_users:
        overlap = set(user["liked_books"]) & set(user_liked_books)
        similarity = len(overlap) / max(len(user["liked_books"]), len(user_liked_books))
```

**Production tip**: In real systems, user preferences themselves are often embedded into vectors, enabling even faster similarity lookups between users.

### Metadata Filtering

Note how the system uses metadata filters to scope searches:

```python
results = index.query(
    vector=preference_vector,
    filter={"type": "book"}  # Only match book entries, not text chunks
)
```

This ensures recommendations come from the right category of items in the vector index.

## Running the Demo

```bash
# From the project root
python module_4/product_recommendation/main.py
```

Follow the interactive prompts to:
1. View the book catalog
2. Enter numbers of books you like (e.g., `1,3,5`)
3. See recommendations from both algorithms

## Key Takeaways

1. **Embeddings capture meaning**: Similar items cluster together in vector space
2. **Aggregation works**: Averaging vectors creates valid "composite" preferences
3. **Hybrid is best**: Combining content and collaborative signals produces superior recommendations
4. **Metadata matters**: Filters ensure recommendations stay within appropriate boundaries
5. **Speed at scale**: Vector databases return similar items in milliseconds, even across millions of options
