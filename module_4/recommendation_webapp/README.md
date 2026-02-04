# Product Recommendations

See how vector databases power recommendation engines.

## Run

```bash
cd module_4/recommendation_webapp
python app.py
```

Open http://localhost:5001

**Note**: This uses the same port as the RAG demo. Stop one before running the other.

## What It Shows

Two recommendation approaches side-by-side:

### Content-Based Filtering
"You liked books about X, here are more books about X"

1. Average the embeddings of books you liked
2. Find nearest neighbors to that "preference vector"
3. Recommend similar content

### Collaborative Filtering  
"Users like you also enjoyed these books"

1. Find users with similar reading history
2. Recommend what they liked that you haven't read
3. Weight by similarity (more overlap = stronger signal)

## Try It

1. Browse user profiles to see their reading preferences
2. Click a user to get recommendations for them
3. Or select your own books and see what gets recommended
4. Compare content-based vs collaborative results
