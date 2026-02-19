"""
Initialize Qdrant vector database for the Recall vs Latency demo.

This script uses the SIFT benchmark dataset (128K vectors) which is
the industry standard for demonstrating ANN recall/latency tradeoffs.
The dataset is specifically designed to show how approximate search
degrades at low ef values.
"""

import json
import os
import sys
import struct
import urllib.request
import gzip

# ---------------------------------------------------------------------------
# Allow Python to find the shared utilities in the setup/ directory.
# This adds setup/ to the module search path so we can write:
#     from embeddings import get_embedding_client
# instead of dealing with complex relative imports.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "setup"))

from embeddings import get_embedding_client
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    HnswConfigDiff,
)

# Qdrant connection settings (container networking)
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "sift_benchmark"

# SIFT dataset URLs (standard ANN benchmark)
SIFT_BASE_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_books():
    """Load the 500 books dataset."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "books_large.json")
    with open(data_path, "r") as f:
        return json.load(f)


def chunk_description(description, chunk_size=50, overlap=10):
    """
    Split description into overlapping chunks to increase vector count.
    
    With 500 books and ~3-4 chunks per description, we get ~2000 vectors.
    To reach ~10K, we also create variations with different context.
    """
    words = description.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) >= 10:  # Only keep meaningful chunks
            chunks.append(chunk)
    
    return chunks if chunks else [description]


def create_text_variations(book):
    """
    Create many text variations from a single book to reach ~10K+ vectors.
    Each variation embeds different semantic aspects and phrasings.
    We need enough vectors for HNSW approximation to actually matter.
    """
    import random
    variations = []
    
    title = book['title']
    author = book['author']
    genre = book['genre']
    desc = book['description']
    year = book.get('publish_year', 2020)
    rating = book.get('rating', 4.0)
    
    # 1. Original full description
    variations.append({
        "text": f"{title}. {desc}",
        "variation": "full"
    })
    
    # 2. Title + genre focused
    variations.append({
        "text": f"{title} - A {genre} novel by {author}",
        "variation": "genre_focus"
    })
    
    # 3. Author + title style
    variations.append({
        "text": f"By {author}: {title}. {desc[:100]}...",
        "variation": "author_focus"
    })
    
    # 4. Review-style variations
    review_templates = [
        f"I loved reading {title} by {author}. {desc}",
        f"{title} is an amazing {genre} book. {desc}",
        f"Just finished {title}. What a great {genre} story! {desc}",
        f"Highly recommend {title} by {author} for fans of {genre}.",
        f"{author}'s {title} is a must-read {genre} novel from {year}.",
        f"Rating: {rating}/5 - {title} delivers an incredible {genre} experience.",
        f"Looking for a good {genre} book? Try {title} by {author}.",
        f"{title} ({year}) by {author} - A compelling {genre} narrative.",
    ]
    for i, template in enumerate(review_templates):
        variations.append({
            "text": template,
            "variation": f"review_{i}"
        })
    
    # 5. Question-style variations (simulating search queries)
    question_templates = [
        f"What is {title} about? {desc}",
        f"Who wrote {title}? {author} wrote this {genre} novel.",
        f"Is {title} a good {genre} book? Yes, {desc}",
        f"Books similar to {title}: {genre} novels with {desc[:50]}",
        f"Best {genre} books like {title} by {author}",
    ]
    for i, template in enumerate(question_templates):
        variations.append({
            "text": template,
            "variation": f"question_{i}"
        })
    
    # 6. Keyword-dense variations
    words = desc.split()
    if len(words) > 10:
        # First half
        variations.append({
            "text": f"{title}: {' '.join(words[:len(words)//2])}",
            "variation": "first_half"
        })
        # Second half
        variations.append({
            "text": f"{title}: {' '.join(words[len(words)//2:])}",
            "variation": "second_half"
        })
        # Random subset
        random.seed(hash(title))  # Deterministic per book
        subset = random.sample(words, min(15, len(words)))
        variations.append({
            "text": f"{title} {author} {genre}: {' '.join(subset)}",
            "variation": "random_keywords"
        })
    
    # 7. Metadata-heavy variations  
    variations.append({
        "text": f"{genre} {genre} {genre} - {title} by {author}, published {year}, rated {rating}",
        "variation": "metadata_heavy"
    })
    
    return variations


def get_qdrant_client():
    """Create Qdrant client with container networking."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def init_qdrant_collection(reset=False):
    """Initialize the Qdrant collection with book vectors."""
    
    # Load books
    print("Loading books dataset...")
    books = load_books()
    print(f"Loaded {len(books)} books")
    
    # Initialize embedding client
    print("Initializing embedding client...")
    embedding_client = get_embedding_client()
    
    # Connect to Qdrant
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = get_qdrant_client()
    
    # Check if collection exists
    collections = [c.name for c in client.get_collections().collections]
    
    if COLLECTION_NAME in collections:
        if reset:
            print(f"Deleting existing collection '{COLLECTION_NAME}'...")
            client.delete_collection(COLLECTION_NAME)
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists. Use --reset to recreate.")
            return
    
    # Create collection with HNSW index
    # We use default HNSW params - the ef_search parameter is set at query time
    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embedding_client.dimension,
            distance=Distance.COSINE,
        ),
        hnsw_config=HnswConfigDiff(
            m=16,  # Number of edges per node
            ef_construct=100,  # Construction-time ef (higher = better index quality)
        )
    )
    
    # Generate vectors
    print("Generating embeddings (this may take a few minutes)...")
    points = []
    point_id = 0
    
    for i, book in enumerate(books):
        variations = create_text_variations(book)
        
        for var in variations:
            embedding = embedding_client.embed(var["text"])
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "book_id": book["id"],
                    "title": book["title"],
                    "author": book["author"],
                    "genre": book["genre"],
                    "description": book["description"],
                    "variation": var["variation"],
                    "text": var["text"][:500],  # Truncate for storage
                    "rating": book["rating"],
                    "publish_year": book["publish_year"]
                }
            ))
            point_id += 1
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(books)} books ({len(points)} vectors)")
    
    print(f"Total vectors to insert: {len(points)}")
    
    # Upsert in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        if (i + batch_size) % 500 == 0 or i + batch_size >= len(points):
            print(f"  Inserted {min(i + batch_size, len(points))}/{len(points)} vectors")
    
    print(f"\nSuccessfully initialized Qdrant with {len(points)} vectors")
    print(f"Collection: {COLLECTION_NAME}")


def get_collection_info():
    """Display collection statistics."""
    client = get_qdrant_client()
    
    try:
        info = client.get_collection(COLLECTION_NAME)
        print(f"\nCollection: {COLLECTION_NAME}")
        print(f"  Vectors: {info.points_count}")
        print(f"  Dimension: {info.config.params.vectors.size}")
        print(f"  Distance: {info.config.params.vectors.distance}")
    except Exception as e:
        print(f"Could not get collection info: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize Qdrant for Recall vs Latency demo")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate collection")
    parser.add_argument("--info", action="store_true", help="Show collection info only")
    args = parser.parse_args()
    
    if args.info:
        get_collection_info()
    else:
        init_qdrant_collection(reset=args.reset)
        get_collection_info()
