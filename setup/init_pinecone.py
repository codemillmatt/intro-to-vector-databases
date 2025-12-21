"""
Initialize the Pinecone vector database with bookstore data.
This script populates the vector database with book embeddings
and their associated metadata.
"""

import json
import os
import time

from pinecone import Pinecone, ServerlessSpec
from embeddings import get_embedding_client

# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "local")
PINECONE_HOST = os.getenv("PINECONE_HOST", "http://localhost:5081")
INDEX_NAME = "bookstore"


def load_books_data():
    """Load books data from JSON file."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "books.json")
    with open(data_path, "r") as f:
        return json.load(f)


def load_book_texts():
    """Load full book texts for RAG."""
    texts_dir = os.path.join(os.path.dirname(__file__), "data", "book_texts")
    book_texts = {}
    
    if os.path.exists(texts_dir):
        for filename in os.listdir(texts_dir):
            if filename.endswith(".md"):
                book_id = filename.replace(".md", "")
                with open(os.path.join(texts_dir, filename), "r") as f:
                    book_texts[book_id] = f.read()
    
    return book_texts


def get_pinecone_client():
    """Get a Pinecone client instance."""
    return Pinecone(api_key=PINECONE_API_KEY, host=PINECONE_HOST)


def create_index(pc, embedding_dim=384):
    """Create the Pinecone index if it doesn't exist."""
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        # Wait for index to be ready
        time.sleep(2)
        print(f"Created index '{INDEX_NAME}'")
    else:
        print(f"Index '{INDEX_NAME}' already exists")
    
    return pc.Index(INDEX_NAME)


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks


def init_vector_database(include_book_texts=True):
    """Initialize the vector database with book data."""
    # Load data
    books = load_books_data()
    book_texts = load_book_texts() if include_book_texts else {}
    
    # Initialize embedding client
    embedding_client = get_embedding_client()
    
    # Initialize Pinecone
    pc = get_pinecone_client()
    index = create_index(pc, embedding_client.dimension)
    
    # Clear existing vectors
    try:
        index.delete(delete_all=True)
    except Exception:
        pass  # Index might be empty
    
    vectors = []
    
    # Create embeddings for book descriptions (for semantic search)
    print("Creating book description embeddings...")
    for book in books:
        # Combine title and description for richer embedding
        text = f"{book['title']}. {book['description']}"
        embedding = embedding_client.embed(text)
        
        vectors.append({
            "id": book["id"],
            "values": embedding,
            "metadata": {
                "type": "book",
                "title": book["title"],
                "author": book["author"],
                "genre": book["genre"],
                "publish_date": book["publish_date"],
                "rating": book["rating"],
                "price": book["price"],
                "in_stock": book["in_stock"],
                "tags": book.get("tags", []),
                "text": book["description"]
            }
        })
    
    # Create embeddings for book text chunks (for RAG)
    if book_texts:
        print("Creating book text chunk embeddings...")
        for book_id, full_text in book_texts.items():
            # Find the book metadata
            book_meta = next((b for b in books if b["id"] == book_id), {})
            
            chunks = chunk_text(full_text)
            for i, chunk in enumerate(chunks):
                embedding = embedding_client.embed(chunk)
                
                vectors.append({
                    "id": f"{book_id}_chunk_{i}",
                    "values": embedding,
                    "metadata": {
                        "type": "chunk",
                        "book_id": book_id,
                        "title": book_meta.get("title", ""),
                        "author": book_meta.get("author", ""),
                        "chunk_index": i,
                        "text": chunk
                    }
                })
    
    # Upsert vectors in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"Upserted batch {i // batch_size + 1}/{(len(vectors) + batch_size - 1) // batch_size}")
    
    print(f"Successfully inserted {len(vectors)} vectors into Pinecone")


def reset_vector_database():
    """Delete and recreate the vector database index."""
    pc = get_pinecone_client()
    
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME in existing_indexes:
        pc.delete_index(INDEX_NAME)
        print(f"Deleted index '{INDEX_NAME}'")
        time.sleep(2)
    
    print("Vector database reset successfully")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize Pinecone vector database")
    parser.add_argument("--reset", action="store_true", help="Reset the index")
    parser.add_argument("--no-texts", action="store_true", help="Skip book text chunks")
    args = parser.parse_args()
    
    if args.reset:
        reset_vector_database()
    
    init_vector_database(include_book_texts=not args.no_texts)
