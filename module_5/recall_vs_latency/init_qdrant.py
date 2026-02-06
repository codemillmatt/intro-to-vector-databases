"""
Initialize Qdrant vector database for the Recall vs Latency demo.

This script generates 100K random vectors to demonstrate the 
recall/latency tradeoff when tuning the HNSW ef parameter.
Random vectors spread uniformly across the space, making 
approximate search more challenging than clustered real data.
"""

import os
import numpy as np

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
COLLECTION_NAME = "recall_demo"

# Dataset parameters
VECTOR_DIM = 128
NUM_VECTORS = 100_000  # 100K vectors - enough to show recall degradation


def get_qdrant_client():
    """Create Qdrant client with container networking."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def generate_vectors(num_vectors, dim, seed=42):
    """
    Generate random unit vectors.
    
    Random vectors spread uniformly in high-dimensional space,
    making nearest neighbor search harder (good for demo).
    """
    np.random.seed(seed)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    # Normalize to unit vectors for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    return vectors


def init_qdrant_collection(reset=False, num_vectors=NUM_VECTORS):
    """Initialize the Qdrant collection with random vectors."""
    
    print(f"Generating {num_vectors:,} random vectors ({VECTOR_DIM}D)...")
    vectors = generate_vectors(num_vectors, VECTOR_DIM)
    print(f"Generated vectors with shape: {vectors.shape}")
    
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
    # Using low m value to make approximation effects more visible
    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE,
        ),
        hnsw_config=HnswConfigDiff(
            m=8,  # Lower m = less accurate but faster (default is 16)
            ef_construct=64,  # Lower = faster indexing, less accurate
        )
    )
    
    # Insert vectors in batches
    print("Inserting vectors...")
    batch_size = 1000
    for i in range(0, num_vectors, batch_size):
        end_idx = min(i + batch_size, num_vectors)
        batch_vectors = vectors[i:end_idx]
        
        points = [
            PointStruct(
                id=j,
                vector=batch_vectors[j - i].tolist(),
                payload={"index": j}
            )
            for j in range(i, end_idx)
        ]
        
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        
        if (i + batch_size) % 10000 == 0 or end_idx == num_vectors:
            print(f"  Inserted {end_idx:,}/{num_vectors:,} vectors")
    
    print(f"\nSuccessfully initialized Qdrant with {num_vectors:,} vectors")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Vector dimension: {VECTOR_DIM}")
    print(f"HNSW config: m=8, ef_construct=64")


def get_collection_info():
    """Display collection statistics."""
    client = get_qdrant_client()
    
    try:
        info = client.get_collection(COLLECTION_NAME)
        print(f"\nCollection: {COLLECTION_NAME}")
        print(f"  Vectors: {info.points_count:,}")
        print(f"  Dimension: {info.config.params.vectors.size}")
        print(f"  Distance: {info.config.params.vectors.distance}")
    except Exception as e:
        print(f"Could not get collection info: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize Qdrant for Recall vs Latency demo")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate collection")
    parser.add_argument("--vectors", type=int, default=NUM_VECTORS, 
                        help=f"Number of vectors to generate (default: {NUM_VECTORS:,})")
    parser.add_argument("--info", action="store_true", help="Show collection info only")
    args = parser.parse_args()
    
    if args.info:
        get_collection_info()
    else:
        init_qdrant_collection(reset=args.reset, num_vectors=args.vectors)
        get_collection_info()
