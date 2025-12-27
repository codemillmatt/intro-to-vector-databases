# Finding Meaning Demo

## Overview

This demo compares traditional SQL keyword-based search against semantic vector similarity search to demonstrate the value of vector databases. It shows how vector databases can understand the meaning and context of queries, not just match keywords.

## What It Does

The demo provides three modes of operation:

1. **Interactive Comparison Mode (default)**: Compare SQL and semantic search side-by-side
2. **View All Books Mode (`-a`)**: Display all book titles and descriptions from PostgreSQL
3. **View Embeddings Mode (`-i`)**: Display sample embedding vectors from Pinecone

### Key Features

- **SQL Search**: Traditional keyword matching using PostgreSQL's `ILIKE` operator
- **Semantic Search**: Vector similarity search using Pinecone with embedding models
- **Side-by-Side Comparison**: See how each approach handles the same query
- **Rich Display**: Formatted tables showing search results with ratings and scores

## How to Run

### Prerequisites

Make sure you have:
1. Initialized the databases (see main repository README)
2. PostgreSQL running with book data loaded
3. Pinecone index created with embeddings
4. Ollama or sentence-transformers available for embeddings

### Running the Demo

Navigate to the demo directory:
```bash
cd module_1/finding_meaning
```

**Interactive Mode (default):**
```bash
python main.py
```
This starts an interactive session where you can enter search queries and see how SQL and semantic search compare.

**Display All Books:**
```bash
python main.py -a
```
or
```bash
python main.py --all
```
Shows all book titles and descriptions stored in PostgreSQL in a table format.

**Display Pinecone Sample Data:**
```bash
python main.py -i
```
or
```bash
python main.py --index-sample
```
Shows sample embedding vectors from Pinecone, including vector dimensions and the first few values of each vector.

## Example Queries

When running in interactive mode, try these example queries to see semantic search in action:

- "books about space exploration and adventure"
- "stories dealing with loss and healing"
- "thriller with technology and hacking"
- "fantasy with magical creatures"

## Understanding the Results

### SQL Search Results
- Uses keyword matching (`ILIKE`) on title, description, and genre
- Only finds books containing the exact words from your query
- Results are ordered by rating

### Semantic Search Results
- Uses vector similarity to find conceptually related books
- Can find relevant books even without keyword matches
- Includes a similarity score (higher = more similar)
- Results are ordered by vector similarity score

## Technical Details

- **Database**: PostgreSQL for traditional SQL search
- **Vector Database**: Pinecone Local for semantic search
- **Embeddings**: Generated using Ollama (all-minilm model) or sentence-transformers
- **Display**: Rich library for formatted console output
