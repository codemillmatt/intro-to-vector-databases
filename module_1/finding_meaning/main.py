"""
Module 1: Finding Meaning
SQL vs Semantic Search Comparison Demo

This demo compares traditional SQL keyword-based search against 
semantic vector similarity search to demonstrate the value of 
vector databases.
"""

import argparse
import os
import sys

# Add setup directory to path for shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "setup"))

import psycopg2
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt

from embeddings import get_embedding_client
from pinecone_utils import get_pinecone_index

# Configuration
POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "database": os.getenv("POSTGRES_DB", "bookstore"),
    "user": os.getenv("POSTGRES_USER", "bookstore"),
    "password": os.getenv("POSTGRES_PASSWORD", "bookstore"),
}

console = Console()


def sql_search(query: str) -> list[dict]:
    """
    Search books using traditional SQL LIKE pattern matching.
    
    This represents how traditional databases handle text search -
    keyword matching that misses semantic meaning.
    """
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    try:
        with conn.cursor() as cur:
            # Use ILIKE for case-insensitive search
            # Search in title and description
            sql = """
                SELECT id, title, author, description, genre, rating
                FROM books
                WHERE title ILIKE %s 
                   OR description ILIKE %s
                   OR genre ILIKE %s
                ORDER BY rating DESC
                LIMIT 5
            """
            search_pattern = f"%{query}%"
            cur.execute(sql, (search_pattern, search_pattern, search_pattern))
            
            columns = ["id", "title", "author", "description", "genre", "rating"]
            results = [dict(zip(columns, row)) for row in cur.fetchall()]
            return results
    finally:
        conn.close()


def semantic_search(query: str) -> list[dict]:
    """
    Search books using semantic vector similarity.
    
    This demonstrates how vector databases understand meaning -
    finding conceptually similar content even without keyword matches.
    """
    # Get embedding for the query
    embedding_client = get_embedding_client()
    query_embedding = embedding_client.embed(query)
    
    # Search in Pinecone
    index = get_pinecone_index()
    
    # Only search book descriptions, not text chunks
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True,
        filter={"type": "book"}
    )
    
    # Format results
    formatted = []
    for match in results.matches:
        formatted.append({
            "id": match.id,
            "title": match.metadata.get("title", ""),
            "author": match.metadata.get("author", ""),
            "description": match.metadata.get("text", ""),
            "genre": match.metadata.get("genre", ""),
            "rating": match.metadata.get("rating", 0),
            "score": match.score
        })
    
    return formatted


def get_all_books() -> list[dict]:
    """
    Get all books from PostgreSQL database.
    
    Returns all book titles and descriptions.
    """
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    try:
        with conn.cursor() as cur:
            sql = """
                SELECT id, title, author, description, genre, rating
                FROM books
                ORDER BY title
            """
            cur.execute(sql)
            
            columns = ["id", "title", "author", "description", "genre", "rating"]
            results = [dict(zip(columns, row)) for row in cur.fetchall()]
            return results
    finally:
        conn.close()


def get_pinecone_sample() -> tuple[list[dict], dict]:
    """
    Get a limited sample of embedding data from Pinecone.
    
    Returns a few vectors with their metadata to show what's stored.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY, host=PINECONE_HOST)
    index_host = _get_pinecone_index_host()
    index = pc.Index(INDEX_NAME, host=index_host) if index_host else pc.Index(INDEX_NAME)
    
    # Query with a dummy vector to get some results
    stats = index.describe_index_stats()
    
    # Get dimension from embedding client to ensure consistency
    embedding_client = get_embedding_client()
    dummy_dimension = embedding_client.dimension
    dummy_vector = [0.0] * dummy_dimension
    
    results = index.query(
        vector=dummy_vector,
        top_k=5,
        include_metadata=True,
        include_values=True,
        filter={"type": "book"}
    )
    
    # Format results
    formatted = []
    for match in results.matches:
        # Only show first few dimensions of the vector
        vector_sample = match.values[:10] if match.values else []
        formatted.append({
            "id": match.id,
            "vector_sample": vector_sample,
            "vector_dim": len(match.values) if match.values else 0,
            "metadata": {
                "title": match.metadata.get("title", ""),
                "author": match.metadata.get("author", ""),
                "type": match.metadata.get("type", "")
            }
        })
    
    return formatted, stats


def display_results(title: str, results: list[dict], show_score: bool = False, show_description: bool = False):
    """Display search results in a formatted table."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    
    table.add_column("Title", style="cyan", width=30)
    table.add_column("Author", style="green", width=20)
    if show_description:
        table.add_column("Description", style="blue", width=50)
    table.add_column("Genre", style="yellow", width=20)
    table.add_column("Rating", justify="center", width=8)
    if show_score:
        table.add_column("Score", justify="center", width=10)
    
    if not results:
        empty_row = ["[dim]No results found[/dim]", ""]
        if show_description:
            empty_row.append("")
        empty_row.extend(["", ""])
        if show_score:
            empty_row.append("")
        table.add_row(*empty_row)
    else:
        for book in results:
            row = [
                book.get("title", "")[:30],
                book.get("author", "")[:20]
            ]
            if show_description:
                row.append(book.get("description", "")[:50])
            row.extend([
                book.get("genre", "")[:20],
                f"{book.get('rating', 0):.1f}"
            ])
            if show_score:
                row.append(f"{book.get('score', 0):.3f}")
            table.add_row(*row)
    
    console.print(table)
    console.print()


def display_all_books():
    """Display all books from PostgreSQL."""
    console.print("[bold cyan]Fetching all books from PostgreSQL...[/bold cyan]")
    console.print()
    
    books = get_all_books()
    
    table = Table(title="All Books in Database", show_header=True, header_style="bold magenta")
    table.add_column("Title", style="cyan", width=40)
    table.add_column("Description", style="blue", width=60)
    
    for book in books:
        table.add_row(
            book.get("title", ""),
            book.get("description", "")[:60]
        )
    
    console.print(table)
    console.print(f"\n[dim]Total books: {len(books)}[/dim]\n")


def display_pinecone_sample():
    """Display sample embedding data from Pinecone."""
    console.print("[bold green]Fetching sample embedding data from Pinecone...[/bold green]")
    console.print()
    
    samples, stats = get_pinecone_sample()
    
    # Display index stats
    console.print(Panel(
        f"[bold]Index Statistics:[/bold]\n"
        f"Total vectors: {stats.get('total_vector_count', 0)}\n"
        f"Namespaces: {list(stats.get('namespaces', {}).keys())}",
        title="Pinecone Index Info",
        style="green"
    ))
    console.print()
    
    # Display sample vectors
    table = Table(title="Sample Embedding Vectors", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", width=20)
    table.add_column("Title", style="green", width=30)
    table.add_column("Vector Dimensions", justify="center", width=18)
    table.add_column("First 10 Values", style="yellow", width=60)
    
    for sample in samples:
        vector_preview = ", ".join([f"{v:.4f}" for v in sample["vector_sample"][:10]])
        table.add_row(
            sample["id"][:20],
            sample["metadata"].get("title", "")[:30],
            str(sample["vector_dim"]),
            f"[{vector_preview}...]"
        )
    
    console.print(table)
    console.print(f"\n[dim]Showing {len(samples)} sample vectors[/dim]\n")


def run_comparison(query: str):
    """Run both searches and display comparison."""
    console.print(Panel(f"[bold]Search Query:[/bold] {query}", style="blue"))
    console.print()
    
    # SQL Search
    console.print("[bold cyan]Running SQL keyword search...[/bold cyan]")
    sql_results = sql_search(query)
    display_results("SQL Search Results (Keyword Matching)", sql_results)
    
    # Semantic Search
    console.print("[bold green]Running semantic vector search...[/bold green]")
    semantic_results = semantic_search(query)
    display_results("Semantic Search Results (Vector Similarity)", semantic_results, show_score=True)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Finding Meaning - SQL vs Semantic Search Comparison"
    )
    parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="Display all book titles and descriptions from PostgreSQL"
    )
    parser.add_argument(
        "-i", "--index-sample",
        action="store_true",
        help="Display sample embedding data from Pinecone"
    )
    
    args = parser.parse_args()
    
    # Handle -a flag: display all books
    if args.all:
        display_all_books()
        return
    
    # Handle -i flag: display Pinecone sample
    if args.index_sample:
        display_pinecone_sample()
        return
    
    # Default interactive mode
    console.print(Panel.fit(
        "[bold blue]Module 1: Finding Meaning[/bold blue]\n"
        "SQL vs Semantic Search Comparison",
        style="bold"
    ))
    console.print()
    
    # Example queries that demonstrate semantic understanding
    example_queries = [
        "books about space exploration and adventure",
        "stories dealing with loss and healing",
        "thriller with technology and hacking",
        "fantasy with magical creatures",
    ]
    
    console.print("[dim]Example queries to try:[/dim]")
    for i, q in enumerate(example_queries, 1):
        console.print(f"  {i}. {q}")
    console.print()
    
    while True:
        query = Prompt.ask(
            "[bold]Enter your search query[/bold] (or 'quit' to exit)"
        )
        
        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break
        
        if query.strip():
            run_comparison(query)


if __name__ == "__main__":
    main()
