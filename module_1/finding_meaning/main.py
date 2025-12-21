"""
Module 1: Finding Meaning
SQL vs Semantic Search Comparison Demo

This demo compares traditional SQL keyword-based search against 
semantic vector similarity search to demonstrate the value of 
vector databases.
"""

import os
import sys

# Add setup directory to path for shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "setup"))

import psycopg2
from pinecone import Pinecone
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt

from embeddings import get_embedding_client

# Configuration
POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "database": os.getenv("POSTGRES_DB", "bookstore"),
    "user": os.getenv("POSTGRES_USER", "bookstore"),
    "password": os.getenv("POSTGRES_PASSWORD", "bookstore"),
}

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "local")
PINECONE_HOST = os.getenv("PINECONE_HOST", "http://localhost:5081")
INDEX_NAME = "bookstore"

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
    pc = Pinecone(api_key=PINECONE_API_KEY, host=PINECONE_HOST)
    index = pc.Index(INDEX_NAME)
    
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


def display_results(title: str, results: list[dict], show_score: bool = False):
    """Display search results in a formatted table."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    
    table.add_column("Title", style="cyan", width=30)
    table.add_column("Author", style="green", width=20)
    table.add_column("Genre", style="yellow", width=20)
    table.add_column("Rating", justify="center", width=8)
    if show_score:
        table.add_column("Score", justify="center", width=10)
    
    if not results:
        table.add_row("[dim]No results found[/dim]", "", "", "")
    else:
        for book in results:
            row = [
                book.get("title", "")[:30],
                book.get("author", "")[:20],
                book.get("genre", "")[:20],
                f"{book.get('rating', 0):.1f}"
            ]
            if show_score:
                row.append(f"{book.get('score', 0):.3f}")
            table.add_row(*row)
    
    console.print(table)
    console.print()


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
    
    # Comparison insight
    console.print(Panel(
        "[bold]Key Insight:[/bold]\n\n"
        "• SQL search only finds books containing the exact keywords\n"
        "• Semantic search understands the [italic]meaning[/italic] of your query\n"
        "• Related concepts are found even without matching words\n"
        "• The similarity score shows how conceptually close each result is",
        title="Why This Matters",
        style="yellow"
    ))


def main():
    """Main demo loop."""
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
