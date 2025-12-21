"""
Module 3: Metadata Filtering Demo

This demo extends the semantic search from Module 1 with metadata filters,
showing how vector databases can combine semantic similarity with 
structured data filtering.
"""

import os
import sys

# Add setup directory to path for shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "setup"))

from pinecone import Pinecone
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from embeddings import get_embedding_client

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "local")
PINECONE_HOST = os.getenv("PINECONE_HOST", "http://localhost:5081")
INDEX_NAME = "bookstore"

console = Console()


def semantic_search_with_filter(
    query: str,
    genre: str | None = None,
    min_rating: float | None = None,
    max_price: float | None = None,
    in_stock_only: bool = False
) -> list[dict]:
    """
    Search books using semantic similarity with metadata filters.
    
    This demonstrates the core capability of vector databases to combine
    semantic understanding with traditional database-style filtering.
    """
    # Get embedding for the query
    embedding_client = get_embedding_client()
    query_embedding = embedding_client.embed(query)
    
    # Build metadata filter
    filters = {"type": "book"}
    
    if genre:
        filters["genre"] = genre
    
    if min_rating is not None:
        filters["rating"] = {"$gte": min_rating}
    
    if max_price is not None:
        if "price" in filters:
            filters["price"]["$lte"] = max_price
        else:
            filters["price"] = {"$lte": max_price}
    
    if in_stock_only:
        filters["in_stock"] = True
    
    # Search in Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY, host=PINECONE_HOST)
    index = pc.Index(INDEX_NAME)
    
    results = index.query(
        vector=query_embedding,
        top_k=10,
        include_metadata=True,
        filter=filters
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
            "price": match.metadata.get("price", 0),
            "in_stock": match.metadata.get("in_stock", False),
            "score": match.score
        })
    
    return formatted


def display_results(
    title: str, 
    results: list[dict], 
    filters_applied: dict
):
    """Display search results with applied filters."""
    # Show applied filters
    filter_text = []
    if filters_applied.get("genre"):
        filter_text.append(f"Genre: {filters_applied['genre']}")
    if filters_applied.get("min_rating"):
        filter_text.append(f"Min Rating: {filters_applied['min_rating']}★")
    if filters_applied.get("max_price"):
        filter_text.append(f"Max Price: ${filters_applied['max_price']}")
    if filters_applied.get("in_stock_only"):
        filter_text.append("In Stock Only")
    
    filters_str = " | ".join(filter_text) if filter_text else "None"
    
    table = Table(
        title=f"{title}\n[dim]Filters: {filters_str}[/dim]",
        show_header=True,
        header_style="bold magenta"
    )
    
    table.add_column("Title", style="cyan", width=25)
    table.add_column("Author", style="green", width=18)
    table.add_column("Genre", style="yellow", width=18)
    table.add_column("Rating", justify="center", width=8)
    table.add_column("Price", justify="right", width=8)
    table.add_column("Stock", justify="center", width=8)
    table.add_column("Score", justify="center", width=8)
    
    if not results:
        table.add_row(
            "[dim]No results found[/dim]", "", "", "", "", "", ""
        )
    else:
        for book in results:
            stock_status = "✓" if book.get("in_stock") else "✗"
            stock_style = "green" if book.get("in_stock") else "red"
            
            table.add_row(
                book.get("title", "")[:25],
                book.get("author", "")[:18],
                book.get("genre", "")[:18],
                f"{book.get('rating', 0):.1f}★",
                f"${book.get('price', 0):.2f}",
                f"[{stock_style}]{stock_status}[/{stock_style}]",
                f"{book.get('score', 0):.3f}"
            )
    
    console.print(table)
    console.print()


def run_comparison(query: str):
    """Run search with and without filters to show the difference."""
    console.print(Panel(f"[bold]Search Query:[/bold] {query}", style="blue"))
    console.print()
    
    # Search without filters
    console.print("[bold cyan]Search WITHOUT filters:[/bold cyan]")
    results_no_filter = semantic_search_with_filter(query)
    display_results("All Matching Books", results_no_filter, {})
    
    # Get filter options from user
    console.print("[bold yellow]Now let's add some filters...[/bold yellow]")
    console.print()
    
    # Genre filter
    genre = Prompt.ask(
        "Filter by genre (or press Enter to skip)",
        default=""
    )
    
    # Rating filter
    min_rating_str = Prompt.ask(
        "Minimum rating (1-5, or press Enter to skip)",
        default=""
    )
    min_rating = float(min_rating_str) if min_rating_str else None
    
    # Price filter
    max_price_str = Prompt.ask(
        "Maximum price (or press Enter to skip)",
        default=""
    )
    max_price = float(max_price_str) if max_price_str else None
    
    # Stock filter
    in_stock_only = Confirm.ask("Only show in-stock items?", default=False)
    
    # Search with filters
    console.print()
    console.print("[bold green]Search WITH filters:[/bold green]")
    
    results_filtered = semantic_search_with_filter(
        query,
        genre=genre if genre else None,
        min_rating=min_rating,
        max_price=max_price,
        in_stock_only=in_stock_only
    )
    
    filters_applied = {
        "genre": genre if genre else None,
        "min_rating": min_rating,
        "max_price": max_price,
        "in_stock_only": in_stock_only
    }
    
    display_results("Filtered Results", results_filtered, filters_applied)
    
    # Insight
    console.print(Panel(
        "[bold]Key Insight:[/bold]\n\n"
        "• Metadata filtering happens [italic]after[/italic] vector similarity\n"
        "• The semantic meaning is preserved while narrowing results\n"
        "• This is different from SQL WHERE clauses - we're filtering\n"
        "  conceptually similar items, not just keyword matches\n"
        "• This enables powerful queries like: 'adventure books under $15'\n"
        "  combining semantic search with business logic",
        title="Why Metadata Filtering Matters",
        style="yellow"
    ))


def interactive_demo():
    """Run an interactive demonstration of metadata filtering."""
    console.print(Panel.fit(
        "[bold blue]Module 3: Metadata Filtering[/bold blue]\n"
        "Combining Semantic Search with Structured Filters",
        style="bold"
    ))
    console.print()
    
    # Available genres for reference
    genres = [
        "Fantasy", "Science Fiction", "Mystery", "Historical Fiction",
        "Literary Fiction", "Thriller", "Romance", "Memoir"
    ]
    
    console.print("[dim]Available genres:[/dim]")
    console.print(f"  {', '.join(genres)}")
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
    interactive_demo()
