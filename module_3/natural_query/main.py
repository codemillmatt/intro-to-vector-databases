"""
Module 3: Natural Language Query Demo

This demo shows how to parse a single natural language query
into both semantic search AND metadata filters automatically.

Example: "fantasy books under $15" becomes:
  - Semantic query: "fantasy books"
  - Metadata filter: price <= 15
"""

import os
import re
import sys

# Add setup directory to path for shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "setup"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt

from embeddings import get_embedding_client
from pinecone_utils import get_pinecone_index

console = Console()

# Known genres for extraction
KNOWN_GENRES = {
    "fantasy": "Fantasy",
    "sci-fi": "Science Fiction",
    "science fiction": "Science Fiction",
    "scifi": "Science Fiction",
    "mystery": "Mystery",
    "thriller": "Thriller",
    "romance": "Romance",
    "historical fiction": "Historical Fiction",
    "historical": "Historical Fiction",
    "literary fiction": "Literary Fiction",
    "literary": "Literary Fiction",
    "memoir": "Memoir",
}


def parse_natural_query(query: str) -> dict:
    """
    Parse a natural language query into semantic text and metadata filters.
    
    Examples:
        "fantasy under $15" -> semantic="fantasy", filters={price: {$lte: 15}}
        "mystery rated above 4 stars" -> semantic="mystery", filters={rating: {$gte: 4}}
        "sci-fi in stock" -> semantic="sci-fi", filters={in_stock: True}
        "adventure books under $20 rated 4+" -> semantic="adventure books", filters={...}
    
    Returns:
        dict with keys: semantic_query, filters, extracted (what was parsed)
    """
    original_query = query
    query_lower = query.lower()
    filters = {"type": "book"}
    extracted = []
    
    # --- Extract price filters ---
    # Patterns: "under $15", "below $20", "less than $15", "< $15", "max $15"
    price_under_patterns = [
        r'under\s+\$?(\d+(?:\.\d{2})?)',
        r'below\s+\$?(\d+(?:\.\d{2})?)',
        r'less\s+than\s+\$?(\d+(?:\.\d{2})?)',
        r'<\s*\$?(\d+(?:\.\d{2})?)',
        r'max\s+\$?(\d+(?:\.\d{2})?)',
        r'\$(\d+(?:\.\d{2})?)\s+or\s+less',
    ]
    for pattern in price_under_patterns:
        match = re.search(pattern, query_lower)
        if match:
            price = float(match.group(1))
            filters["price"] = {"$lte": price}
            extracted.append(f"price ≤ ${price:.2f}")
            # Remove the matched text from semantic query
            query_lower = re.sub(pattern, '', query_lower)
            break
    
    # Patterns: "over $15", "above $20", "more than $15", "> $15", "min $15"
    price_over_patterns = [
        r'over\s+\$?(\d+(?:\.\d{2})?)',
        r'above\s+\$?(\d+(?:\.\d{2})?)',
        r'more\s+than\s+\$?(\d+(?:\.\d{2})?)',
        r'>\s*\$?(\d+(?:\.\d{2})?)',
        r'min\s+\$?(\d+(?:\.\d{2})?)',
        r'\$(\d+(?:\.\d{2})?)\s+or\s+more',
    ]
    for pattern in price_over_patterns:
        match = re.search(pattern, query_lower)
        if match:
            price = float(match.group(1))
            filters["price"] = {"$gte": price}
            extracted.append(f"price ≥ ${price:.2f}")
            query_lower = re.sub(pattern, '', query_lower)
            break
    
    # --- Extract rating filters ---
    # Patterns: "rated 4+", "4+ stars", "above 4 stars", "rating over 4", "4 stars or better"
    rating_patterns = [
        r'rated\s+(\d+(?:\.\d)?)\+',
        r'(\d+(?:\.\d)?)\+\s*stars?',
        r'above\s+(\d+(?:\.\d)?)\s*stars?',
        r'over\s+(\d+(?:\.\d)?)\s*stars?',
        r'rating\s+(?:over|above)\s+(\d+(?:\.\d)?)',
        r'(\d+(?:\.\d)?)\s*stars?\s+or\s+(?:better|higher|more)',
        r'at\s+least\s+(\d+(?:\.\d)?)\s*stars?',
        r'minimum\s+(\d+(?:\.\d)?)\s*stars?',
    ]
    for pattern in rating_patterns:
        match = re.search(pattern, query_lower)
        if match:
            rating = float(match.group(1))
            filters["rating"] = {"$gte": rating}
            extracted.append(f"rating ≥ {rating}★")
            query_lower = re.sub(pattern, '', query_lower)
            break
    
    # --- Extract in-stock filter ---
    stock_patterns = [
        r'\bin\s*stock\b',
        r'\bavailable\b',
        r'\bin\s*inventory\b',
    ]
    for pattern in stock_patterns:
        if re.search(pattern, query_lower):
            filters["in_stock"] = True
            extracted.append("in stock only")
            query_lower = re.sub(pattern, '', query_lower)
            break
    
    # --- Extract genre filter ---
    for genre_key, genre_value in KNOWN_GENRES.items():
        # Match genre as a word boundary to avoid partial matches
        pattern = r'\b' + re.escape(genre_key) + r'\b'
        if re.search(pattern, query_lower):
            filters["genre"] = genre_value
            extracted.append(f"genre = {genre_value}")
            # Don't remove genre from semantic query - it adds context
            break
    
    # --- Clean up semantic query ---
    # Remove common filler words that were part of filter patterns
    filler_patterns = [
        r'\bbooks?\b',
        r'\bthat\s+are\b',
        r'\bwith\b',
        r'\band\b',
        r'\bthe\b',
    ]
    semantic_query = query_lower
    for pattern in filler_patterns:
        semantic_query = re.sub(pattern, ' ', semantic_query)
    
    # Clean up whitespace
    semantic_query = ' '.join(semantic_query.split()).strip()
    
    # If we stripped everything, use the genre or a default
    if not semantic_query:
        if "genre" in filters:
            semantic_query = filters["genre"].lower()
        else:
            semantic_query = "books"
    
    return {
        "original_query": original_query,
        "semantic_query": semantic_query,
        "filters": filters,
        "extracted": extracted
    }


def search_with_parsed_query(parsed: dict) -> list[dict]:
    """Execute the search using parsed query components."""
    embedding_client = get_embedding_client()
    query_embedding = embedding_client.embed(parsed["semantic_query"])
    
    index = get_pinecone_index()
    
    results = index.query(
        vector=query_embedding,
        top_k=10,
        include_metadata=True,
        filter=parsed["filters"]
    )
    
    formatted = []
    for match in results.matches:
        formatted.append({
            "id": match.id,
            "title": match.metadata.get("title", ""),
            "author": match.metadata.get("author", ""),
            "genre": match.metadata.get("genre", ""),
            "rating": match.metadata.get("rating", 0),
            "price": match.metadata.get("price", 0),
            "in_stock": match.metadata.get("in_stock", False),
            "score": match.score
        })
    
    return formatted


def display_parsed_query(parsed: dict):
    """Show how the query was parsed."""
    console.print()
    console.print(Panel(
        f"[bold]Original query:[/bold] {parsed['original_query']}\n\n"
        f"[bold]Semantic search:[/bold] \"{parsed['semantic_query']}\"\n\n"
        f"[bold]Extracted filters:[/bold] {', '.join(parsed['extracted']) if parsed['extracted'] else 'None'}",
        title="Query Parsing",
        style="cyan"
    ))
    console.print()


def display_results(results: list[dict]):
    """Display search results."""
    table = Table(
        title="Search Results",
        show_header=True,
        header_style="bold magenta"
    )
    
    table.add_column("Title", style="cyan", width=28)
    table.add_column("Author", style="green", width=18)
    table.add_column("Genre", style="yellow", width=16)
    table.add_column("Rating", justify="center", width=8)
    table.add_column("Price", justify="right", width=8)
    table.add_column("Stock", justify="center", width=6)
    table.add_column("Score", justify="center", width=7)
    
    if not results:
        table.add_row("[dim]No results found[/dim]", "", "", "", "", "", "")
    else:
        for book in results:
            stock = "[green]✓[/green]" if book["in_stock"] else "[red]✗[/red]"
            table.add_row(
                book["title"][:28],
                book["author"][:18],
                book["genre"][:16],
                f"{book['rating']:.1f}★",
                f"${book['price']:.2f}",
                stock,
                f"{book['score']:.3f}"
            )
    
    console.print(table)
    console.print()


def interactive_demo():
    """Run the natural language query demo."""
    console.print(Panel.fit(
        "[bold blue]Module 3: Natural Language Query[/bold blue]\n"
        "Combined Semantic + Metadata Search",
        style="bold"
    ))
    console.print()
    
    console.print("[bold]Try queries like:[/bold]")
    examples = [
        "fantasy under $15",
        "mystery books rated 4+ stars",
        "science fiction in stock",
        "adventure stories under $20 with 4 stars or better",
        "romance available",
        "thriller below $18",
    ]
    for ex in examples:
        console.print(f"  • [cyan]{ex}[/cyan]")
    console.print()
    
    while True:
        query = Prompt.ask(
            "[bold]Enter your query[/bold] (or 'quit' to exit)"
        )
        
        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break
        
        if query.strip():
            # Parse the query
            parsed = parse_natural_query(query)
            display_parsed_query(parsed)
            
            # Execute search
            results = search_with_parsed_query(parsed)
            display_results(results)
            
            # Show insight on first run
            console.print(Panel(
                "[bold]How it works:[/bold]\n\n"
                "1. Your natural language query is parsed for metadata patterns\n"
                "   (prices, ratings, genres, stock status)\n\n"
                "2. Extracted filters become Pinecone metadata filters\n\n"
                "3. Remaining text becomes the semantic search query\n\n"
                "4. Both are combined in a single vector database query",
                title="Behind the Scenes",
                style="dim"
            ))


if __name__ == "__main__":
    interactive_demo()
