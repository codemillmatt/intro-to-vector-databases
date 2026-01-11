"""
Module 4: Product Recommendation Demo

This demo shows how to use vector databases for product recommendations,
combining content-based and collaborative filtering approaches.
"""

import os
import sys
import json

# Add setup directory to path for shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "setup"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt

from embeddings import get_embedding_client
from pinecone_utils import get_pinecone_index

console = Console()


def load_books():
    """Load books data."""
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "setup", "data", "books.json"
    )
    with open(data_path, "r") as f:
        return {book["id"]: book for book in json.load(f)}


def load_users():
    """Load user reading habits data."""
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "setup", "data", "users.json"
    )
    with open(data_path, "r") as f:
        return json.load(f)


def get_book_embeddings(book_ids: list[str]) -> list[list[float]]:
    """Get embeddings for a list of books."""
    index = get_pinecone_index()
    
    # Fetch book vectors
    results = index.fetch(ids=book_ids)
    
    embeddings = []
    for book_id in book_ids:
        if book_id in results.vectors:
            embeddings.append(results.vectors[book_id].values)
    
    return embeddings


def average_embeddings(embeddings: list[list[float]]) -> list[float]:
    """Calculate the average of multiple embedding vectors."""
    if not embeddings:
        return [0.0] * 384
    
    num_dims = len(embeddings[0])
    avg = []
    for i in range(num_dims):
        total = sum(emb[i] for emb in embeddings)
        avg.append(total / len(embeddings))
    
    return avg


def recommend_similar_books(
    liked_books: list[str],
    exclude_books: list[str] | None = None,
    top_k: int = 5
) -> list[dict]:
    """
    Recommend books similar to the user's liked books.
    
    This is content-based filtering - finding books with similar
    semantic content to what the user already enjoys.
    """
    # Get embeddings for liked books
    embeddings = get_book_embeddings(liked_books)
    
    if not embeddings:
        return []
    
    # Create a "preference vector" by averaging liked book embeddings
    preference_vector = average_embeddings(embeddings)
    
    # Search for similar books
    index = get_pinecone_index()
    
    results = index.query(
        vector=preference_vector,
        top_k=top_k + len(liked_books or []) + len(exclude_books or []),
        include_metadata=True,
        filter={"type": "book"}
    )
    
    # Filter out already-read books
    exclude_set = set(liked_books or []) | set(exclude_books or [])
    
    recommendations = []
    for match in results.matches:
        if match.id not in exclude_set:
            recommendations.append({
                "id": match.id,
                "title": match.metadata.get("title", ""),
                "author": match.metadata.get("author", ""),
                "genre": match.metadata.get("genre", ""),
                "rating": match.metadata.get("rating", 0),
                "score": match.score
            })
            
            if len(recommendations) >= top_k:
                break
    
    return recommendations


def recommend_collaborative(
    user_liked_books: list[str],
    all_users: list[dict],
    exclude_books: list[str] | None = None,
    top_k: int = 5
) -> list[dict]:
    """
    Recommend books based on what similar users liked.
    
    This is collaborative filtering - finding users with similar
    tastes and recommending books they liked.
    """
    books = load_books()
    exclude_set = set(user_liked_books or []) | set(exclude_books or [])
    
    # Find users with overlapping tastes
    similar_users = []
    for user in all_users:
        overlap = set(user["liked_books"]) & set(user_liked_books)
        if overlap:
            similarity = len(overlap) / max(
                len(user["liked_books"]), len(user_liked_books)
            )
            similar_users.append({
                "user": user,
                "similarity": similarity,
                "overlap": overlap
            })
    
    # Sort by similarity
    similar_users.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Collect book recommendations from similar users
    book_scores = {}
    for similar_user in similar_users:
        for book_id in similar_user["user"]["liked_books"]:
            if book_id not in exclude_set:
                if book_id not in book_scores:
                    book_scores[book_id] = 0
                book_scores[book_id] += similar_user["similarity"]
    
    # Sort by score and return top k
    sorted_books = sorted(
        book_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:top_k]
    
    recommendations = []
    for book_id, score in sorted_books:
        if book_id in books:
            book = books[book_id]
            recommendations.append({
                "id": book_id,
                "title": book["title"],
                "author": book["author"],
                "genre": book["genre"],
                "rating": book["rating"],
                "score": score
            })
    
    return recommendations


def display_books(books: list[dict], title: str):
    """Display a list of books in a table."""
    console.print(f"\n[bold]{title}[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", justify="center", width=3)
    table.add_column("Title", style="cyan", width=30)
    table.add_column("Author", style="green", width=20)
    table.add_column("Genre", style="yellow", width=20)
    table.add_column("Rating", justify="center", width=8)
    
    for i, book in enumerate(books, 1):
        table.add_row(
            str(i),
            book.get("title", "")[:30],
            book.get("author", "")[:20],
            book.get("genre", "")[:20],
            f"{book.get('rating', 0):.1f}★"
        )
    
    console.print(table)


def display_recommendations(
    recommendations: list[dict], 
    title: str,
    show_score: bool = True
):
    """Display recommendations with scores."""
    console.print(f"\n[bold green]{title}[/bold green]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", justify="center", width=3)
    table.add_column("Title", style="cyan", width=30)
    table.add_column("Author", style="green", width=20)
    table.add_column("Genre", style="yellow", width=20)
    if show_score:
        table.add_column("Match", justify="center", width=8)
    
    if not recommendations:
        table.add_row("", "[dim]No recommendations found[/dim]", "", "")
    else:
        for i, rec in enumerate(recommendations, 1):
            row = [
                str(i),
                rec.get("title", "")[:30],
                rec.get("author", "")[:20],
                rec.get("genre", "")[:20]
            ]
            if show_score:
                row.append(f"{rec.get('score', 0):.2f}")
            table.add_row(*row)
    
    console.print(table)


def interactive_demo():
    """Run an interactive recommendation demonstration."""
    console.print(Panel.fit(
        "[bold blue]Module 4: Product Recommendations[/bold blue]\n"
        "Content-Based and Collaborative Filtering",
        style="bold"
    ))
    console.print()
    
    # Load data
    books = load_books()
    users = load_users()
    
    # Display available books
    display_books(list(books.values()), "📚 Available Books")
    console.print()
    
    # Get user preferences
    console.print("[bold]Let's build your reading profile![/bold]")
    console.print("Enter the numbers of books you've enjoyed (comma-separated)")
    console.print("[dim]Example: 1,3,5[/dim]")
    console.print()
    
    while True:
        selection = Prompt.ask(
            "[bold]Your liked books[/bold] (or 'quit' to exit)"
        )
        
        if selection.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break
        
        try:
            # Parse selection
            indices = [int(x.strip()) for x in selection.split(",")]
            book_list = list(books.values())
            liked_books = [book_list[i-1]["id"] for i in indices if 0 < i <= len(book_list)]
            
            if not liked_books:
                console.print("[red]No valid books selected. Try again.[/red]")
                continue
            
            # Show selected books
            selected = [books[bid] for bid in liked_books]
            display_books(selected, "✅ Your Liked Books")
            
            # Content-based recommendations
            console.print("\n[bold cyan]Method 1: Content-Based Filtering[/bold cyan]")
            console.print("[dim]Finding books semantically similar to your preferences...[/dim]")
            
            content_recs = recommend_similar_books(liked_books)
            display_recommendations(
                content_recs, 
                "Books with Similar Content",
                show_score=True
            )
            
            # Collaborative filtering
            console.print("\n[bold cyan]Method 2: Collaborative Filtering[/bold cyan]")
            console.print("[dim]Finding what similar readers enjoyed...[/dim]")
            
            collab_recs = recommend_collaborative(liked_books, users)
            display_recommendations(
                collab_recs,
                "Books Liked by Similar Readers",
                show_score=True
            )
            
            # Insight
            console.print(Panel(
                "[bold]Two Recommendation Approaches:[/bold]\n\n"
                "1. [cyan]Content-Based[/cyan]: Uses vector similarity to find\n"
                "   books with similar themes/topics to what you liked\n\n"
                "2. [cyan]Collaborative[/cyan]: Uses behavior patterns to find\n"
                "   books enjoyed by readers with similar tastes\n\n"
                "[dim]Production systems typically combine both approaches.[/dim]",
                title="How Recommendations Work",
                style="yellow"
            ))
            console.print()
            
        except (ValueError, IndexError) as e:
            console.print(f"[red]Invalid input: {e}. Try again.[/red]")


if __name__ == "__main__":
    interactive_demo()
