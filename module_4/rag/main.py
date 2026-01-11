"""
Module 4: RAG (Retrieval Augmented Generation) Demo

This demo shows how to use vector databases for RAG - retrieving relevant
passages from books to answer questions about their content.

Note: This demo retrieves the relevant context but simulates the LLM
response. In a production environment, you would send the context
to an actual LLM like GPT-4 or Ollama.
"""

import os
import sys

# Add setup directory to path for shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "setup"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown

from embeddings import get_embedding_client
from pinecone_utils import get_pinecone_index

console = Console()

# Try to import ollama for actual LLM responses
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


def retrieve_relevant_chunks(
    query: str, 
    book_id: str | None = None,
    top_k: int = 3
) -> list[dict]:
    """
    Retrieve relevant text chunks from books based on the query.
    
    This is the "Retrieval" part of RAG - finding the most relevant
    passages that can help answer the user's question.
    """
    # Get embedding for the query
    embedding_client = get_embedding_client()
    query_embedding = embedding_client.embed(query)
    
    # Build filter for text chunks
    filters = {"type": "chunk"}
    if book_id:
        filters["book_id"] = book_id
    
    # Search in Pinecone
    index = get_pinecone_index()
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filters
    )
    
    # Format results
    chunks = []
    for match in results.matches:
        chunks.append({
            "book_id": match.metadata.get("book_id", ""),
            "title": match.metadata.get("title", ""),
            "author": match.metadata.get("author", ""),
            "chunk_index": match.metadata.get("chunk_index", 0),
            "text": match.metadata.get("text", ""),
            "score": match.score
        })
    
    return chunks


def get_available_books() -> list[dict]:
    """Get list of books that have full text available."""
    index = get_pinecone_index()
    
    # Query for chunk type to find books with text
    results = index.query(
        vector=[0.0] * 384,  # Dummy vector, we just want metadata
        top_k=100,
        include_metadata=True,
        filter={"type": "chunk"}
    )
    
    # Get unique books
    books = {}
    for match in results.matches:
        book_id = match.metadata.get("book_id", "")
        if book_id and book_id not in books:
            books[book_id] = {
                "id": book_id,
                "title": match.metadata.get("title", ""),
                "author": match.metadata.get("author", "")
            }
    
    return list(books.values())


def generate_answer(query: str, context: str) -> str:
    """
    Generate an answer using the retrieved context.
    
    This is the "Generation" part of RAG. In production, you would
    use an actual LLM. Here we try Ollama if available, otherwise
    we return a simulated response.
    """
    prompt = f"""Based on the following context from a book, answer the question.
If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
    
    if OLLAMA_AVAILABLE:
        try:
            response = ollama.chat(
                model="llama3.2:1b",  # Use a small model for quick responses
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            console.print(f"[dim]Ollama error: {e}[/dim]")
    
    # Fallback: return the context with a note
    return (
        f"[LLM not available - showing retrieved context]\n\n"
        f"The retrieved passages may help answer your question:\n\n"
        f"{context[:1000]}..."
    )


def display_chunks(chunks: list[dict]):
    """Display retrieved chunks."""
    table = Table(
        title="Retrieved Passages",
        show_header=True,
        header_style="bold magenta"
    )
    
    table.add_column("#", justify="center", width=3)
    table.add_column("Book", style="cyan", width=30)
    table.add_column("Score", justify="center", width=8)
    table.add_column("Preview", style="dim", width=50)
    
    for i, chunk in enumerate(chunks, 1):
        preview = chunk["text"][:100].replace("\n", " ") + "..."
        table.add_row(
            str(i),
            f"{chunk['title']}",
            f"{chunk['score']:.3f}",
            preview
        )
    
    console.print(table)
    console.print()


def ask_question(query: str, book_id: str | None = None):
    """Process a question using RAG."""
    console.print(Panel(f"[bold]Question:[/bold] {query}", style="blue"))
    console.print()
    
    # Step 1: Retrieve relevant chunks
    console.print("[bold cyan]Step 1: Retrieving relevant passages...[/bold cyan]")
    chunks = retrieve_relevant_chunks(query, book_id=book_id, top_k=3)
    
    if not chunks:
        console.print("[red]No relevant passages found.[/red]")
        return
    
    display_chunks(chunks)
    
    # Step 2: Build context from chunks
    console.print("[bold cyan]Step 2: Building context...[/bold cyan]")
    context = "\n\n---\n\n".join([
        f"From '{chunk['title']}' by {chunk['author']}:\n{chunk['text']}"
        for chunk in chunks
    ])
    
    # Step 3: Generate answer
    console.print("[bold cyan]Step 3: Generating answer...[/bold cyan]")
    console.print()
    
    answer = generate_answer(query, context)
    
    console.print(Panel(
        Markdown(answer),
        title="Answer",
        style="green"
    ))
    
    # Insight
    console.print(Panel(
        "[bold]How RAG Works:[/bold]\n\n"
        "1. [cyan]Retrieve[/cyan]: Use vector similarity to find relevant passages\n"
        "2. [cyan]Augment[/cyan]: Add retrieved text to the LLM prompt as context\n"
        "3. [cyan]Generate[/cyan]: LLM produces answer grounded in retrieved facts\n\n"
        "[dim]This prevents hallucination by grounding responses in actual data.[/dim]",
        title="RAG Pattern",
        style="yellow"
    ))


def interactive_demo():
    """Run an interactive RAG demonstration."""
    console.print(Panel.fit(
        "[bold blue]Module 4: RAG Demo[/bold blue]\n"
        "Retrieval Augmented Generation for Book Q&A",
        style="bold"
    ))
    console.print()
    
    # Show available books
    books = get_available_books()
    if books:
        console.print("[dim]Books with full text available:[/dim]")
        for book in books:
            console.print(f"  • {book['title']} by {book['author']}")
        console.print()
    
    # Example questions
    example_questions = [
        "What happens in the garden at midnight?",
        "How does the physicist save reality?",
        "What is the lighthouse keeper's name?",
        "How does the AI become conscious?",
    ]
    
    console.print("[dim]Example questions to try:[/dim]")
    for q in example_questions:
        console.print(f"  • {q}")
    console.print()
    
    while True:
        query = Prompt.ask(
            "[bold]Ask a question about the books[/bold] (or 'quit' to exit)"
        )
        
        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break
        
        if query.strip():
            ask_question(query)


if __name__ == "__main__":
    interactive_demo()
