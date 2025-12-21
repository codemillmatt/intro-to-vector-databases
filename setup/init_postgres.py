"""
Initialize the PostgreSQL database with bookstore data.
This script is used for Module 1's SQL vs Vector comparison demo.
"""

import json
import os
import psycopg2
from psycopg2.extras import execute_values

# Database connection settings
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "database": os.getenv("POSTGRES_DB", "bookstore"),
    "user": os.getenv("POSTGRES_USER", "bookstore"),
    "password": os.getenv("POSTGRES_PASSWORD", "bookstore"),
}


def load_books_data():
    """Load books data from JSON file."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "books.json")
    with open(data_path, "r") as f:
        return json.load(f)


def create_tables(conn):
    """Create the necessary database tables."""
    with conn.cursor() as cur:
        # Books table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS books (
                id VARCHAR(20) PRIMARY KEY,
                title VARCHAR(500) NOT NULL,
                author VARCHAR(200) NOT NULL,
                description TEXT,
                genre VARCHAR(100),
                publish_date DATE,
                pages INTEGER,
                rating NUMERIC(3,2),
                price NUMERIC(10,2),
                in_stock BOOLEAN DEFAULT true
            )
        """)
        
        # Book tags table (for many-to-many relationship)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS book_tags (
                book_id VARCHAR(20) REFERENCES books(id),
                tag VARCHAR(100),
                PRIMARY KEY (book_id, tag)
            )
        """)
        
        # Create indexes for common queries
        cur.execute("CREATE INDEX IF NOT EXISTS idx_books_genre ON books(genre)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_books_author ON books(author)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_books_rating ON books(rating)")
        
        conn.commit()


def insert_books(conn, books):
    """Insert books into the database."""
    with conn.cursor() as cur:
        # Clear existing data
        cur.execute("DELETE FROM book_tags")
        cur.execute("DELETE FROM books")
        
        # Insert books
        book_values = [
            (
                book["id"],
                book["title"],
                book["author"],
                book["description"],
                book["genre"],
                book["publish_date"],
                book["pages"],
                book["rating"],
                book["price"],
                book["in_stock"]
            )
            for book in books
        ]
        
        execute_values(
            cur,
            """
            INSERT INTO books (id, title, author, description, genre, 
                             publish_date, pages, rating, price, in_stock)
            VALUES %s
            """,
            book_values
        )
        
        # Insert tags
        tag_values = [
            (book["id"], tag)
            for book in books
            for tag in book.get("tags", [])
        ]
        
        if tag_values:
            execute_values(
                cur,
                "INSERT INTO book_tags (book_id, tag) VALUES %s",
                tag_values
            )
        
        conn.commit()
        print(f"Inserted {len(books)} books with their tags")


def reset_database():
    """Reset the database to a clean state."""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS book_tags CASCADE")
            cur.execute("DROP TABLE IF EXISTS books CASCADE")
            conn.commit()
        
        print("Database reset successfully")
    finally:
        if conn:
            conn.close()


def init_database():
    """Initialize the database with sample data."""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        # Create tables
        create_tables(conn)
        
        # Load and insert books
        books = load_books_data()
        insert_books(conn, books)
        
        print("Database initialized successfully")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize PostgreSQL database")
    parser.add_argument("--reset", action="store_true", help="Reset database tables")
    args = parser.parse_args()
    
    if args.reset:
        reset_database()
    
    init_database()
