"""
Enhanced PostgreSQL setup for Module 1: Finding Meaning (Enhanced).

Adds a full_text column to the existing books table, populated from
the book text markdown files. This gives the SQL ILIKE search more
content to match against, while still showing the limitations of
pure substring matching compared to semantic search.

Prerequisites:
    - The base 'books' table must already exist (run setup/init_postgres.py first).

Usage:
    python init_enhanced_postgres.py          # Add full_text column
    python init_enhanced_postgres.py --reset  # Remove added column
"""

import argparse
import glob
import os

import psycopg2

# Database connection settings
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "database": os.getenv("POSTGRES_DB", "bookstore"),
    "user": os.getenv("POSTGRES_USER", "bookstore"),
    "password": os.getenv("POSTGRES_PASSWORD", "bookstore"),
}

# Path to book text markdown files
BOOK_TEXTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "setup", "data", "book_texts"
)


def load_book_texts() -> dict[str, str]:
    """
    Load book text content from markdown files.

    Returns:
        A dict mapping book_id (e.g. 'book_001') to the full text content.
    """
    texts: dict[str, str] = {}
    pattern = os.path.join(BOOK_TEXTS_DIR, "book_*.md")
    for filepath in sorted(glob.glob(pattern)):
        filename = os.path.basename(filepath)
        book_id = filename.replace(".md", "")  # e.g. "book_001"
        with open(filepath, "r", encoding="utf-8") as f:
            texts[book_id] = f.read()
    return texts


def add_full_text_column(conn):
    """Add the full_text column to the books table (idempotent)."""
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE books
            ADD COLUMN IF NOT EXISTS full_text TEXT DEFAULT ''
        """)
        conn.commit()
    print("  ✓ Added full_text column")


def populate_full_text(conn, book_texts: dict[str, str]):
    """Populate the full_text column from book text markdown files."""
    updated = 0
    with conn.cursor() as cur:
        for book_id, text in book_texts.items():
            cur.execute(
                "UPDATE books SET full_text = %s WHERE id = %s",
                (text, book_id),
            )
            if cur.rowcount > 0:
                updated += 1
        conn.commit()
    print(f"  ✓ Populated full_text for {updated} books (from {len(book_texts)} text files)")


def init_enhanced():
    """Add full-text search capabilities to the books table."""
    book_texts = load_book_texts()
    if not book_texts:
        print("⚠️  No book text files found in", BOOK_TEXTS_DIR)
        print("   Make sure setup/data/book_texts/ contains book_*.md files.")
        return

    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)

        # Verify that the books table exists
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'books'
                )
            """)
            if not cur.fetchone()[0]:
                print("❌ The 'books' table does not exist.")
                print("   Run 'python setup/init_postgres.py' first.")
                return

        print("📚 Enhancing PostgreSQL with book text content...")
        add_full_text_column(conn)
        populate_full_text(conn, book_texts)
        print("✅ Enhanced PostgreSQL setup complete!")

    except psycopg2.OperationalError as e:
        print(f"❌ Could not connect to PostgreSQL: {e}")
        print("   Make sure you're running inside the DevContainer.")
    finally:
        if conn:
            conn.close()


def reset_enhanced():
    """Remove the full_text column and associated index."""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE books DROP COLUMN IF EXISTS full_text")
            conn.commit()
        print("✅ Enhanced PostgreSQL changes removed.")
    except psycopg2.OperationalError as e:
        print(f"❌ Could not connect to PostgreSQL: {e}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced PostgreSQL setup — adds book text content"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove the full_text column",
    )
    args = parser.parse_args()

    if args.reset:
        reset_enhanced()
    else:
        init_enhanced()
