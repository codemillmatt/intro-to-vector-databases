#!/bin/bash
# Initialize all databases after DevContainer starts
# This script waits for services to be ready before running init scripts

set -e

echo "🚀 Starting database initialization..."

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL..."
for i in {1..30}; do
    if pg_isready -h localhost -p 5432 -U bookstore -q 2>/dev/null; then
        echo "✅ PostgreSQL is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  PostgreSQL not ready after 30 seconds, continuing anyway..."
    fi
    sleep 1
done

# Wait for Pinecone to be ready
echo "⏳ Waiting for Pinecone..."
for i in {1..30}; do
    if curl -s http://pinecone:5081/health > /dev/null 2>&1; then
        echo "✅ Pinecone is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  Pinecone not ready after 30 seconds, continuing anyway..."
    fi
    sleep 1
done

# Run PostgreSQL initialization
echo "📦 Initializing PostgreSQL..."
cd /workspace/setup
python init_postgres.py

# Run Pinecone initialization
echo "🔢 Initializing Pinecone (this may take a minute)..."
python init_pinecone.py

# Add enhanced full-text column for Module 1
echo "📖 Adding full book text to PostgreSQL (Module 1 enhanced)..."
cd /workspace/module_1/finding_meaning_enhanced
python init_enhanced_postgres.py

echo ""
echo "✨ Database initialization complete!"
echo "   You can now run any of the demo modules."
echo ""
echo "   Note: Module 5 (Recall vs Latency) requires separate Qdrant setup:"
echo "     cd module_5/recall_vs_latency && python init_qdrant.py --reset"
echo ""
