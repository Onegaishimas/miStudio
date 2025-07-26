#!/bin/bash
# Development server script for miStudioExplain

set -e

echo "🔧 Starting miStudioExplain development server..."

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Run ./scripts/setup.sh first"
    exit 1
fi

# Set development environment variables
export LOG_LEVEL=DEBUG
export DATA_PATH=./data
export OLLAMA_SERVICE_NAME=ollama

# Start the development server
echo "🚀 Starting FastAPI development server..."
python -m src.main

