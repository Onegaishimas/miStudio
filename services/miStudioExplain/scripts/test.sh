#!/bin/bash
# Test runner script for miStudioExplain

set -e

echo "🧪 Running miStudioExplain tests..."

# Activate virtual environment
source venv/bin/activate

# Run unit tests
echo "📋 Running unit tests..."
pytest tests/unit/ -v --tb=short

# Run integration tests (if Ollama is available)
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "🔗 Running integration tests..."
    pytest tests/integration/ -v --tb=short
else
    echo "⚠️  Skipping integration tests (Ollama not available)"
fi

# Run code quality checks
echo "🔍 Running code quality checks..."
python -m flake8 src/ --max-line-length=100 --ignore=E203,W503
python -m black --check src/

echo "✅ All tests passed!"

