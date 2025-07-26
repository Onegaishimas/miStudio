#!/bin/bash
# Code formatting script for miStudioExplain

set -e

echo "🎨 Formatting miStudioExplain code..."

# Activate virtual environment
source venv/bin/activate

# Format with black
echo "📝 Running Black formatter..."
python -m black src/ tests/

# Sort imports
echo "📝 Sorting imports..."
python -m isort src/ tests/

# Lint with flake8
echo "🔍 Running flake8 linter..."
python -m flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503

echo "✅ Code formatting complete!"

