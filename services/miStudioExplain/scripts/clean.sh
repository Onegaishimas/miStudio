#!/bin/bash
# Cleanup script for miStudioExplain

set -e

echo "🧹 Cleaning up miStudioExplain artifacts..."

# Clean Python cache
echo "🐍 Cleaning Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Clean build artifacts
echo "🏗️ Cleaning build artifacts..."
rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true

# Clean test artifacts
echo "🧪 Cleaning test artifacts..."
rm -rf .pytest_cache/ .coverage htmlcov/ 2>/dev/null || true

# Clean Docker images (optional)
read -p "Clean Docker images? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🐳 Cleaning Docker images..."
    docker system prune -f
fi

echo "✅ Cleanup complete!"

