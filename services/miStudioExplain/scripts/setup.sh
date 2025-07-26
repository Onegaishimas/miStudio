#!/bin/bash
# Setup script for miStudioExplain development environment

set -e

echo "🚀 Setting up miStudioExplain development environment..."

# Create Python virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3.11 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r deployment/docker/requirements.txt

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/input data/output data/cache logs

# Copy sample configuration
echo "⚙️  Setting up configuration..."
if [ ! -f "config/local.yaml" ]; then
    cp config/service.yaml config/local.yaml
fi

# Set permissions
echo "🔐 Setting permissions..."
chmod +x scripts/*.sh

echo "✅ Setup complete!"
echo "💡 Next steps:"
echo "   1. Activate environment: source venv/bin/activate"
echo "   2. Run tests: pytest tests/"
echo "   3. Start development server: python -m src.main"

