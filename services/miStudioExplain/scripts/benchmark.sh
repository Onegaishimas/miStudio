#!/bin/bash
# Performance benchmarking script for miStudioExplain

set -e

echo "⚡ Running miStudioExplain performance benchmarks..."

# Activate virtual environment
source venv/bin/activate

# Check if service is running
if ! curl -s http://localhost:8002/api/v1/health > /dev/null; then
    echo "❌ miStudioExplain service not running. Start with ./scripts/dev.sh"
    exit 1
fi

echo "📊 Running performance tests..."

# Run benchmark tests
pytest tests/integration/test_performance.py -v --benchmark-only

echo "🔍 GPU utilization during tests:"
nvidia-smi

echo "✅ Benchmarks complete!"

