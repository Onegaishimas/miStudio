"""
Pytest configuration and fixtures
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_feature_data():
    """Sample feature data for testing"""
    return {
        "feature_id": 1,
        "coherence_score": 0.65,
        "quality_level": "good",
        "pattern_category": "technical",
        "pattern_keywords": ["json", "schema"],
        "top_activations": [
            {
                "text": "JSON schema validation example",
                "activation_strength": 0.85,
                "context": "API documentation"
            }
        ],
        "activation_statistics": {
            "mean": 0.15,
            "std": 0.08,
            "frequency": 0.023
        }
    }


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response"""
    return {
        "model": "llama3.1:8b",
        "response": "This feature detects JSON schema validation patterns in API documentation.",
        "done": True,
        "total_duration": 1234567890,
        "load_duration": 123456789,
        "prompt_eval_count": 50,
        "eval_count": 25
    }

