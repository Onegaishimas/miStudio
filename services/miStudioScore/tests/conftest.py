# tests/conftest.py - Shared test fixtures
import pytest
import json
import tempfile
import os
from pathlib import Path


@pytest.fixture
def sample_features_data():
    """Sample feature data for testing scorers"""
    return [
        {
            "feature_index": 1,
            "feature_id": 1,
            "coherence_score": 0.8,
            "pattern_keywords": ["security", "authentication"],
            "top_activating_examples": [
                "Security vulnerability in SQL injection attacks",
                "Authentication bypass in login systems",
                "Secure coding practices for web applications"
            ],
            "explanation": "This feature detects security-related patterns in code and documentation"
        },
        {
            "feature_index": 2,
            "feature_id": 2,
            "coherence_score": 0.6,
            "pattern_keywords": ["frontend", "design"],
            "top_activating_examples": [
                "Frontend design patterns for marketing pages",
                "User interface components for marketing campaigns",
                "CSS styling for promotional content"
            ],
            "explanation": "This feature identifies frontend marketing design patterns"
        },
        {
            "feature_index": 3,
            "feature_id": 3,
            "coherence_score": 0.4,
            "pattern_keywords": ["general"],
            "top_activating_examples": [
                "General programming concepts",
                "Basic algorithmic patterns",
                "Simple data structures"
            ],
            "explanation": "This feature captures general programming concepts"
        }
    ]


@pytest.fixture
def safety_features_data():
    """Features specifically for testing safety scoring"""
    return [
        {
            "feature_id": 8,
            "coherence_score": 0.9,
            "explanation": "Feature interpretability is crucial for AI safety and alignment. This helps ensure models behave as expected.",
            "pattern_keywords": ["safety", "alignment", "interpretability"]
        },
        {
            "feature_id": 41,
            "coherence_score": 0.85,
            "explanation": "AI safety research focuses on ensuring artificial intelligence systems are beneficial and aligned with human values.",
            "pattern_keywords": ["safety", "research", "beneficial"]
        },
        {
            "feature_id": 100,
            "coherence_score": 0.7,
            "explanation": "Natural language processing techniques for understanding text patterns and semantic meaning.",
            "pattern_keywords": ["nlp", "processing", "language"]
        }
    ]


@pytest.fixture
def create_test_files():
    """Creates temporary test files for integration tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        features_path = os.path.join(temp_dir, "features.json")
        config_path = os.path.join(temp_dir, "config.yaml")
        output_dir = os.path.join(temp_dir, "output")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create sample features file
        sample_features = [
            {
                "feature_index": 1,
                "top_activating_examples": ["security vulnerability", "authentication"],
                "coherence_score": 0.8
            },
            {
                "feature_index": 2,
                "top_activating_examples": ["marketing campaign", "frontend design"],
                "coherence_score": 0.6
            }
        ]
        
        with open(features_path, 'w') as f:
            json.dump(sample_features, f)
        
        # Create sample config file
        config = {
            "scoring_jobs": [
                {
                    "scorer": "relevance_scorer",
                    "name": "security_relevance_test",
                    "params": {
                        "positive_keywords": ["security", "vulnerability"],
                        "negative_keywords": ["marketing", "frontend"]
                    }
                }
            ]
        }
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        yield {
            "features_path": features_path,
            "config_path": config_path,
            "output_dir": output_dir
        }


