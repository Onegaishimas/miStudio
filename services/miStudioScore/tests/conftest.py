# tests/conftest.py
"""
Pytest fixtures for the miStudioScore service tests.
"""
import pytest
import json
import yaml
import os

@pytest.fixture(scope="function")
def temp_test_dir(tmp_path):
    """
    Creates a temporary directory structure for a single test function.
    'tmp_path' is a built-in pytest fixture.
    """
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    config_dir = tmp_path / "config"

    input_dir.mkdir()
    output_dir.mkdir()
    config_dir.mkdir()

    return {
        "base": tmp_path,
        "input": input_dir,
        "output": output_dir,
        "config": config_dir,
    }

@pytest.fixture
def sample_features_data():
    """Provides a sample list of feature dictionaries."""
    return [
        {
            "feature_index": 1,
            "top_activating_examples": [
                "This is a security vulnerability.",
                "Warning: potential SQL injection detected.",
            ],
        },
        {
            "feature_index": 2,
            "top_activating_examples": [
                "Our new marketing campaign is live.",
                "Check out the new frontend design.",
            ],
        },
        {"feature_index": 3, "top_activating_examples": []}, # Feature with no examples
    ]

@pytest.fixture
def sample_relevance_config_data():
    """Provides a sample configuration for the relevance scorer."""
    return {
        "scoring_jobs": [
            {
                "scorer": "relevance_scorer",
                "name": "security_relevance_test",
                "params": {
                    "positive_keywords": ["security", "vulnerability", "injection"],
                    "negative_keywords": ["marketing", "frontend"],
                },
            }
        ]
    }

@pytest.fixture
def sample_ablation_config_data(temp_test_dir):
    """Provides a sample configuration for the ablation scorer."""
    # Create a dummy benchmark file for the config to point to
    benchmark_content = """
def run_benchmark(model, tokenizer, device):
    # A dummy benchmark that returns a fixed value
    return 1.0
"""
    benchmark_path = temp_test_dir["input"] / "dummy_benchmark.py"
    benchmark_path.write_text(benchmark_content)

    return {
        "scoring_jobs": [
            {
                "scorer": "ablation_scorer",
                "name": "qa_utility_test",
                "params": {
                    "benchmark_dataset_path": str(benchmark_path),
                    "target_model_name": "mock-model",
                    "target_model_layer": "mock.layer",
                    "device": "cpu",
                },
            }
        ]
    }


@pytest.fixture
def create_test_files(temp_test_dir, sample_features_data, sample_relevance_config_data):
    """
    A fixture that creates the dummy input files for relevance scoring.
    """
    # Create features.json
    features_path = temp_test_dir["input"] / "features.json"
    with open(features_path, "w") as f:
        json.dump(sample_features_data, f)

    # Create service_config.yaml
    config_path = temp_test_dir["config"] / "service_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_relevance_config_data, f)

    return {
        "features_path": str(features_path),
        "config_path": str(config_path),
        "output_dir": str(temp_test_dir["output"]),
    }
