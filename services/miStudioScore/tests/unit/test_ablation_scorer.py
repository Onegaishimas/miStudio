# tests/unit/test_ablation_scorer.py
"""
Unit tests for the AblationScorer module.
"""
import pytest
from unittest.mock import MagicMock, patch
from src.scorers.ablation_scorer import AblationScorer

@pytest.fixture
def mock_dependencies(mocker):
    """Mocks all external dependencies for the AblationScorer."""
    mocker.patch("src.scorers.ablation_scorer.load_huggingface_model", return_value=(MagicMock(), MagicMock()))
    mocker.patch("src.scorers.ablation_scorer.load_sae_model", return_value=MagicMock())
    
    # Mock the dynamic import of the benchmark function
    mock_benchmark_module = MagicMock()
    # Make the mocked benchmark function return different values on subsequent calls
    mock_benchmark_module.run_benchmark.side_effect = [
        1.0,  # Baseline score
        1.2,  # Score for feature 1
        0.9,  # Score for feature 2
    ]
    mocker.patch("src.scorers.ablation_scorer.importlib.util.spec_from_file_location")
    mocker.patch("src.scorers.ablation_scorer.importlib.util.module_from_spec", return_value=mock_benchmark_module)
    
    # Mock the model's layer and hook registration
    mock_layer = MagicMock()
    mock_hook = MagicMock()
    mock_layer.register_forward_hook.return_value = mock_hook
    mocker.patch("src.scorers.ablation_scorer.get_target_layer", return_value=mock_layer)
    
    return {"benchmark_fn": mock_benchmark_module.run_benchmark, "hook": mock_hook}


def test_ablation_scorer_logic(sample_features_data, sample_ablation_config_data, mock_dependencies):
    """
    Tests the core logic of the AblationScorer, ensuring it calculates utility scores correctly.
    """
    # Arrange
    scorer = AblationScorer()
    # We only need the params from the config
    params = sample_ablation_config_data["scoring_jobs"][0]["params"]
    params["name"] = sample_ablation_config_data["scoring_jobs"][0]["name"]
    
    # We only test the first two features which have an index
    features_to_test = sample_features_data[:2]

    # Act
    scored_features = scorer.score(features_to_test, **params)

    # Assert
    # Baseline score is 1.0
    # Feature 1 ablated score is 1.2 -> utility = 1.2 - 1.0 = 0.2
    # Feature 2 ablated score is 0.9 -> utility = 0.9 - 1.0 = -0.1
    assert scored_features[0]["qa_utility_test"] == pytest.approx(0.2)
    assert scored_features[1]["qa_utility_test"] == pytest.approx(-0.1)

    # Verify that the benchmark function was called correctly (1 for baseline + 2 for features)
    assert mock_dependencies["benchmark_fn"].call_count == 3
    
    # Verify that the hook was registered and removed
    mock_dependencies["hook"].remove.assert_called_once()

def test_ablation_scorer_missing_params():
    """
    Tests that the scorer raises a ValueError if essential parameters are missing.
    """
    scorer = AblationScorer()
    with pytest.raises(ValueError):
        scorer.score(features=[], name="test") # Missing other params
