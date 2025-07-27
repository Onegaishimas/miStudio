# tests/unit/test_orchestrator.py
"""
Unit tests for the ScoringOrchestrator module.
"""
import pytest
from unittest.mock import patch, MagicMock
from src.core.orchestrator import ScoringOrchestrator

# The create_test_files fixture sets up our test environment
def test_orchestrator_run(create_test_files, mocker):
    """
    Tests the main 'run' method of the orchestrator.
    """
    # Arrange
    # Mock the scorer's 'score' method to isolate the orchestrator's logic
    mock_relevance_scorer = MagicMock()
    mock_relevance_scorer.return_value.score.return_value = [{"feature_index": 1, "security_relevance_test": 0.5}]

    # Use patch to replace the RelevanceScorer class with our mock
    mocker.patch(
        "src.core.orchestrator.importlib.import_module",
        # This is a bit complex, but it effectively mocks the dynamic import
        return_value=MagicMock(RelevanceScorer=mock_relevance_scorer)
    )
    
    # We need to reload the orchestrator's scorers map
    orchestrator = ScoringOrchestrator(config_path=create_test_files["config_path"])
    
    # Manually patch the loaded scorer map for simplicity in this test
    orchestrator.scorers = {"relevance_scorer": mock_relevance_scorer}

    # Act
    output_path, added_scores = orchestrator.run(
        features_path=create_test_files["features_path"],
        output_dir=create_test_files["output_dir"]
    )

    # Assert
    # 1. Check that the scorer's 'score' method was called exactly once
    mock_relevance_scorer.return_value.score.assert_called_once()
    
    # 2. Check the arguments it was called with
    call_args, call_kwargs = mock_relevance_scorer.return_value.score.call_args
    assert "positive_keywords" in call_kwargs
    assert call_kwargs["name"] == "security_relevance_test"

    # 3. Check the orchestrator's return values
    assert output_path is not None
    assert "scores_" in output_path
    assert added_scores == ["security_relevance_test"]

def test_orchestrator_handles_missing_scorer(create_test_files, caplog):
    """
    Tests that the orchestrator logs a warning and continues if a
    scorer in the config is not found.
    """
    # Arrange
    orchestrator = ScoringOrchestrator(config_path=create_test_files["config_path"])
    orchestrator.scorers = {} # Empty the available scorers
    
    # Act
    orchestrator.run(
        features_path=create_test_files["features_path"],
        output_dir=create_test_files["output_dir"]
    )

    # Assert
    assert "Scorer 'relevance_scorer' not found. Skipping job." in caplog.text
