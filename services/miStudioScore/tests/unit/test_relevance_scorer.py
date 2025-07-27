# tests/unit/test_relevance_scorer.py
"""
Unit tests for the RelevanceScorer module.
"""
import pytest
from src.scorers.relevance_scorer import RelevanceScorer

# We can use the fixture directly by naming it as an argument
def test_relevance_scorer_calculation(sample_features_data):
    """
    Tests the core calculation logic of the RelevanceScorer.
    """
    # Arrange
    scorer = RelevanceScorer()
    params = {
        "name": "security_relevance_test",
        "positive_keywords": ["security", "vulnerability", "injection"],
        "negative_keywords": ["marketing", "frontend"],
    }
    
    # Act
    scored_features = scorer.score(sample_features_data, **params)

    # Assert
    # Total keywords = 3 positive + 2 negative = 5
    # Feature 1: 3 positive, 0 negative -> score = (3 - 0) / 5 = 0.6
    # Feature 2: 0 positive, 2 negative -> score = (0 - 2) / 5 = -0.4
    # Feature 3: 0 positive, 0 negative -> score = 0.0
    assert scored_features[0]["security_relevance_test"] == 0.6
    assert scored_features[1]["security_relevance_test"] == -0.4
    assert scored_features[2]["security_relevance_test"] == 0.0

def test_relevance_scorer_no_keywords():
    """
    Tests that the scorer handles cases with no keywords gracefully.
    """
    # Arrange
    scorer = RelevanceScorer()
    features = [{"feature_index": 1, "top_activating_examples": ["some text"]}]
    params = {"name": "test_score", "positive_keywords": [], "negative_keywords": []}
    
    # Act
    scored_features = scorer.score(features, **params)

    # Assert
    assert scored_features[0]["test_score"] == 0.0

def test_relevance_scorer_missing_name_param():
    """
    Tests that the scorer raises an error if the 'name' parameter is missing.
    """
    # Arrange
    scorer = RelevanceScorer()
    
    # Act & Assert
    with pytest.raises(ValueError, match="RelevanceScorer requires a 'name' parameter."):
        scorer.score(features=[], positive_keywords=["test"])
