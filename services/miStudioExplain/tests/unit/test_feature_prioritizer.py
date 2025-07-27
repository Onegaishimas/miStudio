import pytest
from src.core.feature_prioritizer import FeaturePrioritizer
from src.core.input_manager import ExplanationRequest # Corrected import

@pytest.fixture
def prioritizer():
    """Provides a fresh instance of FeaturePrioritizer for each test."""
    return FeaturePrioritizer()

def test_prioritize_from_text(prioritizer: FeaturePrioritizer):
    """Tests that the prioritizer can extract keywords from raw text."""
    text = "This is a test. The test is important. A test helps verify code."
    request_data = {
        "request_id": "test-req",
        "input_data": {"text_corpus": text * 5} # Ensure it's long enough
    }
    request = ExplanationRequest.model_validate(request_data)
    features = prioritizer.prioritize_features(request)
    assert "test" in features