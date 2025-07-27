import pytest
from src.core.input_manager import InputManager, ExplanationRequest
from pydantic import ValidationError

@pytest.fixture
def manager():
    return InputManager()

def test_process_valid_request(manager):
    valid_data = {
        "request_id": "test-123",
        "input_data": {"text_corpus": "a" * 100}
    }
    request = manager.process_request(valid_data)
    assert isinstance(request, ExplanationRequest)
    assert request.request_id == "test-123"

def test_process_invalid_request(manager):
    invalid_data = {"request_id": "test-123"} # Missing input_data
    with pytest.raises(ValueError, match="Invalid request payload"):
        manager.process_request(invalid_data)