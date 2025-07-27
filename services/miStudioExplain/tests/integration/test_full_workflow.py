import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from src.main import app

# This single fixture mocks all necessary external calls for the integration test.
@pytest.fixture
def mock_external_calls():
    # Mock the OllamaManager's initialization health check
    with patch('src.infrastructure.ollama_manager.OllamaManager.initialize', new_callable=AsyncMock) as mock_init:
        mock_init.return_value = True
        # Mock the explanation generation call itself
        with patch('src.infrastructure.ollama_manager.OllamaManager.generate_explanation', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = {
                "success": True,
                "response": "This is a mocked explanation about the test corpus.",
                "model_used": "mock-model:latest",
                "token_count": 10
            }
            # Yield the mock so it's active during the test
            yield mock_generate

def test_full_workflow_success(mock_external_calls):
    """
    Tests the full API endpoint with a valid request, mocking the LLM call.
    We pass the fixture as an argument to ensure it's active.
    """
    with TestClient(app) as client:
        request_payload = {
            "request_id": "integration-test-001",
            "input_data": {
                "text_corpus": "This is a long test corpus designed to pass all the validation steps and test the full internal workflow of the application without making a real network call to the large language model."
            }
        }

        response = client.post("/explain", json=request_payload)

        # Assert that the API call was successful
        assert response.status_code == 200

        # Assert the response body is correct
        response_json = response.json()
        assert response_json["status"] == "success"
        assert response_json["request_id"] == "integration-test-001"
        assert "mock-model:latest" in response_json["explanation"]["model_used"]