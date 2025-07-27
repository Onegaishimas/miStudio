# tests/integration/test_scoring_workflow.py
"""
Integration tests for the miStudioScore service API.
"""
import json
from fastapi.testclient import TestClient
from src.main import app

# The TestClient allows us to make requests to our FastAPI app in tests
client = TestClient(app)

def test_health_check_endpoint():
    """Tests the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_score_endpoint_success(create_test_files):
    """
    Tests a successful end-to-end run of the /score endpoint
    with the relevance scorer.
    """
    # Arrange
    request_payload = {
        "features_path": create_test_files["features_path"],
        "config_path": create_test_files["config_path"],
        "output_dir": create_test_files["output_dir"],
    }

    # Act
    response = client.post("/score", json=request_payload)
    response_data = response.json()

    # Assert
    assert response.status_code == 200
    assert response_data["message"] == "Scoring job completed successfully."
    assert "scores_" in response_data["output_path"]
    assert response_data["scores_added"] == ["security_relevance_test"]

    # Verify the content of the output file
    with open(response_data["output_path"], "r") as f:
        output_data = json.load(f)
    
    assert len(output_data) == 3
    assert "security_relevance_test" in output_data[0]
    assert output_data[0]["security_relevance_test"] == 0.6

def test_score_endpoint_file_not_found():
    """
    Tests that the API returns a 400 error if an input file is not found.
    """
    # Arrange
    request_payload = {
        "features_path": "non_existent_dir/features.json",
        "config_path": "non_existent_dir/config.yaml",
        "output_dir": "non_existent_dir/output",
    }

    # Act
    response = client.post("/score", json=request_payload)

    # Assert
    assert response.status_code == 400
    assert "Input file not found" in response.json()["detail"]
