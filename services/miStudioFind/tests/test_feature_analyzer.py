# tests/test_input_manager.py
"""
Unit tests for InputManager module.

Tests input file loading, validation, and error handling.
"""

import pytest
import json
import torch
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.input_manager import InputManager, InputValidationError
from models.analysis_models import InputArtifacts, ActivationData


class TestInputManager:
    """Test suite for InputManager functionality."""

    @pytest.fixture
    def temp_data_path(self):
        """Create temporary data directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_metadata(self):
        """Sample metadata for testing."""
        return {
            "job_id": "test_job_123",
            "service": "miStudioTrain",
            "version": "1.0.0",
            "model_info": {"model_name": "test/model", "hidden_size": 512},
            "sae_config": {"hidden_dim": 64, "input_dim": 512},
            "training_results": {"final_loss": 0.01},
            "ready_for_find_service": True,
        }

    @pytest.fixture
    def sample_sae_model(self):
        """Sample SAE model data for testing."""
        return {
            "model_state_dict": {"encoder.weight": torch.randn(64, 512)},
            "input_dim": 512,
            "hidden_dim": 64,
            "sparsity_coeff": 0.001,
        }

    @pytest.fixture
    def sample_activation_data(self):
        """Sample activation data for testing."""
        return {
            "feature_activations": torch.randn(100, 64),
            "original_activations": torch.randn(100, 512),
            "texts": [f"Sample text {i}" for i in range(100)],
            "feature_count": 64,
            "activation_dim": 512,
        }

    def test_input_manager_initialization(self, temp_data_path):
        """Test InputManager initialization."""
        manager = InputManager(str(temp_data_path))
        assert manager.data_path == temp_data_path

    def test_discover_input_files_success(self, temp_data_path):
        """Test successful input file discovery."""
        # Create directory structure
        job_id = "test_job_123"
        models_dir = temp_data_path / "models" / job_id
        activations_dir = temp_data_path / "activations" / job_id

        models_dir.mkdir(parents=True)
        activations_dir.mkdir(parents=True)

        # Create required files
        (models_dir / "sae_model.pt").touch()
        (activations_dir / "feature_activations.pt").touch()
        (activations_dir / "metadata.json").touch()

        manager = InputManager(str(temp_data_path))
        artifacts = manager.discover_input_files(job_id)

        assert artifacts.job_id == job_id
        assert Path(artifacts.sae_model_path).exists()
        assert Path(artifacts.feature_activations_path).exists()
        assert Path(artifacts.metadata_path).exists()

    def test_discover_input_files_missing_files(self, temp_data_path):
        """Test input file discovery with missing files."""
        manager = InputManager(str(temp_data_path))

        with pytest.raises(InputValidationError, match="Missing input files"):
            manager.discover_input_files("nonexistent_job")

    def test_load_metadata_success(self, temp_data_path, sample_metadata):
        """Test successful metadata loading."""
        # Create metadata file
        metadata_file = temp_data_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(sample_metadata, f)

        manager = InputManager(str(temp_data_path))
        loaded_metadata = manager.load_metadata(str(metadata_file))

        assert loaded_metadata["job_id"] == sample_metadata["job_id"]
        assert loaded_metadata["service"] == "miStudioTrain"
        assert loaded_metadata["ready_for_find_service"] is True

    def test_load_metadata_invalid_service(self, temp_data_path):
        """Test metadata loading with invalid service."""
        invalid_metadata = {
            "job_id": "test",
            "service": "WrongService",
            "ready_for_find_service": True,
        }

        metadata_file = temp_data_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(invalid_metadata, f)

        manager = InputManager(str(temp_data_path))

        with pytest.raises(InputValidationError, match="Invalid source service"):
            manager.load_metadata(str(metadata_file))

    def test_load_sae_model_success(self, temp_data_path, sample_sae_model):
        """Test successful SAE model loading."""
        model_file = temp_data_path / "sae_model.pt"
        torch.save(sample_sae_model, model_file)

        manager = InputManager(str(temp_data_path))
        loaded_model = manager.load_sae_model(str(model_file))

        assert loaded_model["input_dim"] == 512
        assert loaded_model["hidden_dim"] == 64
        assert "model_state_dict" in loaded_model

    def test_load_activation_data_success(self, temp_data_path, sample_activation_data):
        """Test successful activation data loading."""
        activation_file = temp_data_path / "activations.pt"
        torch.save(sample_activation_data, activation_file)

        manager = InputManager(str(temp_data_path))
        loaded_data = manager.load_activation_data(str(activation_file))

        assert isinstance(loaded_data, ActivationData)
        assert loaded_data.feature_count == 64
        assert loaded_data.activation_dim == 512
        assert len(loaded_data.texts) == 100

    def test_activation_data_consistency_validation(self, temp_data_path):
        """Test activation data consistency validation."""
        # Create inconsistent data
        inconsistent_data = {
            "feature_activations": torch.randn(100, 64),
            "original_activations": torch.randn(50, 512),  # Wrong size
            "texts": [f"Text {i}" for i in range(100)],
            "feature_count": 64,
            "activation_dim": 512,
        }

        activation_file = temp_data_path / "activations.pt"
        torch.save(inconsistent_data, activation_file)

        manager = InputManager(str(temp_data_path))

        with pytest.raises(InputValidationError, match="consistency check failed"):
            manager.load_activation_data(str(activation_file))

    def test_validate_input_consistency_mismatch(
        self, sample_metadata, sample_sae_model
    ):
        """Test input consistency validation with mismatched dimensions."""
        manager = InputManager()

        # Create mismatched activation data
        mismatched_data = ActivationData(
            feature_activations=torch.randn(100, 32),  # Wrong feature count
            original_activations=torch.randn(100, 512),
            texts=[f"Text {i}" for i in range(100)],
            feature_count=32,  # Doesn't match metadata (64)
            activation_dim=512,
        )

        with pytest.raises(InputValidationError, match="Feature count mismatch"):
            manager.validate_input_consistency(
                sample_metadata, sample_sae_model, mismatched_data
            )
