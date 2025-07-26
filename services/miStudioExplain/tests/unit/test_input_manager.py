"""
Complete Unit Tests for InputManager

Working test examples with comprehensive coverage, fixtures, and error scenarios.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open

from src.core.input_manager import (
    InputManager, 
    FeatureData, 
    JobMetadata, 
    ValidationError, 
    FileNotFoundError
)


class TestInputManager:
    """Comprehensive test suite for InputManager functionality."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_feature_data(self):
        """Sample feature data for testing."""
        return {
            "feature_id": 123,
            "coherence_score": 0.75,
            "quality_level": "good",
            "pattern_category": "technical",
            "pattern_keywords": ["json", "schema", "validation"],
            "top_activations": [
                {
                    "text": "JSON schema validation example",
                    "activation_strength": 0.85,
                    "context": "API documentation"
                },
                {
                    "text": "Schema validation in REST APIs",
                    "activation_strength": 0.78,
                    "context": "technical writing"
                }
            ],
            "activation_statistics": {
                "mean": 0.15,
                "std": 0.08,
                "frequency": 0.023
            }
        }
    
    @pytest.fixture
    def sample_job_metadata(self):
        """Sample job metadata for testing."""
        return {
            "job_id": "find_20250726_123456",
            "source_training_job": "train_20250726_120000",
            "model_name": "microsoft/phi-4",
            "total_features_processed": 512,
            "processing_time": "24.5 minutes",
            "service_version": "1.0.0",
            "completion_timestamp": "2025-07-26T15:30:45Z"
        }
    
    @pytest.fixture
    def complete_mistudio_results(self, sample_job_metadata, sample_feature_data):
        """Complete miStudioFind results for testing."""
        return {
            "job_metadata": sample_job_metadata,
            "features": [sample_feature_data] * 3,  # 3 identical features for testing
            "summary_insights": {
                "high_quality_features": 2,
                "total_features_analyzed": 3,
                "average_coherence": 0.75
            }
        }
    
    def test_initialization_with_valid_path(self, temp_data_dir):
        """Test InputManager initialization with valid data path."""
        manager = InputManager(str(temp_data_dir))
        
        assert manager.data_path == temp_data_dir
        assert temp_data_dir.exists()
        assert temp_data_dir.is_dir()
    
    def test_initialization_with_invalid_path(self):
        """Test InputManager initialization with invalid path."""
        with pytest.raises(ValidationError, match="data_path must be a non-empty string"):
            InputManager("")
        
        with pytest.raises(ValidationError, match="data_path must be a non-empty string"):
            InputManager(None)
    
    def test_load_valid_mistudio_results(self, temp_data_dir, complete_mistudio_results):
        """Test loading valid miStudioFind JSON results."""
        # Setup
        job_id = "find_20250726_123456"
        results_file = temp_data_dir / f"{job_id}_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(complete_mistudio_results, f)
        
        manager = InputManager(str(temp_data_dir))
        
        # Test
        result = manager.load_mistudio_find_results(job_id)
        
        # Assert
        assert result["job_metadata"]["job_id"] == job_id
        assert len(result["features"]) == 3
        assert result["features"][0]["feature_id"] == 123
        assert result["summary_insights"]["total_features_analyzed"] == 3
    
    def test_load_results_with_alternative_filename(self, temp_data_dir, complete_mistudio_results):
        """Test loading results with alternative filename patterns."""
        job_id = "test_job_456"
        
        # Test alternative filename pattern
        results_file = temp_data_dir / f"{job_id}.json"
        
        with open(results_file, 'w') as f:
            json.dump(complete_mistudio_results, f)
        
        manager = InputManager(str(temp_data_dir))
        result = manager.load_mistudio_find_results(job_id)
        
        assert result["job_metadata"]["job_id"] == "find_20250726_123456"  # Original ID in data
        assert len(result["features"]) == 3
    
    def test_load_results_file_not_found(self, temp_data_dir):
        """Test error handling when results file is not found."""
        manager = InputManager(str(temp_data_dir))
        
        with pytest.raises(FileNotFoundError) as exc_info:
            manager.load_mistudio_find_results("nonexistent_job")
        
        assert "Results file not found" in str(exc_info.value)
        assert "nonexistent_job" in str(exc_info.value)
    
    def test_load_results_invalid_json(self, temp_data_dir):
        """Test error handling for invalid JSON files."""
        job_id = "invalid_json_job"
        results_file = temp_data_dir / f"{job_id}_results.json"
        
        # Write invalid JSON
        with open(results_file, 'w') as f:
            f.write('{"invalid": json content}')
        
        manager = InputManager(str(temp_data_dir))
        
        with pytest.raises(json.JSONDecodeError):
            manager.load_mistudio_find_results(job_id)
    
    def test_validate_input_data_valid(self, complete_mistudio_results):
        """Test input data validation with valid data."""
        manager = InputManager()
        
        result = manager.validate_input_data(complete_mistudio_results)
        
        assert result is True
    
    def test_validate_input_data_missing_root_keys(self):
        """Test validation failure with missing root keys."""
        manager = InputManager()
        invalid_data = {"features": []}  # Missing job_metadata and summary_insights
        
        with pytest.raises(ValidationError, match="Missing required root keys"):
            manager.validate_input_data(invalid_data)
    
    def test_validate_input_data_invalid_type(self):
        """Test validation failure with invalid data type."""
        manager = InputManager()
        
        with pytest.raises(TypeError, match="Input data must be a dictionary"):
            manager.validate_input_data("not a dictionary")
    
    def test_validate_input_data_empty_features(self, sample_job_metadata):
        """Test validation failure with empty features list."""
        manager = InputManager()
        invalid_data = {
            "job_metadata": sample_job_metadata,
            "features": [],  # Empty features
            "summary_insights": {}
        }
        
        with pytest.raises(ValidationError, match="features list cannot be empty"):
            manager.validate_input_data(invalid_data)
    
    def test_extract_features_valid_data(self, complete_mistudio_results):
        """Test feature extraction from valid data."""
        manager = InputManager()
        
        features = manager.extract_features(complete_mistudio_results)
        
        assert len(features) == 3
        assert all(isinstance(f, FeatureData) for f in features)
        assert features[0].feature_id == 123
        assert features[0].coherence_score == 0.75
        assert features[0].pattern_category == "technical"
        assert len(features[0].pattern_keywords) == 3
        assert len(features[0].top_activations) == 2
    
    def test_extract_features_invalid_feature_data(self, sample_job_metadata):
        """Test feature extraction with invalid feature data."""
        manager = InputManager()
        
        # Data with malformed feature
        invalid_data = {
            "job_metadata": sample_job_metadata,
            "features": [
                {"feature_id": "not_an_int"},  # Invalid feature_id type
                {"feature_id": 456, "coherence_score": 0.6}  # Valid feature
            ],
            "summary_insights": {}
        }
        
        features = manager.extract_features(invalid_data)
        
        # Should skip malformed feature and extract valid one
        assert len(features) == 1
        assert features[0].feature_id == 456
    
    def test_extract_features_no_valid_features(self, sample_job_metadata):
        """Test feature extraction when no valid features can be extracted."""
        manager = InputManager()
        
        invalid_data = {
            "job_metadata": sample_job_metadata,
            "features": [
                {"invalid": "feature"},
                {"also": "invalid"}
            ],
            "summary_insights": {}
        }
        
        with pytest.raises(ValidationError, match="No valid features could be extracted"):
            manager.extract_features(invalid_data)
    
    def test_get_job_metadata_valid(self, complete_mistudio_results):
        """Test job metadata extraction from valid data."""
        manager = InputManager()
        
        metadata = manager.get_job_metadata(complete_mistudio_results)
        
        assert isinstance(metadata, JobMetadata)
        assert metadata.job_id == "find_20250726_123456"
        assert metadata.source_training_job == "train_20250726_120000"
        assert metadata.model_name == "microsoft/phi-4"
        assert metadata.total_features == 512
        assert metadata.processing_time == "24.5 minutes"
        assert metadata.service_version == "1.0.0"
    
    def test_get_job_metadata_missing_metadata(self):
        """Test metadata extraction failure when metadata is missing."""
        manager = InputManager()
        invalid_data = {"features": []}  # No job_metadata
        
        with pytest.raises(ValidationError, match="Missing job_metadata"):
            manager.get_job_metadata(invalid_data)
    
    def test_get_job_metadata_invalid_values(self):
        """Test metadata extraction with invalid values."""
        manager = InputManager()
        
        invalid_data = {
            "job_metadata": {
                "job_id": "",  # Empty job_id
                "source_training_job": "valid_job",
                "total_features_processed": -1  # Invalid count
            }
        }
        
        with pytest.raises(ValidationError):
            manager.get_job_metadata(invalid_data)
    
    def test_list_available_jobs(self, temp_data_dir):
        """Test listing available job IDs."""
        # Create test files with different naming patterns
        test_files = [
            "job1_results.json",
            "job2.json",
            "mistudio_find_job3.json",
            "feature_explanations_job4.json",
            "not_a_job_file.txt"  # Should be ignored
        ]
        
        for filename in test_files:
            (temp_data_dir / filename).write_text('{"test": "data"}')
        
        manager = InputManager(str(temp_data_dir))
        available_jobs = manager.list_available_jobs()
        
        expected_jobs = ["job1", "job2", "job3", "job4"]
        assert sorted(available_jobs) == sorted(expected_jobs)
    
    def test_get_job_summary(self, temp_data_dir, complete_mistudio_results):
        """Test getting job summary without loading full data."""
        job_id = "summary_test_job"
        results_file = temp_data_dir / f"{job_id}_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(complete_mistudio_results, f)
        
        manager = InputManager(str(temp_data_dir))
        summary = manager.get_job_summary(job_id)
        
        assert summary["job_id"] == "find_20250726_123456"
        assert summary["feature_count"] == 3
        assert summary["avg_coherence_score"] == 0.75
        assert summary["max_coherence_score"] == 0.75
        assert summary["high_quality_features"] == 3  # All features have score >= 0.6
    
    def test_get_job_summary_error_handling(self, temp_data_dir):
        """Test job summary error handling for non-existent job."""
        manager = InputManager(str(temp_data_dir))
        summary = manager.get_job_summary("nonexistent_job")
        
        assert summary["job_id"] == "nonexistent_job"
        assert "error" in summary
        assert "Results file not found" in summary["error"]
    
    def test_feature_data_validation(self):
        """Test FeatureData validation in __post_init__."""
        # Valid feature data
        valid_feature = FeatureData(
            feature_id=1,
            coherence_score=0.5,
            quality_level="good",
            pattern_category="technical",
            pattern_keywords=["test"],
            top_activations=[{"text": "example"}],
            activation_statistics={"mean": 0.1}
        )
        assert valid_feature.feature_id == 1
        
        # Invalid feature_id
        with pytest.raises(ValidationError, match="Invalid feature_id"):
            FeatureData(
                feature_id=-1,  # Negative ID
                coherence_score=0.5,
                quality_level="good",
                pattern_category="technical",
                pattern_keywords=["test"],
                top_activations=[],
                activation_statistics={}
            )
        
        # Invalid coherence_score
        with pytest.raises(ValidationError, match="Invalid coherence_score"):
            FeatureData(
                feature_id=1,
                coherence_score=1.5,  # > 1.0
                quality_level="good",
                pattern_category="technical",
                pattern_keywords=["test"],
                top_activations=[],
                activation_statistics={}
            )
    
    def test_job_metadata_validation(self):
        """Test JobMetadata validation in __post_init__."""
        # Valid metadata
        valid_metadata = JobMetadata(
            job_id="test_job",
            source_training_job="source_job",
            model_name="test/model",
            total_features=100,
            processing_time="10 minutes"
        )
        assert valid_metadata.job_id == "test_job"
        
        # Invalid job_id
        with pytest.raises(ValidationError, match="job_id must be a non-empty string"):
            JobMetadata(
                job_id="",  # Empty string
                source_training_job="source_job",
                model_name="test/model",
                total_features=100,
                processing_time="10 minutes"
            )
        
        # Invalid total_features
        with pytest.raises(ValidationError, match="total_features must be a positive integer"):
            JobMetadata(
                job_id="test_job",
                source_training_job="source_job",
                model_name="test/model",
                total_features=0,  # Must be positive
                processing_time="10 minutes"
            )
    
    def test_data_structure_validation_consistency_checks(self, temp_data_dir, complete_mistudio_results):
        """Test internal data structure validation with consistency checks."""
        # Modify data to have mismatched feature count
        modified_results = complete_mistudio_results.copy()
        modified_results["job_metadata"]["total_features_processed"] = 10  # Mismatch with 3 actual features
        
        job_id = "consistency_test"
        results_file = temp_data_dir / f"{job_id}_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(modified_results, f)
        
        manager = InputManager(str(temp_data_dir))
        
        # Should still load but log warning about mismatch
        result = manager.load_mistudio_find_results(job_id)
        assert len(result["features"]) == 3  # Actual count
        assert result["job_metadata"]["total_features_processed"] == 10  # Declared count
    
    def test_duplicate_feature_ids_validation(self, temp_data_dir, sample_job_metadata):
        """Test validation failure with duplicate feature IDs."""
        # Create data with duplicate feature IDs
        duplicate_data = {
            "job_metadata": sample_job_metadata,
            "features": [
                {"feature_id": 1, "coherence_score": 0.5},
                {"feature_id": 1, "coherence_score": 0.7}  # Duplicate ID
            ],
            "summary_insights": {}
        }
        
        job_id = "duplicate_test"
        results_file = temp_data_dir / f"{job_id}_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(duplicate_data, f)
        
        manager = InputManager(str(temp_data_dir))
        
        with pytest.raises(ValidationError, match="Duplicate feature IDs found"):
            manager.load_mistudio_find_results(job_id)


# Additional integration-style tests
class TestInputManagerIntegration:
    """Integration tests for InputManager with realistic scenarios."""
    
    def test_complete_workflow_realistic_data(self, temp_data_dir):
        """Test complete workflow with realistic miStudioFind data."""
        # Create realistic test data
        realistic_data = {
            "job_metadata": {
                "job_id": "find_20250726_154523_abc123",
                "source_training_job": "train_20250726_151321_d9675dce",
                "model_name": "microsoft/phi-4",
                "total_features_processed": 512,
                "processing_time": "23.7 minutes",
                "service_version": "1.0.0",
                "completion_timestamp": "2025-07-26T15:45:23Z"
            },
            "features": [
                {
                    "feature_id": 348,
                    "coherence_score": 0.501,
                    "quality_level": "good",
                    "pattern_category": "technical",
                    "pattern_keywords": ["json", "schema", "validation", "api", "structure"],
                    "top_activations": [
                        {
                            "text": "JSON schema validation in REST API endpoints",
                            "activation_strength": 0.89,
                            "context": "API documentation"
                        },
                        {
                            "text": "Schema validation for request/response structures",
                            "activation_strength": 0.84,
                            "context": "technical specification"
                        }
                    ],
                    "activation_statistics": {
                        "mean": 0.15,
                        "std": 0.08,
                        "frequency": 0.023,
                        "max": 0.89,
                        "min": 0.02
                    }
                }
            ],
            "summary_insights": {
                "high_quality_features": 58,
                "total_features_analyzed": 512,
                "average_coherence": 0.42,
                "processing_efficiency": "94.3%"
            }
        }
        
        # Save realistic data
        job_id = "find_20250726_154523_abc123"
        results_file = temp_data_dir / f"{job_id}_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(realistic_data, f, indent=2)
        
        # Test complete workflow
        manager = InputManager(str(temp_data_dir))
        
        # 1. Load results
        results = manager.load_mistudio_find_results(job_id)
        assert results["job_metadata"]["job_id"] == job_id
        
        # 2. Validate data
        is_valid = manager.validate_input_data(results)
        assert is_valid
        
        # 3. Extract features
        features = manager.extract_features(results)
        assert len(features) == 1
        assert features[0].feature_id == 348
        assert features[0].coherence_score == 0.501
        
        # 4. Get metadata
        metadata = manager.get_job_metadata(results)
        assert metadata.total_features == 512
        assert metadata.model_name == "microsoft/phi-4"
        
        # 5. Get summary
        summary = manager.get_job_summary(job_id)
        assert summary["feature_count"] == 1
        assert summary["avg_coherence_score"] == 0.501


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])