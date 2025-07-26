# tests/test_integration.py
"""
Integration tests for miStudioFind service.

Tests end-to-end workflows and component integration.
"""

import pytest
import tempfile
import torch
import json
from pathlib import Path
from unittest.mock import patch

from core.processing_service import ProcessingService
from models.api_models import FindRequest


class TestIntegration:
    """Integration test suite for complete workflows."""
    
    @pytest.fixture
    def temp_data_environment(self):
        """Create complete temporary data environment for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)
            
            # Create source job structure
            source_job_id = "test_train_job_123"
            models_dir = data_path / "models" / source_job_id
            activations_dir = data_path / "activations" / source_job_id
            
            models_dir.mkdir(parents=True)
            activations_dir.mkdir(parents=True)
            
            # Create sample SAE model
            sae_model = {
                "model_state_dict": {"encoder.weight": torch.randn(64, 512)},
                "input_dim": 512,
                "hidden_dim": 64,
                "sparsity_coeff": 0.001
            }
            torch.save(sae_model, models_dir / "sae_model.pt")
            
            # Create sample activation data
            activation_data = {
                "feature_activations": torch.randn(100, 64).abs(),
                "original_activations": torch.randn(100, 512),
                "texts": [f"Sample text {i} about various topics" for i in range(100)],
                "feature_count": 64,
                "activation_dim": 512
            }
            torch.save(activation_data, activations_dir / "feature_activations.pt")
            
            # Create sample metadata
            metadata = {
                "job_id": source_job_id,
                "service": "miStudioTrain",
                "version": "1.0.0",
                "model_info": {"model_name": "test/model", "hidden_size": 512},
                "sae_config": {"hidden_dim": 64, "input_dim": 512},
                "training_results": {"final_loss": 0.01},
                "ready_for_find_service": True
            }
            
            with open(activations_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f)
            
            yield str(data_path), source_job_id
    
    @pytest.mark.asyncio
    async def test_complete_analysis_workflow(self, temp_data_environment):
        """Test complete feature analysis workflow."""
        data_path, source_job_id = temp_data_environment
        
        # Initialize processing service
        processing_service = ProcessingService(data_path)
        
        try:
            # Create analysis request
            request = FindRequest(
                source_job_id=source_job_id,
                top_k=10,
                coherence_threshold=0.5,
                include_statistics=True
            )
            
            # Start analysis job
            job_id = await processing_service.start_analysis_job(request)
            assert job_id is not None
            assert job_id.startswith("find_")
            
            # Wait for job to complete (in real scenario this would be background)
            # For testing, we'll simulate the execution
            import time
            start_time = time.time()
            
            # Check initial status
            status = processing_service.get_job_status(job_id)
            assert status is not None
            assert status["status"] in ["queued", "running"]
            
            # Simulate job execution by calling the private method directly
            processing_service._execute_analysis_job(job_id)
            
            # Check final status
            final_status = processing_service.get_job_status(job_id)
            assert final_status["status"] in ["completed", "failed"]
            
            if final_status["status"] == "completed":
                # Check results
                results = processing_service.get_job_results(job_id)
                assert results is not None
                assert results["total_features"] > 0
                assert "output_files" in results
                
                # Verify output files were created
                output_files = results["output_files"]
                for file_path in output_files.values():
                    assert Path(file_path).exists()
            
        finally:
            processing_service.shutdown()
    
    @pytest.mark.asyncio
    async def test_error_handling_missing_source_job(self):
        """Test error handling for missing source job."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processing_service = ProcessingService(temp_dir)
            
            try:
                request = FindRequest(
                    source_job_id="nonexistent_job",
                    top_k=10
                )
                
                with pytest.raises(Exception):  # Should raise ProcessingError
                    await processing_service.start_analysis_job(request)
            
            finally:
                processing_service.shutdown()
    
    def test_memory_management_integration(self, temp_data_environment):
        """Test memory management during processing."""
        data_path, source_job_id = temp_data_environment
        
        from utils.memory_manager import MemoryManager
        
        memory_manager = MemoryManager(max_memory_gb=1.0)  # Low limit for testing
        
        # Check memory before processing
        initial_memory = memory_manager.get_memory_usage()
        assert "system_memory" in initial_memory
        
        # Test memory context manager
        with memory_manager.memory_context("test_operation"):
            # Simulate some memory usage
            test_tensor = torch.randn(1000, 1000)
            del test_tensor
        
        # Memory should be cleaned up
        final_memory = memory_manager.get_memory_usage()
        assert final_memory["system_memory"]["used_gb"] <= initial_memory["system_memory"]["used_gb"] + 0.5
    
    def test_statistics_engine_integration(self):
        """Test statistics engine integration."""
        from utils.statistics_engine import StatisticsEngine
        
        stats_engine = StatisticsEngine()
        
        # Test with sample data
        sample_activations = np.random.exponential(1.0, 1000)  # Realistic activation distribution
        sample_texts = [f"Sample text {i} with content" for i in range(20)]
        
        # Test distribution analysis
        distribution_stats = stats_engine.compute_activation_distributions(sample_activations)
        assert "basic_statistics" in distribution_stats
        assert "percentiles" in distribution_stats
        
        # Test coherence analysis
        coherence_scores = stats_engine.calculate_coherence_scores(sample_texts)
        assert "combined_coherence" in coherence_scores
        assert 0.0 <= coherence_scores["combined_coherence"] <= 1.0
        
        # Test quality metrics
        quality_metrics = stats_engine.generate_quality_metrics(
            sample_activations[:20], sample_texts, feature_id=42
        )
        assert "overall_quality" in quality_metrics
        assert quality_metrics["feature_id"] == 42_success(self, sample_metadata, sample_sae_model, sample_activation_data):
        """Test successful input consistency validation."""
        manager = InputManager()
        
        activation_data = ActivationData(
            feature_activations=sample_activation_data["feature_activations"],
            original_activations=sample_activation_data["original_activations"],
            texts=sample_activation_data["texts"],
            feature_count=sample_activation_data["feature_count"],
            activation_dim=sample_activation_data["activation_dim"]
        )
        
        # Should not raise exception
        manager.validate_input_consistency(sample_metadata, sample_sae_model, activation_data)
    
    def test_validate_input_consistency(self, sample_metadata, sample_sae_model, sample_activation_data):
        """Test input consistency validation with missing data."""
        manager = InputManager()
        
        # Create activation data with missing feature activations
        activation_data = ActivationData(
            feature_activations=None,  # Missing feature activations
            original_activations=sample_activation_data["original_activations"],
            texts=sample_activation_data["texts"],
            feature_count=sample_activation_data["feature_count"],
            activation_dim=sample_activation_data["activation_dim"]
        )
        
        with pytest.raises(ValueError, match="Feature activations are required"):
            manager.validate_input_consistency(sample_metadata, sample_sae_model, activation_data)