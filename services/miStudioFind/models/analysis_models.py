# models/analysis_models.py
"""
Internal data models for feature analysis processing in miStudioFind service.

These models are used internally by the analysis engine and are not exposed via the API.
They handle the core data structures for processing SAE features and managing analysis workflows.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch
import numpy as np
import os
from pathlib import Path


@dataclass
class InputArtifacts:
    """
    Container for input files from miStudioTrain.
    
    Manages the three required input files and provides validation methods
    to ensure all necessary artifacts are available for processing.
    """
    
    sae_model_path: str
    feature_activations_path: str
    metadata_path: str
    job_id: str
    
    def validate_paths(self) -> None:
        """
        Validate that all required files exist and are accessible.
        
        Raises:
            FileNotFoundError: If any required input file is missing
        """
        required_files = [
            self.sae_model_path,
            self.feature_activations_path,
            self.metadata_path
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required input file not found: {file_path}")
    
    def get_file_sizes(self) -> Dict[str, int]:
        """Get file sizes for all input files in bytes."""
        sizes = {}
        file_paths = {
            "sae_model": self.sae_model_path,
            "feature_activations": self.feature_activations_path,
            "metadata": self.metadata_path
        }
        
        for name, path in file_paths.items():
            try:
                sizes[name] = os.path.getsize(path)
            except OSError:
                sizes[name] = -1  # File not accessible
        
        return sizes


@dataclass
class ActivationData:
    """
    Container for loaded activation data from miStudioTrain.
    
    Holds the complete dataset including SAE features, original activations,
    and corresponding text snippets with validation methods.
    """
    
    feature_activations: torch.Tensor  # Shape: [n_samples, n_features]
    original_activations: torch.Tensor  # Shape: [n_samples, original_dim]
    texts: List[str]
    feature_count: int
    activation_dim: int
    
    def validate_consistency(self) -> None:
        """
        Validate data consistency across tensors and metadata.
        
        Ensures that all components have matching dimensions and sizes.
        
        Raises:
            ValueError: If data dimensions are inconsistent
        """
        n_samples = len(self.texts)
        
        # Check feature activations consistency
        if self.feature_activations.shape[0] != n_samples:
            raise ValueError(
                f"Feature activations batch size ({self.feature_activations.shape[0]}) "
                f"doesn't match text count ({n_samples})"
            )
        
        # Check original activations consistency
        if self.original_activations.shape[0] != n_samples:
            raise ValueError(
                f"Original activations batch size ({self.original_activations.shape[0]}) "
                f"doesn't match text count ({n_samples})"
            )
        
        # Check feature dimension consistency
        if self.feature_activations.shape[1] != self.feature_count:
            raise ValueError(
                f"Feature activations dimension ({self.feature_activations.shape[1]}) "
                f"doesn't match feature count ({self.feature_count})"
            )
        
        # Check activation dimension consistency
        if self.original_activations.shape[1] != self.activation_dim:
            raise ValueError(
                f"Original activations dimension ({self.original_activations.shape[1]}) "
                f"doesn't match expected dimension ({self.activation_dim})"
            )
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the activation data."""
        return {
            "sample_count": len(self.texts),
            "feature_count": self.feature_count,
            "activation_dim": self.activation_dim,
            "feature_activations_shape": list(self.feature_activations.shape),
            "original_activations_shape": list(self.original_activations.shape),
            "feature_activations_dtype": str(self.feature_activations.dtype),
            "original_activations_dtype": str(self.original_activations.dtype),
            "text_sample_lengths": {
                "min": min(len(text) for text in self.texts) if self.texts else 0,
                "max": max(len(text) for text in self.texts) if self.texts else 0,
                "mean": np.mean([len(text) for text in self.texts]) if self.texts else 0
            }
        }
    
    def get_feature_activation_stats(self, feature_id: int) -> Dict[str, float]:
        """Get basic statistics for a specific feature's activations."""
        if feature_id >= self.feature_count or feature_id < 0:
            raise ValueError(f"Feature ID {feature_id} out of range [0, {self.feature_count-1}]")
        
        activations = self.feature_activations[:, feature_id]
        
        return {
            "mean": float(torch.mean(activations)),
            "std": float(torch.std(activations)),
            "min": float(torch.min(activations)),
            "max": float(torch.max(activations)),
            "median": float(torch.median(activations)),
            "nonzero_count": int(torch.sum(activations > 0)),
            "nonzero_ratio": float(torch.sum(activations > 0)) / len(activations)
        }


@dataclass
class FeatureAnalysisResult:
    """
    Results of analyzing a single feature.
    
    Contains all analysis outputs for one SAE feature including top activations,
    statistics, quality assessment, and pattern information.
    """
    
    feature_id: int
    top_activations: List[Dict[str, Any]]
    activation_statistics: Dict[str, float]
    coherence_score: float
    quality_level: str
    pattern_keywords: List[str]
    
    def to_api_model(self):
        """
        Convert to API response model.
        
        Returns:
            FeatureAnalysis object suitable for API responses
        """
        # Import here to avoid circular imports
        try:
            from models.api_models import FeatureAnalysis, FeatureActivation, FeatureStatistics
        except ImportError:
            # Fallback if API models not available
            return self._to_dict()
        
        # Convert top activations to API format
        activations = [
            FeatureActivation(
                text=act["text"],
                activation_value=act["activation_value"],
                text_index=act["text_index"],
                ranking=act["ranking"]
            )
            for act in self.top_activations
        ]
        
        # Convert statistics to API format
        statistics = FeatureStatistics(
            mean_activation=self.activation_statistics.get("mean_activation", 0.0),
            max_activation=self.activation_statistics.get("max_activation", 0.0),
            activation_frequency=self.activation_statistics.get("activation_frequency", 0.0),
            coherence_score=self.coherence_score
        )
        
        return FeatureAnalysis(
            feature_id=self.feature_id,
            top_activations=activations,
            statistics=statistics,
            quality_assessment=self.quality_level
        )
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format as fallback."""
        return {
            "feature_id": self.feature_id,
            "top_activations": self.top_activations,
            "activation_statistics": self.activation_statistics,
            "coherence_score": self.coherence_score,
            "quality_level": self.quality_level,
            "pattern_keywords": self.pattern_keywords
        }
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get a summary of the feature quality assessment."""
        return {
            "feature_id": self.feature_id,
            "quality_level": self.quality_level,
            "coherence_score": self.coherence_score,
            "top_activation_count": len(self.top_activations),
            "pattern_keyword_count": len(self.pattern_keywords),
            "max_activation_value": max(
                (act["activation_value"] for act in self.top_activations),
                default=0.0
            ),
            "interpretable": self.coherence_score >= 0.5 and self.quality_level in ["high", "medium"]
        }


@dataclass
class ProcessingJob:
    """
    Container for tracking a processing job.
    
    Manages job state, progress tracking, and result storage for
    background feature analysis tasks.
    """
    
    job_id: str
    source_job_id: str
    config: Dict[str, Any]
    status: str
    start_time: float
    progress: Dict[str, Any]
    results: Optional[List[FeatureAnalysisResult]] = None
    error_message: Optional[str] = None
    
    def update_progress(self, features_processed: int, total_features: int, current_feature: int) -> None:
        """
        Update job progress information.
        
        Args:
            features_processed: Number of features completed
            total_features: Total number of features to process
            current_feature: Currently processing feature ID
        """
        import time
        
        self.progress.update({
            "features_processed": features_processed,
            "total_features": total_features,
            "current_feature": current_feature,
            "last_updated": time.time()
        })
        
        # Estimate remaining time based on current progress
        if features_processed > 0:
            elapsed_time = time.time() - self.start_time
            time_per_feature = elapsed_time / features_processed
            remaining_features = total_features - features_processed
            estimated_remaining = time_per_feature * remaining_features
            self.progress["estimated_time_remaining"] = int(estimated_remaining)
        else:
            self.progress["estimated_time_remaining"] = None
    
    def get_progress_percentage(self) -> float:
        """Calculate progress as percentage."""
        total = self.progress.get("total_features", 0)
        processed = self.progress.get("features_processed", 0)
        
        if total == 0:
            return 0.0
        
        return (processed / total) * 100.0
    
    def get_processing_rate(self) -> float:
        """Get features processed per second."""
        import time
        
        elapsed_time = time.time() - self.start_time
        processed = self.progress.get("features_processed", 0)
        
        if elapsed_time == 0:
            return 0.0
        
        return processed / elapsed_time
    
    def add_result(self, result: FeatureAnalysisResult) -> None:
        """Add a feature analysis result to the job."""
        if self.results is None:
            self.results = []
        
        self.results.append(result)
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of analysis results."""
        if not self.results:
            return {
                "total_results": 0,
                "quality_distribution": {"high": 0, "medium": 0, "low": 0},
                "mean_coherence": 0.0
            }
        
        # Quality distribution
        quality_counts = {"high": 0, "medium": 0, "low": 0}
        coherence_scores = []
        
        for result in self.results:
            quality_counts[result.quality_level] += 1
            coherence_scores.append(result.coherence_score)
        
        return {
            "total_results": len(self.results),
            "quality_distribution": quality_counts,
            "mean_coherence": np.mean(coherence_scores) if coherence_scores else 0.0,
            "interpretable_features": sum(
                1 for result in self.results 
                if result.coherence_score >= 0.5 and result.quality_level in ["high", "medium"]
            )
        }
    
    def is_completed(self) -> bool:
        """Check if the job has completed successfully."""
        return self.status == "completed" and self.results is not None
    
    def is_failed(self) -> bool:
        """Check if the job has failed."""
        return self.status == "failed" or self.error_message is not None


@dataclass
class AnalysisConfiguration:
    """
    Configuration parameters for a specific analysis job.
    
    Stores job-specific settings that may differ from global defaults.
    """
    
    top_k: int = 20
    coherence_threshold: float = 0.7
    include_statistics: bool = True
    memory_optimization: bool = True
    batch_size: int = 100
    min_activation_threshold: float = 0.01
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.top_k <= 0 or self.top_k > 100:
            raise ValueError("top_k must be between 1 and 100")
        
        if not 0.0 <= self.coherence_threshold <= 1.0:
            raise ValueError("coherence_threshold must be between 0.0 and 1.0")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.min_activation_threshold < 0:
            raise ValueError("min_activation_threshold must be non-negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "top_k": self.top_k,
            "coherence_threshold": self.coherence_threshold,
            "include_statistics": self.include_statistics,
            "memory_optimization": self.memory_optimization,
            "batch_size": self.batch_size,
            "min_activation_threshold": self.min_activation_threshold
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AnalysisConfiguration":
        """Create configuration from dictionary."""
        return cls(
            top_k=config_dict.get("top_k", 20),
            coherence_threshold=config_dict.get("coherence_threshold", 0.7),
            include_statistics=config_dict.get("include_statistics", True),
            memory_optimization=config_dict.get("memory_optimization", True),
            batch_size=config_dict.get("batch_size", 100),
            min_activation_threshold=config_dict.get("min_activation_threshold", 0.01)
        )


@dataclass
class ModelInfo:
    """
    Information about the source model and SAE configuration.
    
    Extracted from miStudioTrain metadata for validation and processing context.
    """
    
    model_name: str
    architecture: str
    total_layers: int
    hidden_size: int
    vocab_size: int
    sae_input_dim: int
    sae_hidden_dim: int
    sparsity_coeff: float
    layer_analyzed: int
    
    def validate_compatibility(self, activation_data: ActivationData) -> None:
        """
        Validate that model info is compatible with activation data.
        
        Args:
            activation_data: Loaded activation data to validate against
            
        Raises:
            ValueError: If model info and activation data are incompatible
        """
        if self.sae_input_dim != activation_data.activation_dim:
            raise ValueError(
                f"SAE input dimension ({self.sae_input_dim}) doesn't match "
                f"activation dimension ({activation_data.activation_dim})"
            )
        
        if self.sae_hidden_dim != activation_data.feature_count:
            raise ValueError(
                f"SAE hidden dimension ({self.sae_hidden_dim}) doesn't match "
                f"feature count ({activation_data.feature_count})"
            )
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of model information."""
        return {
            "model_name": self.model_name,
            "architecture": self.architecture,
            "model_scale": f"{self.total_layers} layers, {self.hidden_size} hidden size",
            "sae_configuration": {
                "input_dim": self.sae_input_dim,
                "hidden_dim": self.sae_hidden_dim,
                "sparsity_coeff": self.sparsity_coeff,
                "layer_analyzed": self.layer_analyzed
            },
            "vocab_size": self.vocab_size
        }
    
    @classmethod
    def from_metadata(cls, metadata: Dict[str, Any]) -> "ModelInfo":
        """Create ModelInfo from miStudioTrain metadata."""
        model_info = metadata.get("model_info", {})
        sae_config = metadata.get("sae_config", {})
        
        return cls(
            model_name=model_info.get("model_name", "unknown"),
            architecture=model_info.get("architecture", "unknown"),
            total_layers=model_info.get("total_layers", 0),
            hidden_size=model_info.get("hidden_size", 0),
            vocab_size=model_info.get("vocab_size", 0),
            sae_input_dim=sae_config.get("input_dim", 0),
            sae_hidden_dim=sae_config.get("hidden_dim", 0),
            sparsity_coeff=sae_config.get("sparsity_coeff", 0.0),
            layer_analyzed=sae_config.get("layer_number", 0)
        )