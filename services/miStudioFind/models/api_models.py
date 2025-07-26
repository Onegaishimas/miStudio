# models/api_models.py
"""
API data models for miStudioFind service.

This module defines the request and response models used by the FastAPI endpoints.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    """Enumeration of possible job statuses."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FindRequest(BaseModel):
    """Request model for starting a feature analysis job."""

    source_job_id: str = Field(
        description="Job ID from miStudioTrain that produced the input files"
    )
    top_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of top activating texts to find per feature",
    )
    coherence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum coherence score for feature quality assessment",
    )
    include_statistics: bool = Field(
        default=True, description="Whether to include detailed statistical analysis"
    )

    @validator("source_job_id")
    def validate_job_id(cls, v):
        """Validate job ID format."""
        if not v or len(v) < 10:
            raise ValueError("source_job_id must be a valid job identifier")
        return v


class FeatureActivation(BaseModel):
    """Model for a single feature activation record."""

    text: str = Field(description="Text snippet that activated the feature")
    activation_value: float = Field(description="Strength of feature activation")
    text_index: int = Field(description="Index of text in original dataset")
    ranking: int = Field(description="Ranking among top activations for this feature")


class FeatureStatistics(BaseModel):
    """Statistical information about a feature."""

    mean_activation: float = Field(description="Mean activation value across dataset")
    max_activation: float = Field(description="Maximum activation value")
    activation_frequency: float = Field(
        description="Frequency of activation above threshold"
    )
    coherence_score: float = Field(
        description="Coherence score for feature interpretability"
    )


class FeatureAnalysis(BaseModel):
    """Complete analysis results for a single feature."""

    feature_id: int = Field(description="Feature identifier")
    top_activations: List[FeatureActivation] = Field(
        description="Top activating text snippets"
    )
    statistics: FeatureStatistics = Field(description="Statistical analysis of feature")
    quality_assessment: str = Field(description="Quality assessment (high/medium/low)")


class ProcessingProgress(BaseModel):
    """Progress information for a running analysis job."""

    features_processed: int = Field(description="Number of features processed")
    total_features: int = Field(description="Total number of features to process")
    current_feature: int = Field(description="Currently processing feature ID")
    estimated_time_remaining: Optional[int] = Field(
        description="Estimated seconds remaining"
    )

    @property
    def progress_percentage(self) -> float:
        """Calculate progress as percentage."""
        if self.total_features == 0:
            return 0.0
        return (self.features_processed / self.total_features) * 100.0


class FindStatus(BaseModel):
    """Status response for a feature analysis job."""

    job_id: str = Field(description="Unique job identifier")
    status: JobStatus = Field(description="Current job status")
    progress: ProcessingProgress = Field(description="Processing progress information")
    message: str = Field(description="Human-readable status message")
    start_time: datetime = Field(description="Job start timestamp")
    completion_time: Optional[datetime] = Field(description="Job completion timestamp")
    error_message: Optional[str] = Field(description="Error message if job failed")


class QualityDistribution(BaseModel):
    """Distribution of feature quality across the analysis."""

    high_quality_features: int = Field(description="Number of high-quality features")
    medium_quality_features: int = Field(
        description="Number of medium-quality features"
    )
    low_quality_features: int = Field(description="Number of low-quality features")
    mean_coherence_score: float = Field(
        description="Mean coherence score across all features"
    )
    coherence_scores: List[float] = Field(
        description="Coherence scores for all features"
    )


class FindResult(BaseModel):
    """Complete results of a feature analysis job."""

    job_id: str = Field(description="Unique job identifier")
    source_job_id: str = Field(description="Source training job ID")
    status: JobStatus = Field(description="Job completion status")
    features_analyzed: int = Field(description="Total number of features analyzed")
    processing_time_seconds: float = Field(description="Total processing time")
    quality_distribution: QualityDistribution = Field(
        description="Quality assessment distribution"
    )
    output_files: Dict[str, str] = Field(description="Paths to generated output files")
    ready_for_explain_service: bool = Field(
        description="Whether results are ready for explanation"
    )


class FeaturePreview(BaseModel):
    """Preview of top coherent features for quick inspection."""

    feature_id: int = Field(description="Feature identifier")
    coherence_score: float = Field(description="Coherence score")
    pattern_description: str = Field(
        description="Brief description of identified pattern"
    )
    example_texts: List[str] = Field(description="Sample activating texts")


class FindPreview(BaseModel):
    """Preview response showing top features from analysis."""

    job_id: str = Field(description="Job identifier")
    top_coherent_features: List[FeaturePreview] = Field(
        description="Highest quality features"
    )
    total_features: int = Field(description="Total number of features analyzed")
    quality_summary: QualityDistribution = Field(
        description="Overall quality distribution"
    )


# models/analysis_models.py
"""
Internal data models for feature analysis processing.

These models are used internally by the analysis engine and are not exposed via the API.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch
import numpy as np


@dataclass
class InputArtifacts:
    """Container for input files from miStudioTrain."""

    sae_model_path: str
    feature_activations_path: str
    metadata_path: str
    job_id: str

    def validate_paths(self) -> None:
        """Validate that all required files exist."""
        import os

        required_files = [
            self.sae_model_path,
            self.feature_activations_path,
            self.metadata_path,
        ]

        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required input file not found: {file_path}")


@dataclass
class ActivationData:
    """Container for loaded activation data."""

    feature_activations: torch.Tensor  # Shape: [n_samples, n_features]
    original_activations: torch.Tensor  # Shape: [n_samples, original_dim]
    texts: List[str]
    feature_count: int
    activation_dim: int

    def validate_consistency(self) -> None:
        """Validate data consistency across tensors and metadata."""
        n_samples = len(self.texts)

        if self.feature_activations.shape[0] != n_samples:
            raise ValueError("Feature activations batch size doesn't match text count")

        if self.original_activations.shape[0] != n_samples:
            raise ValueError("Original activations batch size doesn't match text count")

        if self.feature_activations.shape[1] != self.feature_count:
            raise ValueError(
                "Feature activations dimension doesn't match feature count"
            )

        if self.original_activations.shape[1] != self.activation_dim:
            raise ValueError(
                "Original activations dimension doesn't match expected dimension"
            )


@dataclass
class FeatureAnalysisResult:
    """Results of analyzing a single feature."""

    feature_id: int
    top_activations: List[Dict[str, Any]]
    activation_statistics: Dict[str, float]
    coherence_score: float
    quality_level: str
    pattern_keywords: List[str]

    def to_api_model(self) -> FeatureAnalysis:
        """Convert to API response model."""
        from models.api_models import (
            FeatureAnalysis,
            FeatureActivation,
            FeatureStatistics,
        )

        activations = [
            FeatureActivation(
                text=act["text"],
                activation_value=act["activation_value"],
                text_index=act["text_index"],
                ranking=act["ranking"],
            )
            for act in self.top_activations
        ]

        statistics = FeatureStatistics(
            mean_activation=self.activation_statistics["mean_activation"],
            max_activation=self.activation_statistics["max_activation"],
            activation_frequency=self.activation_statistics["activation_frequency"],
            coherence_score=self.coherence_score,
        )

        return FeatureAnalysis(
            feature_id=self.feature_id,
            top_activations=activations,
            statistics=statistics,
            quality_assessment=self.quality_level,
        )


@dataclass
class ProcessingJob:
    """Container for tracking a processing job."""

    job_id: str
    source_job_id: str
    config: Dict[str, Any]
    status: str
    start_time: float
    progress: Dict[str, Any]
    results: Optional[List[FeatureAnalysisResult]] = None
    error_message: Optional[str] = None

    def update_progress(
        self, features_processed: int, total_features: int, current_feature: int
    ) -> None:
        """Update job progress information."""
        import time

        self.progress.update(
            {
                "features_processed": features_processed,
                "total_features": total_features,
                "current_feature": current_feature,
                "last_updated": time.time(),
            }
        )

        # Estimate remaining time based on current progress
        if features_processed > 0:
            elapsed_time = time.time() - self.start_time
            time_per_feature = elapsed_time / features_processed
            remaining_features = total_features - features_processed
            estimated_remaining = time_per_feature * remaining_features
            self.progress["estimated_time_remaining"] = int(estimated_remaining)
