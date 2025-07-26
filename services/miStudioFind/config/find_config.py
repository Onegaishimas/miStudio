# config/find_config.py
"""
Configuration management for miStudioFind service.

This module defines all configuration parameters for the feature analysis service,
including processing parameters, quality thresholds, and resource limits.
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class FindConfig:
    """Configuration parameters for miStudioFind service."""
    
    # Core processing parameters
    top_k_selections: int = 20
    coherence_threshold: float = 0.7
    batch_size: int = 100
    memory_optimization: bool = True
    
    # Quality assessment parameters
    min_activation_threshold: float = 0.01
    diversity_threshold: float = 0.5
    outlier_detection_threshold: float = 2.5
    
    # Performance parameters
    max_concurrent_jobs: int = 4
    processing_timeout_minutes: int = 60
    memory_limit_gb: float = 8.0
    
    # Output parameters
    save_intermediate_results: bool = True
    compress_outputs: bool = False
    feature_preview_count: int = 50
    
    # Service configuration
    service_name: str = "miStudioFind"
    service_version: str = "1.0.0"
    data_path: str = "/data"
    log_level: str = "INFO"
    
    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    max_request_size_mb: int = 100
    
    @classmethod
    def from_env(cls) -> "FindConfig":
        """Create configuration from environment variables."""
        return cls(
            top_k_selections=int(os.getenv("TOP_K_SELECTIONS", "20")),
            coherence_threshold=float(os.getenv("COHERENCE_THRESHOLD", "0.7")),
            batch_size=int(os.getenv("BATCH_SIZE", "100")),
            memory_optimization=os.getenv("MEMORY_OPTIMIZATION", "true").lower() == "true",
            processing_timeout_minutes=int(os.getenv("PROCESSING_TIMEOUT_MINUTES", "60")),
            data_path=os.getenv("DATA_PATH", "/data"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8001")),
        )
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.top_k_selections <= 0:
            raise ValueError("top_k_selections must be positive")
        
        if not 0.0 <= self.coherence_threshold <= 1.0:
            raise ValueError("coherence_threshold must be between 0.0 and 1.0")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.processing_timeout_minutes <= 0:
            raise ValueError("processing_timeout_minutes must be positive")
        
        if self.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be positive")
        
        if not os.path.exists(self.data_path):
            raise ValueError(f"data_path does not exist: {self.data_path}")


# Global configuration instance
config = FindConfig.from_env()