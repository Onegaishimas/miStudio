# =============================================================================
# config/settings.py - Unified Configuration for miStudioTrain
# =============================================================================

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    """Unified training configuration that uses environment variables consistently"""
    
    # Required fields for training requests
    model_name: str
    corpus_file: str
    
    # Model configuration
    layer_number: int = 12
    huggingface_token: Optional[str] = None
    
    # SAE configuration
    hidden_dim: int = 1024
    sparsity_coeff: float = 1e-3
    
    # Training configuration
    learning_rate: float = 1e-4
    batch_size: int = 16
    max_epochs: int = 50
    min_loss: float = 0.01
    
    # Hardware configuration
    gpu_id: Optional[int] = None
    max_sequence_length: int = 512
    
    # Data path configuration - unified approach
    data_path: str = os.getenv("DATA_PATH", "/data")
    
    # Service metadata
    service_name: str = "miStudioTrain"
    service_version: str = "v1.2.0"
    
    def __post_init__(self):
        """Ensure data path exists and create Path object for convenience"""
        self.data_path_obj = Path(self.data_path)
        self.data_path_obj.mkdir(parents=True, exist_ok=True)
    
    @property
    def models_dir(self) -> Path:
        """Directory where trained SAE models are saved"""
        return self.data_path_obj / "models"
    
    @property
    def activations_dir(self) -> Path:
        """Directory where feature activations are saved"""
        return self.data_path_obj / "activations"
    
    @property
    def samples_dir(self) -> Path:
        """Directory where input corpus files are stored"""
        return self.data_path_obj / "samples"
    
    @property
    def cache_dir(self) -> Path:
        """Directory for temporary/cache files"""
        return self.data_path_obj / "cache"
    
    @property
    def logs_dir(self) -> Path:
        """Directory for service logs"""
        return self.data_path_obj / "logs" / "train"


class ServiceConfig:
    """Global service configuration - environment-first approach"""
    
    def __init__(self):
        # Primary data path - same pattern for all services
        self.data_path = os.getenv("DATA_PATH", "/data")
        self.data_path_obj = Path(self.data_path)
        self.data_path_obj.mkdir(parents=True, exist_ok=True)
        
        # API configuration
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8001"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Service metadata
        self.service_name = "miStudioTrain"
        self.service_version = "v1.2.0"
        
        # GPU configuration
        self.cuda_memory_fraction = float(os.getenv("CUDA_MEMORY_FRACTION", "0.9"))
        self.max_concurrent_jobs = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
        
        # Create subdirectories
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.data_path_obj / "models",
            self.data_path_obj / "activations", 
            self.data_path_obj / "samples",
            self.data_path_obj / "cache",
            self.data_path_obj / "logs" / "train"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def models_dir(self) -> Path:
        return self.data_path_obj / "models"
    
    @property
    def activations_dir(self) -> Path:
        return self.data_path_obj / "activations"
    
    @property
    def samples_dir(self) -> Path:
        return self.data_path_obj / "samples"
    
    @property
    def cache_dir(self) -> Path:
        return self.data_path_obj / "cache"
    
    @property
    def logs_dir(self) -> Path:
        return self.data_path_obj / "logs" / "train"


# Global configuration instance
config = ServiceConfig()


# Legacy support - keep TrainConfig for backwards compatibility
def create_train_config(**kwargs) -> TrainConfig:
    """Factory function to create TrainConfig with unified data path"""
    # Ensure data_path uses the global config
    if 'data_path' not in kwargs:
        kwargs['data_path'] = config.data_path
    
    return TrainConfig(**kwargs)