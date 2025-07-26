# =============================================================================
# models/api_models.py
# =============================================================================

import requests
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class TrainingRequest(BaseModel):
    """Enhanced training request with dynamic model loading and validation"""
    # Required fields
    model_name: str = Field(description="HuggingFace model name (e.g., 'microsoft/Phi-4')")
    corpus_file: str = Field(description="Input corpus file name in samples/")

    # Model configuration
    layer_number: int = Field(default=12, ge=0, le=48, description="Layer to extract activations from")
    huggingface_token: Optional[str] = Field(default=None, description="HuggingFace authentication token")

    # SAE configuration
    hidden_dim: int = Field(default=1024, description="SAE hidden dimension")
    sparsity_coeff: float = Field(default=1e-3, description="L1 sparsity coefficient")

    # Training configuration
    learning_rate: float = Field(default=1e-4, description="Training learning rate")
    batch_size: int = Field(default=16, description="Training batch size")
    max_epochs: int = Field(default=50, description="Maximum training epochs")
    min_loss: float = Field(default=0.01, description="Early stopping loss threshold")

    # Hardware configuration
    gpu_id: Optional[int] = Field(default=None, description="Specific GPU ID (auto-select if None)")
    max_sequence_length: int = Field(default=512, description="Maximum sequence length for tokenization")

    @validator('model_name')
    def validate_model_name(cls, v):
        """Validate model exists on HuggingFace Hub"""
        try:
            # Quick check if model exists
            url = f"https://huggingface.co/api/models/{v}"
            response = requests.head(url, timeout=10)
            if response.status_code == 404:
                raise ValueError(f"Model '{v}' not found on HuggingFace Hub")
        except requests.RequestException:
            # If network fails, allow it (will fail later with better error)
            pass
        return v
    
    @validator('layer_number')
    def validate_layer_number(cls, v, values):
        """Validate layer number is reasonable"""
        if v < 0:
            raise ValueError("Layer number must be non-negative")
        if v > 48:  # Most models don't have more than 48 layers
            raise ValueError("Layer number seems too high (max 48)")
        return v

    @validator('hidden_dim')
    def validate_hidden_dim(cls, v):
        """Validate SAE hidden dimension"""
        if v < 64:
            raise ValueError("Hidden dimension too small (min 64)")
        if v > 32768:
            raise ValueError("Hidden dimension too large (max 32768)")
        return v

    @validator('batch_size')
    def validate_batch_size(cls, v):
        """Validate batch size"""
        if v < 1:
            raise ValueError("Batch size must be at least 1")
        if v > 128:
            raise ValueError("Batch size too large (max 128)")
        return v

    @validator('sparsity_coeff')
    def validate_sparsity_coeff(cls, v):
        """Validate sparsity coefficient"""
        if v <= 0:
            raise ValueError("Sparsity coefficient must be positive")
        if v > 1.0:
            raise ValueError("Sparsity coefficient should be <= 1.0")
        return v


class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    actual_model_loaded: str
    architecture: str
    total_layers: int
    selected_layer: int
    hidden_size: int
    vocab_size: int
    requires_token: bool


class TrainingStatus(BaseModel):
    """Training status response"""
    job_id: str
    status: str  # 'queued', 'running', 'completed', 'failed'
    progress: float  # 0.0 to 1.0
    current_epoch: int
    current_loss: float
    estimated_time_remaining: Optional[int]
    message: str
    model_info: Optional[ModelInfo] = None
    last_checkpoint: Optional[str] = None


class TrainingResult(BaseModel):
    """Training completion result"""
    job_id: str
    status: str
    model_path: str
    activations_path: str
    metadata_path: str
    training_stats: Dict[str, Any]
    feature_count: int
    model_info: ModelInfo
    ready_for_find_service: bool
    checkpoints: Optional[list] = None
