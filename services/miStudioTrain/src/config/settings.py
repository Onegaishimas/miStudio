# =============================================================================
# config/settings.py
# =============================================================================

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """Internal training configuration"""
    model_name: str
    corpus_file: str
    layer_number: int = 12
    huggingface_token: Optional[str] = None
    hidden_dim: int = 1024
    sparsity_coeff: float = 1e-3
    learning_rate: float = 1e-4
    batch_size: int = 16
    max_epochs: int = 50
    min_loss: float = 0.01
    gpu_id: Optional[int] = None
    max_sequence_length: int = 512
    data_path: str = "/data"
    service_name: str = "miStudioTrain"
    service_version: str = "v1.2.0"

