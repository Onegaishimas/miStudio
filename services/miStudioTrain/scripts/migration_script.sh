#!/bin/bash
# Migration script to convert miStudioTrain to modular architecture

set -e

SERVICE_DIR="$HOME/app/miStudio/services/miStudioTrain"
SRC_DIR="$SERVICE_DIR/src"

echo "🔄 Migrating miStudioTrain to modular architecture..."

# Backup original main.py
if [ -f "$SRC_DIR/main.py" ]; then
    echo "📦 Backing up original main.py..."
    cp "$SRC_DIR/main.py" "$SRC_DIR/main.py.backup.$(date +%Y%m%d_%H%M%S)"
fi

# Create directory structure
echo "📁 Creating modular directory structure..."
mkdir -p "$SRC_DIR/config"
mkdir -p "$SRC_DIR/models"
mkdir -p "$SRC_DIR/core"
mkdir -p "$SRC_DIR/utils"

# Create __init__.py files
echo "📝 Creating __init__.py files..."
touch "$SRC_DIR/__init__.py"
touch "$SRC_DIR/config/__init__.py"
touch "$SRC_DIR/models/__init__.py"
touch "$SRC_DIR/core/__init__.py"
touch "$SRC_DIR/utils/__init__.py"

# Create config/settings.py
echo "⚙️  Creating config/settings.py..."
cat > "$SRC_DIR/config/settings.py" << 'EOF'
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
EOF

# Create models/api_models.py
echo "📊 Creating models/api_models.py..."
cat > "$SRC_DIR/models/api_models.py" << 'EOF'
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class TrainingRequest(BaseModel):
    """Enhanced training request with dynamic model loading"""
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
EOF

# Create utils/logging_config.py
echo "📋 Creating utils/logging_config.py..."
cat > "$SRC_DIR/utils/logging_config.py" << 'EOF'
import logging


def setup_logging(level: str = "INFO"):
    """Configure logging for the application"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Set specific loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
EOF

# Note: The SAE, GPU Manager, Activation Extractor, and Training Service modules
# need to be created manually from the artifacts provided above

echo "✅ Basic modular structure created!"
echo ""
echo "📋 Next steps:"
echo "1. Copy the remaining modules from the artifacts:"
echo "   - models/sae.py"
echo "   - core/gpu_manager.py" 
echo "   - core/activation_extractor.py"
echo "   - core/training_service.py"
echo "2. Replace main.py with the simplified version"
echo "3. Test the modular version"
echo ""
echo "🔗 Files ready for manual creation:"
echo "   $SRC_DIR/models/sae.py"
echo "   $SRC_DIR/core/gpu_manager.py"
echo "   $SRC_DIR/core/activation_extractor.py"
echo "   $SRC_DIR/core/training_service.py"
echo "   $SRC_DIR/main.py (simplified)"

# Create a test script
echo "🧪 Creating test script..."
cat > "$SERVICE_DIR/test_modular.py" << 'EOF'
#!/usr/bin/env python3
"""
Test script for modular miStudioTrain
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported"""
    try:
        from config.settings import TrainConfig
        print("✅ config.settings imported successfully")
        
        from models.api_models import TrainingRequest, ModelInfo
        print("✅ models.api_models imported successfully")
        
        from utils.logging_config import setup_logging
        print("✅ utils.logging_config imported successfully")
        
        # These will be available after copying the full modules
        try:
            from models.sae import SparseAutoencoder
            print("✅ models.sae imported successfully")
        except ImportError:
            print("⚠️  models.sae not yet available (copy from artifacts)")
        
        try:
            from core.gpu_manager import GPUManager
            print("✅ core.gpu_manager imported successfully")
        except ImportError:
            print("⚠️  core.gpu_manager not yet available (copy from artifacts)")
            
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing modular miStudioTrain imports...")
    success = test_imports()
    
    if success:
        print("\n🎉 Basic modular structure is working!")
    else:
        print("\n💥 Issues detected - check the imports")
EOF

chmod +x "$SERVICE_DIR/test_modular.py"

echo ""
echo "🎯 Migration preparation complete!"
echo "Run: python $SERVICE_DIR/test_modular.py to test basic imports"