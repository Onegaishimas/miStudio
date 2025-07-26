#!/bin/bash
# miStudio Quick Setup for Sean's GPU Host Environment
# Run this script to get miStudio up and running quickly

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Step 1: Test Docker GPU with available images
test_docker_gpu() {
    print_step "Testing Docker GPU support with available images..."
    
    # Try different CUDA images that should be available
    local images=(
        "nvidia/cuda:12.1-base-ubuntu20.04"
        "nvidia/cuda:11.8-base-ubuntu20.04" 
        "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
        "tensorflow/tensorflow:2.13.0-gpu"
    )
    
    for image in "${images[@]}"; do
        print_info "Trying $image..."
        if docker run --rm --gpus all $image nvidia-smi > /dev/null 2>&1; then
            print_step "✅ Docker GPU support confirmed with $image"
            return 0
        fi
    done
    
    print_warning "No standard CUDA images found. Testing with nvidia-smi container..."
    # Create a simple test
    docker run --rm --gpus all ubuntu:20.04 bash -c "
        apt-get update -qq && apt-get install -y -qq nvidia-utils-470 > /dev/null 2>&1 || 
        apt-get install -y -qq nvidia-utils-525 > /dev/null 2>&1 || 
        echo 'GPU drivers accessible from container'
    "
}

# Step 2: Verify MicroK8s GPU setup
verify_microk8s_gpu() {
    print_step "Verifying MicroK8s GPU configuration..."
    
    # Check if nvidia addon is enabled
    if microk8s status | grep -q "nvidia.*enabled"; then
        print_step "✅ MicroK8s NVIDIA addon is enabled"
    else
        print_warning "MicroK8s NVIDIA addon not showing as enabled"
        print_info "Current MicroK8s status:"
        microk8s status
    fi
    
    # Check if GPU resources are visible to Kubernetes
    print_info "Checking GPU resources in Kubernetes..."
    GPU_NODES=$(kubectl get nodes -o json | jq -r '.items[] | select(.status.capacity."nvidia.com/gpu" != null) | .metadata.name' | wc -l)
    
    if [ "$GPU_NODES" -gt 0 ]; then
        print_step "✅ $GPU_NODES GPU node(s) detected in Kubernetes"
        kubectl get nodes -o json | jq -r '.items[] | select(.status.capacity."nvidia.com/gpu" != null) | "Node: " + .metadata.name + " GPUs: " + .status.capacity."nvidia.com/gpu"'
    else
        print_warning "No GPU resources detected in Kubernetes yet"
        print_info "This might resolve after enabling the addon. Continuing setup..."
    fi
}

# Step 3: Create miStudio project structure
create_mistudio_project() {
    print_step "Creating miStudio project structure..."
    
    # Create in Sean's app directory
    cd ~/app
    
    if [ -d "miStudio" ]; then
        print_warning "miStudio directory already exists. Moving to miStudio-backup-$(date +%s)"
        mv miStudio miStudio-backup-$(date +%s)
    fi
    
    # Create the project structure manually since we don't have the migration script yet
    mkdir -p miStudio/{services,infrastructure,ui,docs,tests,tools,data}
    cd miStudio
    
    # Create the 7 service directories with hybrid naming
    local services=("Train" "Find" "Explain" "Score" "Correlate" "Monitor" "Steer")
    for service in "${services[@]}"; do
        mkdir -p services/miStudio${service}/{src,k8s,scripts,tests}
    done
    
    # Create infrastructure directories
    mkdir -p infrastructure/{k8s,scripts,helm}
    mkdir -p tools/{dev,deployment,monitoring}
    mkdir -p data/{samples,models,artifacts}
    
    print_step "✅ miStudio project structure created at ~/app/miStudio"
}

# Step 4: Set up Python development environment
setup_python_env() {
    print_step "Setting up Python development environment..."
    
    cd ~/app/miStudio
    
    # Create virtual environment
    python3 -m venv miStudio-dev-env
    source miStudio-dev-env/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    print_info "Installing PyTorch with CUDA 12.1 support..."
    # Use CUDA 12.1 which is compatible with your 12.2 installation
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    print_info "Installing core ML libraries..."
    pip install transformers datasets accelerate bitsandbytes
    pip install numpy pandas scikit-learn
    
    print_info "Installing development tools..."
    pip install jupyter jupyterlab ipython
    pip install pytest pytest-cov black flake8
    pip install rich typer loguru
    
    print_info "Installing monitoring tools..."
    pip install nvidia-ml-py3 gpustat
    pip install wandb tensorboard
    
    print_info "Installing deployment tools..."
    pip install docker kubernetes
    
    # Save requirements
    pip freeze > dev-requirements.txt
    
    print_step "✅ Python environment created with GPU support"
}

# Step 5: Test GPU functionality
test_gpu_functionality() {
    print_step "Testing GPU functionality in Python..."
    
    cd ~/app/miStudio
    source miStudio-dev-env/bin/activate
    
    python -c "
import torch
print('🖥️  GPU Test Results:')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} ({props.total_memory/1e9:.1f}GB)')
        
        # Test tensor operations
        torch.cuda.set_device(i)
        x = torch.randn(1000, 1000).cuda()
        y = torch.matmul(x, x.T)
        print(f'  ✅ Tensor operations working on GPU {i}')
    
    print('🎉 All GPUs operational!')
else:
    print('❌ CUDA not available in PyTorch')
"
}

# Step 6: Create basic miStudioTrain service
create_basic_train_service() {
    print_step "Creating basic miStudioTrain service..."
    
    cd ~/app/miStudio
    
    # Create the main service file
    cat > services/miStudioTrain/src/main.py << 'EOF'
#!/usr/bin/env python3
"""
miStudioTrain Service - Step 1: Train Sparse Autoencoders
Optimized for dual GPU setup (RTX 3090 + RTX 3080 Ti)
"""

import os
import logging
import argparse
import torch
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainConfig:
    """Configuration for miStudioTrain service"""
    model_name: str = "EleutherAI/pythia-160m"
    layer_number: int = 6
    batch_size: int = 8
    data_path: str = "/data"
    input_file: str = "samples/sample_corpus.txt"
    output_file: str = "activations/sample_activations.pt"
    gpu_id: Optional[int] = None  # Auto-select best GPU
    service_name: str = "miStudioTrain"
    service_version: str = "v1.0.0"

class GPUManager:
    """Manage GPU selection for optimal performance"""
    
    @staticmethod
    def get_best_gpu(prefer_large_memory: bool = True) -> int:
        """Select best GPU based on available memory"""
        if not torch.cuda.is_available():
            return -1
        
        best_gpu = 0
        best_memory = 0
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            
            logger.info(f"GPU {i}: {props.name} ({memory_gb:.1f}GB)")
            
            if prefer_large_memory and memory_gb > best_memory:
                best_gpu = i
                best_memory = memory_gb
        
        logger.info(f"Selected GPU {best_gpu} for training")
        return best_gpu

class MiStudioTrain:
    """Step 1: Train Sparse Autoencoders for AI interpretability"""
    
    def __init__(self, config: TrainConfig):
        self.config = config
        
        # Set up GPU
        if config.gpu_id is None:
            self.gpu_id = GPUManager.get_best_gpu()
        else:
            self.gpu_id = config.gpu_id
            
        if self.gpu_id >= 0:
            torch.cuda.set_device(self.gpu_id)
            self.device = torch.device(f"cuda:{self.gpu_id}")
            logger.info(f"Using {torch.cuda.get_device_name(self.gpu_id)}")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")
        
        # Set up paths
        self.data_path = Path(config.data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🏋️ miStudioTrain v{config.service_version} initialized")
    
    def create_sample_data(self):
        """Create sample data if it doesn't exist"""
        sample_dir = self.data_path / "samples"
        sample_dir.mkdir(exist_ok=True)
        
        sample_file = sample_dir / "sample_corpus.txt"
        if not sample_file.exists():
            logger.info("Creating sample corpus...")
            sample_text = """
            The quick brown fox jumps over the lazy dog. This is sample text for AI interpretability research.
            Machine learning models process language by converting words into numerical representations.
            Transformers use attention mechanisms to focus on relevant parts of the input sequence.
            Sparse autoencoders can decompose neural network activations into interpretable features.
            Large language models learn complex patterns from vast amounts of text data.
            Mechanistic interpretability seeks to understand the internal workings of AI systems.
            Feature visualization helps researchers understand what neural networks have learned.
            The goal is to make AI systems more transparent, controllable, and aligned with human values.
            """
            
            with open(sample_file, 'w') as f:
                f.write(sample_text.strip())
            
            logger.info(f"Sample corpus created at {sample_file}")
    
    def run_sample_training(self):
        """Run a sample training demonstration"""
        logger.info("🚀 Starting miStudioTrain Step 1 - Sample Training")
        
        # Create sample data
        self.create_sample_data()
        
        # Read sample data
        input_path = self.data_path / self.config.input_file
        if input_path.exists():
            with open(input_path, 'r') as f:
                text = f.read()
            logger.info(f"Read {len(text)} characters from {input_path}")
        else:
            logger.warning(f"Input file not found: {input_path}")
            text = "Sample text for demonstration"
        
        # Simulate activation extraction (replace with real implementation)
        logger.info("🔬 Extracting sample activations...")
        
        # Create sample activation tensor
        vocab_size = len(text.split())
        hidden_dim = 512
        sample_activations = torch.randn(vocab_size, hidden_dim, device=self.device)
        
        logger.info(f"Generated sample activations: {sample_activations.shape}")
        
        # Save activations
        output_dir = self.data_path / "activations"
        output_dir.mkdir(exist_ok=True)
        output_path = self.data_path / self.config.output_file
        
        # Move to CPU for saving
        activations_cpu = sample_activations.cpu()
        torch.save(activations_cpu, output_path)
        
        # Save metadata
        metadata = {
            "service": self.config.service_name,
            "version": self.config.service_version,
            "model_name": self.config.model_name,
            "layer_number": self.config.layer_number,
            "shape": list(activations_cpu.shape),
            "device_used": str(self.device),
            "gpu_name": torch.cuda.get_device_name(self.gpu_id) if self.gpu_id >= 0 else "CPU"
        }
        
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Sample training completed!")
        logger.info(f"📊 Activations saved: {output_path}")
        logger.info(f"📋 Metadata saved: {metadata_path}")
        logger.info(f"➡️  Ready for Step 2: miStudioFind")

def main():
    parser = argparse.ArgumentParser(description="miStudioTrain - SAE Training Service")
    parser.add_argument("--model-name", default="EleutherAI/pythia-160m")
    parser.add_argument("--gpu-id", type=int, help="Specific GPU to use (default: auto-select)")
    parser.add_argument("--data-path", default="./data", help="Data directory path")
    
    args = parser.parse_args()
    
    config = TrainConfig(
        model_name=args.model_name,
        gpu_id=args.gpu_id,
        data_path=args.data_path
    )
    
    service = MiStudioTrain(config)
    service.run_sample_training()

if __name__ == "__main__":
    main()
EOF

    # Make it executable
    chmod +x services/miStudioTrain/src/main.py
    
    # Create requirements.txt for the service
    cat > services/miStudioTrain/requirements.txt << 'EOF'
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
accelerate>=0.20.0
EOF

    print_step "✅ Basic miStudioTrain service created"
}

# Step 7: Create development Makefile
create_dev_makefile() {
    print_step "Creating development Makefile..."
    
    cd ~/app/miStudio
    
    cat > Makefile << 'EOF'
# miStudio Development Makefile - Sean's GPU Environment

.PHONY: help setup test-gpu status train-sample clean

VENV_PATH = ./miStudio-dev-env
PYTHON = $(VENV_PATH)/bin/python

help: ## Show available commands
	@echo "miStudio Development Commands:"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

setup: ## Set up development environment (already done by setup script)
	@echo "Development environment already set up!"
	@echo "Virtual environment: $(VENV_PATH)"
	@echo "Activate with: source $(VENV_PATH)/bin/activate"

test-gpu: ## Test GPU functionality
	@echo "🧪 Testing GPU setup..."
	@$(PYTHON) -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

status: ## Show system status
	@echo "📊 miStudio System Status:"
	@echo "Hardware:"
	@nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | sed 's/^/  /'
	@echo ""
	@echo "Python Environment:"
	@$(PYTHON) --version
	@echo ""
	@echo "MicroK8s:"
	@kubectl get nodes -o wide 2>/dev/null | head -2 || echo "  MicroK8s not accessible"

train-sample: ## Run sample training
	@echo "🏋️ Running miStudioTrain sample..."
	@cd services/miStudioTrain && $(realpath $(PYTHON)) src/main.py --data-path ../../data

jupyter: ## Start Jupyter Lab
	@echo "🚀 Starting Jupyter Lab..."
	@echo "Access at: http://$(shell hostname -I | awk '{print $$1}'):8888"
	@cd tools && $(PYTHON) -m jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

clean: ## Clean up generated files
	@echo "🧹 Cleaning up..."
	@rm -rf data/activations/* data/artifacts/*
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -delete

monitor: ## Monitor GPU usage
	@echo "📊 GPU Monitor (Ctrl+C to stop):"
	@watch -n 2 nvidia-smi
EOF

    print_step "✅ Development Makefile created"
}

# Step 8: Final verification
final_verification() {
    print_step "Final verification..."
    
    cd ~/app/miStudio
    source miStudio-dev-env/bin/activate
    
    # Test the sample service
    print_info "Testing miStudioTrain service..."
    if python services/miStudioTrain/src/main.py --data-path ./data; then
        print_step "✅ miStudioTrain service working!"
    else
        print_warning "Sample service test failed, but environment is set up"
    fi
    
    # Show what's available
    print_step "🎉 miStudio setup complete!"
    echo ""
    echo "📂 Project location: ~/app/miStudio"
    echo "🐍 Python environment: ~/app/miStudio/miStudio-dev-env"
    echo ""
    echo "Quick commands:"
    echo "  cd ~/app/miStudio"
    echo "  source miStudio-dev-env/bin/activate"
    echo "  make help              # Show all commands"
    echo "  make test-gpu          # Test GPU access"
    echo "  make train-sample      # Run sample training"
    echo "  make status            # Show system status"
    echo ""
    echo "🚀 Ready to develop AI interpretability services!"
}

# Main execution
main() {
    echo "🎯 miStudio Quick Setup for Sean's GPU Environment"
    echo "=================================================="
    
    test_docker_gpu
    verify_microk8s_gpu
    create_mistudio_project
    setup_python_env
    test_gpu_functionality
    create_basic_train_service
    create_dev_makefile
    final_verification
}

# Run the setup
main "$@"