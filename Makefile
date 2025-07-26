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
