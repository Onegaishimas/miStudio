#!/usr/bin/env python3
"""Quick test for GPU/CPU functionality"""

import torch
import sys

def test_environment():
    print("🧪 miStudioTrain Environment Test")
    print("=" * 35)
    
    # Basic info
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    
    # CUDA test
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"GPU Count: {gpu_count}")
        
        for i in range(gpu_count):
            try:
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1e9
                print(f"GPU {i}: {props.name} ({memory_gb:.1f}GB)")
                
                # Test GPU computation
                device = torch.device(f"cuda:{i}")
                x = torch.randn(100, 100).to(device)
                y = torch.mm(x, x.t())
                print(f"   ✅ GPU {i} computation successful")
                
            except Exception as e:
                print(f"   ❌ GPU {i} test failed: {e}")
    
    # CPU fallback test
    print("\nTesting CPU fallback:")
    try:
        device = torch.device("cpu")
        x = torch.randn(100, 100).to(device)
        y = torch.mm(x, x.t())
        print("   ✅ CPU computation successful")
        
        # Test simple neural network
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 100)
        )
        
        output = model(x)
        loss = nn.MSELoss()(output, x)
        print(f"   ✅ Neural network test: loss = {loss.item():.4f}")
        
    except Exception as e:
        print(f"   ❌ CPU test failed: {e}")
    
    # Summary
    print("\n" + "=" * 35)
    if cuda_available:
        print("✅ NVIDIA GPU available - high performance mode")
    else:
        print("⚠️  No GPU - CPU fallback mode")
        print("   Service will work but train slower")
    
    print("✅ miStudioTrain ready to run!")

if __name__ == "__main__":
    test_environment()
