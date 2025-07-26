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
