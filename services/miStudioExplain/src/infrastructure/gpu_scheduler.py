"""
GPU Scheduler for miStudioExplain Service

Dynamic GPU allocation and resource management.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class GPUDevice(Enum):
    """Available GPU devices"""
    RTX_3090 = "cuda:0"
    RTX_3080_TI = "cuda:1"


@dataclass
class GPUStatus:
    """Current GPU status information"""
    device_id: str
    name: str
    total_memory_mb: int
    used_memory_mb: int
    available_memory_mb: int
    utilization_percent: float


@dataclass
class ResourceAllocation:
    """GPU resource allocation for a model"""
    allocation_id: str
    device_id: str
    allocated_memory_mb: int
    model_name: str
    allocation_time: datetime


class GPUScheduler:
    """Manages GPU resource allocation and scheduling"""
    
    GPU_CAPACITIES = {
        "cuda:0": {"total": 24576, "name": "RTX_3090"},
        "cuda:1": {"total": 12288, "name": "RTX_3080_Ti"}
    }
    
    def __init__(self):
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.allocation_lock = asyncio.Lock()
        
    async def get_gpu_status(self, device_id: str) -> Optional[GPUStatus]:
        """Get current GPU status"""
        try:
            # Simplified GPU status (would normally use nvidia-ml-py or nvidia-smi)
            capacity = self.GPU_CAPACITIES.get(device_id, {})
            if not capacity:
                return None
            
            # Calculate used memory from allocations
            used_memory = sum(
                alloc.allocated_memory_mb 
                for alloc in self.allocations.values() 
                if alloc.device_id == device_id
            )
            
            total_memory = capacity["total"]
            available_memory = total_memory - used_memory
            
            return GPUStatus(
                device_id=device_id,
                name=capacity["name"],
                total_memory_mb=total_memory,
                used_memory_mb=used_memory,
                available_memory_mb=available_memory,
                utilization_percent=0.0  # Would get from nvidia-smi
            )
            
        except Exception as e:
            logger.error(f"❌ Error getting GPU status: {e}")
            return None
        
    async def allocate_gpu_for_model(self, model_name: str, required_memory_mb: int) -> Optional[str]:
        """Allocate GPU resources for a model"""
        async with self.allocation_lock:
            try:
                optimal_gpu = self.get_optimal_gpu_for_model(model_name)
                gpu_status = await self.get_gpu_status(optimal_gpu)
                
                if not gpu_status:
                    return None
                
                if gpu_status.available_memory_mb < required_memory_mb:
                    logger.warning(f"⚠️ Insufficient memory on {optimal_gpu}")
                    return None
                
                # Create allocation
                allocation_id = f"{model_name}_{datetime.now().strftime(%Y%m%d_%H%M%S)}"
                allocation = ResourceAllocation(
                    allocation_id=allocation_id,
                    device_id=optimal_gpu,
                    allocated_memory_mb=required_memory_mb,
                    model_name=model_name,
                    allocation_time=datetime.now()
                )
                
                self.allocations[allocation_id] = allocation
                logger.info(f"✅ Allocated {required_memory_mb}MB on {optimal_gpu} for {model_name}")
                return allocation_id
                
            except Exception as e:
                logger.error(f"❌ Error allocating GPU: {e}")
                return None
        
    async def release_gpu_allocation(self, allocation_id: str) -> bool:
        """Release GPU allocation"""
        async with self.allocation_lock:
            try:
                if allocation_id in self.allocations:
                    allocation = self.allocations.pop(allocation_id)
                    logger.info(f"✅ Released allocation {allocation_id}")
                    return True
                return False
            except Exception as e:
                logger.error(f"❌ Error releasing allocation: {e}")
                return False
        
    def get_optimal_gpu_for_model(self, model_name: str) -> str:
        """Determine optimal GPU for a specific model"""
        try:
            from .ollama_manager import OllamaManager
            
            model_config = OllamaManager.MODEL_CONFIGS.get(model_name)
            if model_config:
                if model_config.target_gpu == "RTX_3090":
                    return "cuda:0"
                elif model_config.target_gpu == "RTX_3080_Ti":
                    return "cuda:1"
            
            return "cuda:1"  # Default to RTX 3080 Ti
            
        except Exception as e:
            logger.error(f"❌ Error determining optimal GPU: {e}")
            return "cuda:1"
        
    async def monitor_gpu_utilization(self) -> Dict[str, GPUStatus]:
        """Monitor GPU utilization across all devices"""
        try:
            results = {}
            for device_id in self.GPU_CAPACITIES.keys():
                status = await self.get_gpu_status(device_id)
                if status:
                    results[device_id] = status
            return results
        except Exception as e:
            logger.error(f"❌ Error monitoring GPU utilization: {e}")
            return {}

