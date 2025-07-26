"""
GPU Scheduler for miStudioExplain Service

Manages GPU resource allocation and prevents resource contention.
"""

import asyncio
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class GPUScheduler:
    """
    Manages and schedules access to GPU resources within the cluster.

    This scheduler prevents multiple models from being loaded onto the same GPU
    if it would exceed the GPU's memory capacity.
    """

    def __init__(self, gpu_inventory: Dict[str, int]):
        """
        Initializes the scheduler with a dictionary of available GPUs and their memory.

        Args:
            gpu_inventory: A dictionary where keys are GPU identifiers (e.g., "gpu-0")
                           and values are their total memory in MB (e.g., 12288).
        """
        self._gpu_inventory = gpu_inventory
        self._gpu_status: Dict[str, Dict] = {
            gpu_id: {"total_memory_mb": mem, "used_memory_mb": 0, "model_using": None}
            for gpu_id, mem in gpu_inventory.items()
        }
        self._lock = asyncio.Lock()
        logger.info(f"🔧 GPUScheduler initialized with inventory: {self._gpu_inventory}")

    async def acquire_gpu(self, model_name: str, memory_required_mb: int) -> Optional[str]:
        """
        Acquires an available GPU for a model if resources are sufficient.

        This method is atomic and safe from race conditions.

        Args:
            model_name: The name of the model requesting the GPU.
            memory_required_mb: The amount of GPU memory the model needs.

        Returns:
            The ID of the acquired GPU if successful, otherwise None.
        """
        async with self._lock:
            logger.info(f"Attempting to acquire GPU for {model_name} (requires {memory_required_mb}MB)...")

            # Find a GPU with enough free memory
            for gpu_id, status in self._gpu_status.items():
                if status["model_using"] is None:  # Check if GPU is free
                    if status["total_memory_mb"] >= memory_required_mb:
                        # Allocate the GPU
                        status["used_memory_mb"] = memory_required_mb
                        status["model_using"] = model_name
                        logger.info(f"✅ Acquired {gpu_id} for model {model_name}.")
                        return gpu_id

            logger.warning(f"❌ No available GPU with sufficient memory for {model_name}.")
            return None

    async def release_gpu(self, gpu_id: str):
        """
        Releases a GPU, making it available for other tasks.

        Args:
            gpu_id: The ID of the GPU to release.
        """
        async with self._lock:
            if gpu_id in self._gpu_status:
                model_name = self._gpu_status[gpu_id]["model_using"]
                if model_name:
                    logger.info(f"Releasing {gpu_id} from model {model_name}...")
                    # Reset the status
                    self._gpu_status[gpu_id]["used_memory_mb"] = 0
                    self._gpu_status[gpu_id]["model_using"] = None
                    logger.info(f"✅ {gpu_id} has been released and is now free.")
                else:
                    logger.warning(f"Attempted to release {gpu_id}, but it was already free.")
            else:
                logger.error(f"Attempted to release an unknown GPU: {gpu_id}")

    def get_status(self) -> Dict[str, Dict]:
        """Returns the current status of all managed GPUs."""
        return self._gpu_status