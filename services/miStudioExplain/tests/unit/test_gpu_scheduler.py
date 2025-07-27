import pytest
from src.infrastructure.gpu_scheduler import GPUScheduler

@pytest.fixture
def scheduler():
    """Provides a scheduler with a predefined GPU inventory for testing."""
    inventory = {"gpu-0": 12000, "gpu-1": 24000}
    return GPUScheduler(gpu_inventory=inventory)

@pytest.mark.asyncio
async def test_gpu_allocation(scheduler: GPUScheduler):
    """Tests that a GPU can be successfully acquired and released."""
    # Acquire a GPU
    gpu_id = await scheduler.acquire_gpu(model_name="test-model", memory_required_mb=8000)
    assert gpu_id == "gpu-0"

    # Verify status
    status = scheduler.get_status()
    assert status["gpu-0"]["model_using"] == "test-model"
    assert status["gpu-0"]["used_memory_mb"] == 8000

    # Release the GPU
    await scheduler.release_gpu(gpu_id)
    status = scheduler.get_status()
    assert status["gpu-0"]["model_using"] is None
    assert status["gpu-0"]["used_memory_mb"] == 0

@pytest.mark.asyncio
async def test_gpu_allocation_fails_if_no_memory(scheduler: GPUScheduler):
    """Tests that allocation fails if no GPU has enough memory."""
    gpu_id = await scheduler.acquire_gpu(model_name="large-model", memory_required_mb=30000)
    assert gpu_id is None