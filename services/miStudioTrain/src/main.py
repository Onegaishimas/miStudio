# main.py - Memory Optimized Version with Fixed Multiprocessing Issues

import os
import torch
import aiofiles
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from transformers import AutoTokenizer, AutoConfig

# Set memory optimization environment variables before any PyTorch imports
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Fix multiprocessing issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer parallelism
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)  # Use spawn instead of fork

from utils.logging_config import setup_logging
from models.api_models import TrainingRequest, TrainingStatus, TrainingResult
from core.training_service import MiStudioTrainService
from core.gpu_manager import GPUManager

# Setup logging
setup_logging()

# Initialize service
data_path = os.getenv("DATA_PATH", "/data")
train_service = MiStudioTrainService(data_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger = __import__('logging').getLogger(__name__)
    
    # Clear GPU cache at startup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared GPU cache at startup")
    
    logger.info("miStudioTrain API starting up with memory optimizations...")
    
    if torch.cuda.is_available():
        # Clear any residual GPU memory
        torch.cuda.empty_cache()
        
        # Log initial GPU status
        gpu_status = GPUManager.monitor_memory_usage()
        logger.info(f"Startup GPU status: {len(gpu_status)} GPUs available")
        
        for gpu_id, info in gpu_status.items():
            if "error" not in info:
                logger.info(f"GPU {gpu_id}: {info['name']} - {info['free_gb']:.1f}GB free")
    
    yield
    
    # Shutdown
    logger.info("miStudioTrain API shutting down...")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared GPU cache on shutdown")


# FastAPI app with lifespan
app = FastAPI(
    title="miStudioTrain API - Memory Optimized",
    description="Sparse Autoencoder Training Service with Memory Optimization and Dynamic Model Loading",
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# Add signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger = __import__('logging').getLogger(__name__)
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    
    # Clear GPU cache on forced shutdown
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared GPU cache during signal shutdown")
    
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with memory information"""
    return await train_service.get_health_status()


@app.get("/gpu/status")
async def gpu_status():
    """Get detailed GPU status and memory information"""
    try:
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        gpu_status = GPUManager.monitor_memory_usage()
        return {
            "cuda_available": True,
            "gpu_count": torch.cuda.device_count(),
            "gpus": gpu_status,
            "memory_optimization_active": True,
            "recommendations": "Use /api/v1/check-memory/{model_name} to verify model compatibility"
        }
    except Exception as e:
        return {"error": f"Failed to get GPU status: {str(e)}"}


@app.get("/gpu/clear-cache")
async def clear_gpu_cache():
    """Manually clear GPU cache"""
    try:
        if torch.cuda.is_available():
            GPUManager.clear_gpu_cache()
            return {"status": "success", "message": "GPU cache cleared"}
        else:
            return {"status": "skipped", "message": "CUDA not available"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to clear cache: {str(e)}"}


@app.post("/api/v1/train", response_model=Dict[str, str])
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start SAE training job with memory optimization and dynamic model loading"""
    try:
        # Validate corpus file exists
        corpus_path = train_service.base_data_path / "samples" / request.corpus_file
        if not corpus_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Corpus file not found: {request.corpus_file}. Please upload to /data/samples/",
            )

        # Check memory requirements before starting
        memory_check = GPUManager.check_memory_for_model(request.model_name)
        if not memory_check["sufficient"]:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient GPU memory for {request.model_name}: {memory_check.get('reason', 'Unknown reason')}. "
                       f"Required: {memory_check['recommendations']['min_gpu_memory_gb']}GB"
            )

        # Start training job
        job_id = await train_service.start_training(request)

        # Add training job to background tasks
        background_tasks.add_task(train_service.run_training_job, job_id)

        return {
            "job_id": job_id,
            "status": "queued",
            "message": f"Training job started for {request.model_name}",
            "model_name": request.model_name,
            "requires_token": str(request.huggingface_token is not None),
            "memory_check": "passed",
            "optimizations_applied": f"Applied optimizations for {request.model_name}: {memory_check.get('recommendations', {}).get('is_large_model', False)}",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/train/{job_id}/status", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    """Get training job status with model information"""
    return train_service.get_job_status(job_id)


@app.get("/api/v1/train/{job_id}/result", response_model=TrainingResult)
async def get_training_result(job_id: str):
    """Get training job result with model information"""
    return train_service.get_job_result(job_id)


@app.get("/api/v1/jobs")
async def list_jobs():
    """List all training jobs with model information"""
    return await train_service.list_jobs()


@app.get("/api/v1/check-memory/{model_name}")
async def check_memory_requirements(model_name: str):
    """Check memory requirements and GPU compatibility for a specific model"""
    try:
        memory_check = GPUManager.check_memory_for_model(model_name)
        optimizations = GPUManager.optimize_for_model(model_name)
        
        return {
            "model_name": model_name,
            "memory_check": memory_check,
            "optimizations": optimizations,
            "gpu_status": GPUManager.monitor_memory_usage() if torch.cuda.is_available() else None,
            "recommendation": (
                "Model can be trained with current setup" if memory_check["sufficient"]
                else f"Insufficient memory: {memory_check.get('reason', 'Unknown')}"
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory check failed: {str(e)}")


@app.post("/api/v1/validate-model")
async def validate_model(model_name: str, huggingface_token: Optional[str] = None):
    """Validate if a model can be loaded with memory check"""
    try:
        # First check memory requirements
        memory_check = GPUManager.check_memory_for_model(model_name)
        
        # Try to load just the tokenizer to validate access
        use_auth_token = huggingface_token if huggingface_token else None

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_auth_token=use_auth_token, trust_remote_code=True
        )

        # Try to get model config
        config = AutoConfig.from_pretrained(
            model_name, use_auth_token=use_auth_token, trust_remote_code=True
        )

        return {
            "valid": True,
            "model_name": model_name,
            "architecture": getattr(config, "model_type", "unknown"),
            "hidden_size": getattr(config, "hidden_size", getattr(config, "d_model", "unknown")),
            "num_layers": getattr(config, "num_hidden_layers", getattr(config, "n_layer", "unknown")),
            "vocab_size": getattr(config, "vocab_size", len(tokenizer)),
            "requires_token": huggingface_token is not None,
            "memory_sufficient": memory_check["sufficient"],
            "memory_requirements": memory_check,
            "optimizations": GPUManager.optimize_for_model(model_name),
            "message": "Model validation successful",
        }

    except Exception as e:
        memory_check = GPUManager.check_memory_for_model(model_name)
        return {
            "valid": False,
            "model_name": model_name,
            "error": str(e),
            "memory_sufficient": memory_check["sufficient"],
            "memory_requirements": memory_check,
            "message": "Model validation failed",
            "suggestions": [
                "Check model name spelling",
                "Verify model exists on HuggingFace Hub",
                "Provide HuggingFace token if model requires authentication",
                "Ensure model supports transformers library",
                f"Ensure GPU has at least {memory_check['recommendations']['min_gpu_memory_gb']}GB memory"
            ],
        }


@app.post("/api/v1/upload")
async def upload_corpus_file(file: UploadFile = File(...)):
    """Upload a corpus file for training"""
    try:
        # Validate file type
        if not file.filename.endswith(('.txt', '.csv', '.json')):
            raise HTTPException(
                status_code=400,
                detail="Only .txt, .csv, and .json files are supported"
            )

        # Ensure samples directory exists
        samples_dir = train_service.base_data_path / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        # Save the uploaded file
        file_path = samples_dir / file.filename

        # Use aiofiles for async file writing
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        # Get file stats
        file_size = len(content)
        line_count = content.decode('utf-8').count('\n') + 1

        return {
            "status": "success",
            "message": f"File uploaded successfully",
            "filename": file.filename,
            "file_path": str(file_path),
            "file_size_bytes": file_size,
            "estimated_lines": line_count,
            "ready_for_training": True
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/v1/files")
async def list_corpus_files():
    """List available corpus files"""
    try:
        samples_dir = train_service.base_data_path / "samples"

        if not samples_dir.exists():
            return {"files": [], "message": "No samples directory found"}

        files = []
        for file_path in samples_dir.glob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "extension": file_path.suffix
                })

        return {
            "files": files,
            "total_files": len(files),
            "samples_directory": str(samples_dir)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@app.delete("/api/v1/files/{filename}")
async def delete_corpus_file(filename: str):
    """Delete a corpus file"""
    try:
        file_path = train_service.base_data_path / "samples" / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File {filename} not found")

        file_path.unlink()

        return {
            "status": "success",
            "message": f"File {filename} deleted successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Configure uvicorn for better multiprocessing handling
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,  # miStudioTrain service port
        log_level="info", 
        access_log=True,
        # Disable workers in development to avoid multiprocessing issues
        workers=1,
        # Use lifespan for proper startup/shutdown
        lifespan="on"
    )