# core/training_service.py - Memory Optimized Version with Enhanced GPU Cleanup

import torch
import torch.nn.functional as F
import numpy as np
import logging
import time
import json
import math
import os
import gc
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import Dict, Any

from fastapi import HTTPException

from config.settings import TrainConfig
from models.api_models import TrainingRequest, TrainingStatus, TrainingResult, ModelInfo
from models.sae import SparseAutoencoder
from core.gpu_manager import GPUManager
from core.activation_extractor import EnhancedActivationExtractor

logger = logging.getLogger(__name__)


def safe_float_for_json(value):
    """Convert float values to JSON-safe values"""
    if value is None:
        return None
    if math.isinf(value) or math.isnan(value):
        return 0.0
    return float(value)


class GPUMemoryManager:
    """Context manager for comprehensive GPU memory management"""
    
    def __init__(self, device, job_id: str):
        self.device = device
        self.job_id = job_id
        self.initial_memory = None
        self.objects_to_cleanup = []
    
    def __enter__(self):
        if self.device.type == "cuda":
            try:
                device_id = self.device.index if hasattr(self.device, 'index') else 0
                self.initial_memory = GPUManager.get_memory_info(device_id)
                logger.info(f"Job {self.job_id}: Initial GPU memory: {self.initial_memory.get('allocated_gb', 0):.1f}GB allocated")
            except Exception as e:
                logger.warning(f"Could not get initial GPU memory info: {e}")
        return self
    
    def register_for_cleanup(self, *objects):
        """Register objects for automatic cleanup"""
        self.objects_to_cleanup.extend(objects)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Comprehensive cleanup on context exit"""
        if self.device.type == "cuda":
            self._perform_comprehensive_cleanup()
    
    def _perform_comprehensive_cleanup(self):
        """Perform thorough GPU memory cleanup"""
        logger.info(f"Job {self.job_id}: Starting comprehensive GPU memory cleanup...")
        
        # Step 1: Clean up registered objects
        for i, obj in enumerate(self.objects_to_cleanup):
            try:
                if obj is not None:
                    if hasattr(obj, 'cleanup_gpu_resources'):
                        obj.cleanup_gpu_resources()
                    elif hasattr(obj, 'cleanup'):
                        obj.cleanup()
                    elif hasattr(obj, 'cpu'):
                        obj.cpu()
                    del obj
            except Exception as e:
                logger.warning(f"Error cleaning up object {i}: {e}")
        
        # Step 2: Clear the cleanup list
        self.objects_to_cleanup.clear()
        
        # Step 3: Force Python garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collector freed {collected} objects")
        
        # Step 4: Clear PyTorch CUDA cache
        torch.cuda.empty_cache()
        
        # Step 5: Synchronize to ensure all operations complete
        torch.cuda.synchronize()
        
        # Step 6: Log final memory state
        try:
            device_id = self.device.index if hasattr(self.device, 'index') else 0
            final_memory = GPUManager.get_memory_info(device_id)
            
            if self.initial_memory and 'allocated_gb' in final_memory:
                memory_freed = self.initial_memory.get('allocated_gb', 0) - final_memory.get('allocated_gb', 0)
                logger.info(f"Job {self.job_id}: Released {memory_freed:.1f}GB GPU memory")
                logger.info(f"Job {self.job_id}: Final GPU memory: {final_memory['allocated_gb']:.1f}GB allocated, {final_memory.get('free_gb', 0):.1f}GB free")
            else:
                logger.info(f"Job {self.job_id}: GPU memory cleanup completed")
                
        except Exception as e:
            logger.warning(f"Could not get final GPU memory info: {e}")


class MiStudioTrainService:
    """Enhanced training service with comprehensive memory management"""

    def __init__(self, base_data_path: str = "/data"):
        self.base_data_path = Path(base_data_path)
        self.base_data_path.mkdir(parents=True, exist_ok=True)

        # Training job tracking
        self.active_jobs: Dict[str, Dict[str, Any]] = {}

        # Set memory optimization environment variables
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        logger.info("🏋️ miStudioTrain API Service v1.2.0 initialized (Enhanced Memory Management)")

    def generate_job_id(self) -> str:
        """Generate unique job ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"train_{timestamp}_{torch.randint(1000, 9999, (1,)).item()}"

    async def start_training(self, request: TrainingRequest) -> str:
        """Start training job with dynamic model loading"""
        job_id = self.generate_job_id()

        # Create job tracking entry
        self.active_jobs[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "current_epoch": 0,
            "current_loss": 0.0,
            "start_time": time.time(),
            "estimated_time_remaining": None,
            "message": f"Job queued for training with {request.model_name}",
            "config": request.model_dump(),
            "result": None,
            "model_info": None,
        }

        logger.info(f"Created training job {job_id} for model {request.model_name}")
        return job_id

    def get_job_status(self, job_id: str) -> TrainingStatus:
        """Get current job status with safe float conversion"""
        if job_id not in self.active_jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        job_info = self.active_jobs[job_id]

        # Convert model_info if present
        model_info = None
        if job_info.get("model_info"):
            model_info = ModelInfo(**job_info["model_info"])

        return TrainingStatus(
            job_id=job_id,
            status=job_info["status"],
            progress=job_info["progress"],
            current_epoch=job_info["current_epoch"],
            current_loss=safe_float_for_json(job_info["current_loss"]),
            estimated_time_remaining=job_info["estimated_time_remaining"],
            message=job_info["message"],
            model_info=model_info,
        )

    def get_job_result(self, job_id: str) -> TrainingResult:
        """Get job result (only for completed jobs)"""
        if job_id not in self.active_jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        job_info = self.active_jobs[job_id]
        if job_info["status"] != "completed":
            raise HTTPException(status_code=400, detail=f"Job {job_id} not completed yet")

        return job_info["result"]

    async def run_training_job(self, job_id: str):
        """Execute the training job with enhanced memory management"""
        job_info = self.active_jobs[job_id]
        
        # Initialize variables for cleanup
        sae = None
        optimizer = None
        extractor = None
        dataloader = None
        device = None

        try:
            job_info["status"] = "running"
            job_info["message"] = f"Initializing training with {job_info['config']['model_name']}..."

            # Convert request to config
            config = TrainConfig(**job_info["config"])

            # Check memory requirements for the model
            memory_check = GPUManager.check_memory_for_model(config.model_name)
            if not memory_check["sufficient"]:
                raise RuntimeError(f"Insufficient GPU memory: {memory_check.get('reason', 'Unknown reason')}")

            # GPU setup with memory optimization
            gpu_id = config.gpu_id
            if gpu_id is None:
                # Use the recommended GPU from memory check
                gpu_id = memory_check.get("device_id", -1)

            if gpu_id >= 0:
                device = torch.device(f"cuda:{gpu_id}")
                torch.cuda.set_device(gpu_id)
                # Clear any residual memory
                torch.cuda.empty_cache()
                logger.info(f"Cleared GPU cache before training")
            else:
                device = torch.device("cpu")
                logger.warning("Using CPU - this will be very slow for large models")

            logger.info(f"Job {job_id}: Using device {device}")

            # Use GPU memory manager for comprehensive cleanup
            with GPUMemoryManager(device, job_id) as memory_manager:
                
                # Get model-specific optimizations
                model_opts = GPUManager.optimize_for_model(config.model_name)
                
                # Apply model-specific optimizations to config
                if model_opts["is_large_model"]:
                    config.batch_size = min(config.batch_size, model_opts["recommended_batch_size"])
                    config.max_sequence_length = min(config.max_sequence_length, model_opts["recommended_sequence_length"])
                    logger.info(f"Applied optimizations for large model: batch_size={config.batch_size}, seq_len={config.max_sequence_length}")

                # Load corpus
                corpus_path = self.base_data_path / "samples" / config.corpus_file
                if not corpus_path.exists():
                    raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

                with open(corpus_path, "r", encoding="utf-8") as f:
                    corpus_text = f.read()

                # Process text
                texts = [chunk.strip() for chunk in corpus_text.split("\n") if chunk.strip()]
                if len(texts) < 10:
                    import re
                    texts = [s.strip() for s in re.split(r"[.!?]+", corpus_text) if s.strip()]

                logger.info(f"Job {job_id}: Processing {len(texts)} text chunks")

                # Extract activations with dynamic model loading
                job_info["message"] = f"Loading {config.model_name} and extracting activations..."
                job_info["progress"] = 0.1

                extractor = EnhancedActivationExtractor(
                    model_name=config.model_name,
                    layer_number=config.layer_number,
                    device=device,
                    huggingface_token=config.huggingface_token,
                    max_sequence_length=config.max_sequence_length,
                )

                # Register extractor for cleanup
                memory_manager.register_for_cleanup(extractor)

                # Store model info
                model_info = extractor.get_model_info()
                job_info["model_info"] = model_info.model_dump()

                # Memory monitoring before activation extraction
                if device.type == "cuda":
                    mem_info = GPUManager.get_memory_info(gpu_id)
                    logger.info(f"GPU memory before activation extraction: {mem_info.get('allocated_gb', 0):.1f}GB allocated")

                activations = extractor.extract_activations(texts, batch_size=config.batch_size)
                
                # Clean up extractor immediately after use
                extractor.cleanup()

                # Clear GPU memory after extraction
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    mem_info = GPUManager.get_memory_info(gpu_id)
                    logger.info(f"GPU memory after activation extraction: {mem_info.get('allocated_gb', 0):.1f}GB allocated")

                logger.info(f"Job {job_id}: Extracted activations shape: {activations.shape}")

                # Initialize SAE
                input_dim = activations.shape[1]
                hidden_dim = config.hidden_dim

                job_info["message"] = f"Training SAE ({input_dim} -> {hidden_dim})..."
                job_info["progress"] = 0.2

                sae = SparseAutoencoder(input_dim, hidden_dim, config.sparsity_coeff)
                # Ensure dtype consistency - convert activations to float32 for training
                activations = activations.float()
                sae = sae.to(device)
                optimizer = torch.optim.Adam(sae.parameters(), lr=config.learning_rate)

                # Register SAE and optimizer for cleanup
                memory_manager.register_for_cleanup(sae, optimizer)

                # Initialize mixed precision scaler for large models
                scaler = None
                if device.type == "cuda" and model_opts["is_large_model"]:
                    scaler = torch.cuda.amp.GradScaler()
                    logger.info("Enabled mixed precision training for memory efficiency")

                # Prepare data with memory-efficient dataloader
                dataset = torch.utils.data.TensorDataset(activations)
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=0,  # Set to 0 to avoid multiprocessing issues with CUDA
                    pin_memory=False,  # Disable pin_memory to save memory
                )

                # Register dataloader for cleanup
                memory_manager.register_for_cleanup(dataloader)

                # Training loop with memory management
                training_stats = {
                    "losses": [],
                    "sparsity_levels": [],
                    "reconstruction_errors": [],
                }

                sae.train()
                best_loss = float("inf")
                epochs_without_improvement = 0

                for epoch in range(config.max_epochs):
                    epoch_losses = []
                    epoch_sparsity = []
                    epoch_recon_errors = []

                    # Clear cache at the start of each epoch
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    for batch_idx, (batch_data,) in enumerate(dataloader):
                        try:
                            batch_data = batch_data.to(device)

                            optimizer.zero_grad()
                            
                            # Use mixed precision for large models
                            if scaler is not None:
                                with torch.cuda.amp.autocast():
                                    reconstruction, hidden, total_loss = sae(batch_data)
                                    recon_error = F.mse_loss(reconstruction, batch_data)
                                    sparsity_level = torch.mean(torch.sum(hidden > 0, dim=1).float()) / hidden_dim

                                scaler.scale(total_loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                reconstruction, hidden, total_loss = sae(batch_data)
                                recon_error = F.mse_loss(reconstruction, batch_data)
                                sparsity_level = torch.mean(torch.sum(hidden > 0, dim=1).float()) / hidden_dim

                                total_loss.backward()
                                optimizer.step()

                            epoch_losses.append(total_loss.item())
                            epoch_sparsity.append(sparsity_level.item())
                            epoch_recon_errors.append(recon_error.item())

                            # Clear intermediate tensors
                            del batch_data, reconstruction, hidden
                            
                            # Periodic memory cleanup during training
                            if batch_idx % 50 == 0 and device.type == "cuda":
                                torch.cuda.empty_cache()
                            
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                logger.error(f"CUDA OOM in epoch {epoch}, batch {batch_idx}. Clearing cache and continuing...")
                                torch.cuda.empty_cache()
                                continue
                            else:
                                raise e

                    # Skip epoch if no successful batches
                    if not epoch_losses:
                        logger.warning(f"No successful batches in epoch {epoch}, skipping...")
                        continue

                    # Epoch statistics
                    avg_loss = np.mean(epoch_losses)
                    avg_sparsity = np.mean(epoch_sparsity)
                    avg_recon_error = np.mean(epoch_recon_errors)

                    training_stats["losses"].append(avg_loss)
                    training_stats["sparsity_levels"].append(avg_sparsity)
                    training_stats["reconstruction_errors"].append(avg_recon_error)

                    # Update job progress
                    progress = 0.2 + 0.7 * (epoch + 1) / config.max_epochs
                    job_info["progress"] = progress
                    job_info["current_epoch"] = epoch + 1
                    job_info["current_loss"] = avg_loss
                    job_info["message"] = f"Epoch {epoch+1}/{config.max_epochs} - Loss: {avg_loss:.4f}, Sparsity: {avg_sparsity:.3f}"

                    # Estimate time remaining
                    elapsed_time = time.time() - job_info["start_time"]
                    if epoch > 0:
                        time_per_epoch = elapsed_time / (epoch + 1)
                        remaining_epochs = config.max_epochs - (epoch + 1)
                        job_info["estimated_time_remaining"] = int(time_per_epoch * remaining_epochs)

                    logger.info(f"Job {job_id}: {job_info['message']}")

                    # Memory monitoring every 5 epochs
                    if device.type == "cuda" and epoch % 5 == 0:
                        mem_info = GPUManager.get_memory_info(gpu_id)
                        logger.info(f"Epoch {epoch}: GPU memory utilization: {mem_info.get('utilization_pct', 0):.1f}%")

                    # Early stopping
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                    if avg_loss < config.min_loss or epochs_without_improvement >= 10:
                        logger.info(f"Job {job_id}: Early stopping at epoch {epoch+1}")
                        break

                # Save results
                job_info["message"] = "Saving trained model and outputs..."
                job_info["progress"] = 0.9

                # Create output directories
                output_dir = self.base_data_path / "models" / job_id
                output_dir.mkdir(parents=True, exist_ok=True)

                activations_dir = self.base_data_path / "activations" / job_id
                activations_dir.mkdir(parents=True, exist_ok=True)

                # Save SAE model
                model_path = output_dir / "sae_model.pt"
                torch.save({
                    "model_state_dict": sae.state_dict(),
                    "config": asdict(config),
                    "training_stats": training_stats,
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                    "sparsity_coeff": config.sparsity_coeff,
                    "model_info": model_info.model_dump(),
                    "memory_optimizations": model_opts,
                }, model_path)

                # Generate feature activations for miStudioFind with memory management
                sae.eval()
                feature_activations = []
                original_activations = []

                # Process in smaller batches for memory efficiency
                eval_batch_size = min(config.batch_size, 4)
                
                with torch.no_grad():
                    for i in range(0, len(texts), eval_batch_size):
                        try:
                            batch_texts = texts[i : i + eval_batch_size]
                            batch_activations = activations[i : i + len(batch_texts)].to(device)

                            if scaler is not None:
                                with torch.cuda.amp.autocast():
                                    features = sae.encode(batch_activations)
                            else:
                                features = sae.encode(batch_activations)
                            
                            feature_activations.append(features.cpu())
                            original_activations.append(batch_activations.cpu())
                            
                            # Clear GPU tensors
                            del batch_activations, features
                            if device.type == "cuda" and i % 100 == 0:
                                torch.cuda.empty_cache()
                                
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                logger.warning(f"OOM during feature generation at batch {i}, skipping...")
                                torch.cuda.empty_cache()
                                continue
                            else:
                                raise e

                if feature_activations:
                    feature_activations = torch.cat(feature_activations, dim=0)
                    original_activations = torch.cat(original_activations, dim=0)
                else:
                    raise RuntimeError("Failed to generate feature activations due to memory constraints")

                # Save activations for miStudioFind
                activations_path = activations_dir / "feature_activations.pt"
                torch.save({
                    "feature_activations": feature_activations,
                    "original_activations": original_activations,
                    "texts": texts,
                    "feature_count": hidden_dim,
                    "activation_dim": input_dim,
                    "model_info": model_info.model_dump(),
                }, activations_path)

                # Compute feature statistics with memory management
                stats_dataloader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(original_activations),
                    batch_size=eval_batch_size,
                    shuffle=False,
                    num_workers=0,
                )
                feature_stats = sae.get_feature_stats(stats_dataloader)

                # Save comprehensive metadata for miStudioFind
                metadata = {
                    "job_id": job_id,
                    "service": "miStudioTrain",
                    "version": config.service_version,
                    "timestamp": datetime.now().isoformat(),
                    "model_info": model_info.model_dump(),
                    "memory_optimizations_applied": model_opts,
                    "sae_config": {
                        "input_dim": input_dim,
                        "hidden_dim": hidden_dim,
                        "sparsity_coeff": config.sparsity_coeff,
                        "learning_rate": config.learning_rate,
                    },
                    "training_results": {
                        "final_loss": float(best_loss),
                        "epochs_trained": len(training_stats["losses"]),
                        "final_sparsity": (
                            float(training_stats["sparsity_levels"][-1])
                            if training_stats["sparsity_levels"]
                            else 0.0
                        ),
                    },
                    "feature_statistics": feature_stats,
                    "data_info": {
                        "text_count": len(texts),
                        "activation_shape": list(original_activations.shape),
                        "feature_activation_shape": list(feature_activations.shape),
                    },
                    "paths": {
                        "model_path": str(model_path),
                        "activations_path": str(activations_path),
                        "metadata_path": str(activations_dir / "metadata.json"),
                    },
                    "ready_for_find_service": True,
                }

                metadata_path = activations_dir / "metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                # Complete job
                job_info["status"] = "completed"
                job_info["progress"] = 1.0
                job_info["message"] = f"Training completed successfully with {config.model_name}"
                job_info["result"] = TrainingResult(
                    job_id=job_id,
                    status="completed",
                    model_path=str(model_path),
                    activations_path=str(activations_path),
                    metadata_path=str(metadata_path),
                    training_stats=training_stats,
                    feature_count=hidden_dim,
                    model_info=model_info,
                    ready_for_find_service=True,
                )

                logger.info(f"Job {job_id}: Training completed successfully")

        except Exception as e:
            logger.error(f"Job {job_id}: Training failed: {str(e)}")
            job_info["status"] = "failed"
            job_info["message"] = f"Training failed: {str(e)}"
            raise

        finally:
            # Additional explicit cleanup for any remaining references
            try:
                # Clean up local variables that might not be in memory manager
                if 'texts' in locals():
                    del texts
                if 'activations' in locals():
                    del activations
                if 'dataset' in locals():
                    del dataset
                if 'feature_activations' in locals():
                    del feature_activations
                if 'original_activations' in locals():
                    del original_activations
                if 'scaler' in locals():
                    del scaler
                
                # Final garbage collection
                gc.collect()
                
                # Final GPU cleanup if device exists
                if device is not None and device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                logger.info(f"Job {job_id}: Final cleanup completed")
                
            except Exception as cleanup_error:
                logger.error(f"Job {job_id}: Error during final cleanup: {cleanup_error}")

    async def list_jobs(self):
        """List all training jobs with model information"""
        jobs = []
        for job_id, job_info in self.active_jobs.items():
            model_info = job_info.get("model_info", {})
            jobs.append({
                "job_id": job_id,
                "status": job_info["status"],
                "progress": job_info["progress"],
                "message": job_info["message"],
                "model_name": job_info["config"].get("model_name", "unknown"),
                "model_info": model_info,
                "start_time": job_info["start_time"],
                "duration_seconds": time.time() - job_info["start_time"],
            })

        return {"jobs": jobs}

    async def get_health_status(self):
        """Enhanced health check endpoint with memory information"""
        gpu_info = []
        cuda_status = "unavailable"

        try:
            if torch.cuda.is_available():
                cuda_status = "available"
                for i in range(torch.cuda.device_count()):
                    try:
                        memory_info = GPUManager.get_memory_info(i)
                        if "error" not in memory_info:
                            gpu_info.append({
                                "id": i,
                                "name": memory_info["name"],
                                "memory_total_gb": memory_info["total_gb"],
                                "memory_available_gb": memory_info["free_gb"],
                                "memory_used_gb": memory_info["allocated_gb"],
                                "utilization_pct": memory_info["utilization_pct"],
                                "suitable_for_phi4": memory_info["free_gb"] > 8.0,
                                "available": memory_info["free_gb"] > 4.0,
                                "compute_capability": memory_info.get("compute_capability", "unknown"),
                            })
                        else:
                            gpu_info.append({
                                "id": i,
                                "name": "Unknown",
                                "error": memory_info["error"],
                                "available": False,
                                "suitable_for_phi4": False,
                            })
                    except Exception as e:
                        gpu_info.append({
                            "id": i,
                            "name": "Unknown",
                            "error": str(e),
                            "available": False,
                            "suitable_for_phi4": False,
                        })
        except Exception as e:
            cuda_status = f"error: {e}"

        return {
            "status": "healthy",
            "service": "miStudioTrain",
            "version": "1.2.0",
            "features": [
                "dynamic_model_loading",
                "huggingface_token_support",
                "enhanced_memory_optimization",
                "comprehensive_gpu_cleanup",
                "mixed_precision_training",
                "automatic_model_optimization",
                "gpu_auto_selection",
            ],
            "cuda_status": cuda_status,
            "gpu_count": len(gpu_info),
            "gpus": gpu_info,
            "active_jobs": len(self.active_jobs),
            "memory_optimizations": {
                "quantization_support": True,
                "gradient_checkpointing": True,
                "mixed_precision": True,
                "automatic_batch_sizing": True,
                "comprehensive_cleanup": True,
                "memory_monitoring": True,
            },
            "supported_models": [
                "microsoft/Phi-4",
                "microsoft/Phi-2",
                "EleutherAI/pythia-*",
                "gpt2",
                "microsoft/DialoGPT-*",
                "meta-llama/Llama-*",
                "mistralai/Mistral-*",
                "Any HuggingFace transformer model",
            ],
            "timestamp": datetime.now().isoformat(),
        }