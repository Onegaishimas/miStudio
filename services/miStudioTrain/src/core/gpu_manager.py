# =============================================================================
# core/gpu_manager.py - GPU Resource Management with Enhanced Memory Monitoring
# =============================================================================

import torch
import logging
import gc
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class GPUManager:
    """GPU management with enhanced detection, memory management, and monitoring"""

    @staticmethod
    def get_best_gpu(prefer_large_memory: bool = True, min_memory_gb: float = 4.0) -> int:
        """Enhanced GPU selection for large model requirements"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU fallback")
            return -1

        try:
            best_gpu = 0
            best_score = 0

            for i in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / 1e9

                    torch.cuda.set_device(i)
                    memory_allocated = torch.cuda.memory_allocated(i) / 1e9
                    memory_available = memory_gb - memory_allocated

                    logger.info(f"GPU {i}: {props.name} ({memory_gb:.1f}GB total, {memory_available:.1f}GB available)")

                    # Check if GPU has sufficient memory for large models
                    if memory_available < min_memory_gb:
                        logger.warning(f"GPU {i} insufficient for large models ({memory_available:.1f}GB < {min_memory_gb}GB)")
                        continue

                    score = memory_available
                    if prefer_large_memory:
                        score += memory_gb * 0.1

                    if score > best_score:
                        best_gpu = i
                        best_score = score

                except Exception as e:
                    logger.error(f"Error checking GPU {i}: {e}")
                    continue

            if best_score == 0:
                logger.warning(f"No GPU with {min_memory_gb}GB available")
                return -1

            logger.info(f"Selected GPU {best_gpu} for training")
            return best_gpu

        except Exception as e:
            logger.error(f"GPU selection failed: {e}")
            return -1

    @staticmethod
    def clear_gpu_cache(force_gc: bool = True):
        """Enhanced GPU cache clearing with optional garbage collection"""
        if torch.cuda.is_available():
            try:
                # Force Python garbage collection first if requested
                if force_gc:
                    collected = gc.collect()
                    logger.debug(f"Garbage collection freed {collected} objects before GPU cache clear")
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Log memory stats for all GPUs
                for i in range(torch.cuda.device_count()):
                    try:
                        allocated = torch.cuda.memory_allocated(i) / 1e9
                        reserved = torch.cuda.memory_reserved(i) / 1e9
                        logger.info(f"GPU {i} after cache clear - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")
                    except Exception as e:
                        logger.warning(f"Could not get memory info for GPU {i}: {e}")
                
                # Synchronize to ensure all operations complete
                torch.cuda.synchronize()
                logger.debug("GPU cache cleared and synchronized")
                
            except Exception as e:
                logger.error(f"Error during GPU cache clearing: {e}")

    @staticmethod
    def get_memory_info(device_id: int = 0) -> dict:
        """Get detailed memory information for a GPU with error handling"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        try:
            # Ensure device_id is valid
            if device_id >= torch.cuda.device_count():
                return {"error": f"Device {device_id} not available (only {torch.cuda.device_count()} GPUs)"}
            
            props = torch.cuda.get_device_properties(device_id)
            allocated = torch.cuda.memory_allocated(device_id) / 1e9
            reserved = torch.cuda.memory_reserved(device_id) / 1e9
            total = props.total_memory / 1e9
            
            return {
                "device_id": device_id,
                "name": props.name,
                "total_gb": total,
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "free_gb": total - allocated,
                "utilization_pct": (allocated / total) * 100,
                "compute_capability": f"{props.major}.{props.minor}",
                "multiprocessor_count": props.multi_processor_count,
                "memory_fragmentation_ratio": reserved / allocated if allocated > 0 else 0.0,
            }
        except Exception as e:
            return {"error": f"Could not get info for GPU {device_id}: {str(e)}"}

    @staticmethod
    def monitor_memory_usage(device_id: int = None) -> dict:
        """Monitor memory usage across all or specific GPU with enhanced metrics"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        results = {}
        
        if device_id is not None:
            results[device_id] = GPUManager.get_memory_info(device_id)
        else:
            for i in range(torch.cuda.device_count()):
                results[i] = GPUManager.get_memory_info(i)
        
        return results

    @staticmethod
    def optimize_for_model(model_name: str) -> dict:
        """Get optimization recommendations for specific models with enhanced recommendations"""
        model_name_lower = model_name.lower()
        
        recommendations = {
            "model_name": model_name,
            "is_large_model": False,
            "recommended_batch_size": 8,
            "recommended_sequence_length": 512,
            "use_quantization": False,
            "use_gradient_checkpointing": False,
            "min_gpu_memory_gb": 4.0,
            "cleanup_frequency": 50,  # How often to clear cache during training
            "use_mixed_precision": False,
        }
        
        # Large model detection and optimization
        if any(name in model_name_lower for name in ["phi-2", "phi-4", "llama", "mistral", "gpt-3", "gpt-4"]):
            recommendations.update({
                "is_large_model": True,
                "recommended_batch_size": 1,
                "recommended_sequence_length": 256,
                "use_quantization": True,
                "use_gradient_checkpointing": True,
                "min_gpu_memory_gb": 8.0,
                "cleanup_frequency": 20,  # More frequent cleanup for large models
                "use_mixed_precision": True,
            })
        elif any(name in model_name_lower for name in ["phi-1", "gpt2", "bert", "distilbert"]):
            recommendations.update({
                "recommended_batch_size": 16,
                "min_gpu_memory_gb": 2.0,
                "cleanup_frequency": 100,  # Less frequent cleanup for small models
                "use_mixed_precision": True,
            })
        
        return recommendations

    @staticmethod
    def check_memory_for_model(model_name: str, device_id: int = None) -> dict:
        """Enhanced memory compatibility check with detailed recommendations"""
        if not torch.cuda.is_available():
            return {"sufficient": False, "reason": "CUDA not available"}
        
        # Get optimization requirements
        reqs = GPUManager.optimize_for_model(model_name)
        min_memory_gb = reqs["min_gpu_memory_gb"]
        
        # Check specific device or find best device
        if device_id is not None:
            memory_info = GPUManager.get_memory_info(device_id)
            if "error" in memory_info:
                return {"sufficient": False, "reason": memory_info["error"]}
            
            available_gb = memory_info["free_gb"]
            sufficient = available_gb >= min_memory_gb
            
            return {
                "sufficient": sufficient,
                "device_id": device_id,
                "available_gb": available_gb,
                "required_gb": min_memory_gb,
                "memory_gap_gb": max(0, min_memory_gb - available_gb),
                "recommendations": reqs,
                "fragmentation_warning": memory_info.get("memory_fragmentation_ratio", 0) > 1.5,
            }
        else:
            # Check all devices and find best
            best_device = GPUManager.get_best_gpu(min_memory_gb=min_memory_gb)
            if best_device >= 0:
                memory_info = GPUManager.get_memory_info(best_device)
                return {
                    "sufficient": True,
                    "device_id": best_device,
                    "available_gb": memory_info["free_gb"],
                    "required_gb": min_memory_gb,
                    "memory_gap_gb": 0,
                    "recommendations": reqs,
                    "fragmentation_warning": memory_info.get("memory_fragmentation_ratio", 0) > 1.5,
                }
            else:
                return {
                    "sufficient": False,
                    "reason": f"No GPU with {min_memory_gb}GB available",
                    "recommendations": reqs,
                    "suggested_actions": [
                        "Clear GPU cache manually",
                        "Reduce batch size",
                        "Use quantization",
                        "Try CPU training (slower)",
                    ],
                }

    @staticmethod
    def force_memory_cleanup(device_id: Optional[int] = None):
        """Force comprehensive memory cleanup for specific or all GPUs"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available for memory cleanup")
            return
        
        logger.info("Starting forced memory cleanup...")
        
        try:
            # Step 1: Python garbage collection
            collected = gc.collect()
            logger.debug(f"Garbage collection freed {collected} objects")
            
            # Step 2: Clear CUDA cache for specific or all devices
            if device_id is not None:
                torch.cuda.set_device(device_id)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info(f"Forced cleanup completed for GPU {device_id}")
            else:
                # Clear cache for all devices
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("Forced cleanup completed for all GPUs")
            
            # Step 3: Log final memory state
            memory_status = GPUManager.monitor_memory_usage(device_id)
            for gpu_id, info in memory_status.items():
                if "error" not in info:
                    logger.info(f"GPU {gpu_id} post-cleanup: {info['allocated_gb']:.1f}GB allocated, {info['free_gb']:.1f}GB free")
            
        except Exception as e:
            logger.error(f"Error during forced memory cleanup: {e}")

    @staticmethod
    def get_memory_summary() -> Dict[str, Any]:
        """Get comprehensive memory summary across all GPUs"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        summary = {
            "total_gpus": torch.cuda.device_count(),
            "gpus": {},
            "system_totals": {
                "total_memory_gb": 0,
                "allocated_memory_gb": 0,
                "free_memory_gb": 0,
            },
            "recommendations": []
        }
        
        for i in range(torch.cuda.device_count()):
            gpu_info = GPUManager.get_memory_info(i)
            summary["gpus"][i] = gpu_info
            
            if "error" not in gpu_info:
                summary["system_totals"]["total_memory_gb"] += gpu_info["total_gb"]
                summary["system_totals"]["allocated_memory_gb"] += gpu_info["allocated_gb"]
                summary["system_totals"]["free_memory_gb"] += gpu_info["free_gb"]
                
                # Generate recommendations
                if gpu_info["utilization_pct"] > 90:
                    summary["recommendations"].append(f"GPU {i}: High memory usage ({gpu_info['utilization_pct']:.1f}%) - consider cleanup")
                
                fragmentation_ratio = gpu_info.get("memory_fragmentation_ratio", 0)
                if fragmentation_ratio > 2.0:
                    summary["recommendations"].append(f"GPU {i}: High memory fragmentation - consider restart")
        
        return summary

    @staticmethod
    def log_memory_state(job_id: str = None, context: str = ""):
        """Log current memory state for debugging purposes"""
        prefix = f"Job {job_id}: " if job_id else ""
        context_str = f" ({context})" if context else ""
        
        if not torch.cuda.is_available():
            logger.info(f"{prefix}Memory state{context_str}: CUDA not available")
            return
        
        try:
            summary = GPUManager.get_memory_summary()
            logger.info(f"{prefix}Memory state{context_str}:")
            
            for gpu_id, info in summary["gpus"].items():
                if "error" not in info:
                    logger.info(f"  GPU {gpu_id}: {info['allocated_gb']:.1f}GB/{info['total_gb']:.1f}GB ({info['utilization_pct']:.1f}% used)")
                else:
                    logger.warning(f"  GPU {gpu_id}: {info['error']}")
            
            if summary["recommendations"]:
                logger.warning(f"{prefix}Memory recommendations: {'; '.join(summary['recommendations'])}")
                
        except Exception as e:
            logger.error(f"Error logging memory state: {e}")