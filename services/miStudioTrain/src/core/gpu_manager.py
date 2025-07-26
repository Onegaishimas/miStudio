# =============================================================================
# core/gpu_manager.py - Memory Management Enhanced Version
# =============================================================================

import torch
import logging

logger = logging.getLogger(__name__)


class GPUManager:
    """GPU management with enhanced detection and memory management"""

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
    def clear_gpu_cache():
        """Clear GPU cache and log memory stats"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                try:
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    reserved = torch.cuda.memory_reserved(i) / 1e9
                    logger.info(f"GPU {i} after cache clear - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")
                except Exception as e:
                    logger.warning(f"Could not get memory info for GPU {i}: {e}")

    @staticmethod
    def get_memory_info(device_id: int = 0) -> dict:
        """Get detailed memory information for a GPU"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        try:
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
                "multiprocessor_count": props.multi_processor_count
            }
        except Exception as e:
            return {"error": f"Could not get info for GPU {device_id}: {str(e)}"}

    @staticmethod
    def monitor_memory_usage(device_id: int = None) -> dict:
        """Monitor memory usage across all or specific GPU"""
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
        """Get optimization recommendations for specific models"""
        model_name_lower = model_name.lower()
        
        recommendations = {
            "model_name": model_name,
            "is_large_model": False,
            "recommended_batch_size": 8,
            "recommended_sequence_length": 512,
            "use_quantization": False,
            "use_gradient_checkpointing": False,
            "min_gpu_memory_gb": 4.0
        }
        
        # Large model detection and optimization
        if any(name in model_name_lower for name in ["phi-2", "phi-4", "llama", "mistral", "gpt-3", "gpt-4"]):
            recommendations.update({
                "is_large_model": True,
                "recommended_batch_size": 1,
                "recommended_sequence_length": 256,
                "use_quantization": True,
                "use_gradient_checkpointing": True,
                "min_gpu_memory_gb": 8.0
            })
        elif any(name in model_name_lower for name in ["phi-1", "gpt2", "bert", "distilbert"]):
            recommendations.update({
                "recommended_batch_size": 16,
                "min_gpu_memory_gb": 2.0
            })
        
        return recommendations

    @staticmethod
    def check_memory_for_model(model_name: str, device_id: int = None) -> dict:
        """Check if there's sufficient memory for a specific model"""
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
                "recommendations": reqs
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
                    "recommendations": reqs
                }
            else:
                return {
                    "sufficient": False,
                    "reason": f"No GPU with {min_memory_gb}GB available",
                    "recommendations": reqs
                }
