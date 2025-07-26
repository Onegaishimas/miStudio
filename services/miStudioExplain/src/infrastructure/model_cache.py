"""
Model Cache for miStudioExplain Service

Intelligent model loading and caching system.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class CachedModel:
    """Information about a cached model"""
    model_name: str
    loaded_at: datetime
    last_used: datetime
    status: str  # "loading", "ready", "error"
    usage_count: int = 0


class ModelCache:
    """Manages model loading, caching, and lifecycle"""
    
    def __init__(self, ollama_manager, gpu_scheduler):
        self.ollama_manager = ollama_manager
        self.gpu_scheduler = gpu_scheduler
        self.cached_models: Dict[str, CachedModel] = {}
        self.cache_lock = asyncio.Lock()
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_requests": 0
        }
        
    async def ensure_model_loaded(self, model_name: str) -> bool:
        """Ensure model is loaded and ready for inference"""
        async with self.cache_lock:
            try:
                self.stats["total_requests"] += 1
                
                # Check cache
                if model_name in self.cached_models:
                    cached_model = self.cached_models[model_name]
                    
                    if cached_model.status == "ready":
                        cached_model.last_used = datetime.now()
                        cached_model.usage_count += 1
                        self.stats["cache_hits"] += 1
                        logger.debug(f"✅ Cache hit for {model_name}")
                        return True
                
                # Cache miss - load model
                self.stats["cache_misses"] += 1
                logger.info(f"📥 Loading model: {model_name}")
                
                # Create cache entry
                self.cached_models[model_name] = CachedModel(
                    model_name=model_name,
                    loaded_at=datetime.now(),
                    last_used=datetime.now(),
                    status="loading"
                )
                
                # Ensure model is available
                success = await self.ollama_manager.ensure_model_available(model_name)
                
                if success:
                    self.cached_models[model_name].status = "ready"
                    logger.info(f"✅ Model {model_name} loaded successfully")
                else:
                    self.cached_models[model_name].status = "error"
                    logger.error(f"❌ Failed to load {model_name}")
                
                return success
                
            except Exception as e:
                logger.error(f"❌ Error loading model: {e}")
                if model_name in self.cached_models:
                    self.cached_models[model_name].status = "error"
                return False
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload model from cache"""
        async with self.cache_lock:
            try:
                if model_name in self.cached_models:
                    del self.cached_models[model_name]
                    logger.info(f"🗑️ Unloaded model {model_name}")
                    return True
                return False
            except Exception as e:
                logger.error(f"❌ Error unloading model: {e}")
                return False
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get current cache status"""
        try:
            hit_rate = 0.0
            if self.stats["total_requests"] > 0:
                hit_rate = self.stats["cache_hits"] / self.stats["total_requests"]
            
            return {
                "cached_models": {
                    name: {
                        "status": model.status,
                        "usage_count": model.usage_count,
                        "last_used": model.last_used.isoformat()
                    }
                    for name, model in self.cached_models.items()
                },
                "stats": {
                    "hit_rate": hit_rate,
                    "total_requests": self.stats["total_requests"],
                    "cache_hits": self.stats["cache_hits"],
                    "cache_misses": self.stats["cache_misses"]
                }
            }
        except Exception as e:
            logger.error(f"❌ Error getting cache status: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup cache resources"""
        try:
            async with self.cache_lock:
                self.cached_models.clear()
                self.stats = {"cache_hits": 0, "cache_misses": 0, "total_requests": 0}
                logger.info("✅ Model cache cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

