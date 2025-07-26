"""
Ollama Manager for miStudioExplain Service

Local LLM orchestration and management via Ollama.
"""

import asyncio
import logging
import httpx
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    gpu_memory_mb: int
    target_gpu: str
    use_cases: List[str]
    max_concurrent: int
    parameters: Dict[str, Any]


@dataclass
class OllamaServiceInfo:
    """Information about Ollama service in cluster"""
    endpoint: str
    status: str
    available_models: List[str]
    health_check_passed: bool


class OllamaManager:
    """Manages Ollama service integration and model orchestration"""
    
    MODEL_CONFIGS = {
        "llama3.1:8b": ModelConfig(
            name="llama3.1:8b",
            gpu_memory_mb=8192,
            target_gpu="RTX_3080_Ti", 
            use_cases=["simple_patterns", "quick_explanations"],
            max_concurrent=2,
            parameters={"temperature": 0.1, "top_p": 0.9, "max_tokens": 200}
        ),
        "llama3.1:70b": ModelConfig(
            name="llama3.1:70b", 
            gpu_memory_mb=20480,
            target_gpu="RTX_3090",
            use_cases=["complex_behavioral", "detailed_analysis"],
            max_concurrent=1,
            parameters={"temperature": 0.1, "top_p": 0.9, "max_tokens": 300}
        ),
        "codellama:13b": ModelConfig(
            name="codellama:13b",
            gpu_memory_mb=12288,
            target_gpu="RTX_3080_Ti", 
            use_cases=["technical_patterns", "code_analysis"],
            max_concurrent=1,
            parameters={"temperature": 0.0, "top_p": 0.95, "max_tokens": 250}
        )
    }
    
    def __init__(self, namespace: str = "mistudio-services"):
        self.namespace = namespace
        self.ollama_endpoint = None
        self.service_info = None
        self.http_client = None
        
    async def initialize(self) -> bool:
        """Initialize connection to Ollama service"""
        try:
            logger.info("🔧 Initializing Ollama connection...")
            
            # Simple endpoint discovery
            self.ollama_endpoint = f"http://ollama.{self.namespace}.svc.cluster.local:11434"
            
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
            
            # Test connection
            health_check = await self.health_check()
            
            if health_check:
                logger.info("✅ Ollama connection initialized successfully")
                return True
            else:
                logger.error("❌ Ollama health check failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ollama connection: {e}")
            return False
        
    async def health_check(self) -> bool:
        """Check Ollama service health"""
        try:
            response = await self.http_client.get(f"{self.ollama_endpoint}/api/tags")
            
            if response.status_code == 200:
                data = response.json()
                available_models = [model["name"] for model in data.get("models", [])]
                
                self.service_info = OllamaServiceInfo(
                    endpoint=self.ollama_endpoint,
                    status="healthy",
                    available_models=available_models,
                    health_check_passed=True
                )
                
                logger.info(f"✅ Ollama health check passed. Models: {len(available_models)}")
                return True
            else:
                logger.warning(f"⚠️ Ollama health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ollama health check failed: {e}")
            return False
        
    async def ensure_model_available(self, model_name: str) -> bool:
        """Ensure model is pulled and ready"""
        try:
            # Check if model is already available
            if (self.service_info and 
                model_name in self.service_info.available_models):
                logger.debug(f"✅ Model {model_name} already available")
                return True
            
            # Pull model if not available
            logger.info(f"📥 Pulling model: {model_name}")
            
            pull_data = {"name": model_name}
            response = await self.http_client.post(
                f"{self.ollama_endpoint}/api/pull",
                json=pull_data,
                timeout=httpx.Timeout(600.0)
            )
            
            if response.status_code == 200:
                await self.health_check()  # Refresh model list
                logger.info(f"✅ Model {model_name} pulled successfully")
                return True
            else:
                logger.error(f"❌ Failed to pull model {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error ensuring model availability: {e}")
            return False
        
    async def generate_explanation(self, model_name: str, prompt: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate explanation using specified model"""
        try:
            if not await self.ensure_model_available(model_name):
                return {
                    "success": False,
                    "error": f"Model {model_name} not available"
                }
            
            # Prepare generation request
            generation_data = {
                "model": model_name,
                "prompt": prompt,
                "options": parameters or {},
                "stream": False
            }
            
            # Send generation request
            response = await self.http_client.post(
                f"{self.ollama_endpoint}/api/generate",
                json=generation_data,
                timeout=httpx.Timeout(300.0)
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "response": data.get("response", ""),
                    "model_used": model_name,
                    "token_count": data.get("eval_count", 0)
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            logger.error(f"❌ Exception during generation: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        
    def get_optimal_model_for_task(self, task_type: str, complexity: str) -> str:
        """Select optimal model based on task requirements"""
        try:
            if task_type == "technical" or "code" in task_type.lower():
                return "codellama:13b"
            elif complexity == "high" or "complex" in complexity.lower():
                return "llama3.1:70b"
            else:
                return "llama3.1:8b"
                
        except Exception as e:
            logger.error(f"❌ Error selecting optimal model: {e}")
            return "llama3.1:8b"
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.http_client:
                await self.http_client.aclose()
            logger.info("✅ Ollama manager cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

