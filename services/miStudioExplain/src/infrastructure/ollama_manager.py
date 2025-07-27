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

    def __init__(self, namespace: str = "mistudio-services", ollama_endpoint: Optional[str] = None):
        """
        Initializes the manager.

        Args:
            namespace: The Kubernetes namespace for service discovery (fallback).
            ollama_endpoint: A specific endpoint URL for the Ollama service.
        """
        self.namespace = namespace
        # Prioritize the provided endpoint, otherwise use the default for in-cluster discovery.
        self.ollama_endpoint = ollama_endpoint or f"http://ollama.{self.namespace}.svc.cluster.local:11434"
        self.service_info = None
        self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))

    async def initialize(self) -> bool:
        """Initialize connection to Ollama service and perform a health check."""
        logger.info(f"🔧 Initializing Ollama connection to endpoint: {self.ollama_endpoint}")
        if await self.health_check():
            logger.info(f"✅ Ollama connection initialized successfully.")
            return True
        else:
            logger.error(f"❌ Ollama health check failed at {self.ollama_endpoint}")
            return False

    async def health_check(self) -> bool:
        """Check Ollama service health and list available models."""
        try:
            # The API endpoint for listing models is /api/tags
            response = await self.http_client.get(f"{self.ollama_endpoint}/api/tags")
            response.raise_for_status()
            data = response.json()
            available_models = [model["name"] for model in data.get("models", [])]

            self.service_info = OllamaServiceInfo(
                endpoint=self.ollama_endpoint,
                status="healthy",
                available_models=available_models,
                health_check_passed=True
            )
            logger.info(f"✅ Ollama health check passed. Available models: {available_models}")
            return True
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f"❌ Ollama health check failed: {e}")
            self.service_info = OllamaServiceInfo(
                endpoint=self.ollama_endpoint,
                status="unhealthy",
                available_models=[],
                health_check_passed=False
            )
            return False

    async def ensure_model_available(self, model_name: str) -> bool:
        """Ensure a model is pulled and ready for use."""
        if self.service_info and model_name in self.service_info.available_models:
            logger.debug(f"✅ Model {model_name} is already available.")
            return True

        logger.info(f"📥 Model {model_name} not found. Pulling from registry...")
        try:
            pull_data = {"name": model_name}
            async with self.http_client.stream("POST", f"{self.ollama_endpoint}/api/pull", json=pull_data, timeout=600.0) as response:
                async for line in response.aiter_lines():
                    pass # Process streaming output for progress if needed
            
            await self.health_check() # Refresh model list after pulling
            if self.service_info and model_name in self.service_info.available_models:
                 logger.info(f"✅ Model {model_name} pulled successfully")
                 return True
            else:
                 logger.error(f"❌ Failed to pull model {model_name}.")
                 return False

        except Exception as e:
            logger.error(f"❌ Error ensuring model availability for {model_name}: {e}")
            return False

    async def generate_explanation(self, model_name: str, prompt: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate an explanation using the specified model."""
        if not await self.ensure_model_available(model_name):
            return {"success": False, "error": f"Model {model_name} not available"}

        try:
            # The system prompt is now part of the main prompt content
            generation_data = {
                "model": model_name,
                "prompt": prompt,
                "options": parameters or {},
                "stream": False
            }

            response = await self.http_client.post(
                f"{self.ollama_endpoint}/api/generate",
                json=generation_data
            )
            response.raise_for_status()
            data = response.json()

            return {
                "success": True,
                "response": data.get("response", "").strip(),
                "model_used": model_name,
                "token_count": data.get("eval_count", 0)
            }
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f"❌ Exception during generation with {model_name}: {e}")
            return {"success": False, "error": str(e)}

    def get_optimal_model_for_task(self, task_type: str, complexity: str) -> str:
        """Select the optimal model based on task requirements."""
        task_type = task_type.lower()
        complexity = complexity.lower()

        if "technical" in task_type or "code" in task_type:
            return "codellama:13b"
        if "complex" in complexity or "behavioral" in task_type:
            return "llama3.1:70b"
        return "llama3.1:8b"

    async def cleanup(self):
        """Clean up resources, like the HTTP client."""
        if self.http_client:
            await self.http_client.aclose()
            logger.info("✅ Ollama manager resources cleaned up.")