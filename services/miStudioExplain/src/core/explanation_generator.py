"""
Explanation Generator for miStudioExplain Service

Orchestrates the call to the LLM to generate the final explanation.
"""

import logging
from typing import Dict, Any

from ..infrastructure.ollama_manager import OllamaManager
from .input_manager import ExplanationRequest

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """
    Takes a prepared context and generates a natural language explanation
    by interfacing with the OllamaManager.
    """

    def __init__(self, ollama_manager: OllamaManager):
        """
        Initializes the generator with an instance of the OllamaManager.

        Args:
            ollama_manager: An initialized OllamaManager to handle LLM communication.
        """
        self.ollama_manager = ollama_manager
        logger.info("🔧 ExplanationGenerator initialized.")

    async def generate_explanation(
        self,
        request: ExplanationRequest,
        prompt_context: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Generates an explanation by calling the appropriate LLM.

        Args:
            request: The original, validated request object.
            prompt_context: A dictionary containing the 'system_prompt' and 'user_prompt'.

        Returns:
            A dictionary containing the explanation result or an error.
        """
        logger.info(f"Generating explanation for request {request.request_id}...")

        # 1. Check if a model was specified in the request; otherwise, choose the optimal one.
        if request.model:
            model_name = request.model
            logger.info(f"Using model specified in request: {model_name}")
        else:
            model_name = self.ollama_manager.get_optimal_model_for_task(
                task_type=request.analysis_type,
                complexity=request.complexity
            )
            logger.info(f"Selected optimal model: {model_name}")

        # 2. Get the model's specific parameters from the manager's config
        model_config = self.ollama_manager.MODEL_CONFIGS.get(model_name)
        if not model_config:
            error_msg = f"Configuration for model '{model_name}' not found."
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        parameters = model_config.parameters

        # 3. Combine system and user prompts for Ollama
        full_prompt = f"{prompt_context['system_prompt']}\n\n{prompt_context['user_prompt']}"


        # 4. Call the OllamaManager to get the explanation
        result = await self.ollama_manager.generate_explanation(
            model_name=model_name,
            prompt=full_prompt,
            parameters=parameters
        )

        if result.get("success"):
            logger.info(f"✅ Successfully generated explanation for request {request.request_id}.")
            return {
                "success": True,
                "explanation_text": result.get("response"),
                "model_used": result.get("model_used"),
                "token_count": result.get("token_count"),
            }
        else:
            logger.error(f"❌ Failed to generate explanation for request {request.request_id}: {result.get('error')}")
            return {
                "success": False,
                "error": f"LLM generation failed: {result.get('error')}",
            }