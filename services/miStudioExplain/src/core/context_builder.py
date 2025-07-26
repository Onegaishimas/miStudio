"""
Context Builder for miStudioExplain Service

Constructs the final prompt for the LLM based on prioritized features.
"""

import logging
from typing import List, Dict

from .input_manager import ExplanationRequest

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds a focused and effective prompt for the LLM.
    """

    # System prompts set the persona and overall goal for the AI model.
    _SYSTEM_PROMPTS = {
        "complex_behavioral": (
            "You are an expert systems analyst. Your task is to explain complex behavioral "
            "patterns from the provided text in a clear, concise, and insightful manner."
        ),
        "technical_patterns": (
            "You are a senior software engineer specializing in code analysis. Your task is to "
            "explain the technical patterns, potential bugs, or architectural significance "
            "of the provided code or log data."
        ),
        "default": (
            "You are a helpful AI assistant. Your goal is to provide a clear explanation "
            "for the key topics in the provided text."
        ),
    }

    def __init__(self):
        logger.info("🔧 ContextBuilder initialized.")

    def build_prompt(
        self,
        request: ExplanationRequest,
        prioritized_features: List[str],
        full_corpus: str,
    ) -> Dict[str, str]:
        """
        Constructs the system and user prompts for the LLM.

        Args:
            request: The original validated request.
            prioritized_features: The list of important features to focus on.
            full_corpus: The complete text data for context.

        Returns:
            A dictionary containing the 'system_prompt' and 'user_prompt'.
        """
        logger.info(f"Building prompt for request {request.request_id}...")

        system_prompt = self._SYSTEM_PROMPTS.get(
            request.analysis_type, self._SYSTEM_PROMPTS["default"]
        )

        # Build the user prompt piece by piece
        user_prompt_parts = []
        user_prompt_parts.append(
            "Please provide an explanation based on the following context."
        )

        # Add instructions based on complexity
        if request.complexity == "high":
            user_prompt_parts.append(
                "Provide a detailed, step-by-step analysis. Be thorough and specific."
            )
        else:
            user_prompt_parts.append(
                "Provide a high-level summary that is easy to understand."
            )

        # Add the features to focus on
        if prioritized_features:
            features_str = ", ".join(f"'{feature}'" for feature in prioritized_features)
            user_prompt_parts.append(
                f"\nYour explanation MUST focus on the following key topics: {features_str}."
            )
        else:
            user_prompt_parts.append("\nExplain the main themes of the text.")

        # Add the full text as context
        user_prompt_parts.append("\n--- CONTEXT BEGINS ---")
        user_prompt_parts.append(full_corpus)
        user_prompt_parts.append("--- CONTEXT ENDS ---")

        user_prompt = "\n".join(user_prompt_parts)

        logger.debug(f"Generated System Prompt: {system_prompt}")
        logger.debug(f"Generated User Prompt (first 100 chars): {user_prompt[:100]}...")

        return {"system_prompt": system_prompt, "user_prompt": user_prompt}