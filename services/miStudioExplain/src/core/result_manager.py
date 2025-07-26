"""
Result Manager for miStudioExplain Service

Handles the persistence of the final, validated explanation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

from .input_manager import ExplanationRequest

logger = logging.getLogger(__name__)


class ResultManager:
    """
    Saves the final explanation result to a persistent storage location.
    """

    def __init__(self, output_directory: str = "data/output"):
        """
        Initializes the ResultManager.

        Args:
            output_directory: The path to the directory where results will be saved.
        """
        self.output_path = Path(output_directory)
        # Ensure the output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"🔧 ResultManager initialized. Output will be saved to {self.output_path}")

    def save_result(
        self,
        request: ExplanationRequest,
        explanation_result: Dict[str, Any],
        validation_passed: bool
    ) -> str:
        """
        Saves the result of an explanation task to a JSON file.

        Args:
            request: The original request object.
            explanation_result: The dictionary containing the explanation text, model, etc.
            validation_passed: A boolean indicating if quality validation passed.

        Returns:
            The file path of the saved result.
        """
        result_id = request.request_id
        file_path = self.output_path / f"{result_id}_explanation.json"
        logger.info(f"Saving result for request {result_id} to {file_path}...")

        # Structure the final output object
        final_output = {
            "request_id": result_id,
            "analysis_type": request.analysis_type,
            "validation_passed": validation_passed,
            "explanation": explanation_result,
            "original_input": request.input_data.model_dump(),
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=4)
            logger.info(f"✅ Successfully saved result to {file_path}")
            return str(file_path)
        except IOError as e:
            logger.error(f"❌ Failed to write result file for {result_id}: {e}")
            raise  # Re-raise the exception to be handled by the main application loop