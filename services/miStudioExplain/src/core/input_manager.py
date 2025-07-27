"""
Input Manager for miStudioExplain Service

Handles validation, parsing, and structuring of incoming requests.
"""

import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class FindResultInput(BaseModel):
    """Structured input from a miStudioFind result."""
    find_job_id: str = Field(..., description="The unique ID of the miStudioFind job.")
    feature_analysis: Dict[str, Any] = Field(..., description="Dictionary containing feature analysis data.")
    summary_report: str = Field(..., description="A text summary from the find job.")


class RawTextInput(BaseModel):
    """Direct raw text input for explanation."""
    text_corpus: str = Field(..., min_length=100, description="The raw text to be explained.")
    source_description: str = Field("ad-hoc text input", description="Description of the text source.")


class ExplanationRequest(BaseModel):
    """
    The main structured request model for the explanation service.
    This model defines the contract for any client calling the service.
    """
    request_id: str = Field(..., description="A unique identifier for this explanation request.")
    analysis_type: str = Field("complex_behavioral", description="The type of analysis requested (e.g., 'technical_patterns', 'complex_behavioral').")
    complexity: str = Field("medium", description="The complexity of the explanation required ('low', 'medium', 'high').")
    model: Optional[str] = Field(None, description="Optionally specify a model to use (e.g., 'llama3.1:8b').")
    input_data: RawTextInput | FindResultInput = Field(..., description="The data to be explained, which can be raw text or a structured find result.")


class InputManager:
    """Manages the intake and validation of explanation requests."""

    def __init__(self):
        logger.info("🔧 InputManager initialized.")

    def process_request(self, request_data: Dict[str, Any]) -> ExplanationRequest:
        """
        Validates and parses the incoming request data.

        Args:
            request_data: A dictionary containing the raw request data.

        Returns:
            A validated ExplanationRequest object.

        Raises:
            ValueError: If the request data is invalid or fails validation.
        """
        try:
            logger.debug(f"Processing request data: {request_data}")
            validated_request = ExplanationRequest.model_validate(request_data)
            logger.info(f"✅ Request {validated_request.request_id} validated successfully.")
            return validated_request
        except ValidationError as e:
            logger.error(f"❌ Validation failed for request: {e}")
            raise ValueError(f"Invalid request payload: {e}")
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred during request processing: {e}")
            raise ValueError("An unexpected error occurred while processing the request.")

    def get_corpus_from_request(self, request: ExplanationRequest) -> str:
        """
        Extracts the primary text corpus to be analyzed from the request.
        """
        if isinstance(request.input_data, RawTextInput):
            logger.info(f"Extracting corpus from RawTextInput for request {request.request_id}.")
            return request.input_data.text_corpus
        elif isinstance(request.input_data, FindResultInput):
            logger.info(f"Extracting corpus from FindResultInput for request {request.request_id}.")
            corpus = (
                f"Summary Report from Find Job {request.input_data.find_job_id}:\n"
                f"{request.input_data.summary_report}\n\n"
                f"Key Feature Analysis:\n"
                f"{', '.join(request.input_data.feature_analysis.keys())}"
            )
            return corpus
        else:
            logger.error(f"Unsupported input data type for request {request.request_id}.")
            raise TypeError(f"Unsupported input data type: {type(request.input_data)}")