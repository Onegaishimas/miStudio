"""
Main entry point for the miStudioExplain FastAPI service.

This service receives explanation requests, orchestrates the workflow using
the core and infrastructure components, and returns the final result.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .core.context_builder import ContextBuilder
from .core.explanation_generator import ExplanationGenerator
from .core.feature_prioritizer import FeaturePrioritizer
from .core.input_manager import InputManager
from .core.quality_validator import QualityValidator
from .core.result_manager import ResultManager
from .infrastructure.ollama_manager import OllamaManager
from .utils.logging import setup_logging

# --- Application Setup ---

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize all the service components
# Use your custom Ollama endpoint here
ollama_manager = OllamaManager(ollama_endpoint="http://ollama.mcslab.io")
input_manager = InputManager()
feature_prioritizer = FeaturePrioritizer()
context_builder = ContextBuilder()
explanation_generator = ExplanationGenerator(ollama_manager=ollama_manager)
quality_validator = QualityValidator()
result_manager = ResultManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    """
    logger.info("🚀 Service starting up...")
    await ollama_manager.initialize()
    yield  # The application is now running
    logger.info("🛑 Service shutting down...")
    await ollama_manager.cleanup()


# Create the FastAPI app instance with the lifespan manager
app = FastAPI(
    title="miStudioExplain Service",
    description="Generates explanations for complex data patterns.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- API Endpoint ---


@app.post("/explain", response_class=JSONResponse)
async def create_explanation(request_data: dict):
    """
    Main endpoint to receive a request and generate an explanation.
    """
    try:
        # 1. Validate and structure the input request
        validated_request = input_manager.process_request(request_data)
        request_id = validated_request.request_id
        logger.info(f"Processing request: {request_id}")

        # 2. Extract the text corpus to be analyzed
        corpus = input_manager.get_corpus_from_request(validated_request)

        # 3. Identify the most important features to focus on
        features = feature_prioritizer.prioritize_features(validated_request)

        # 4. Build the prompt for the LLM
        prompt = context_builder.build_prompt(
            request=validated_request,
            prioritized_features=features,
            full_corpus=corpus,
        )

        # 5. Generate the explanation from the LLM
        explanation_result = await explanation_generator.generate_explanation(
            request=validated_request, prompt_context=prompt
        )
        if not explanation_result.get("success"):
            raise HTTPException(
                status_code=500, detail=f"LLM failed: {explanation_result.get('error')}"
            )

        # 6. Validate the quality of the generated text
        is_valid, failures = quality_validator.validate(
            explanation_text=explanation_result.get("explanation_text", ""),
            prioritized_features=features,
        )
        if not is_valid:
            logger.warning(f"Validation failed for {request_id}: {failures}")
            # Still save the result, but flag it as failed validation
            explanation_result["validation_failures"] = failures

        # 7. Save the final result
        result_path = result_manager.save_result(
            request=validated_request,
            explanation_result=explanation_result,
            validation_passed=is_valid,
        )

        logger.info(f"✅ Successfully completed request {request_id}")
        return {
            "status": "success",
            "request_id": request_id,
            "validation_passed": is_valid,
            "result_location": result_path,
            "explanation": explanation_result,
        }

    except ValueError as e:
        # Handle validation errors from the InputManager
        logger.error(f"Validation Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle all other unexpected errors
        logger.exception("An unhandled exception occurred.")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")