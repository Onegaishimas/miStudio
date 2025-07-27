# src/main.py
"""
FastAPI application entry point for the miStudioScore service.
"""
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from src.models.api_models import ScoreRequest, ScoreResponse, ErrorResponse
from src.utils.logging_config import setup_logging
from src.core.orchestrator import ScoringOrchestrator

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="miStudioScore API",
    description="A service for scoring features from a trained SAE.",
    version="1.0.0"
)

def run_scoring_task(request: ScoreRequest):
    """
    The background task function that runs the orchestrator.
    This function is designed to be run by FastAPI's BackgroundTasks.
    """
    try:
        logger.info(f"Starting background scoring task for features at {request.features_path}")
        orchestrator = ScoringOrchestrator(config_path=request.config_path)
        orchestrator.run(
            features_path=request.features_path,
            output_dir=request.output_dir
        )
        logger.info(f"Background scoring task for {request.features_path} completed.")
    except Exception as e:
        # In a real application, you'd have more robust error handling,
        # perhaps updating a database with the job status.
        logger.error(f"Background scoring task failed: {e}", exc_info=True)


@app.post("/score",
          response_model=ScoreResponse,
          responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def score_features(request: ScoreRequest, background_tasks: BackgroundTasks):
    """
    Accepts a scoring request and initiates the process as a background task.
    
    This endpoint immediately returns a confirmation and runs the potentially
    long-running scoring process in the background to avoid timeouts.
    """
    logger.info(f"Received scoring request for features: {request.features_path}")
    
    # For now, we are returning a mock response.
    # The actual result path would be determined by the background task.
    # A more advanced system would use a task queue (like Celery) and a
    # separate endpoint to check job status.
    
    # Add the long-running job to the background
    # background_tasks.add_task(run_scoring_task, request)

    # For this iteration, we will run it synchronously to provide immediate feedback.
    # In a production system, the async approach with background tasks is preferred.
    try:
        orchestrator = ScoringOrchestrator(config_path=request.config_path)
        output_path, added_scores = orchestrator.run(
            features_path=request.features_path,
            output_dir=request.output_dir
        )
        
        if output_path is None:
             raise HTTPException(status_code=404, detail="Scoring job produced no output. Check logs.")

        return ScoreResponse(
            message="Scoring job completed successfully.",
            output_path=output_path,
            features_scored=len(orchestrator.config.get("scoring_jobs", [])), # This is a simplification
            scores_added=added_scores
        )
    except FileNotFoundError as e:
        logger.error(f"File not found during scoring request: {e}")
        raise HTTPException(status_code=400, detail=f"Input file not found: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@app.get("/health", status_code=200)
def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}
