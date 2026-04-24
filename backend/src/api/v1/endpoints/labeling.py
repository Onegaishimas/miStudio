"""
Feature labeling API endpoints.

Provides REST API for independent semantic labeling of extracted SAE features.
"""

import logging
from typing import Optional
import httpx
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import select
from src.core.config import settings
from src.core.deps import get_db
from src.services.labeling_service import LabelingService
from src.workers.labeling_tasks import label_features_task
from src.schemas.labeling import (
    LabelingConfigRequest,
    LabelingStatusResponse,
    LabelingListResponse
)
from src.models.extraction_job import ExtractionJob
from src.models.training import Training
from src.models.external_sae import ExternalSAE
from src.models.model import Model

logger = logging.getLogger(__name__)

router = APIRouter()


async def _enrich_labeling_responses(
    db: AsyncSession,
    responses: list[LabelingStatusResponse]
) -> list[LabelingStatusResponse]:
    """Enrich labeling responses with extraction context (model, layer, hook, SAE name)."""
    if not responses:
        return responses

    # Batch-load extraction jobs
    ext_ids = {r.extraction_job_id for r in responses}
    result = await db.execute(
        select(ExtractionJob).where(ExtractionJob.id.in_(ext_ids))
    )
    ext_map = {ej.id: ej for ej in result.scalars().all()}

    # Collect training/SAE IDs for batch loading
    training_ids = {ej.training_id for ej in ext_map.values() if ej.training_id}
    sae_ids = {ej.external_sae_id for ej in ext_map.values() if ej.external_sae_id}

    # Batch-load trainings
    trainings_map = {}
    model_ids = set()
    if training_ids:
        result = await db.execute(select(Training).where(Training.id.in_(training_ids)))
        for t in result.scalars().all():
            trainings_map[t.id] = t
            if t.model_id:
                model_ids.add(t.model_id)

    # Batch-load external SAEs
    saes_map = {}
    if sae_ids:
        result = await db.execute(select(ExternalSAE).where(ExternalSAE.id.in_(sae_ids)))
        for s in result.scalars().all():
            saes_map[s.id] = s
            if hasattr(s, 'model_id') and s.model_id:
                model_ids.add(s.model_id)

    # Batch-load models
    models_map = {}
    if model_ids:
        result = await db.execute(
            select(Model.id, Model.name).where(Model.id.in_(model_ids))
        )
        for row in result.all():
            models_map[row[0]] = row[1]

    # Enrich each response
    for resp in responses:
        ej = ext_map.get(resp.extraction_job_id)
        if not ej:
            continue

        resp.layer_index = ej.layer_index
        resp.hook_type = ej.hook_type

        if ej.training_id and ej.training_id in trainings_map:
            training = trainings_map[ej.training_id]
            resp.model_name = models_map.get(training.model_id, training.model_id)
        elif ej.external_sae_id and ej.external_sae_id in saes_map:
            sae = saes_map[ej.external_sae_id]
            resp.sae_name = getattr(sae, 'name', None) or sae.id
            if hasattr(sae, 'model_id') and sae.model_id:
                resp.model_name = models_map.get(sae.model_id, sae.model_id)

    return responses


@router.post(
    "/labeling",
    response_model=LabelingStatusResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start feature labeling"
)
async def start_labeling(
    config: LabelingConfigRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Start a semantic labeling job for a completed extraction.

    This creates a labeling job and queues it for async processing. Features
    are labeled independently from extraction, allowing re-labeling without
    re-extraction.

    Args:
        config: Labeling configuration (extraction_job_id, labeling_method, etc.)

    Returns:
        LabelingStatusResponse with job details

    Raises:
        404: Extraction not found
        409: Active labeling already exists for this extraction
        422: Extraction not completed or has no features
    """
    labeling_service = LabelingService(db)

    try:
        # Start labeling job (creates record in QUEUED status)
        labeling_job = await labeling_service.start_labeling(
            extraction_job_id=config.extraction_job_id,
            config=config.model_dump()
        )

        # Queue Celery task for async labeling
        task = label_features_task.delay(labeling_job.id)

        # Update with Celery task ID
        labeling_job.celery_task_id = task.id
        await db.commit()
        await db.refresh(labeling_job)

        logger.info(
            f"Started labeling job {labeling_job.id} for extraction "
            f"{config.extraction_job_id} with task {task.id}"
        )

        return LabelingStatusResponse.model_validate(labeling_job)

    except ValueError as e:
        error_message = str(e)

        # Check for specific error conditions
        if "not found" in error_message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_message
            )
        elif "already has an active labeling" in error_message:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_message
            )
        else:
            # Must be completed, has features, etc.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_message
            )


@router.get(
    "/labeling/{labeling_job_id}",
    response_model=LabelingStatusResponse,
    summary="Get labeling job status"
)
async def get_labeling_status(
    labeling_job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the status of a specific labeling job.

    Args:
        labeling_job_id: ID of the labeling job

    Returns:
        LabelingStatusResponse with status, progress, and statistics

    Raises:
        404: Labeling job not found
    """
    labeling_service = LabelingService(db)

    labeling_job = await labeling_service.get_labeling_job(labeling_job_id)

    if not labeling_job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Labeling job {labeling_job_id} not found"
        )

    response = LabelingStatusResponse.model_validate(labeling_job)
    enriched = await _enrich_labeling_responses(db, [response])
    return enriched[0]


@router.get(
    "/labeling",
    response_model=LabelingListResponse,
    summary="List labeling jobs"
)
async def list_labeling_jobs(
    extraction_job_id: Optional[str] = Query(None, description="Filter by extraction job ID"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a paginated list of labeling jobs.

    Args:
        extraction_job_id: Optional filter by extraction job ID
        limit: Maximum number of results to return (1-100)
        offset: Number of results to skip for pagination

    Returns:
        LabelingListResponse with list of labeling jobs and metadata
    """
    labeling_service = LabelingService(db)

    # Get labeling jobs
    jobs_list, total = await labeling_service.list_labeling_jobs(
        extraction_job_id=extraction_job_id,
        limit=limit,
        offset=offset
    )

    responses = [LabelingStatusResponse.model_validate(job) for job in jobs_list]
    responses = await _enrich_labeling_responses(db, responses)

    return LabelingListResponse(
        data=responses,
        meta={
            "total": total,
            "limit": limit,
            "offset": offset
        }
    )


@router.post(
    "/labeling/{labeling_job_id}/cancel",
    status_code=status.HTTP_200_OK,
    summary="Cancel labeling job"
)
async def cancel_labeling(
    labeling_job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Cancel an active labeling job.

    Args:
        labeling_job_id: ID of the labeling job to cancel

    Returns:
        Success message

    Raises:
        404: Labeling job not found
        409: Labeling job not in cancellable state
    """
    labeling_service = LabelingService(db)

    try:
        await labeling_service.cancel_labeling_job(labeling_job_id)
        logger.info(f"Cancelled labeling job {labeling_job_id}")
        return {"message": "Labeling job cancelled successfully"}
    except ValueError as e:
        error_message = str(e)
        if "not found" in error_message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_message
            )
        else:
            # Cannot cancel due to status
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_message
            )


@router.delete(
    "/labeling/{labeling_job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete labeling job"
)
async def delete_labeling(
    labeling_job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a labeling job record.

    This does NOT delete the features or their labels, only the labeling job
    record itself. Feature labels will remain intact.

    If the job is currently active (queued or labeling), it will be automatically
    cancelled before deletion by revoking the Celery task.

    Args:
        labeling_job_id: ID of the labeling job to delete

    Raises:
        404: Labeling job not found
    """
    labeling_service = LabelingService(db)

    try:
        await labeling_service.delete_labeling_job(labeling_job_id)
        logger.info(f"Deleted labeling job {labeling_job_id}")
        return None  # 204 No Content
    except ValueError as e:
        error_message = str(e)
        # Only possible error now is "not found"
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_message
        )


@router.post(
    "/extractions/{extraction_id}/label",
    response_model=LabelingStatusResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Label extraction (convenience endpoint)"
)
async def label_extraction(
    extraction_id: str,
    config: LabelingConfigRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Convenience endpoint to start labeling for an extraction.

    This is a shorthand for POST /labeling with extraction_job_id in the body.
    The extraction_id from the URL takes precedence over config.extraction_job_id.

    Args:
        extraction_id: ID of the extraction to label
        config: Labeling configuration (labeling_method, openai_model, etc.)

    Returns:
        LabelingStatusResponse with job details

    Raises:
        404: Extraction not found
        409: Active labeling already exists
        422: Extraction not completed or has no features
    """
    # Override extraction_job_id with URL parameter
    config.extraction_job_id = extraction_id

    # Delegate to main labeling endpoint
    return await start_labeling(config, db)


@router.get(
    "/labeling/models/available",
    summary="List available Ollama models"
)
async def list_available_ollama_models(db: AsyncSession = Depends(get_db)):
    """
    List available Ollama models for local labeling.

    Queries the Ollama API to get all available models. Returns both
    the raw model list and a formatted list with display names.

    Returns:
        Dict with 'models' array containing model information

    Raises:
        503: Ollama service unavailable
    """
    # Resolve Ollama URL: DB setting takes precedence over env var
    from src.models.app_setting import AppSetting
    result = await db.execute(select(AppSetting).where(AppSetting.key == "ollama_url"))
    db_setting = result.scalar_one_or_none()
    ollama_url = (db_setting.value if db_setting else None) or settings.ollama_url

    try:
        # Query Ollama/OpenAI-compatible API for available models
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Try OpenAI-compatible /v1/models first, fall back to Ollama /api/tags
            try:
                response = await client.get(f"{ollama_url}/v1/models")
                response.raise_for_status()
                data = response.json()
                models = []
                for model in data.get("data", []):
                    model_name = model.get("id", "")
                    models.append({
                        "name": model_name,
                        "display_name": model_name,
                        "size": 0,
                        "size_gb": 0,
                        "modified_at": "",
                        "details": {},
                    })
                return {"models": models, "total": len(models)}
            except Exception:
                pass
            # Fall back to Ollama /api/tags
            response = await client.get(f"{ollama_url}/api/tags")
            response.raise_for_status()
            data = response.json()

            # Extract model information
            models = []
            for model in data.get("models", []):
                model_name = model.get("name", "")
                model_size = model.get("size", 0)
                modified_at = model.get("modified_at", "")

                # Format size in GB
                size_gb = model_size / (1024**3) if model_size else 0

                # Create display name
                display_name = f"{model_name}"
                if size_gb > 0:
                    display_name += f" ({size_gb:.1f}GB)"

                models.append({
                    "name": model_name,
                    "display_name": display_name,
                    "size": model_size,
                    "size_gb": round(size_gb, 2),
                    "modified_at": modified_at,
                    "details": model.get("details", {})
                })

            return {
                "models": models,
                "total": len(models)
            }

    except httpx.RequestError as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama service is not available. Please ensure Ollama is running."
        )
    except Exception as e:
        logger.error(f"Error listing Ollama models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list Ollama models: {str(e)}"
        )


class FetchModelsRequest(BaseModel):
    api_key: Optional[str] = None
    endpoint_url: str = "https://api.openai.com/v1"


@router.post(
    "/labeling/models/openai",
    summary="Fetch available models from OpenAI or compatible endpoint"
)
async def fetch_openai_models(
    request: FetchModelsRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Fetch available models from any OpenAI-compatible endpoint.
    Proxied through the backend to avoid CORS and DNS resolution issues.

    API key resolution order:
      1. request.api_key (explicit in POST body)
      2. Database AppSetting 'openai_api_key' (set via Settings → API Keys)
      3. Environment variable settings.openai_api_key
      4. None (unauthenticated — fine for Ollama / vLLM / miLLM)
    """
    api_key = request.api_key
    if not api_key:
        from src.models.app_setting import AppSetting
        from src.core.encryption import decrypt_value

        result = await db.execute(
            select(AppSetting).where(AppSetting.key == "openai_api_key")
        )
        db_setting = result.scalar_one_or_none()
        if db_setting and db_setting.value:
            api_key = (
                decrypt_value(db_setting.value, setting_key="openai_api_key")
                if db_setting.is_sensitive
                else db_setting.value
            )
    if not api_key:
        api_key = getattr(settings, 'openai_api_key', None)

    # Validate URL scheme — only http/https allowed to prevent SSRF via file://, ftp://, etc.
    from urllib.parse import urlparse
    parsed = urlparse(request.endpoint_url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="endpoint_url must use http or https scheme")

    # Normalize endpoint URL
    base_url = request.endpoint_url.rstrip('/')
    if not base_url.endswith('/v1'):
        base_url = f"{base_url}/v1"

    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{base_url}/models",
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            models = []

            # OpenAI format: { "data": [{ "id": "gpt-4o-mini", ... }] }
            if "data" in data and isinstance(data["data"], list):
                for m in data["data"]:
                    model_id = m.get("id", "")
                    if model_id:
                        models.append({
                            "id": model_id,
                            "owned_by": m.get("owned_by", ""),
                        })

            # Ollama format: { "models": [{ "name": "gemma2:2b", ... }] }
            elif "models" in data and isinstance(data["models"], list):
                for m in data["models"]:
                    model_name = m.get("name", "") or m.get("id", "")
                    if model_name:
                        models.append({
                            "id": model_name,
                            "owned_by": m.get("owned_by", "ollama"),
                        })

            # Sort: gpt models first, then alphabetical
            models.sort(key=lambda m: (
                0 if m["id"].startswith("gpt-") else 1,
                m["id"]
            ))

            return {
                "models": models,
                "total": len(models)
            }

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key."
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"API returned HTTP {e.response.status_code}: {e.response.text[:200]}"
        )
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot connect to {base_url}. Check the endpoint URL."
        )
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch models: {str(e)}"
        )
