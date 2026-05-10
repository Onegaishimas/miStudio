"""
Task Queue API endpoints.

This module provides endpoints for viewing and managing background task operations,
including failed tasks that can be manually retried.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.deps import get_db
from ....services.app_setting_service import AppSettingService
from ....services.task_queue_service import TaskQueueService
from ....schemas.task_queue import (
    TaskQueueResponse,
    TaskQueueListResponse,
    TaskQueueRetryRequest,
    TaskQueueRetryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("", response_model=TaskQueueListResponse)
async def list_tasks(
    status: Optional[str] = None,
    entity_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    List all task queue entries with optional filtering.

    Query Parameters:
        status: Filter by status (queued, running, failed, completed, cancelled)
        entity_type: Filter by entity type (model, dataset, training)

    Returns:
        List of task queue entries with entity information
    """
    tasks = await TaskQueueService.get_all_tasks(db, status=status, entity_type=entity_type)

    # Enrich tasks with entity information
    enriched_tasks = []
    for task in tasks:
        entity_info = await TaskQueueService.get_entity_info(
            db, task.entity_id, task.entity_type
        )

        task_dict = {
            "id": task.id,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "entity_id": task.entity_id,
            "entity_type": task.entity_type,
            "status": task.status,
            "progress": task.progress,
            "error_message": task.error_message,
            "retry_params": task.retry_params,
            "retry_count": task.retry_count,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
            "entity_info": entity_info,
        }
        enriched_tasks.append(task_dict)

    return {"data": enriched_tasks}


@router.get("/failed", response_model=TaskQueueListResponse)
async def list_failed_tasks(db: AsyncSession = Depends(get_db)):
    """
    List all failed task queue entries.

    Returns:
        List of failed tasks with entity information
    """
    tasks = await TaskQueueService.get_failed_tasks(db)

    enriched_tasks = []
    for task in tasks:
        entity_info = await TaskQueueService.get_entity_info(
            db, task.entity_id, task.entity_type
        )

        task_dict = {
            "id": task.id,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "entity_id": task.entity_id,
            "entity_type": task.entity_type,
            "status": task.status,
            "progress": task.progress,
            "error_message": task.error_message,
            "retry_params": task.retry_params,
            "retry_count": task.retry_count,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
            "entity_info": entity_info,
        }
        enriched_tasks.append(task_dict)

    return {"data": enriched_tasks}


@router.get("/active", response_model=TaskQueueListResponse)
async def list_active_tasks(db: AsyncSession = Depends(get_db)):
    """
    List all active (queued or running) operations across all job types.

    Queries task_queue table plus labeling_jobs and neuronpedia push jobs
    to provide a unified view of all background operations.

    Returns:
        List of active tasks with entity information
    """
    from sqlalchemy import select, or_, text

    tasks = await TaskQueueService.get_active_tasks(db)

    enriched_tasks = []
    for task in tasks:
        entity_info = await TaskQueueService.get_entity_info(
            db, task.entity_id, task.entity_type
        )

        task_dict = {
            "id": task.id,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "entity_id": task.entity_id,
            "entity_type": task.entity_type,
            "status": task.status,
            "progress": task.progress,
            "error_message": task.error_message,
            "retry_params": task.retry_params,
            "retry_count": task.retry_count,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
            "entity_info": entity_info,
        }
        enriched_tasks.append(task_dict)

    # Also include active labeling jobs
    try:
        labeling_result = await db.execute(
            text("""
                SELECT lj.id, lj.extraction_job_id, lj.labeling_method,
                       lj.openai_compatible_model, lj.openai_model, lj.local_model,
                       lj.status, lj.progress, lj.features_labeled, lj.total_features,
                       lj.created_at, lj.updated_at
                FROM labeling_jobs lj
                WHERE lj.status IN ('queued', 'labeling')
                ORDER BY lj.created_at DESC
            """)
        )
        for row in labeling_result:
            model_name = row.openai_compatible_model or row.openai_model or row.local_model or 'unknown'
            progress_pct = (row.features_labeled / row.total_features * 100) if row.total_features > 0 else 0
            enriched_tasks.append({
                "id": row.id,
                "task_id": row.id,
                "task_type": "labeling",
                "entity_id": row.extraction_job_id,
                "entity_type": "labeling",
                "status": "running" if row.status == "labeling" else "queued",
                "progress": progress_pct,
                "error_message": None,
                "retry_params": None,
                "retry_count": 0,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "started_at": row.created_at.isoformat() if row.created_at else None,
                "completed_at": None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "entity_info": {
                    "name": f"Labeling ({model_name})",
                    "details": f"{row.features_labeled}/{row.total_features} features",
                },
            })
    except Exception as e:
        logger.warning(f"Failed to query labeling jobs: {e}")

    # Also include active Neuronpedia push jobs
    try:
        push_result = await db.execute(
            text("""
                SELECT np.id, np.sae_id, np.status, np.progress,
                       np.features_pushed, np.total_features,
                       np.created_at, np.updated_at,
                       es.name as sae_name
                FROM neuronpedia_pushes np
                LEFT JOIN external_saes es ON es.id = np.sae_id
                WHERE np.status IN ('queued', 'pushing', 'preparing')
                ORDER BY np.created_at DESC
            """)
        )
        for row in push_result:
            progress_pct = float(row.progress) if row.progress else 0
            enriched_tasks.append({
                "id": row.id,
                "task_id": row.id,
                "task_type": "neuronpedia_push",
                "entity_id": row.sae_id,
                "entity_type": "neuronpedia",
                "status": "running" if row.status in ("pushing", "preparing") else "queued",
                "progress": progress_pct,
                "error_message": None,
                "retry_params": None,
                "retry_count": 0,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "started_at": row.created_at.isoformat() if row.created_at else None,
                "completed_at": None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "entity_info": {
                    "name": f"Push to Neuronpedia",
                    "details": row.sae_name or row.sae_id,
                },
            })
    except Exception as e:
        logger.warning(f"Failed to query neuronpedia pushes: {e}")

    return {"data": enriched_tasks}


@router.get("/{task_queue_id}", response_model=TaskQueueResponse)
async def get_task(
    task_queue_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific task queue entry by ID.

    Args:
        task_queue_id: Task queue entry ID

    Returns:
        Task queue entry with entity information

    Raises:
        HTTPException: If task not found
    """
    task = await TaskQueueService.get_task_by_id(db, task_queue_id)

    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Task queue entry '{task_queue_id}' not found"
        )

    entity_info = await TaskQueueService.get_entity_info(
        db, task.entity_id, task.entity_type
    )

    return {
        "data": {
            "id": task.id,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "entity_id": task.entity_id,
            "entity_type": task.entity_type,
            "status": task.status,
            "progress": task.progress,
            "error_message": task.error_message,
            "retry_params": task.retry_params,
            "retry_count": task.retry_count,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
            "entity_info": entity_info,
        }
    }


@router.post("/{task_queue_id}/retry", response_model=TaskQueueRetryResponse)
async def retry_task(
    task_queue_id: str,
    request: TaskQueueRetryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Retry a failed task with optional parameter overrides.

    This endpoint:
    1. Verifies the task exists and is in failed state
    2. Updates retry count and status
    3. Dispatches a new Celery task with retry parameters
    4. Returns the new task information

    Args:
        task_queue_id: Task queue entry ID
        request: Optional parameter overrides for retry

    Returns:
        Retry status and new task information

    Raises:
        HTTPException: If task not found or not in failed state
    """
    task = await TaskQueueService.get_task_by_id(db, task_queue_id)

    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Task queue entry '{task_queue_id}' not found"
        )

    if task.status != "failed":
        raise HTTPException(
            status_code=400,
            detail=f"Task is in '{task.status}' state, can only retry failed tasks"
        )

    # Merge retry_params with any overrides from request
    retry_params = task.retry_params.copy() if task.retry_params else {}
    if request.param_overrides:
        retry_params.update(request.param_overrides)

    # Fetch HF token from encrypted app_settings — never stored in retry_params
    hf_token = await AppSettingService.get_decrypted_value(db, "hf_token")

    # Increment retry count
    await TaskQueueService.increment_retry_count(db, task_queue_id)

    # Dispatch appropriate Celery task based on task type
    if task.task_type == "download" and task.entity_type == "model":
        from ....workers.model_tasks import download_and_load_model

        celery_task = download_and_load_model.delay(
            model_id=task.entity_id,
            repo_id=retry_params.get("repo_id"),
            quantization=retry_params.get("quantization"),
            access_token=hf_token,
            trust_remote_code=retry_params.get("trust_remote_code", False),
        )

        # Update task_queue entry with new Celery task ID
        task.task_id = celery_task.id
        await db.commit()

        logger.info(f"Retried model download task {task_queue_id}, new Celery task: {celery_task.id}")

    elif task.task_type == "download" and task.entity_type == "dataset":
        from ....workers.dataset_tasks import download_dataset_task

        celery_task = download_dataset_task.delay(
            dataset_id=task.entity_id,
            repo_id=retry_params.get("repo_id"),
            access_token=hf_token,
            split=retry_params.get("split"),
            config=retry_params.get("config"),
        )

        task.task_id = celery_task.id
        await db.commit()

        logger.info(f"Retried dataset download task {task_queue_id}, new Celery task: {celery_task.id}")

    elif task.task_type == "tokenization" and task.entity_type == "dataset":
        from ....workers.dataset_tasks import tokenize_dataset_task

        celery_task = tokenize_dataset_task.delay(
            dataset_id=task.entity_id,
            tokenizer_name=retry_params.get("tokenizer_name"),
            max_length=retry_params.get("max_length", 512),
            stride=retry_params.get("stride", 0),
            padding=retry_params.get("padding", "max_length"),
            truncation=retry_params.get("truncation", "longest_first"),
            add_special_tokens=retry_params.get("add_special_tokens", True),
            text_column=retry_params.get("text_column"),
            enable_cleaning=retry_params.get("enable_cleaning", True),
        )

        task.task_id = celery_task.id
        await db.commit()

        logger.info(f"Retried tokenization task {task_queue_id}, new Celery task: {celery_task.id}")

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported task type '{task.task_type}' for entity type '{task.entity_type}'"
        )

    return {
        "success": True,
        "message": f"Task retry initiated (attempt {task.retry_count + 1})",
        "task_queue_id": task_queue_id,
        "celery_task_id": task.task_id,
        "retry_count": task.retry_count,
    }


@router.delete("/{task_queue_id}", status_code=204)
async def delete_task(
    task_queue_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a task queue entry.

    This removes the task from the queue. For active tasks, you should
    cancel them first using the cancel endpoint.

    Args:
        task_queue_id: Task queue entry ID

    Raises:
        HTTPException: If task not found
    """
    success = await TaskQueueService.delete_task(db, task_queue_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Task queue entry '{task_queue_id}' not found"
        )

    logger.info(f"Deleted task queue entry {task_queue_id}")
