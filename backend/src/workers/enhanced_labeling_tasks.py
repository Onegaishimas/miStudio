"""
Celery task for enhanced per-feature two-pass LLM labeling.

Triggered from the Feature Detail modal. Runs a two-pass strategy:
  Pass 1 — parallel per-example summarization (workers concurrent HTTP calls)
  Pass 2 — synthesis into a structured label with reasoning

On completion, writes name / category / description / notes /
label_source='enhanced_llm' / labeled_at to the feature row.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict
from urllib.parse import urlparse

import requests

from src.core.celery_app import celery_app
from src.workers.base_task import DatabaseTask

logger = logging.getLogger(__name__)

_MODEL_LOAD_TIMEOUT = 300  # seconds to wait for miLLM model to reach LOADED status


def _ensure_model_loaded(endpoint_url: str, model_name: str) -> None:
    """
    If `endpoint_url` points to a miLLM instance, ensure the named model is loaded.

    Silently no-ops when:
    - The endpoint is not a miLLM instance (non-200 from /api/models)
    - The model is already loaded
    - Any unexpected error occurs (so the main task can still attempt the call)
    """
    try:
        parsed = urlparse(endpoint_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        resp = requests.get(f"{base_url}/api/models", timeout=10)
        if resp.status_code != 200:
            return  # Not a miLLM instance or unreachable

        payload = resp.json()
        if not payload.get("success"):
            return

        models = payload.get("data") or []

        # Find the model matching by name (case-insensitive for robustness)
        model_entry = None
        for m in models:
            if m.get("name", "").lower() == model_name.lower():
                model_entry = m
                break

        if model_entry is None:
            logger.warning(
                "miLLM model %r not found at %s — proceeding without pre-load",
                model_name,
                base_url,
            )
            return

        model_id = model_entry["id"]
        status = model_entry.get("status", "")

        if status == "loaded":
            return  # Already in GPU — nothing to do

        if status in ("downloading", "error"):
            logger.warning(
                "miLLM model %r has status=%r; cannot load — proceeding anyway",
                model_name,
                status,
            )
            return

        # status is "ready" or "loading" — trigger load if needed
        if status == "ready":
            logger.info(
                "miLLM model %r (id=%s) is not loaded; triggering load...",
                model_name,
                model_id,
            )
            load_resp = requests.post(f"{base_url}/api/models/{model_id}/load", timeout=30)
            if load_resp.status_code not in (200, 202):
                logger.warning(
                    "miLLM load request returned %d — proceeding anyway",
                    load_resp.status_code,
                )
                return

        # Poll until loaded or timeout
        deadline = time.time() + _MODEL_LOAD_TIMEOUT
        while time.time() < deadline:
            time.sleep(5)
            poll = requests.get(f"{base_url}/api/models/{model_id}", timeout=10)
            if poll.status_code != 200:
                break
            poll_payload = poll.json()
            current_status = (poll_payload.get("data") or {}).get("status", "")
            if current_status == "loaded":
                logger.info("miLLM model %r loaded successfully", model_name)
                return
            if current_status == "error":
                logger.error("miLLM model %r failed to load (status=error)", model_name)
                return

        logger.warning(
            "miLLM model %r did not reach loaded status within %ds",
            model_name,
            _MODEL_LOAD_TIMEOUT,
        )

    except Exception as exc:
        logger.warning("_ensure_model_loaded: %s (proceeding anyway)", exc)


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="enhanced_label_feature",
    max_retries=0,
    autoretry_for=(),
)
def enhanced_label_feature_task(self, job_id: str) -> Dict[str, Any]:
    """
    Execute a two-pass enhanced labeling job for a single feature.

    Args:
        job_id: EnhancedLabelingJob primary key.

    Returns:
        Dict with result fields (name, category, description) or empty on skip.
    """
    logger.info("Starting enhanced labeling task for job %s", job_id)

    with self.get_db() as db:
        from src.models.enhanced_labeling_job import EnhancedLabelingJob, EnhancedLabelingStatus, EnhancedLabelingPhase
        from src.models.feature import Feature, LabelSource
        from src.models.feature_activation import FeatureActivation
        from src.services.enhanced_labeling_service import EnhancedLabelingService, EnhancedLabelingError
        from src.workers.websocket_emitter import (
            emit_enhanced_labeling_progress,
            emit_enhanced_labeling_completed,
            emit_enhanced_labeling_failed,
        )

        job = db.query(EnhancedLabelingJob).filter(
            EnhancedLabelingJob.id == job_id
        ).first()

        if not job:
            logger.error("Enhanced labeling job %s not found", job_id)
            return {}

        if job.status == EnhancedLabelingStatus.COMPLETED.value:
            logger.info("Job %s already completed, skipping", job_id)
            return {}

        # Mark as running
        job.status = EnhancedLabelingStatus.RUNNING.value
        job.celery_task_id = self.request.id
        job.phase = EnhancedLabelingPhase.PASS1.value
        job.updated_at = datetime.now(timezone.utc)
        db.commit()

        try:
            # Fetch activation examples
            activation_rows = (
                db.query(FeatureActivation)
                .filter(FeatureActivation.feature_id == job.feature_id)
                .order_by(FeatureActivation.max_activation.desc())
                .limit(job.examples_total)
                .all()
            )

            rows_as_dicts = [
                {
                    "prime_token": row.prime_token,
                    "prefix_tokens": row.prefix_tokens or [],
                    "suffix_tokens": row.suffix_tokens or [],
                    "max_activation": row.max_activation,
                }
                for row in activation_rows
            ]

            if not rows_as_dicts:
                raise EnhancedLabelingError(
                    "No activation examples found for feature — run extraction first"
                )

            # Ensure the target model is loaded in miLLM before making LLM calls
            _ensure_model_loaded(job.endpoint, job.model)

            service = EnhancedLabelingService(
                endpoint=job.endpoint,
                model=job.model,
                workers=job.workers,
            )

            # Progress callback for pass 1
            def _progress_cb(n_completed: int, total: int) -> None:
                job.examples_completed = n_completed
                job.updated_at = datetime.now(timezone.utc)
                db.commit()
                emit_enhanced_labeling_progress(
                    job_id,
                    {
                        "job_id": job_id,
                        "phase": "pass1",
                        "examples_completed": n_completed,
                        "examples_total": total,
                    },
                )

            try:
                result = service.run(
                    activation_rows=rows_as_dicts,
                    max_examples=job.examples_total,
                    progress_cb=_progress_cb,
                )
            finally:
                service.close()

            # Pass 2 in-progress notification
            job.phase = EnhancedLabelingPhase.PASS2.value
            job.updated_at = datetime.now(timezone.utc)
            db.commit()
            emit_enhanced_labeling_progress(
                job_id,
                {
                    "job_id": job_id,
                    "phase": "pass2",
                    "examples_completed": job.examples_total,
                    "examples_total": job.examples_total,
                },
            )

            # Write label to feature
            feature = db.query(Feature).filter(Feature.id == job.feature_id).first()
            if feature:
                feature.name = result["name"]
                feature.category = result["category"]
                feature.description = result["description"]
                feature.notes = result["notes"]
                feature.label_source = LabelSource.ENHANCED_LLM.value if hasattr(LabelSource, "ENHANCED_LLM") else "enhanced_llm"
                feature.labeled_at = datetime.now(timezone.utc)
                feature.updated_at = datetime.now(timezone.utc)

            # Mark job completed
            job.status = EnhancedLabelingStatus.COMPLETED.value
            job.phase = None
            job.pass1_summaries = result["pass1_summaries"]
            job.raw_synthesis = result["raw_synthesis"]
            job.completed_at = datetime.now(timezone.utc)
            job.updated_at = datetime.now(timezone.utc)
            db.commit()

            emit_enhanced_labeling_completed(
                job_id,
                {
                    "job_id": job_id,
                    "name": result["name"],
                    "category": result["category"],
                    "description": result["description"],
                    "notes": result["notes"],
                },
            )

            logger.info(
                "Enhanced labeling job %s completed: name=%r", job_id, result["name"]
            )
            return {
                "name": result["name"],
                "category": result["category"],
                "description": result["description"],
            }

        except Exception as exc:
            logger.error(
                "Enhanced labeling task failed for job %s: %s", job_id, exc, exc_info=True
            )
            try:
                job.status = EnhancedLabelingStatus.FAILED.value
                job.phase = None
                job.error_message = str(exc)
                job.updated_at = datetime.now(timezone.utc)
                db.commit()
            except Exception:
                pass
            emit_enhanced_labeling_failed(job_id, str(exc))
            raise
