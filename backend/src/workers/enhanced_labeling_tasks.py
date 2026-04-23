"""
Celery task for enhanced per-feature two-pass LLM labeling.

Triggered from the Feature Detail modal. Runs a two-pass strategy:
  Pass 1 — parallel per-example summarization (workers concurrent HTTP calls)
  Pass 2 — synthesis into a structured label with reasoning

On completion, writes name / category / description / notes /
label_source='enhanced_llm' / labeled_at to the feature row.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from src.core.celery_app import celery_app
from src.utils.millm_utils import ensure_model_loaded
from src.workers.base_task import DatabaseTask

logger = logging.getLogger(__name__)


def _ensure_model_loaded(endpoint_url: str, model_name: str) -> None:
    """Backward-compatible shim — delegates to the shared utility."""
    ensure_model_loaded(endpoint_url, model_name)


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

            # Write label to feature and mark star aqua (permanent enhanced labeling marker)
            feature = db.query(Feature).filter(Feature.id == job.feature_id).first()
            if feature:
                feature.name = result["name"]
                feature.category = result["category"]
                feature.description = result["description"]
                feature.notes = result["notes"]
                feature.label_source = LabelSource.ENHANCED_LLM.value if hasattr(LabelSource, "ENHANCED_LLM") else "enhanced_llm"
                feature.labeled_at = datetime.now(timezone.utc)
                feature.updated_at = datetime.now(timezone.utc)
                feature.is_favorite = True
                feature.star_color = "aqua"

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
