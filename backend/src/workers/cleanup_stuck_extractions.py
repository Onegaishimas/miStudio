"""
Periodic task to clean up stuck extraction jobs.

This task runs every 10 minutes and marks extraction jobs as FAILED if they've been
stuck for too long without updates.

Thresholds:
- QUEUED jobs without a Celery task (batch waiting): 3 hours
- QUEUED jobs with a Celery task: 1 hour
- EXTRACTING jobs: 1 hour
"""

import logging
from datetime import datetime, timezone, timedelta
from src.core.celery_app import celery_app
from src.workers.base_task import DatabaseTask
from src.models.extraction_job import ExtractionJob, ExtractionStatus
from src.workers.websocket_emitter import emit_extraction_job_progress

logger = logging.getLogger(__name__)

# Jobs waiting in a batch queue get a long grace period
BATCH_QUEUED_THRESHOLD_MINUTES = 180  # 3 hours
# Jobs with a Celery task that haven't started
QUEUED_THRESHOLD_MINUTES = 60  # 1 hour
# Jobs actively extracting that appear stuck
EXTRACTING_THRESHOLD_MINUTES = 60  # 1 hour


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="cleanup_stuck_extractions"
)
def cleanup_stuck_extractions_task(self):
    """
    Clean up extraction jobs that have been stuck for too long.

    An extraction is considered stuck if:
    - QUEUED with no celery_task_id (batch waiting): no update in 3 hours
    - QUEUED with celery_task_id: no update in 1 hour AND task is not running
    - EXTRACTING: no update in 1 hour AND task is not running
    """
    logger.info("Running stuck extraction cleanup task")

    with self.get_db() as db:
        try:
            # Use the shortest threshold to find candidates, then apply per-status logic
            candidate_threshold = datetime.now(timezone.utc) - timedelta(minutes=QUEUED_THRESHOLD_MINUTES)

            candidate_extractions = db.query(ExtractionJob).filter(
                ExtractionJob.status.in_([
                    ExtractionStatus.QUEUED.value,
                    ExtractionStatus.EXTRACTING.value
                ]),
                ExtractionJob.updated_at < candidate_threshold
            ).all()

            cleaned_count = 0
            for extraction in candidate_extractions:
                age_minutes = (datetime.now(timezone.utc) - extraction.updated_at).total_seconds() / 60

                # Batch jobs waiting without a Celery task get a longer grace period
                if (extraction.status == ExtractionStatus.QUEUED.value
                        and extraction.batch_id
                        and not extraction.celery_task_id):
                    if age_minutes < BATCH_QUEUED_THRESHOLD_MINUTES:
                        logger.debug(
                            f"Extraction {extraction.id} is a batch-queued job "
                            f"({age_minutes:.0f}min old, threshold {BATCH_QUEUED_THRESHOLD_MINUTES}min), skipping"
                        )
                        continue

                # Check if Celery task is actually running
                task_is_running = False
                if extraction.celery_task_id:
                    from src.core.celery_app import get_task_status
                    task_status = get_task_status(extraction.celery_task_id)

                    if task_status['state'] in ['PENDING', 'STARTED', 'RETRY']:
                        task_is_running = True
                        logger.info(
                            f"Extraction {extraction.id} has active Celery task "
                            f"{extraction.celery_task_id} ({task_status['state']}), skipping cleanup"
                        )

                if not task_is_running:
                    # Mark as failed
                    threshold_used = (
                        BATCH_QUEUED_THRESHOLD_MINUTES
                        if extraction.batch_id and not extraction.celery_task_id
                        else EXTRACTING_THRESHOLD_MINUTES
                    )
                    logger.warning(
                        f"Marking stuck extraction {extraction.id} as FAILED "
                        f"(status: {extraction.status}, age: {age_minutes:.0f}min, "
                        f"threshold: {threshold_used}min, "
                        f"task_id: {extraction.celery_task_id or 'None'})"
                    )

                    extraction.status = ExtractionStatus.FAILED.value
                    extraction.error_message = (
                        f"Extraction job stuck - no progress for more than {int(age_minutes)} minutes. "
                        "This may indicate a crashed worker or system issue."
                    )
                    extraction.completed_at = datetime.now(timezone.utc)
                    extraction.updated_at = datetime.now(timezone.utc)

                    db.commit()
                    cleaned_count += 1

                    # Emit WebSocket event to notify frontend
                    try:
                        emit_extraction_job_progress(
                            extraction_id=extraction.id,
                            training_id=extraction.training_id,
                            sae_id=extraction.external_sae_id,
                            status="failed",
                            message=extraction.error_message,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to emit WebSocket event for {extraction.id}: {e}")

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} stuck extraction(s)")
            else:
                logger.info("No stuck extractions found")

            return {"cleaned": cleaned_count}

        except Exception as e:
            logger.error(f"Error in stuck extraction cleanup: {e}", exc_info=True)
            raise
