"""Celery task for steered-transcript recording (Steered Transcript Recorder).

Records (dial, prompt, unsteered, steered) transcripts for a circuit / cluster /
feature set on the GPU and persists a `steering_samples` manifest. Holds the
single GPU like calibration; its in-flight marker is a `steering_record_runs` row
(cluster/feature jobs have no circuit row). GPU profile → extraction queue.
"""

import logging
from typing import Any, Dict

from ..core.celery_app import celery_app
from .base_task import DatabaseTask
from .websocket_emitter import (
    emit_circuit_run_completed,
    emit_circuit_run_failed,
    emit_circuit_run_progress,
)

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, base=DatabaseTask,
                 name="src.workers.circuit_record_tasks.run_circuit_record",
                 max_retries=0)
def run_circuit_record(self, record_run_id: str,
                       config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate + record steered transcripts. WS on the "steering-record"
    channel (run_id = record_run_id)."""
    from ..services.steering_recorder_service import SteeringRecorderService

    with self.get_db() as db:
        _set_status(db, record_run_id, "running")
        try:
            result = SteeringRecorderService.record_samples(
                db, config,
                progress_cb=lambda pct: emit_circuit_run_progress(
                    "steering-record", record_run_id, pct))
            _complete(db, record_run_id, result.get("manifest_ref"))
            emit_circuit_run_completed("steering-record", record_run_id,
                                       summary=result)
            return result
        except Exception as e:
            logger.exception("Steering record %s failed", record_run_id)
            _set_status(db, record_run_id, "failed", error=str(e)[:500])
            emit_circuit_run_failed("steering-record", record_run_id, str(e)[:500])
            raise


def _set_status(db, record_run_id, status, error=None):
    from ..models.steering_record_run import SteeringRecordRun
    # Roll back first: a DB-error failure leaves the session aborted, and the
    # status write below would itself raise, leaving the marker set and wedging
    # the single-GPU guard (Feature 20 R2 lesson).
    try:
        db.rollback()
    except Exception:
        logger.exception("Rollback before record status write failed for %s",
                         record_run_id)
    try:
        row = db.query(SteeringRecordRun).filter(
            SteeringRecordRun.id == record_run_id).first()
        if row is not None:
            row.status = status
            if error is not None:
                row.error = error
            db.commit()
    except Exception:
        logger.exception("Could not set record status for %s", record_run_id)


def _complete(db, record_run_id, manifest_ref):
    from ..models.steering_record_run import SteeringRecordRun
    # record_samples committed the manifest, so the session is clean here — but
    # roll back defensively so a lingering aborted state can't block the status
    # write (a completed job whose marker stays 'running' would wedge the GPU
    # guard until cleanup; R1).
    try:
        db.rollback()
    except Exception:
        logger.exception("Rollback before record completion failed for %s",
                         record_run_id)
    try:
        row = db.query(SteeringRecordRun).filter(
            SteeringRecordRun.id == record_run_id).first()
        if row is not None:
            row.status = "completed"
            row.manifest_ref = manifest_ref
            db.commit()
    except Exception:
        logger.exception("Could not complete record run %s", record_run_id)
