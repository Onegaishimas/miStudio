"""Celery task for circuit strength calibration (Feature 20 / IDL-37).

Calibration runs ON a circuit and holds the GPU like faithfulness, so it shares
the same lifecycle: a `calibration_status`/`_task_id` pair on the circuit row
(seen by the single-GPU guard + reclaimed by cleanup), a `calibration` manifest
as the reproducible record, and the band written onto the circuit through the
contract (apply_calibration → update() → validators + version bump).

GPU profile → extraction queue (same as capture/faithfulness). The single-GPU
guard is asserted at the endpoint (advisory lock) before dispatch.
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
                 name="src.workers.circuit_calibration_tasks.run_circuit_calibration",
                 max_retries=0)
def run_circuit_calibration(self, circuit_id: str,
                            config: Dict[str, Any]) -> Dict[str, Any]:
    """Find the usable band and clamp the served dial to it. WS on the
    "calibration" channel (run_id = circuit_id)."""
    from ..services.circuit_calibration_service import CircuitCalibrationService

    with self.get_db() as db:
        _set_status(db, circuit_id, "running")
        try:
            result = CircuitCalibrationService.run(
                db, circuit_id, config,
                progress_cb=lambda pct: emit_circuit_run_progress(
                    "calibration", circuit_id, pct))
            emit_circuit_run_completed("calibration", circuit_id, summary=result)
            return result
        except Exception as e:
            logger.exception("Circuit calibration %s failed", circuit_id)
            # Clear the in-flight marker so the single-GPU guard isn't wedged.
            _set_status(db, circuit_id, "failed")
            emit_circuit_run_failed("calibration", circuit_id, str(e)[:500])
            raise


@celery_app.task(bind=True, base=DatabaseTask,
                 name="src.workers.circuit_calibration_tasks.reproduce_circuit_calibration",
                 max_retries=0)
def reproduce_circuit_calibration(self, manifest_id: str) -> Dict[str, Any]:
    """Re-run a calibration from a stored manifest and record a reproduction
    manifest with the band-delta verdict. Holds the GPU like a fresh run."""
    from ..services.circuit_calibration_service import CircuitCalibrationService

    with self.get_db() as db:
        circuit_id = _circuit_of_manifest(db, manifest_id)
        try:
            result = CircuitCalibrationService.reproduce(
                db, manifest_id,
                progress_cb=lambda pct: emit_circuit_run_progress(
                    "calibration-reproduce", manifest_id, pct))
            # Reproduce marked the circuit in-flight (endpoint) — clear it.
            if circuit_id:
                _set_status(db, circuit_id, "completed")
            emit_circuit_run_completed("calibration-reproduce", manifest_id,
                                       summary=result)
            return result
        except Exception as e:
            logger.exception("Calibration reproduce %s failed", manifest_id)
            if circuit_id:
                _set_status(db, circuit_id, "failed")
            emit_circuit_run_failed("calibration-reproduce", manifest_id,
                                    str(e)[:500])
            raise


def _circuit_of_manifest(db, manifest_id):
    from ..models.validation_manifest import ValidationManifest
    try:
        db.rollback()
    except Exception:
        pass
    m = db.query(ValidationManifest).filter(
        ValidationManifest.id == manifest_id).first()
    return m.circuit_id if m is not None else None


def _set_status(db, circuit_id, status):
    from ..models.circuit import Circuit
    # If the run failed on a DB error, `db` is in an aborted transaction and any
    # query on it raises "current transaction is aborted". Roll back FIRST so the
    # status write lands — otherwise the in-flight marker is never cleared and the
    # single-GPU guard stays wedged until the 60-min cleanup.
    try:
        db.rollback()
    except Exception:
        logger.exception("Rollback before calibration_status write failed for %s",
                         circuit_id)
    try:
        row = db.query(Circuit).filter(Circuit.id == circuit_id).first()
        if row is not None:
            row.calibration_status = status
            db.commit()
    except Exception:
        logger.exception("Could not set calibration_status for %s", circuit_id)
