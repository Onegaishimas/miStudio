"""
Periodic cleanup for stuck circuit capture/discovery/attribution runs
(Feature 016, R2 Q3).

Without this, an OOM-killed or pod-restarted capture leaves its row in
'running' forever — and `assert_no_active_gpu_run` then rejects EVERY future
capture with a 409 (a permanent lockout). Mirrors cleanup_stuck_extractions:
if a run has had no update past a threshold AND its Celery task is no longer
active, mark it failed and rmtree any partial store.
"""

import logging
import shutil
from datetime import datetime, timedelta, timezone

from src.core.celery_app import celery_app
from src.models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun
from src.workers.base_task import DatabaseTask

logger = logging.getLogger(__name__)

STUCK_THRESHOLD_MINUTES = 60  # no update in an hour + task not active → stuck
_ACTIVE_CELERY = {"PENDING", "STARTED", "RETRY", "RECEIVED"}


def _task_is_active(task_id):
    if not task_id:
        return False
    try:
        return celery_app.AsyncResult(task_id).state in _ACTIVE_CELERY
    except Exception:  # broker hiccup — treat as active, don't false-kill
        return True


@celery_app.task(bind=True, base=DatabaseTask, name="cleanup_stuck_circuit_runs")
def cleanup_stuck_circuit_runs_task(self):
    """Fail circuit runs stuck past the threshold with no active task."""
    threshold = datetime.now(timezone.utc) - timedelta(
        minutes=STUCK_THRESHOLD_MINUTES)
    cleaned = 0
    with self.get_db() as db:
        # ── captures ──
        for run in db.query(CircuitCaptureRun).filter(
                CircuitCaptureRun.status.in_(("pending", "estimating", "running")),
                CircuitCaptureRun.updated_at < threshold).all():
            if _task_is_active(run.celery_task_id):
                continue
            run.status = "failed"
            run.error_message = "Stuck run reclaimed by cleanup (worker died?)"
            if run.store_path:
                try:
                    from src.core.config import settings
                    p = settings.resolve_data_path(run.store_path)
                    if p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                except Exception:
                    logger.exception("rmtree failed for %s", run.id)
            cleaned += 1
        # ── discovery + attribution lifecycles ──
        for run in db.query(CircuitDiscoveryRun).filter(
                CircuitDiscoveryRun.updated_at < threshold).all():
            if run.status in ("pending", "running") and not _task_is_active(
                    run.celery_task_id):
                run.status = "failed"
                run.error_message = "Stuck discovery reclaimed by cleanup"
                cleaned += 1
            if run.attribution_status in ("pending", "running") and \
                    not _task_is_active(run.attribution_task_id):
                run.attribution_status = "failed"
                run.attribution_error = "Stuck attribution reclaimed by cleanup"
                cleaned += 1
            if run.validation_status in ("pending", "running") and \
                    not _task_is_active(run.validation_task_id):
                run.validation_status = "failed"
                run.validation_error = "Stuck validation reclaimed by cleanup"
                cleaned += 1
        # ── faithfulness (runs on a circuit, not a discovery run) — R2 B-5 ──
        from src.models.circuit import Circuit
        for circuit in db.query(Circuit).filter(
                Circuit.faithfulness_status.in_(("pending", "running")),
                Circuit.updated_at < threshold).all():
            if not _task_is_active(circuit.faithfulness_task_id):
                circuit.faithfulness_status = "failed"
                cleaned += 1
        # ── calibration (also runs on a circuit + holds the GPU) — Feature 20 ──
        # A crashed calibration would otherwise wedge the single-GPU guard for
        # every circuit task, since assert_no_active_gpu_run checks this status.
        for circuit in db.query(Circuit).filter(
                Circuit.calibration_status.in_(("pending", "running")),
                Circuit.updated_at < threshold).all():
            if not _task_is_active(circuit.calibration_task_id):
                circuit.calibration_status = "failed"
                cleaned += 1
        if cleaned:
            db.commit()
            logger.info("Reclaimed %d stuck circuit run(s)", cleaned)
    return {"cleaned": cleaned}
