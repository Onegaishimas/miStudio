"""
Celery tasks for circuit capture + discovery + attribution (Feature 016).

Routing: extraction queue (GPU profile — same as activation/feature
extraction; the steering worker's busy-marker machinery is steering-specific
and deliberately NOT used here). Cancellation: DB-status polling between
batches (house pattern — training_tasks precedent).
"""

import logging
from typing import Any, Dict, Optional

from ..core.celery_app import celery_app
from .base_task import DatabaseTask
from .websocket_emitter import (
    emit_circuit_run_completed,
    emit_circuit_run_failed,
    emit_circuit_run_progress,
)

logger = logging.getLogger(__name__)


def _cancel_checker(db, model_cls, run_id):
    """Throttled DB-status poll — returns a callable for the service loop."""
    state = {"count": 0}

    def check() -> bool:
        state["count"] += 1
        if state["count"] % 5:
            return False
        row = db.query(model_cls).filter(model_cls.id == run_id).first()
        if row is not None:
            db.refresh(row)
            return row.status == "cancelled"
        return False

    return check


@celery_app.task(bind=True, base=DatabaseTask,
                 name="src.workers.circuit_capture_tasks.capture_circuit_activations",
                 max_retries=0)
def capture_circuit_activations(self, run_id: str, confirmed: bool = False) -> Dict[str, Any]:
    """Probe (estimate) and, when confirmed, full capture for one run."""
    from ..models.circuit_runs import CircuitCaptureRun
    from ..services.circuit_capture_service import CircuitCaptureService

    with self.get_db() as db:
        try:
            result = CircuitCaptureService.run_capture(
                db, run_id, confirmed=confirmed,
                cancel_check=_cancel_checker(db, CircuitCaptureRun, run_id),
                progress_cb=lambda pct: emit_circuit_run_progress(
                    "capture", run_id, pct),
            )
            emit_circuit_run_completed("capture", run_id, summary=result)
            return result
        except Exception as e:
            logger.exception("Circuit capture %s failed", run_id)
            run = db.query(CircuitCaptureRun).filter(
                CircuitCaptureRun.id == run_id).first()
            if run is not None:
                run.status = "failed"
                run.error_message = str(e)[:2000]
                db.commit()
            emit_circuit_run_failed("capture", run_id, str(e)[:500])
            raise


@celery_app.task(bind=True, base=DatabaseTask,
                 name="src.workers.circuit_capture_tasks.run_circuit_discovery",
                 max_retries=0)
def run_circuit_discovery(self, run_id: str) -> Dict[str, Any]:
    """Statistical mining over a completed capture store (CPU-heavy, no GPU)."""
    from ..models.circuit_runs import CircuitDiscoveryRun
    from ..services.circuit_discovery_service import CircuitDiscoveryService

    with self.get_db() as db:
        try:
            result = CircuitDiscoveryService.run(
                db, run_id,
                cancel_check=_cancel_checker(db, CircuitDiscoveryRun, run_id),
                progress_cb=lambda pct: emit_circuit_run_progress(
                    "discovery", run_id, pct),
            )
            emit_circuit_run_completed("discovery", run_id, summary=result)
            return result
        except Exception as e:
            logger.exception("Circuit discovery %s failed", run_id)
            run = db.query(CircuitDiscoveryRun).filter(
                CircuitDiscoveryRun.id == run_id).first()
            if run is not None:
                run.status = "failed"
                run.error_message = str(e)[:2000]
                db.commit()
            emit_circuit_run_failed("discovery", run_id, str(e)[:500])
            raise


@celery_app.task(bind=True, base=DatabaseTask,
                 name="src.workers.circuit_capture_tasks.run_circuit_attribution",
                 max_retries=0)
def run_circuit_attribution(self, run_id: str,
                            prompt_limit: Optional[int] = None) -> Dict[str, Any]:
    """Tier-2 gradient attribution pass over a discovery run's candidates (GPU)."""
    from ..models.circuit_runs import CircuitDiscoveryRun
    from ..services.circuit_attribution_service import CircuitAttributionService

    with self.get_db() as db:
        try:
            result = CircuitAttributionService.run(
                db, run_id, prompt_limit=prompt_limit,
                cancel_check=_cancel_checker(db, CircuitDiscoveryRun, run_id),
                progress_cb=lambda pct: emit_circuit_run_progress(
                    "attribution", run_id, pct),
            )
            emit_circuit_run_completed("attribution", run_id, summary=result)
            return result
        except Exception as e:
            logger.exception("Circuit attribution %s failed", run_id)
            run = db.query(CircuitDiscoveryRun).filter(
                CircuitDiscoveryRun.id == run_id).first()
            if run is not None:
                run.status = "failed"
                run.error_message = str(e)[:2000]
                db.commit()
            emit_circuit_run_failed("attribution", run_id, str(e)[:500])
            raise
