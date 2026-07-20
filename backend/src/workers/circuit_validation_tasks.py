"""
Celery tasks for circuit edge validation + faithfulness (Feature 017).

GPU profile → extraction queue (same as capture/attribution). Separate
lifecycle on the discovery run's validation_* columns (a failed pass never
corrupts the completed discovery). Cancellation via DB-status polling.
"""

import logging
from typing import Any, Dict

from ..core.celery_app import celery_app
from .base_task import DatabaseTask
from .circuit_capture_tasks import _cancel_checker
from .websocket_emitter import (
    emit_circuit_run_completed,
    emit_circuit_run_failed,
    emit_circuit_run_progress,
)

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, base=DatabaseTask,
                 name="src.workers.circuit_validation_tasks.validate_circuit_edges",
                 max_retries=0)
def validate_circuit_edges(self, run_id: str, scope: Dict[str, Any]) -> Dict[str, Any]:
    from ..models.circuit_runs import CircuitDiscoveryRun
    from ..services.circuit_intervention_service import CircuitInterventionService

    with self.get_db() as db:
        try:
            result = CircuitInterventionService.run(
                db, run_id, scope,
                cancel_check=_cancel_checker(db, CircuitDiscoveryRun, run_id,
                                             status_field="validation_status"),
                progress_cb=lambda pct: emit_circuit_run_progress(
                    "validation", run_id, pct))
            emit_circuit_run_completed("validation", run_id, summary=result)
            return result
        except Exception as e:
            logger.exception("Circuit validation %s failed", run_id)
            run = db.query(CircuitDiscoveryRun).filter(
                CircuitDiscoveryRun.id == run_id).first()
            if run is not None:
                run.validation_status = "failed"
                run.validation_error = str(e)[:2000]
                db.commit()
            emit_circuit_run_failed("validation", run_id, str(e)[:500])
            raise
