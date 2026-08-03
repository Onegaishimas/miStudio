"""
Close task_queue rows whose worker stopped reporting.

WHY THIS EXISTS. A row is written when a task is QUEUED and moved by the task
ITSELF. A worker that dies — a pod roll, an eviction, an OOM — writes nothing,
so the row keeps its last progress forever. A J-lens fit killed by a deploy sat
at "running 21.5%" in Active Operations for hours while the GPU was idle at 0%,
and nothing in the product could tell the user it was dead.

Read-time reconciliation already makes the LISTING honest. This makes the STATE
terminal, which is what actually clears the row and what a janitor is for: a GET
must not have side effects, so the write belongs here.

DELIBERATELY NOT A RETRY. A fit that died lost every prompt it had processed —
the accumulator lives in worker memory — so re-running it automatically would
take the GPU for another hour without being asked. The row is marked failed with
a reason the user can act on, and re-running is their call.
"""

from __future__ import annotations

import logging

from ..core.celery_app import celery_app
from .base_task import DatabaseTask

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, base=DatabaseTask, name="cleanup_orphaned_tasks")
def cleanup_orphaned_tasks_task(self):
    """Mark rows whose Celery task stopped beating as failed.

    Only rows claiming to be RUNNING are considered. A queued task has not
    started and legitimately has no heartbeat — condemning those would fail
    everything waiting behind a long job, which on a single-GPU queue is the
    normal state of the world.
    """
    from celery.result import AsyncResult

    from ..core.database import get_sync_db
    from ..models.task_queue import TaskQueue
    from .task_heartbeat import STALE_AFTER_SECONDS, looks_orphaned

    closed = 0
    with get_sync_db() as db:
        rows = (
            db.query(TaskQueue)
            .filter(TaskQueue.status == "running")
            .filter(TaskQueue.task_id.isnot(None))
            .all()
        )
        for row in rows:
            try:
                result = AsyncResult(row.task_id, app=celery_app)
                if not looks_orphaned(result.state, result.info):
                    continue
            except Exception as exc:  # noqa: BLE001 - one bad row must not stop the sweep
                logger.warning("Could not check %s: %s", row.task_id, exc)
                continue

            row.status = "failed"
            row.error_message = (
                "the worker running this stopped reporting for more than "
                f"{STALE_AFTER_SECONDS // 60} minutes and is presumed gone "
                "(a deploy, an eviction or a crash). Any work it completed was "
                "not saved. Re-run it."
            )
            closed += 1
        if closed:
            db.commit()

    if closed:
        logger.info("Closed %d orphaned task_queue row(s)", closed)
    return {"closed": closed}
