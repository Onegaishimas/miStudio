"""
Task-queue rows for J-space work, so long jobs are VISIBLE while they run.

WHY THIS EXISTS. A 45-minute fit burned the GPU with nothing anywhere in the
product saying so. The J-Lens panel's own fit card only knows about a fit THIS
browser tab started — its polling lives in component state, so a fit queued from
the API, from MCP, from another tab, or before a refresh was invisible. The
System Monitor's Active Operations panel reads `task_queue`, and J-space tasks
were never writing rows to it.

This repo has the same defect on record already: finalize and prune create no
task_queue row, so they do not appear in Active Operations either. The fix is
the same shape — write the row where the task is QUEUED, update it where the
task reports progress.

SYNC SESSIONS ON PURPOSE. `TaskQueueService` is async and Celery workers are
not; the workers that already write these rows (model_tasks, dataset_tasks) use
a sync session directly, and this follows them rather than introducing an event
loop into a worker.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

#: `task_type` values. Prefixed so a reader of Active Operations can tell J-space
#: work apart from training and extraction at a glance.
FIT = "jlens_fit"
BAND_REPORT = "jlens_band_report"
INTERVENTION = "jlens_intervention"
READOUT = "jlens_readout"
PROBE = "jlens_probe"

#: Statuses after which a task will never report again. `completed_at` is
#: stamped on entering any of them, so a finished row carries a real duration
#: rather than an open-ended one.
TERMINAL_STATUSES = ("completed", "failed", "cancelled")


def open_row(task_type: str, entity_id: str, task_id: str) -> Optional[str]:
    """Record that a J-space task has been queued. Returns the row id.

    NEVER RAISES. A bookkeeping row failing to write must not fail the fit it
    describes — the work is the point and the row is the narration. Failures
    are logged so a missing row is diagnosable rather than mysterious.
    """
    from ..core.database import get_sync_db
    from ..models.task_queue import TaskQueue

    row_id = f"tq_{uuid.uuid4().hex[:12]}"
    try:
        with get_sync_db() as db:
            db.add(
                TaskQueue(
                    id=row_id,
                    task_id=task_id,
                    task_type=task_type,
                    entity_id=entity_id,
                    entity_type="model",
                    status="queued",
                    progress=0.0,
                    retry_params={},
                    retry_count=0,
                )
            )
            db.commit()
        return row_id
    except Exception as exc:  # noqa: BLE001 - narration must not break the work
        logger.warning("Could not open a task_queue row for %s: %s", task_id, exc)
        return None


def update_row(
    task_id: str,
    status: Optional[str] = None,
    progress: Optional[float] = None,
    error_message: Optional[str] = None,
) -> None:
    """Move a queued row along. Located by CELERY task id, not by row id.

    By task id because the worker knows that and would otherwise have to be
    handed the row id through the task signature — one more argument to forget,
    and forgetting it silently leaves a row stuck at "queued" forever.
    """
    from ..core.database import get_sync_db
    from ..models.task_queue import TaskQueue

    try:
        with get_sync_db() as db:
            row = db.query(TaskQueue).filter(TaskQueue.task_id == task_id).first()
            if row is None:
                return
            if status is not None:
                # STAMP THE CLOCK ON THE TRANSITIONS. Both columns existed and
                # neither was ever written for J-space work, so every J-lens row
                # carried started_at=None and completed_at=None. Any elapsed
                # time a reader derived had to come from `created_at`, which is
                # QUEUE time: an LFM2 fit that waited three hours behind gemma
                # would have reported a four-hour fit after one hour of work.
                if status == "running" and row.started_at is None:
                    row.started_at = datetime.now(timezone.utc)
                if status in TERMINAL_STATUSES and row.completed_at is None:
                    row.completed_at = datetime.now(timezone.utc)
                row.status = status
            if progress is not None:
                # Clamped: a progress bar past 100% reads as a bug in the bar
                # rather than in whatever produced the number.
                row.progress = max(0.0, min(100.0, float(progress)))
            if error_message is not None:
                row.error_message = error_message[:2000]
            db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not update task_queue row for %s: %s", task_id, exc)


def fail_row(task_id: str, exc: BaseException) -> None:
    """Record a task's OWN failure, with its OWN reason.

    THE TASK OWNS ITS TERMINAL STATE. Leaving this to the orphan janitor costs
    three things: up to five minutes of "queued 0%" on an idle GPU while the
    sweep waits for its next beat; the real reason — "unknown primitive
    'aditive'", "swap partner is 2 tokens", the near-parallel refusal — replaced
    by the janitor's prose about the BOOKKEEPING defect, which tells the caller
    nothing about their request; and a blind spot for anything that fails AFTER
    its first progress report, which the sweep's `looks_abandoned` rule
    deliberately never closes because a terminal Celery state is not an orphan.

    The janitor remains the backstop for a worker that dies without running any
    Python at all — an eviction, an OOM kill, a pod roll.
    """
    update_row(task_id, status="failed", error_message=f"{type(exc).__name__}: {exc}")


def owns_its_failure(fn):
    """Decorator: a J-space task records its own failure before re-raising.

    Applied at the task rather than duplicated in five bodies, so a task added
    later inherits it by construction instead of by anyone remembering. The
    exception still propagates — Celery must see the FAILURE, and swallowing it
    here would trade one silent state for another.
    """
    import functools

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
            request_id = getattr(getattr(self, "request", None), "id", None)
            if request_id:
                fail_row(request_id, exc)
            raise

    return wrapper
