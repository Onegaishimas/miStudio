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
ACQUIRE = "jlens_acquire"
PUBLISH = "jlens_publish"

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
) -> bool:
    """Move a queued row along. Located by CELERY task id, not by row id.

    By task id because the worker knows that and would otherwise have to be
    handed the row id through the task signature — one more argument to forget,
    and forgetting it silently leaves a row stuck at "queued" forever.

    RETURNS WHETHER A ROW WAS FOUND. The endpoints open the row AFTER `.delay()`,
    so a task that fails in its first milliseconds can arrive here before the row
    exists — and a silent `return` then leaves it at "queued 0%" forever, which
    reads as a job that never started rather than one that failed instantly. A
    caller that can retry needs to know the difference; one that cannot is
    unaffected, since the value is simply ignored.
    """
    from ..core.database import get_sync_db
    from ..models.task_queue import TaskQueue

    try:
        with get_sync_db() as db:
            row = db.query(TaskQueue).filter(TaskQueue.task_id == task_id).first()
            if row is None:
                return False
            # A TERMINAL ROW NEVER GOES BACK TO RUNNING. Cancellation is
            # cooperative here — the endpoint writes "cancelled" and the task
            # notices at its next checkpoint — so between those two moments the
            # task is still reporting progress. Without this guard its next
            # `update_row(status="running")` overwrites the cancellation and the
            # request is silently lost, which is worse than having no cancel at
            # all: the operator is told it worked.
            #
            # Also covers the janitor: `cleanup_orphaned_tasks` marks an
            # abandoned row "failed", and a straggling heartbeat from a worker
            # on its way out would otherwise revive it.
            if row.status in TERMINAL_STATUSES and status not in (None,) + TERMINAL_STATUSES:
                logger.info(
                    "Ignoring %s update for %s: row is already %s",
                    status, task_id, row.status,
                )
                return False
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
            return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not update task_queue row for %s: %s", task_id, exc)
    return False



def request_cancel(task_id: str, reason: str = "cancelled by operator") -> bool:
    """Ask a running J-space task to stop at its next checkpoint.

    COOPERATIVE BY NECESSITY, not by preference. The GPU worker runs
    `--pool=solo` because CUDA and fork do not mix (celery_app.py), and Celery's
    `revoke(terminate=True)` only signals a POOL CHILD — solo has none. Worse,
    a solo worker executing a task is not reading the control queue at all, so
    the revoke is never even delivered: it returns cleanly and does nothing, and
    the worker does not appear in `inspect()`. Verified on hardware 2026-09-05
    against a running gemma-4-12B fit.

    So the row IS the channel. The task polls it and stops itself.
    """
    from ..core.database import get_sync_db
    from ..models.task_queue import TaskQueue

    try:
        with get_sync_db() as db:
            row = db.query(TaskQueue).filter(TaskQueue.task_id == task_id).first()
            if row is None:
                return False
            if row.status in TERMINAL_STATUSES:
                return False
            row.status = "cancelled"
            row.error_message = reason[:2000]
            if row.completed_at is None:
                row.completed_at = datetime.now(timezone.utc)
            db.commit()
            return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not request cancel for %s: %s", task_id, exc)
        return False


def cancel_checker(task_id: str, every: int = 1):
    """A callable the work loop polls; True once cancellation is requested.

    Throttled like `circuit_capture_tasks._cancel_checker`, and INCLUDING the
    first call — polling every Nth from zero would skip the opening checks, so a
    task cancelled immediately would run to its Nth checkpoint before noticing.
    """
    state = {"count": 0}

    def check() -> bool:
        should_poll = state["count"] % max(every, 1) == 0
        state["count"] += 1
        if not should_poll:
            return False
        from ..core.database import get_sync_db
        from ..models.task_queue import TaskQueue

        try:
            with get_sync_db() as db:
                row = (
                    db.query(TaskQueue)
                    .filter(TaskQueue.task_id == task_id)
                    .first()
                )
                return row is not None and row.status == "cancelled"
        except Exception as exc:  # noqa: BLE001 - a failed poll must not kill the work
            logger.warning("Cancel poll failed for %s: %s", task_id, exc)
            return False

    return check


class TaskCancelled(Exception):
    """Raised inside a J-space task when its row has been set to cancelled."""


def mark_running(task_id: str, progress: float = 1.0, attempts: int = 10) -> bool:
    """First transition to `running`, RETRIED past the row's own creation.

    THE ROW MAY NOT EXIST YET. Every J-space endpoint opens it AFTER `.delay()`,
    so a worker that picks the task up immediately can arrive before the insert
    commits — `update_row` then finds nothing, returns, and the row lands as
    "queued 0%" and never moves. That reads as a job that never started rather
    than one already running.

    FOR THE FIRST-AND-ONLY EARLY TRANSITION. Acquire and publish mark themselves
    running once and then do long work, so a missed write is permanent. The fit,
    band and intervention tasks write `running` from a REPEATING callback — one
    per prompt or per layer — so a first write that loses the race is corrected
    milliseconds later by the next, and putting a sleeping retry inside a hot
    callback would cost more than the race does. They deliberately keep
    `update_row`.
    """
    import time as _time

    for _attempt in range(max(1, attempts)):
        if update_row(task_id, status="running", progress=progress):
            return True
        _time.sleep(0.1)
    logger.warning(
        "No task_queue row for %s after %d attempts; it will not show progress",
        task_id,
        attempts,
    )
    return False


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


#: Attribute stamped on a wrapped task so the guard can ASK rather than guess.
#:
#: The first version of that guard scraped the source with a regex allowing at
#: most one decorator between `@celery_app.task(...)` and `def`. A task carrying
#: any second decorator matched nothing at all, so it never entered the list the
#: assertion checked — the scan failed OPEN, and an undecorated task would have
#: shipped green. `functools.wraps` copies `__dict__`, so this marker survives
#: any number of further wrappers.
OWNERSHIP_MARKER = "__jlens_owns_its_failure__"


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

    setattr(wrapper, OWNERSHIP_MARKER, True)
    return wrapper
