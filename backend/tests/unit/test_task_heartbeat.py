"""
A task whose worker died must stop claiming to be in progress.

OBSERVED, NOT IMAGINED. A band report was killed when a deploy rolled the
backend pod. Celery keeps whatever the task last reported and nothing writes a
terminal state when a worker vanishes, so `/models/tasks/{id}` returned
PROGRESS for forty minutes afterwards — and three separate status checks
reported the job as healthy.

MUTATION CONTROLS (each must turn this file red):
  * beat() omits the timestamp                  -> "stamps" fails
  * looks_orphaned ignores the threshold        -> "only when stale" fails
  * a task with NO heartbeat reads as orphaned  -> "never beat" fails
  * a terminal state is called orphaned         -> "terminal" fails
"""

import time

import pytest

from src.workers.task_heartbeat import (
    STALE_AFTER_SECONDS,
    beat,
    looks_orphaned,
    seconds_since_beat,
)


class TestTheBeat:
    def test_it_stamps_the_clock_alongside_the_caller_s_meta(self):
        meta = beat({"stage": "profiling", "prompt": 3})
        assert meta["stage"] == "profiling"
        assert meta["prompt"] == 3
        assert isinstance(meta["heartbeat"], float)
        assert abs(meta["heartbeat"] - time.time()) < 5

    def test_it_works_with_no_caller_meta_at_all(self):
        assert "heartbeat" in beat()


class TestStaleness:
    def test_a_fresh_beat_is_not_orphaned(self):
        assert looks_orphaned("PROGRESS", beat({"stage": "x"})) is False

    def test_a_task_that_stopped_reporting_IS_orphaned(self):
        old = {"stage": "profiling", "heartbeat": time.time() - STALE_AFTER_SECONDS - 60}
        assert looks_orphaned("PROGRESS", old) is True

    def test_it_is_orphaned_only_once_PAST_the_threshold(self):
        """A slow task must not be declared dead.

        A false 'dead' is worse than a slow truth: it sends someone to re-run a
        job that was going to finish.
        """
        just_inside = {"heartbeat": time.time() - (STALE_AFTER_SECONDS - 30)}
        assert looks_orphaned("PROGRESS", just_inside) is False

    def test_a_task_that_NEVER_beat_is_not_called_orphaned(self):
        """No heartbeat is not the same as a stale one.

        Short tasks never beat at all, and tasks predating this have none.
        Reporting those as dead would condemn most of the queue.
        """
        assert looks_orphaned("PROGRESS", {"stage": "loading_model"}) is False
        assert looks_orphaned("PROGRESS", None) is False
        assert seconds_since_beat({"stage": "x"}) is None

    @pytest.mark.parametrize("state", ["SUCCESS", "FAILURE", "PENDING", "RETRY"])
    def test_a_terminal_or_unstarted_state_is_never_orphaned(self, state):
        """A report finished last week is not orphaned, it is done."""
        ancient = {"heartbeat": time.time() - 86400}
        assert looks_orphaned(state, ancient) is False


class TestTheStatusEndpointActuallyRuns:
    """The endpoint that READS the heartbeat had no test, and I broke it.

    `from ...workers.task_heartbeat import ...` resolves to `src.api.workers`,
    one package short of `src.workers`. Because the import sits INSIDE the
    handler it does not fail at module import, so nothing noticed: the whole
    backend suite passed, CI went green, the image shipped, and every call to
    /models/tasks/{id} returned 500 — which is how a band report that had
    finished successfully read as unreachable.

    MUTATION CONTROLS:
      * restore the ...workers depth  -> "resolves its imports" fails
      * drop the orphan branch        -> "reports orphaned" fails
    """

    def _status(self, state, info):
        import asyncio
        from unittest.mock import MagicMock, patch

        from src.api.v1.endpoints.models import get_task_status

        result = MagicMock()
        result.state = state
        result.info = info
        result.ready.return_value = state in ("SUCCESS", "FAILURE")
        result.successful.return_value = state == "SUCCESS"
        result.failed.return_value = state == "FAILURE"
        result.result = {}

        with patch("celery.result.AsyncResult", return_value=result):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(get_task_status("t-1"))
            finally:
                loop.close()

    def test_it_resolves_its_imports_and_returns(self):
        """Exercises the handler BODY. A module-import check would not have
        caught this — the bad import is inside the function."""
        body = self._status("PROGRESS", beat({"stage": "profiling"}))
        assert body["task_id"] == "t-1"
        assert body["state"] == "PROGRESS"
        assert body["seconds_since_heartbeat"] is not None

    def test_it_reports_orphaned_when_the_beat_goes_stale(self):
        stale = {"stage": "profiling", "heartbeat": time.time() - STALE_AFTER_SECONDS - 60}
        body = self._status("PROGRESS", stale)
        assert body["state"] == "ORPHANED"
        assert body["ready"] is True
        assert "stopped reporting" in body["error"]

    def test_a_healthy_task_is_untouched(self):
        body = self._status("SUCCESS", {"heartbeat": time.time() - 99999})
        assert body["state"] == "SUCCESS"
        assert "error" not in body


class TestActiveOperationsAgreesWithTheHeartbeat:
    """The two surfaces must not disagree about what is running.

    OBSERVED: a fit killed by a pod roll showed "running 21.5%" in Active
    Operations while /models/tasks/{id} correctly reported ORPHANED. The row is
    written when the task is queued and moved by the task itself — so a worker
    that dies writes nothing and the row sits at its last progress forever.

    MUTATION CONTROLS:
      * drop the reconciliation        -> "reports orphaned" fails
      * reconcile QUEUED rows too      -> "queued is not orphaned" fails
    """

    def _row(self, status, state, info):
        from unittest.mock import MagicMock, patch

        from src.api.v1.endpoints.task_queue import _looks_orphaned

        task = MagicMock()
        task.status = status
        task.task_id = "t-1"

        result = MagicMock()
        result.state = state
        result.info = info
        with patch("celery.result.AsyncResult", return_value=result):
            return _looks_orphaned(task)

    def test_it_reports_a_running_row_whose_task_stopped_beating(self):
        stale = {"stage": "fitting", "heartbeat": time.time() - STALE_AFTER_SECONDS - 60}
        assert self._row("running", "PROGRESS", stale) is True

    def test_a_live_row_is_left_alone(self):
        assert self._row("running", "PROGRESS", beat({"stage": "fitting"})) is False

    def test_a_QUEUED_row_is_never_orphaned(self):
        """A task waiting behind a long job has not started and cannot beat.

        Condemning it would mark everything in the queue as dead.
        """
        stale = {"heartbeat": time.time() - STALE_AFTER_SECONDS - 60}
        assert self._row("queued", "PENDING", stale) is False
        assert self._row("queued", "PROGRESS", stale) is False

    def test_a_row_with_no_task_id_is_never_orphaned(self):
        """Federated rows from other job tables carry no Celery id."""
        from unittest.mock import MagicMock

        from src.api.v1.endpoints.task_queue import _looks_orphaned

        task = MagicMock()
        task.status = "running"
        task.task_id = None
        assert _looks_orphaned(task) is False


class TestTheJanitorClosesGhostRows:
    """Read-time reconciliation makes the LISTING honest; this makes it STOP.

    A GET must not have side effects, so the write belongs in a janitor. Without
    one the row stays "running" in the database forever and every consumer that
    reads it directly — not just the reconciled listing — keeps believing it.

    MUTATION CONTROLS:
      * janitor also closes QUEUED rows      -> "only running" fails
      * janitor RETRIES instead of failing   -> "does not resubmit" fails
      * task not registered / wrong queue    -> "reachable" fails
    """

    def test_the_janitor_is_registered_and_routed_to_a_real_queue(self):
        from src.core.celery_app import celery_app

        assert "cleanup_orphaned_tasks" in celery_app.tasks, (
            "the janitor is not in the live registry, so beat would schedule a "
            "task no worker can execute"
        )
        # Short name + explicit queue: a module-path glob never matches these,
        # which this file's own celery_app comment records as a shipped defect.
        assert celery_app.conf.task_routes["cleanup_orphaned_tasks"]["queue"] == (
            "low_priority"
        )
        entry = celery_app.conf.beat_schedule["cleanup-orphaned-tasks"]
        assert entry["task"] == "cleanup_orphaned_tasks"
        assert entry["schedule"] <= STALE_AFTER_SECONDS, (
            "the sweep must run at least as often as the staleness threshold, "
            "or a ghost row stays visible for the sum of both"
        )

    def test_it_closes_only_rows_that_claim_to_be_RUNNING(self):
        """A queued task has not started and cannot beat.

        On a single-GPU queue everything waits behind a long job, so condemning
        queued rows would fail the entire backlog.

        THE FILTER IS INSPECTED, not assumed. An earlier version of this test
        used a fake that returned the same row whatever it was asked, so
        deleting the status filter from the janitor changed nothing observable
        and the mutation survived.
        """
        import src.workers.cleanup_orphaned_tasks as janitor
        from contextlib import contextmanager
        from unittest.mock import MagicMock, patch

        running = MagicMock(status="running", task_id="t-dead", error_message=None)
        criteria = []

        class _Q:
            def query(self, _m):
                return self

            def filter(self, *args, **_kw):
                criteria.extend(str(a) for a in args)
                return self

            def all(self):
                return [running]

            def commit(self):
                pass

        @contextmanager
        def fake_db():
            yield _Q()

        dead = MagicMock()
        dead.state = "PROGRESS"
        dead.info = {"heartbeat": time.time() - STALE_AFTER_SECONDS - 60}

        with patch("src.core.database.get_sync_db", fake_db), patch(
            "celery.result.AsyncResult", return_value=dead
        ):
            out = janitor.cleanup_orphaned_tasks_task.run()

        assert out["closed"] == 1
        assert running.status == "failed"
        assert "presumed gone" in running.error_message

        joined = " ".join(criteria)
        assert "status" in joined, (
            f"the janitor did not filter on status; criteria were {criteria}. "
            "Without it every QUEUED row waiting behind a long job would be "
            "marked failed."
        )
        assert "task_id" in joined, (
            "the janitor did not filter out rows with no Celery id — federated "
            "rows from other job tables have none and cannot be checked"
        )

    def test_it_does_not_resubmit_the_work(self):
        """A dead fit lost every prompt it processed; the accumulator was in
        worker memory. Re-running automatically would take the GPU for another
        hour without being asked."""
        import inspect

        import src.workers.cleanup_orphaned_tasks as janitor

        source = inspect.getsource(janitor)
        for resubmit in (".delay(", ".apply_async(", ".retry("):
            assert resubmit not in source, (
                f"the janitor calls {resubmit} — it must mark the row and stop, "
                "not silently reclaim the GPU"
            )


# ---------------------------------------------------------------------------
# The half of the problem the heartbeat rule cannot see.
#
# A janitor written specifically to clear a dead row could not clear it: the
# worker died AND its result-backend entry expired, so Celery answered PENDING
# with no info. `looks_orphaned` only ever inspects PROGRESS/STARTED, so it
# returned False and the row sat at "running 21.5%" for hours with the GPU idle
# at 0%. Observed on hardware, through a janitor, twice reported by the user.
# ---------------------------------------------------------------------------


def test_a_pending_task_with_an_old_row_is_abandoned():
    """The exact shape that survived the janitor.

    MUTATION CONTROL: delete the `state == "PENDING"` branch of
    `looks_abandoned` and this fails — which is the state the code was in when
    the ghost row was reported.
    """
    from src.workers.task_heartbeat import STALE_AFTER_SECONDS, looks_abandoned

    assert looks_abandoned("PENDING", None, STALE_AFTER_SECONDS + 60)


def test_a_pending_task_whose_row_is_fresh_is_left_alone():
    """A task that has just been dispatched has not failed."""
    from src.workers.task_heartbeat import looks_abandoned

    assert not looks_abandoned("PENDING", None, 5)


def test_a_pending_task_with_no_row_age_is_not_condemned():
    """Absent evidence is not evidence of death — fail safe, not closed."""
    from src.workers.task_heartbeat import looks_abandoned

    assert not looks_abandoned("PENDING", None, None)


def test_a_terminal_task_is_never_abandoned_however_old_its_row():
    """A row left open behind a SUCCESS is stale bookkeeping, not a dead job.

    Marking it failed would tell the user their completed fit died.

    MUTATION CONTROL: drop the TERMINAL_STATES guard and this fails.
    """
    from src.workers.task_heartbeat import looks_abandoned

    for state in ("SUCCESS", "FAILURE", "REVOKED"):
        assert not looks_abandoned(state, None, 999_999), state


def test_a_stale_heartbeat_is_still_caught():
    """The original rule must keep working — this is a widening, not a swap."""
    from src.workers.task_heartbeat import STALE_AFTER_SECONDS, looks_abandoned

    stale = {"heartbeat": 1_000.0}
    now = 1_000.0 + STALE_AFTER_SECONDS + 1
    assert looks_abandoned("PROGRESS", stale, None, now=now)


def test_a_beating_task_is_never_abandoned():
    """A working job must not be condemned by the widened rule."""
    from src.workers.task_heartbeat import looks_abandoned

    fresh = {"heartbeat": 1_000.0}
    assert not looks_abandoned("PROGRESS", fresh, 999_999, now=1_010.0)


def test_row_age_reads_naive_and_aware_timestamps_alike():
    """The column has carried both; comparing across them raises.

    A janitor that raises on one row stops sweeping every row after it.
    """
    from datetime import datetime, timedelta, timezone

    from src.workers.task_heartbeat import seconds_since_row_update

    class _Row:
        def __init__(self, stamp):
            self.updated_at = stamp
            self.created_at = None

    naive = _Row(datetime.utcnow() - timedelta(seconds=900))
    aware = _Row(datetime.now(timezone.utc) - timedelta(seconds=900))

    for row in (naive, aware):
        age = seconds_since_row_update(row)
        assert age is not None and 800 < age < 1000, age


def test_the_janitor_uses_the_widened_rule_not_the_heartbeat_one():
    """Reachability: the fix must be the rule the sweep actually calls.

    The previous janitor imported `looks_orphaned` and was therefore blind to
    the case it existed to clear. Asserting the module imports the widened rule
    is not enough — this drives the sweep and asserts the row is closed.

    MUTATION CONTROL: point the janitor back at `looks_orphaned` and this fails.
    """
    from datetime import datetime, timedelta, timezone
    from unittest.mock import MagicMock, patch

    import src.workers.cleanup_orphaned_tasks as janitor

    class _Row:
        status = "running"
        task_id = "dead-task"
        error_message = None
        updated_at = datetime.now(timezone.utc) - timedelta(hours=3)
        created_at = updated_at

    row = _Row()
    db = MagicMock()
    db.query.return_value.filter.return_value.filter.return_value.all.return_value = [row]

    from contextlib import contextmanager

    @contextmanager
    def fake_db():
        yield db

    # The shape that beat the old janitor: no record left in the backend.
    dead = MagicMock(state="PENDING", info=None)

    with patch("src.core.database.get_sync_db", fake_db), patch(
        "celery.result.AsyncResult", return_value=dead
    ):
        out = janitor.cleanup_orphaned_tasks_task.run()

    assert out["closed"] == 1, (
        "the sweep left a PENDING row with a three-hour-old timestamp open — "
        "this is the ghost job the janitor was written to clear"
    )
    assert row.status == "failed"
    assert "presumed gone" in (row.error_message or "")
