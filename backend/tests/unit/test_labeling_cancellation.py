"""Cancelling a labeling job must actually STOP it.

WHY THIS EXISTS
---------------
`cancel_labeling_job` set status=CANCELLED and called
`celery_app.control.revoke(terminate=True)` — and the job kept running for
hours. Observed in production: a job cancelled at 11:45:58 was still calling the
LLM at 11:48:25, ~150 features later, and only died when the pod was restarted.

The reason is the worker's pool. It runs `--pool=solo -c 1`:
  * a task executes in the worker's MAIN process, so there is no child for
    Celery to signal — `terminate=True` cannot stop a running task
  * while the task runs, the main process never services control messages, so
    the revoke is not even read (`celery inspect ping` times out)
  * with `-c 1` that one task blocks EVERY queue, so nothing else can start

So revoke can never be the mechanism here. The loop has to notice by itself.

MUTATION CONTROLS:
  * remove `self._raise_if_cancelled(...)` from a batch loop -> the loop-stops
    test fails
  * make the checker swallow the CANCELLED status -> same
  * re-raise instead of returning in the task's cancel branch -> the
    clean-cancel test fails
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.models.labeling_job import LabelingStatus
from src.services.labeling_service import LabelingService


class _Query:
    def __init__(self, row):
        self._row = row

    def filter(self, *a):
        return self

    def first(self):
        return self._row


class _Session:
    def __init__(self, row):
        self.row = row
        self.queries = 0

    def query(self, model):
        self.queries += 1
        return _Query(self.row)


def _service(status):
    row = SimpleNamespace(id="job_1", status=status)
    svc = LabelingService.__new__(LabelingService)   # no __init__ side effects
    svc.db = _Session(row)
    return svc, row


class TestCooperativeCancellation:
    def test_raises_when_the_job_has_been_cancelled(self):
        svc, _ = _service(LabelingStatus.CANCELLED.value)
        with pytest.raises(LabelingService._LabelingCancelled):
            svc._raise_if_cancelled("job_1")

    @pytest.mark.parametrize(
        "status", [LabelingStatus.LABELING.value, LabelingStatus.QUEUED.value]
    )
    def test_does_not_raise_while_the_job_is_live(self, status):
        svc, _ = _service(status)
        svc._raise_if_cancelled("job_1")  # must not raise

    def test_missing_row_does_not_raise(self):
        """A vanished row must not turn into a spurious cancellation."""
        svc, _ = _service(LabelingStatus.LABELING.value)
        svc.db.row = None
        svc._raise_if_cancelled("job_1")

    def test_a_db_error_never_breaks_the_job(self):
        """The check is a safety net, not a new failure mode."""
        svc, _ = _service(LabelingStatus.LABELING.value)

        def boom(_model):
            raise RuntimeError("connection reset")

        svc.db.query = boom
        svc._raise_if_cancelled("job_1")  # swallowed

    def test_a_batch_loop_stops_once_the_job_is_cancelled(self):
        """THE regression: the loop must abandon its remaining batches.

        Simulates the real shape — a long batch loop that checks each iteration.
        Before the fix the loop ran to completion regardless of status.
        """
        svc, row = _service(LabelingStatus.LABELING.value)
        processed = []

        with pytest.raises(LabelingService._LabelingCancelled):
            for batch_start in range(0, 1000, 1):
                svc._raise_if_cancelled("job_1")
                processed.append(batch_start)
                if batch_start == 4:            # user cancels mid-run
                    row.status = LabelingStatus.CANCELLED.value

        assert processed == [0, 1, 2, 3, 4], (
            "the loop continued past the cancellation instead of stopping"
        )


class TestTaskTreatsCancelAsCleanStop:
    def test_the_task_catches_cancellation_and_does_not_re_raise(self):
        """A user-initiated stop must not be recorded as a failed run.

        Parsed with AST rather than string matching: a substring check happily
        matched `return`/`raise` belonging to the NEXT except-block, so the test
        passed even when the cancel branch re-raised.
        """
        import ast
        import inspect
        import textwrap

        from src.workers import labeling_tasks

        tree = ast.parse(textwrap.dedent(inspect.getsource(labeling_tasks.label_features_task)))

        cancel_handlers = [
            h for h in ast.walk(tree)
            if isinstance(h, ast.ExceptHandler)
            and h.type is not None
            and "_LabelingCancelled" in ast.dump(h.type)
        ]
        assert cancel_handlers, (
            "the task does not distinguish a cancellation from a failure"
        )

        for handler in cancel_handlers:
            raises = [n for n in ast.walk(ast.Module(body=handler.body, type_ignores=[]))
                      if isinstance(n, ast.Raise)]
            returns = [n for n in ast.walk(ast.Module(body=handler.body, type_ignores=[]))
                       if isinstance(n, ast.Return)]
            assert not raises, (
                "the cancel branch re-raises — a user-initiated stop would be "
                "recorded as a failed run"
            )
            assert returns, "the cancel branch must return so the worker is freed"


class TestLoopsAreInstrumented:
    def test_every_labeling_batch_loop_checks_for_cancellation(self):
        """All three loops must check — one unguarded loop still hangs the worker."""
        import inspect

        src = inspect.getsource(LabelingService)
        loops = src.count("for batch_start in range(0, total_features, LABEL_BATCH_SIZE):")
        checks = src.count("self._raise_if_cancelled(labeling_job_id)")
        assert loops > 0, "loop pattern changed; update this test"
        assert checks >= loops, (
            f"{loops} labeling batch loop(s) but only {checks} cancellation check(s)"
        )
