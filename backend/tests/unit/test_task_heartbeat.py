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
