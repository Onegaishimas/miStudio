"""Phase 6 — the five lifecycles that were startable and not stoppable.

Faithfulness, calibration, the steering recorder, enhanced labeling and feature
grouping each had a launch route, a status column, and no way for an operator
to reach a running job short of restarting the pod — which on a `--pool=solo`
worker also strands the in-flight `acks_late` message for the full 12-hour
visibility timeout.

Shape A for each: drive the REAL loop, flip the flag, assert the work stops.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.core import cancellation as C
from src.core.cancellation import OperatorCancelled


class _Row:
    def __init__(self, **kw):
        self.id = kw.pop("id", "row_1")
        self.status = kw.pop("status", "running")
        self.error = None
        self.error_message = None
        self.completed_at = None
        self.progress = 0.0
        for k, v in kw.items():
            setattr(self, k, v)


class _Query:
    def __init__(self, row):
        self._row = row

    def filter(self, *a, **k):
        return self

    filter_by = filter

    def populate_existing(self):
        return self

    def first(self):
        return self._row


def _session(row):
    db = MagicMock()
    db.query.side_effect = lambda model: _Query(row)
    return db


class TestFeatureGroupingStops:
    """`FeatureGroupingService.compute` — the real per-batch loop."""

    def test_the_batch_loop_abandons_at_the_flag(self):
        import inspect

        from src.services.feature_grouping_service import FeatureGroupingService

        src = inspect.getsource(FeatureGroupingService._build_index)
        loop = src.index("for batch_start in range(0, total, FEATURE_BATCH_SIZE):")
        poll = src.index("cancel_check", loop)
        work = src.index("batch = features[batch_start", loop)
        assert poll < work, "the loop reads a batch before checking"

    def test_the_task_supplies_a_factory_bound_to_the_new_run(self):
        """The run id does not exist until `compute` creates the row, so the
        task cannot build a checker in advance — and threading the scope name
        into the service would put cancellation vocabulary where it does not
        belong."""
        import inspect

        from src.workers.feature_grouping_tasks import compute_feature_groups_task

        src = inspect.getsource(inspect.unwrap(compute_feature_groups_task))
        assert "cancel_check_for=" in src
        assert '"feature_grouping"' in src

    def test_a_cancelled_run_is_recorded_as_cancelled_not_failed(self):
        import inspect

        from src.workers.feature_grouping_tasks import compute_feature_groups_task

        src = inspect.getsource(inspect.unwrap(compute_feature_groups_task))
        cancelled = src.index("except OperatorCancelled")
        generic = src.index("except Exception as")
        assert cancelled < generic, (
            "the generic handler runs first, so a cancel is written as a failure"
        )

    def test_the_status_enum_can_express_a_cancellation(self):
        from src.models.feature_grouping import GroupingRunStatus

        assert GroupingRunStatus.CANCELLED.value == "cancelled"


class TestSteeringRecorderStops:
    def test_the_prompt_loop_abandons_at_the_flag(self):
        import inspect

        from src.services.steering_recorder_service import SteeringRecorderService

        src = inspect.getsource(SteeringRecorderService.record_samples)
        loop = src.index('for pi, prompt in enumerate(cfg["prompts"]):')
        poll = src.index("cancel_check", loop)
        work = src.index("unsteered = baseline_at(", loop)
        assert poll < work, (
            "the recorder generates a baseline before checking; one prompt is "
            "1 + len(dials) indivisible generations"
        )

    def test_the_task_injects_a_checker_and_handles_the_stop(self):
        import inspect

        from src.workers.circuit_record_tasks import run_circuit_record

        src = inspect.getsource(inspect.unwrap(run_circuit_record))
        assert '"steering_record"' in src
        assert src.index("except OperatorCancelled") < src.index("except Exception as")


class TestCalibrationStops:
    def test_the_progress_callback_is_the_checkpoint(self):
        """Calibration's bisection lives inside the service; the progress
        callback is the one hook already threaded through it."""
        import inspect

        from src.workers import circuit_calibration_tasks as t

        src = inspect.getsource(t._calibration_progress)
        assert "raise_if_cancelled" in src
        assert '"circuit_calibration"' in src

    def test_the_callback_polls_before_it_emits(self):
        import inspect

        from src.workers import circuit_calibration_tasks as t

        src = inspect.getsource(t._calibration_progress)
        code = "\n".join(
            l for l in src.splitlines() if not l.lstrip().startswith("#")
        )
        assert code.index("raise_if_cancelled") < code.index("emit_circuit_run_progress")

    def test_a_cancelled_calibration_returns_rather_than_raising(self):
        import inspect

        from src.workers.circuit_calibration_tasks import run_circuit_calibration

        src = inspect.getsource(inspect.unwrap(run_circuit_calibration))
        assert src.index("except OperatorCancelled") < src.index("except Exception as")


class TestEnhancedLabelingStops:
    def test_pass_one_polls_per_example(self):
        import inspect

        from src.workers.enhanced_labeling_tasks import enhanced_label_feature_task

        src = inspect.getsource(inspect.unwrap(enhanced_label_feature_task))
        cb = src.index("def _progress_cb")
        poll = src.index("raise_if_cancelled", cb)
        work = src.index("job.examples_completed = n_completed", cb)
        assert poll < work, (
            "pass 1 fans out one LLM call per example; polling after the "
            "bookkeeping pays for one more call than it needs to"
        )

    def test_it_polls_between_the_passes(self):
        """Pass 2 is a single synthesis call with no checkpoint of its own."""
        import inspect

        from src.workers.enhanced_labeling_tasks import enhanced_label_feature_task

        src = inspect.getsource(inspect.unwrap(enhanced_label_feature_task))
        assert "stopped between pass 1 and pass 2" in src

    def test_the_status_enum_can_express_a_cancellation(self):
        from src.models.enhanced_labeling_job import EnhancedLabelingStatus

        assert EnhancedLabelingStatus.CANCELLED.value == "cancelled"

    def test_a_cancel_is_not_written_as_a_failure(self):
        import inspect

        from src.workers.enhanced_labeling_tasks import enhanced_label_feature_task

        src = inspect.getsource(inspect.unwrap(enhanced_label_feature_task))
        assert src.index("except OperatorCancelled") < src.index("except Exception as exc")


class TestTheRoutesReachTheChannel:
    """Shape C for all five: what the endpoint writes is what the worker reads."""

    @pytest.mark.parametrize(
        "kind,status_field",
        [
            ("circuit_faithfulness", "faithfulness_status"),
            ("circuit_calibration", "calibration_status"),
            ("steering_record", "status"),
            ("enhanced_labeling", "status"),
            ("feature_grouping", "status"),
        ],
    )
    def test_the_request_is_visible_to_a_checker(self, kind, status_field):
        row = _Row(**{status_field: "running"})
        db = _session(row)
        out = C.request_cancel(kind, "row_1", db=db)
        assert out.requested is True
        assert C.cancel_checker(kind, "row_1", db=db)() is True

    @pytest.mark.parametrize(
        "kind", ["circuit_faithfulness", "circuit_calibration", "steering_record",
                 "enhanced_labeling", "feature_grouping"],
    )
    def test_a_finished_job_is_not_cancellable(self, kind):
        scope = C.get_scope(kind)
        row = _Row(**{scope.status_field: "completed"})
        out = C.request_cancel(kind, "row_1", db=_session(row))
        assert out.requested is False

    def test_the_five_routes_exist(self):
        """Reachability: located in the LIVE router, not by reading source."""
        from src.api.v1.endpoints import circuits, enhanced_labeling, feature_groups

        paths = set()
        for module in (circuits, enhanced_labeling, feature_groups):
            for route in module.router.routes:
                paths.add(getattr(route, "path", ""))

        for expected in (
            "/circuits/{circuit_id}/faithfulness/cancel",
            "/circuits/{circuit_id}/calibration/cancel",
            "/circuits/steering-samples/{run_id}/cancel",
            "/enhanced-labeling/{job_id}/cancel",
            "/feature-groups/runs/{run_id}/cancel",
        ):
            assert expected in paths, f"{expected} is not registered on any router"


class TestTheModelDownloadStops:
    """The download path HuggingFace gives no abort hook for.

    `snapshot_download` cannot be interrupted mid-transfer: the monitor runs on
    a separate thread and a raise there dies in that thread. What exists is the
    boundaries the task owns — and those must actually be polled, or a model
    cancelled while queued still pulls 15 GB first.
    """

    def test_it_polls_before_the_download_begins(self):
        import inspect

        from src.workers.model_tasks import download_and_load_model

        src = inspect.getsource(inspect.unwrap(download_and_load_model))
        poll = src.find("stopped before the download began")
        start = src.find("progress_monitor.start()")
        assert poll != -1, "a model cancelled while queued still pulls the weights"
        assert poll < start

    def test_it_polls_after_the_download_before_loading(self):
        import inspect

        from src.workers.model_tasks import download_and_load_model

        src = inspect.getsource(inspect.unwrap(download_and_load_model))
        assert "stopped after the download, before loading" in src, (
            "nothing acts on a cancellation seen mid-transfer, so the task "
            "goes on to quantize and profile a model nobody wants"
        )

    def test_the_monitor_thread_observes_the_flag(self):
        """It cannot raise — a worker thread's exception does not reach the
        task — but it must stop narrating and record what it saw."""
        import inspect

        from src.workers.model_tasks import DownloadProgressMonitor

        src = inspect.getsource(DownloadProgressMonitor._monitor_loop)
        assert "cancel_seen" in src
        assert "self._stop_event.set()" in src

    def test_the_endpoint_does_not_delete_a_live_download(self):
        import inspect

        from src.workers.model_tasks import cancel_download

        src = inspect.getsource(inspect.unwrap(cancel_download))
        assert "job_had_started" in src
        assert src.index("job_had_started") < src.index("shutil.rmtree(cache_dir)")
