"""The J-space endpoints must be reachable, not merely importable.

HOUSE RULE (CLAUDE.md): a capability is not shipped until a test FAILS when its
wiring is removed. This repo's cautionary case is the 16 `millm_circuit_*` MCP
tools — fully implemented, unit-tested and documented while never registered
with the server. Every test passed by importing the module directly, so the
suite was green and the docs said shipped while no caller could reach the
feature.

Asserting `from ... import jlens` succeeds would reproduce exactly that failure.
This asserts membership in the ASSEMBLED api_router instead.

NOTE on the accessor: `api_router.routes` holds `_IncludedRouter` wrappers in
this FastAPI version, not expanded routes, and mounting into a fresh app does
not expand them either — a naive `app.include_router(...)` check returns zero
paths for EVERY endpoint module, which reads as "jlens is broken" when nothing
is. Reach the sub-router through `original_router`.

MUTATION CONTROLS:
  * delete the include_router(jlens.router) line -> registration test fails
  * rename a route path                          -> path test fails
  * change POST to GET                           -> method test fails
"""

from pathlib import Path

import pytest

from src.api.v1.endpoints import jlens
from src.api.v1.router import api_router

EXPECTED = {"/jlens/readout", "/jlens/probe"}


def _reachable_paths() -> set:
    """Every path reachable through the assembled api_router."""
    paths = set()
    for included in api_router.routes:
        origin = getattr(included, "original_router", None)
        if origin is None:
            continue
        for route in getattr(origin, "routes", []):
            path = getattr(route, "path", None)
            if path:
                paths.add(path)
    return paths


class TestEndpointsAreReachable:
    def test_the_jlens_router_is_registered(self):
        """Membership in the assembled router, not importability."""
        registered = any(
            getattr(inc, "original_router", None) is jlens.router
            for inc in api_router.routes
        )
        assert registered, (
            "jlens.router is not registered in api_router. The module imports "
            "fine and its unit tests pass — and no caller can reach it. This "
            "is the exact shape of the unregistered-MCP-tools defect."
        )

    def test_both_paths_are_reachable(self):
        missing = EXPECTED - _reachable_paths()
        assert not missing, f"unreachable jlens paths: {sorted(missing)}"

    def test_the_accessor_sees_other_modules_too(self):
        """Guards the test itself.

        If `_reachable_paths` silently returned nothing, every assertion above
        would pass vacuously on an empty set difference. Prove the accessor
        actually resolves routes.
        """
        paths = _reachable_paths()
        assert len(paths) > 50, (
            f"only {len(paths)} paths resolved — the accessor is broken, so the "
            "reachability assertions above prove nothing"
        )

    @pytest.mark.parametrize("path", sorted(EXPECTED))
    def test_paths_accept_post(self, path):
        for route in jlens.router.routes:
            if getattr(route, "path", None) == path:
                assert "POST" in route.methods, (
                    f"{path} does not accept POST; a readout carries a prompt "
                    "body and cannot be a GET"
                )
                return
        pytest.fail(f"{path} not defined on the jlens router")


class TestBoundBackendStillFailsLoudly:
    """Both backends are now bound, so the old 501 assertion no longer applies.

    It is replaced rather than deleted: the hazard it guarded did not go away
    when the endpoint was implemented, it MOVED. An unbound endpoint could
    fabricate an empty result; a bound one can return an empty result for a
    task that FAILED, which is the same lie with a 200 on it.
    """

    def test_the_polling_routes_exist_alongside_the_submitting_ones(self):
        """Queue-and-poll needs both halves. A POST that returns a task id
        nobody can poll is a capability with no way to collect its result."""
        paths = _reachable_paths()
        for path in ("/jlens/readout/{task_id}", "/jlens/probe/{task_id}"):
            assert path in paths, f"unreachable poll route: {path}"

    @pytest.mark.parametrize(
        "path", ["/jlens/readout/{task_id}", "/jlens/probe/{task_id}"]
    )
    def test_poll_routes_accept_get(self, path):
        for route in jlens.router.routes:
            if getattr(route, "path", None) == path:
                assert "GET" in route.methods
                return
        pytest.fail(f"{path} not defined on the jlens router")

    def test_a_failed_task_carries_its_reason_and_no_results(self):
        """The failure path must be distinguishable from an empty success.

        MUTATION CONTROL: return `scores=[]` on FAILURE instead of None and
        this fails — which is exactly the shape that would let a caller read a
        crashed probe as "this direction scores nowhere".
        """
        import asyncio
        from unittest.mock import MagicMock, patch

        failed = MagicMock()
        failed.state = "FAILURE"
        failed.info = RuntimeError("expected scalar type BFloat16 but found Half")

        with patch("celery.result.AsyncResult", return_value=failed):
            result = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
                jlens.probe_result("t-1")
            )

        assert result.status == "FAILURE"
        assert result.scores is None, (
            "a failed probe returned a score list; an empty list is "
            "indistinguishable from a real probe that found nothing"
        )
        assert "BFloat16" in (result.error or ""), "the failure reason was dropped"

    def test_a_successful_probe_records_which_mode_produced_it(self):
        """Probe and full-ranking scores can disagree (BR-008), so a result
        that does not say which mode it used cannot be compared with one that
        does."""
        import asyncio
        from unittest.mock import MagicMock, patch

        done = MagicMock()
        done.state = "SUCCESS"
        done.result = {
            "scores": [{"layer": 3, "position": 0, "token": " Paris", "score": 1.5}],
            "mode": "probe",
            "lens_type": "LOGIT_LENS",
        }

        with patch("celery.result.AsyncResult", return_value=done):
            result = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
                jlens.probe_result("t-2")
            )

        assert result.mode == "probe"
        assert result.lens_type == "LOGIT_LENS"
        assert result.scores and result.scores[0].token == " Paris"


# ---------------------------------------------------------------------------
# Band-report computation must be REACHABLE
#
# `compute_band_report`, `save_band_report`, `decide_gate` and `save_gate` were
# fully implemented and unit-tested with ZERO production callers — verified by
# grep before this was written. The suite was green while no user or agent could
# produce a band report at all, so the panel's band rendering was permanently
# unreachable and `classify_behaviour` returned UNKNOWN forever. Same shape as
# the 16 MCP tools this repo once shipped registered with nothing.
#
# MUTATION CONTROLS (each must turn this section red):
#   * delete the POST /jlens/band-report route      -> "compute route" fails
#   * delete the POST /jlens/gate route             -> "gate route" fails
#   * drop jlens_band_tasks from celery `include`   -> "task is registered" fails
#   * make the endpoint stop calling .delay         -> "endpoint queues" fails
# ---------------------------------------------------------------------------


BAND_PATHS = {"/jlens/band-report", "/jlens/gate"}


class TestBandReportIsReachable:
    def test_the_compute_and_gate_routes_are_registered(self):
        missing = BAND_PATHS - _reachable_paths()
        assert not missing, (
            f"unreachable band paths: {sorted(missing)}. The band service was "
            "implemented and tested with no caller at all — a route is what "
            "makes it exist for anyone"
        )

    @pytest.mark.parametrize("path", sorted(BAND_PATHS))
    def test_band_paths_accept_post(self, path):
        for route in jlens.router.routes:
            if getattr(route, "path", None) == path:
                assert "POST" in route.methods
                return
        pytest.fail(f"{path} not defined on the jlens router")

    def test_the_band_tasks_are_registered_with_celery(self):
        """A task the worker never imports is a task nothing can run.

        Asserts the TASK NAME in the live registry, not that the module
        imports: `task_routes` globs match the task name, so a short or
        unregistered name lands on the default queue silently.
        """
        from src.core.celery_app import celery_app

        for name in (
            "src.workers.jlens_band_tasks.compute_band_report",
            "src.workers.jlens_band_tasks.record_gate",
        ):
            assert name in celery_app.tasks, (
                f"{name} is not in the live Celery registry; the endpoint would "
                "queue a task no worker can execute"
            )

    def test_the_band_tasks_route_to_a_worker_that_exists(self):
        """Routed to `extraction`, where the GPU worker actually listens."""
        from src.core.celery_app import celery_app

        routes = celery_app.conf.task_routes
        assert routes.get("src.workers.jlens_band_tasks.*", {}).get("queue") == (
            "extraction"
        ), "band tasks are not routed to the extraction queue"

    @pytest.mark.asyncio
    async def test_the_compute_endpoint_queues_with_the_arguments_it_was_given(self):
        """Payload AND call count — "was called" passes against wrong arguments."""
        from unittest.mock import MagicMock, patch

        from src.api.v1.endpoints.jlens import BandReportRequest

        db = MagicMock()
        db.execute = _async_returning(_scalar(object()))

        with patch(
            "src.workers.jlens_band_tasks.compute_band_report_task"
        ) as task:
            task.delay.return_value = MagicMock(id="t-band")
            accepted = await jlens.compute_band_report(
                BandReportRequest(
                    model_id="m_1",
                    prompts=["a", "b"],
                    control_seed=1234,
                    layers=[3, 4],
                    use_artifact=True,
                ),
                db=db,
            )

        assert task.delay.call_count == 1
        sent = task.delay.call_args.kwargs
        assert sent["model_id"] == "m_1"
        assert sent["prompts"] == ["a", "b"]
        # The seed must survive the trip: the autocorrelation null is drawn from
        # it, and a report whose control cannot be reproduced is not evidence.
        assert sent["control_seed"] == 1234
        assert sent["layers"] == [3, 4]
        assert accepted.task_id == "t-band"

    def test_no_band_boundary_can_be_SUPPLIED_through_the_api(self):
        """BR-002 by construction, not by discipline.

        Bands come from the model's own kurtosis profile or they do not exist
        for it. A request field that accepted boundaries would make porting the
        published Sonnet-4.5 numbers a one-line change.
        """
        from src.api.v1.endpoints.jlens import BandReportRequest

        fields = set(BandReportRequest.model_fields)
        for forbidden in ("boundaries", "workspace_start", "motor_start", "bands"):
            assert forbidden not in fields, (
                f"BandReportRequest accepts {forbidden!r}; boundaries measured "
                "on another model must be impossible to supply, not merely "
                "discouraged (BR-002)"
            )


def _scalar(value):
    from unittest.mock import MagicMock

    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


def _async_returning(value):
    async def _call(*_args, **_kwargs):
        return value

    return _call


class TestRestoreSupersededIsReachable:
    """The recovery route for a lens that was published over.

    A service method nobody can call is the shape this repo has shipped before:
    16 MCP tools fully implemented, unit-tested, documented and never
    registered. `restore_superseded` exists to spare anyone a shell rename
    inside the pod, and it only does that if it is reachable over HTTP.

    MUTATION CONTROLS:
      * remove the @router.post decorator      -> the path test fails
      * change the path or the method to GET   -> the method test fails
    """

    PATH = "/jlens/artifacts/{slug}/restore-superseded"

    def test_the_restore_path_is_reachable_through_the_assembled_router(self):
        assert self.PATH in _reachable_paths(), (
            "restore-superseded is not reachable through api_router, so the "
            "only way to recover a displaced artifact is a shell rename inside "
            "the pod — which is what it was written to replace"
        )

    def test_it_accepts_POST_and_not_GET(self):
        """A restore MUTATES. A GET that swaps directories is a trap for any
        crawler, prefetcher or retry."""
        methods = set()
        for included in api_router.routes:
            origin = getattr(included, "original_router", None)
            if origin is None:
                continue
            for route in getattr(origin, "routes", []):
                if getattr(route, "path", None) == self.PATH:
                    methods |= set(getattr(route, "methods", set()))
        assert "POST" in methods, f"restore route methods: {methods or 'none'}"
        assert "GET" not in methods, (
            "a directory swap must not be reachable by GET"
        )


class TestCausalEvidenceIsReachable:
    """The read surface a serving runtime would use.

    Evidence written to disk that nothing can fetch is the same defect as a
    service method with no route: real, tested, and unreachable. This is the
    route miLLM will call once a lens has been pulled down from HuggingFace.

    MUTATION CONTROLS:
      * remove the @router.get decorator -> the path test fails
      * change GET to POST               -> the method test fails
    """

    PATH = "/jlens/artifacts/{slug}/interventions"

    def test_the_causal_path_is_reachable_through_the_assembled_router(self):
        assert self.PATH in _reachable_paths(), (
            "causal evidence is written beside the lens but nothing can read "
            "it back over HTTP"
        )

    def test_it_is_a_GET(self):
        """Reading demonstrated behaviour has no side effects."""
        methods = set()
        for included in api_router.routes:
            origin = getattr(included, "original_router", None)
            if origin is None:
                continue
            for route in getattr(origin, "routes", []):
                if getattr(route, "path", None) == self.PATH:
                    methods |= set(getattr(route, "methods", set()))
        assert "GET" in methods, f"causal route methods: {methods or 'none'}"


class TestAMalformedSwapIsRefusedSYNCHRONOUSLY:
    """Knowable at request time, so it must not cost a queue slot.

    The worker already refuses a swap with one token — but only after this
    endpoint has returned 202 with a task id. The caller is told the request was
    accepted, the job takes a slot on a single-GPU queue, and the refusal
    arrives a minute later behind a poll.

    OBSERVED IN PRODUCTION while probing what had deployed: a swap with one
    token came back `{"task_id": ...}` and failed a minute later. It also made
    the deploy probe read "not landed" for a guard that had landed, because the
    probe looked at the HTTP response and the guard lived a layer deeper.

    MUTATION CONTROLS:
      * drop the validator                     -> "refused before queueing" fails
      * allow target_token == direction_token  -> "the SAME token twice" fails
      * let dynamic_topk_ablation through      -> "unimplemented" fails
    """

    def _request(self, **over):
        from src.api.v1.endpoints.jlens import InterventionRequest

        body = dict(
            model_id="m_1",
            prompt="hello",
            primitive="coordinate_swap",
            layers=[9],
            direction_token=" dog",
        )
        body.update(over)
        return InterventionRequest(**body)

    def test_a_swap_with_one_token_is_refused_BEFORE_queueing(self):
        with pytest.raises(ValueError, match="TWO different tokens"):
            self._request()

    def test_a_swap_with_the_SAME_token_twice_is_refused(self):
        with pytest.raises(ValueError, match="TWO different tokens"):
            self._request(target_token=" dog")

    def test_a_swap_with_two_DIFFERENT_tokens_is_accepted(self):
        """The guard must not block the thing it exists to enable."""
        req = self._request(target_token=" cat")
        assert req.target_token == " cat"

    def test_an_unimplemented_primitive_is_refused_here_too(self):
        with pytest.raises(ValueError, match="not implemented"):
            self._request(primitive="dynamic_topk_ablation", target_token=" cat")

    def test_the_other_primitives_are_untouched(self):
        for primitive in ("additive", "projective_ablation"):
            assert self._request(primitive=primitive).primitive == primitive
