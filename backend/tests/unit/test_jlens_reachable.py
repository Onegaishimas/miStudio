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
