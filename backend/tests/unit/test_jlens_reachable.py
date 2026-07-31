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


class TestUnboundBackendFailsLoudly:
    def test_readout_does_not_fabricate_an_empty_stream(self):
        """An empty readout is indistinguishable from a real one with no
        content. Until the backend is bound, the endpoint must say so."""
        source = Path(jlens.__file__).read_text()
        assert "HTTP_501_NOT_IMPLEMENTED" in source, (
            "the unbound endpoint returns something other than 501; a "
            "fabricated empty readout would be read as a real result"
        )
        assert "indistinguishable" in source, (
            "the reason the endpoint refuses is undocumented, so the next "
            "reader will 'fix' it by returning an empty list"
        )
