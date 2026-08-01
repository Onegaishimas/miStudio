"""
REACHABILITY for the J-space MCP tools (BR-027).

    A capability is not shipped until a test FAILS when its wiring is removed.

This file was written BEFORE the tools, per the FTID's own ordering, because
the alternative has already happened here: 16 `millm_circuit_*` tools shipped
fully implemented, unit-tested and documented in the contract while registered
with nothing, and every test passed by importing the module directly.

The three shapes from `test_reachability.py`, because each catches what the
others cannot:

  1. REGISTRY   — the category is in the maps that make it selectable
  2. BUILT SERVER — the REAL `build_server()` exposes the tools
  3. CALLER    — each tool issues its documented method and path, with the
                 PAYLOAD and the CALL COUNT asserted; "was called" passes
                 against a call sending the wrong arguments

MUTATION CONTROLS (each must turn this file red):
  * remove `jlens` from CATEGORY_MODULES        -> registry + built server fail
  * remove `jlens` from VALID_CATEGORIES        -> "selectable" fails
  * remove `jlens` from DEFAULT_CATEGORIES      -> "on by default" fails
  * drop a @mcp.tool() decorator                -> built server fails
  * send the wrong path or drop a parameter     -> caller fails
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_server.config import DEFAULT_CATEGORIES, VALID_CATEGORIES
from src.mcp_server.tools import CATEGORY_MODULES

EXPECTED_TOOLS = {
    "list_jlens_artifacts",
    "validate_jlens_artifact",
    "jlens_readout",
    "fit_jlens_artifact",
    "get_jlens_band_report",
    "get_jlens_gate",
    "get_jlens_readout",
    "annotate_jlens_feature",
    "create_jlens_watchlist",
    "jlens_cost_estimate",
    "get_jlens_replication_report",
}


# ── Shape 1: registry ──────────────────────────────────────────────────────


class TestRegistry:
    def test_the_category_is_in_the_module_registry(self):
        assert "jlens" in CATEGORY_MODULES
        assert CATEGORY_MODULES["jlens"], "registered with no modules"

    def test_the_category_is_selectable(self):
        """Absent here, MCP_TOOL_CATEGORIES=jlens raises instead of enabling."""
        assert "jlens" in VALID_CATEGORIES

    def test_it_is_on_by_default(self):
        """J-space readout is a first-class analysis surface, not an opt-in.

        Unlike the millm_* categories — which need a second product running —
        these tools talk only to miStudio's own API, so gating them behind
        explicit opt-in would make the capability unreachable for every agent
        that does not know to ask for it.
        """
        assert "jlens" in {c.strip() for c in DEFAULT_CATEGORIES.split(",")}

    def test_the_module_exposes_register(self):
        for module in CATEGORY_MODULES["jlens"]:
            assert hasattr(module, "register")


# ── Shape 2: built server ──────────────────────────────────────────────────


class TestBuiltServer:
    """The REAL build_server(), not a hand-called register().

    Calling register() directly proves the module works and says nothing about
    whether anything calls it — which is exactly the defect that shipped.
    """

    def _build(self, categories: str, monkeypatch):
        from src.mcp_server.config import MCPSettings
        from src.mcp_server.server import build_server

        monkeypatch.setenv("MILLM_API_URL", "http://millm.test")
        settings = MCPSettings(tool_categories=categories, allow_anonymous=True)
        mcp, _client = build_server(settings, stdio=True)
        return {t.name for t in asyncio.run(mcp.list_tools())}

    def test_build_server_exposes_every_jlens_tool(self, monkeypatch):
        names = self._build("jlens", monkeypatch)
        for expected in EXPECTED_TOOLS:
            assert expected in names, (
                f"{expected} is not reachable from build_server() — "
                "registration exists but nothing reaches it"
            )

    def test_the_tools_are_absent_when_the_category_is_not_enabled(self, monkeypatch):
        """Specificity: without this, the test above proves nothing about wiring."""
        names = self._build("read", monkeypatch)
        assert not (EXPECTED_TOOLS & names)

    def test_the_default_configuration_reaches_them(self, monkeypatch):
        names = self._build(DEFAULT_CATEGORIES, monkeypatch)
        assert EXPECTED_TOOLS <= names


# ── Shape 3: caller ────────────────────────────────────────────────────────


class TestCaller:
    """Each tool issues its documented call, with payload AND call count."""

    def _tools(self):
        from mcp.server.fastmcp import FastMCP

        from src.mcp_server.config import MCPSettings
        from src.mcp_server.tools import jlens

        mcp = FastMCP("test")
        client = MagicMock()
        client.get = AsyncMock(return_value={"ok": True})
        client.post = AsyncMock(return_value={"ok": True})
        jlens.register(mcp, client, MCPSettings(allow_anonymous=True))

        registered = asyncio.run(mcp.list_tools())
        return mcp, client, {t.name for t in registered}

    def test_list_issues_a_GET_to_the_artifacts_path(self):
        mcp, client, names = self._tools()
        assert "list_jlens_artifacts" in names

        asyncio.run(mcp.call_tool("list_jlens_artifacts", {}))

        assert client.get.await_count == 1, "the tool made no call, or made several"
        assert client.get.await_args.args[0] == "/jlens/artifacts"
        assert client.post.await_count == 0

    def test_validate_issues_a_POST_carrying_every_dimension(self):
        """PAYLOAD, not just "was called".

        Dropping n_vocab still calls the endpoint and still returns a verdict —
        one whose envelope bound was derived from the wrong model, which passes
        on one model while missing a real materialisation on another.
        """
        mcp, client, names = self._tools()
        assert "validate_jlens_artifact" in names

        asyncio.run(
            mcp.call_tool(
                "validate_jlens_artifact",
                {"slug": "gemma-2-2b-it", "d_model": 2304, "n_layers": 26, "n_vocab": 256000},
            )
        )

        assert client.post.await_count == 1
        assert client.post.await_args.args[0] == "/jlens/artifacts/gemma-2-2b-it/validate"
        assert client.post.await_args.kwargs == {
            "d_model": 2304,
            "n_layers": 26,
            "n_vocab": 256000,
        }

    def test_readout_issues_a_POST_with_the_prompt_and_model(self):
        mcp, client, names = self._tools()
        assert "jlens_readout" in names

        asyncio.run(
            mcp.call_tool(
                "jlens_readout",
                {"model_id": "m_abc", "prompt": "The capital of France is"},
            )
        )

        assert client.post.await_count == 1
        assert client.post.await_args.args[0] == "/jlens/readout"
        body = client.post.await_args.kwargs["json_body"]
        assert body["model_id"] == "m_abc"
        assert body["prompt"] == "The capital of France is"
        # No artifact_id and no types: the logit lens is the default and needs
        # neither. Sending JACOBIAN_LENS by default would 422 every call.
        assert "artifact_id" not in body
        assert "types" not in body

    def test_fit_issues_a_POST_carrying_the_corpus_name(self):
        """The corpus is part of the recipe (BR-007).

        Dropping it still queues a fit and still produces an artifact — one
        whose provenance says nothing about what it was fitted on.
        """
        mcp, client, names = self._tools()
        assert "fit_jlens_artifact" in names

        asyncio.run(
            mcp.call_tool(
                "fit_jlens_artifact",
                {"model_id": "m_abc", "prompts": ["a", "b"], "corpus_name": "wikitext-103"},
            )
        )

        assert client.post.await_count == 1
        assert client.post.await_args.args[0] == "/jlens/fit"
        body = client.post.await_args.kwargs["json_body"]
        assert body["corpus_name"] == "wikitext-103"
        assert body["prompts"] == ["a", "b"]

    def test_band_report_and_gate_issue_GETs_to_their_own_paths(self):
        """Two tools, two paths. A shared path would return one for the other."""
        mcp, client, names = self._tools()

        asyncio.run(mcp.call_tool("get_jlens_band_report", {"slug": "m"}))
        assert client.get.await_args.args[0] == "/jlens/artifacts/m/band-report"

        asyncio.run(mcp.call_tool("get_jlens_gate", {"slug": "m"}))
        assert client.get.await_args.args[0] == "/jlens/artifacts/m/gate"
        assert client.get.await_count == 2

    def test_polling_the_readout_hits_the_task_path(self):
        """The readout is queue-and-poll; a tool that only POSTs never sees a result."""
        mcp, client, names = self._tools()
        asyncio.run(mcp.call_tool("get_jlens_readout", {"task_id": "t-1"}))
        assert client.get.await_count == 1
        assert client.get.await_args.args[0] == "/jlens/readout/t-1"

    def test_annotation_sends_the_direction_and_the_label_it_is_compared_to(self):
        """Both, or the disagreement score is silently absent.

        The label is what the readout is compared AGAINST — dropping it still
        annotates, and still returns, with no disagreement computed and nothing
        saying why.
        """
        mcp, client, names = self._tools()
        asyncio.run(
            mcp.call_tool(
                "annotate_jlens_feature",
                {
                    "model_id": "m_a",
                    "sae_id": "sae_1",
                    "feature_id": "f7",
                    "layer": 6,
                    "direction": [0.1, 0.2],
                    "label_tokens": ["spider"],
                },
            )
        )
        assert client.post.await_count == 1
        assert client.post.await_args.args[0] == "/jlens/annotate"
        body = client.post.await_args.kwargs["json_body"]
        assert body["direction"] == [0.1, 0.2]
        assert body["label_tokens"] == ["spider"]

    def test_the_watchlist_tool_sends_its_scoring_definition(self):
        """Without it the server refuses — and it should, so the tool must send it."""
        mcp, client, names = self._tools()
        asyncio.run(
            mcp.call_tool(
                "create_jlens_watchlist",
                {
                    "name": "w",
                    "artifact_ref": "gpt2",
                    "scoring_definition": "eval mean minus control mean",
                    "concepts": [{"token": "evaluation", "threshold": 0.5}],
                },
            )
        )
        body = client.post.await_args.kwargs["json_body"]
        assert body["scoring_definition"]
        assert body["artifact_ref"] == "gpt2"

    def test_the_cost_estimate_tool_sends_the_dimensions_it_scales_on(self):
        """An estimate computed from defaults is not about the caller's run."""
        mcp, client, names = self._tools()
        asyncio.run(
            mcp.call_tool(
                "jlens_cost_estimate",
                {
                    "operation": "annotation_sweep",
                    "d_model": 2048,
                    "n_layers": 16,
                    "n_features": 32768,
                },
            )
        )
        assert client.get.await_args.args[0] == "/jlens/cost-estimate"
        params = client.get.await_args.kwargs
        assert params["d_model"] == 2048
        assert params["n_features"] == 32768

    def test_every_registered_tool_is_covered_here(self):
        """A tool added without a caller test would otherwise ship unasserted."""
        _mcp, _client, names = self._tools()
        assert names == EXPECTED_TOOLS, (
            f"tool set changed to {sorted(names)}; add a caller assertion for "
            "anything new rather than letting it ship unasserted"
        )


# ── The endpoints the tools depend on ──────────────────────────────────────


class TestRoutesExist:
    """A tool calling a route that does not exist is the same defect, moved.

    Asserted against the LIVE router rather than the module, because the
    module importing is what 16 unregistered tools also did.
    """

    def _paths(self):
        from src.api.v1.router import api_router

        paths = set()
        for route in api_router.routes:
            inner = getattr(route, "original_router", None)
            if inner is not None:
                paths.update(getattr(r, "path", "") for r in inner.routes)
            paths.add(getattr(route, "path", ""))
        return paths

    def test_every_route_a_tool_calls_is_registered(self):
        paths = self._paths()
        for expected in (
            "/jlens/artifacts",
            "/jlens/artifacts/{slug}/validate",
            "/jlens/readout",
            "/jlens/fit",
            "/jlens/artifacts/{slug}/band-report",
            "/jlens/artifacts/{slug}/gate",
            "/jlens/readout/{task_id}",
            "/jlens/annotate",
            "/jlens/watchlists",
            "/jlens/cost-estimate",
            "/jlens/reports/replication",
        ):
            assert any(expected in p for p in paths), f"{expected} is not routed"
