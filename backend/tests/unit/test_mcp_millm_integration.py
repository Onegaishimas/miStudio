"""Unit tests for the unified-MCP miLLM integration (miLLM Feature 9,
cross-repo): envelope-unwrapping client, health gate, tool smoke tests with a
mocked client, and the deployment-topology matrix."""

from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import httpx
import pytest

from src.mcp_server.client import BackendError
from src.mcp_server.config import MCPSettings
from src.mcp_server.health_gate import HealthGate, gated
from src.mcp_server.millm_client import MiLLMClient
from src.mcp_server.server import build_server


def make_settings(**overrides) -> MCPSettings:
    defaults = dict(auth_token="secret-token", _env_file=None)
    defaults.update(overrides)
    return MCPSettings(**defaults)


def make_client(handler) -> MiLLMClient:
    client = MiLLMClient("http://millm:8000")
    client._http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://millm:8000"
    )
    return client


# =============================================================================
# MiLLMClient — envelope unwrap (contract §2)
# =============================================================================


class TestMiLLMClient:
    def test_success_envelope_returns_data_only(self):
        def handler(request):
            return httpx.Response(200, json={
                "success": True, "data": {"clusters": [1, 2]}, "error": None})

        result = anyio.run(make_client(handler).get, "/api/clusters")
        assert result == {"clusters": [1, 2]}  # no {"success": …} leak

    def test_error_envelope_raises_backend_error_with_code(self):
        def handler(request):
            return httpx.Response(422, json={
                "success": False, "data": None,
                "error": {"code": "VALIDATION_ERROR",
                          "message": "meaningless steering",
                          "details": {"d_sae": 100}}})

        with pytest.raises(BackendError) as exc:
            anyio.run(make_client(handler).post, "/api/clusters/x/activate")
        assert exc.value.detail["code"] == "VALIDATION_ERROR"
        assert "meaningless" in exc.value.detail["message"]

    def test_error_envelope_authoritative_even_on_200(self):
        """miLLM's house style returns some caps as 200 + success=false."""
        def handler(request):
            return httpx.Response(200, json={
                "success": False, "data": None,
                "error": {"code": "PAYLOAD_TOO_LARGE", "message": "too big"}})

        with pytest.raises(BackendError) as exc:
            anyio.run(make_client(handler).post, "/api/clusters/import")
        assert exc.value.detail["code"] == "PAYLOAD_TOO_LARGE"

    def test_raw_get_bypasses_envelope(self):
        """Cluster export IS the artifact — no envelope, returned verbatim."""
        def handler(request):
            return httpx.Response(200, json={
                "kind": "mistudio.cluster-definition", "schema_version": "1"})

        result = anyio.run(make_client(handler).raw_get,
                           "/api/clusters/prof_x/export")
        assert result["kind"] == "mistudio.cluster-definition"

    def test_unreachable_raises_backend_error(self):
        def handler(request):
            raise httpx.ConnectError("refused")

        with pytest.raises(BackendError, match="unreachable"):
            anyio.run(make_client(handler).get, "/api/health/detailed")

    def test_repo_id_slash_not_double_encoded(self):
        seen = {}

        def handler(request):
            seen["path"] = request.url.path
            return httpx.Response(200, json={"success": True, "data": [],
                                             "error": None})

        anyio.run(make_client(handler).get,
                  "/api/clusters/hub/org/pack/definitions")
        assert seen["path"] == "/api/clusters/hub/org/pack/definitions"


# =============================================================================
# HealthGate (contract §3)
# =============================================================================


class TestHealthGate:
    def _gate_with(self, responder) -> HealthGate:
        gate = HealthGate("http://millm:8000", ttl_s=10.0)

        async def probe(product):
            return await responder(product)

        return gate

    @staticmethod
    def _stub_http(gate, responder):
        """Replace the gate's long-lived probe client (built in __init__)."""
        stub = MagicMock()
        stub.get = responder
        gate._http = stub

    def test_2xx_is_available_even_degraded(self):
        gate = HealthGate("http://millm:8000")
        self._stub_http(gate, AsyncMock(return_value=httpx.Response(
            200, json={"status": "degraded"})))
        ok, reason = anyio.run(gate.check, "millm")
        assert ok is True and reason == "ok"

    def test_unhealthy_body_refuses(self):
        """Contract: available <=> 2xx AND status != 'unhealthy' (R1 fix)."""
        gate = HealthGate("http://millm:8000")
        self._stub_http(gate, AsyncMock(return_value=httpx.Response(
            200, json={"status": "unhealthy"})))
        ok, reason = anyio.run(gate.check, "millm")
        assert ok is False and "unhealthy" in reason

    def test_3xx_refuses(self):
        """An ingress redirect fronting a dead backend is NOT available."""
        gate = HealthGate("http://millm:8000")
        self._stub_http(gate, AsyncMock(return_value=httpx.Response(
            307, headers={"location": "http://elsewhere"})))
        ok, reason = anyio.run(gate.check, "millm")
        assert ok is False and "307" in reason

    def test_connection_failure_refuses_with_reason(self):
        gate = HealthGate("http://millm:8000")
        self._stub_http(gate, AsyncMock(
            side_effect=httpx.ConnectError("refused")))
        ok, reason = anyio.run(gate.check, "millm")
        assert ok is False
        assert "ConnectError" in reason and "/api/health" in reason

    def test_ttl_caches_probe(self):
        gate = HealthGate("http://millm:8000", ttl_s=60.0)
        calls = []

        async def get(url):
            calls.append(url)
            return httpx.Response(200, json={"status": "healthy"})

        self._stub_http(gate, get)

        async def run():
            await gate.check("millm")
            await gate.check("millm")

        anyio.run(run)
        assert len(calls) == 1  # second check served from cache

    def test_mistudio_product_probes_v1_system_health(self):
        gate = HealthGate("", mistudio_url="http://backend:8000")
        seen = []

        async def get(url):
            seen.append(url)
            return httpx.Response(200, json={"status": "healthy"})

        self._stub_http(gate, get)
        ok, _ = anyio.run(gate.check, "mistudio")
        assert ok is True
        assert seen == ["http://backend:8000/api/v1/system/health"]

    def test_unconfigured_url_refuses(self):
        gate = HealthGate("")
        ok, reason = anyio.run(gate.check, "millm")
        assert ok is False and "MILLM_API_URL" in reason

    def test_gated_decorator_returns_structured_unavailable(self):
        gate = HealthGate("")

        @gated(gate, "millm")
        async def tool():
            return {"data": 1}

        result = anyio.run(tool)
        assert result == {"unavailable": "millm",
                          "reason": "MILLM_API_URL is not configured"}


# =============================================================================
# Tool smoke tests (mocked client; gate open)
# =============================================================================


class _OpenGate(HealthGate):
    def __init__(self):
        super().__init__("http://millm:8000")

    async def check(self, product):
        return True, "ok"


def _register_and_get(module, tool_name):
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("test")
    millm = MagicMock(spec=MiLLMClient)
    millm.get = AsyncMock(return_value={"ok": 1})
    millm.post = AsyncMock(return_value={"ok": 1})
    millm.put = AsyncMock(return_value={"ok": 1})
    millm.raw_get = AsyncMock(return_value={"kind": "x"})
    module.register(mcp, millm, _OpenGate())
    tool = mcp._tool_manager._tools[tool_name]  # noqa: SLF001
    return tool, millm


class TestToolSmoke:
    def test_import_xor_validation(self):
        from src.mcp_server.tools import millm_clusters

        tool, millm = _register_and_get(millm_clusters, "millm_import_cluster")
        both = anyio.run(lambda: tool.run({"definition": {"kind": "x"},
                                           "repo_id": "org/pack"}))
        neither = anyio.run(lambda: tool.run({}))
        assert "exactly one source" in str(both)
        assert "exactly one source" in str(neither)
        millm.post.assert_not_called()

    def test_import_inline_posts_document(self):
        from src.mcp_server.tools import millm_clusters

        tool, millm = _register_and_get(millm_clusters, "millm_import_cluster")
        anyio.run(lambda: tool.run({"definition": {"kind": "d"},
                                    "activate": True}))
        args, kwargs = millm.post.call_args
        assert args[0] == "/api/clusters/import"
        assert kwargs["json_body"] == {"kind": "d"}
        assert kwargs["activate"] == "true"

    def test_import_hub_requires_filename(self):
        from src.mcp_server.tools import millm_clusters

        tool, millm = _register_and_get(millm_clusters, "millm_import_cluster")
        result = anyio.run(lambda: tool.run({"repo_id": "org/pack"}))
        assert "filename" in str(result)

    def test_status_hits_detailed_health(self):
        from src.mcp_server.tools import millm_runtime

        tool, millm = _register_and_get(millm_runtime, "millm_status")
        anyio.run(lambda: tool.run({}))
        millm.get.assert_awaited_once_with("/api/health/detailed")

    def test_set_intensity_puts_active_route(self):
        from src.mcp_server.tools import millm_runtime

        tool, millm = _register_and_get(millm_runtime, "millm_set_intensity")
        anyio.run(lambda: tool.run({"intensity": 1.2}))
        millm.put.assert_awaited_once_with(
            "/api/clusters/active/intensity",
            json_body={"intensity": 1.2, "reapply": True})

    def test_export_uses_raw_get(self):
        from src.mcp_server.tools import millm_clusters

        tool, millm = _register_and_get(millm_clusters, "millm_export_cluster")
        anyio.run(lambda: tool.run({"cluster_id": "prof_x"}))
        millm.raw_get.assert_awaited_once_with("/api/clusters/prof_x/export")

    def test_sensing_tools_routes(self):
        from src.mcp_server.tools import millm_sensing

        tool, millm = _register_and_get(millm_sensing, "millm_sensing_enable")
        anyio.run(lambda: tool.run({"profile_id": "prof_x"}))
        millm.post.assert_awaited_once_with("/api/sensing/prof_x/enable")


# =============================================================================
# Topology matrix (FR-9.4)
# =============================================================================


class TestTopology:
    def _tool_names(self, mcp):
        return {t.name for t in mcp._tool_manager.list_tools()}  # noqa: SLF001

    def test_both_products(self, monkeypatch):
        monkeypatch.setenv("MILLM_API_URL", "http://millm:8000")
        mcp, _ = build_server(make_settings(
            tool_categories="read,millm_runtime,millm_clusters,millm_sensing"))
        names = self._tool_names(mcp)
        assert "millm_status" in names and "millm_import_cluster" in names
        assert any(not n.startswith("millm_") for n in names)  # miStudio too

    def test_mistudio_only_url_unset_skips_categories(self, monkeypatch):
        monkeypatch.delenv("MILLM_API_URL", raising=False)
        mcp, _ = build_server(make_settings(
            tool_categories="read,millm_runtime"))
        names = self._tool_names(mcp)
        assert not any(n.startswith("millm_") for n in names)
        assert len(names) > 0  # miStudio tools unaffected

    def test_millm_down_structured_unavailable(self, monkeypatch):
        """Registered but unreachable: tools answer, with the gate refusal."""
        monkeypatch.setenv("MILLM_API_URL", "http://127.0.0.1:1")  # nothing there
        mcp, _ = build_server(make_settings(tool_categories="millm_runtime"))
        tool = mcp._tool_manager._tools["millm_status"]  # noqa: SLF001
        result = anyio.run(lambda: tool.run({}))
        text = str(result)
        assert "unavailable" in text and "millm" in text

    def test_default_categories_exclude_millm(self):
        cats = make_settings().enabled_categories()
        assert not any(c.startswith("millm_") for c in cats)


class TestRound1Fixes:
    """009 R1 regression pins."""

    def test_activate_profile_sends_required_body(self):
        """R1 #1 (confirmed): a body-less POST 422s on the real route."""
        from src.mcp_server.tools import millm_runtime

        tool, millm = _register_and_get(millm_runtime,
                                        "millm_activate_profile")
        anyio.run(lambda: tool.run({"profile_id": "prof_x"}))
        args, kwargs = millm.post.call_args
        assert args[0] == "/api/profiles/prof_x/activate"
        assert kwargs["json_body"] == {"apply_steering": True}

    def test_set_intensity_null_reapply_means_true(self):
        from src.mcp_server.tools import millm_runtime

        tool, millm = _register_and_get(millm_runtime, "millm_set_intensity")
        anyio.run(lambda: tool.run({"intensity": 1.2, "reapply": None}))
        kwargs = millm.put.call_args.kwargs
        assert kwargs["json_body"]["reapply"] is True

    def test_import_empty_dict_definition_reaches_millm(self):
        """R1 #10: presence, not truthiness — {} goes to the real validator."""
        from src.mcp_server.tools import millm_clusters

        tool, millm = _register_and_get(millm_clusters,
                                        "millm_import_cluster")
        anyio.run(lambda: tool.run({"definition": {}}))
        millm.post.assert_awaited_once()

    def test_import_on_conflict_passthrough_and_validation(self):
        from src.mcp_server.tools import millm_clusters

        tool, millm = _register_and_get(millm_clusters,
                                        "millm_import_cluster")
        result = anyio.run(lambda: tool.run({"definition": {"kind": "x"},
                                             "on_conflict": "explode"}))
        assert "on_conflict" in str(result)
        millm.post.assert_not_called()
        anyio.run(lambda: tool.run({"definition": {"kind": "x"},
                                    "on_conflict": "fail"}))
        assert millm.post.call_args.kwargs["on_conflict"] == "fail"

    def test_sensing_events_passes_since(self):
        from src.mcp_server.tools import millm_sensing

        tool, millm = _register_and_get(millm_sensing, "millm_sensing_events")
        anyio.run(lambda: tool.run({"since": "2026-07-16T00:00:00Z"}))
        assert millm.get.call_args.kwargs["since"] == "2026-07-16T00:00:00Z"

    def test_raw_get_envelope_error_is_structured(self):
        def handler(request):
            return httpx.Response(404, json={
                "success": False, "data": None,
                "error": {"code": "PROFILE_NOT_FOUND", "message": "gone"}})

        with pytest.raises(BackendError) as exc:
            anyio.run(make_client(handler).raw_get,
                      "/api/clusters/ghost/export")
        assert exc.value.detail["code"] == "PROFILE_NOT_FOUND"

    def test_health_route_reports_products(self, monkeypatch):
        monkeypatch.setenv("MILLM_API_URL", "http://127.0.0.1:1")
        mcp, _ = build_server(make_settings(tool_categories="read"))

        async def run():
            import httpx as _httpx
            from starlette.applications import Starlette

            app = mcp.streamable_http_app()
            transport = _httpx.ASGITransport(app=app)
            async with _httpx.AsyncClient(transport=transport,
                                          base_url="http://t") as http:
                return await http.get("/health")

        response = anyio.run(run)
        body = response.json()
        assert body["products"]["millm"]["available"] is False
        assert "mistudio" in body["products"]
