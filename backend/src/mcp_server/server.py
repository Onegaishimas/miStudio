"""
miStudio MCP server assembly: FastMCP instance, bearer auth, category gating.

Auth model (Feature 010, BR-5.1): the streamable-HTTP transport requires
``Authorization: Bearer <MCP_AUTH_TOKEN>``, compared with
``hmac.compare_digest`` (same constant-time pattern as the backend's
X-Internal-Token check). The port binds 0.0.0.0 by default — LAN-reachable by
design (product decision); the token is therefore mandatory: startup is
refused with an empty token unless MCP_ALLOW_ANONYMOUS=true (stdio/local dev).
"""

import hashlib
import hmac
import json
import logging
import time
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from .client import MiStudioClient
from .config import MCPSettings
from .tools import CATEGORY_MODULES, MILLM_CATEGORY_MODULES

logger = logging.getLogger(__name__)

SERVER_INSTRUCTIONS = """\
miStudio MCP server — SAE feature analysis, circuit discovery, and steering.

**CALL `mistudio_howto` FIRST** for circuit or steering work. 92 tools across
13 categories is more than a description can carry; that tool holds the
workflows, the document shapes that cross between miStudio and miLLM, and the
failure modes that look like something else.

Two planes: miStudio DISCOVERS and CALIBRATES (it runs the model to learn);
miLLM SERVES (it runs the model to serve, behind an OpenAI-compatible API).
The boundary is a portable document, not a code dependency.

Feature analysis: list_extractions → search_features → get_feature →
get_feature_examples/token-analysis/logit-lens → update_feature_label.

Circuit discovery (each stage EARNS an evidence rung — see
`mistudio_howto(\"discovery_pipeline\")`): start_circuit_capture →
run_circuit_discovery (rung 0) → run_attribution_pass (rung 1) →
validate_circuit_edges (rung 2, a real intervention) → create_circuit →
run_circuit_faithfulness → promote_circuit → export_circuit_definition.

Steering: steer_sweep / steer_compare / steer_combined auto-start the GPU
worker, so enter_steering_mode is OPTIONAL (it just pays the cold start up
front). exit_steering_mode is MANDATORY — nothing else reaps the worker.
Poll get_steering_result: it also releases a concurrency slot.

Long-running operations return task ids — poll get_task_status /
get_steering_result.

Cross-product (when millm_* tools are present): move a tuned artifact into
production with export_cluster_definition → millm_import_cluster, or
export_circuit_definition → millm_import_circuit → millm_activate_circuit.
A circuit below rung 2 is REFUSED unless you pass acknowledge_unvalidated.
millm_status answers "what is steering right now" in one call. When miLLM is
down its tools return {"unavailable": "millm", "reason": …} — report the
reason, don't retry in a loop.
"""


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Constant-time bearer-token check on every HTTP request except /health."""

    def __init__(self, app: Any, token: str):
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        header = request.headers.get("authorization", "")
        provided = header[7:] if header.lower().startswith("bearer ") else ""
        if not provided or not hmac.compare_digest(provided, self._token):
            return JSONResponse({"detail": "Invalid or missing bearer token"}, status_code=401)
        return await call_next(request)


class AuditToolLogger:
    """Structured one-line audit log per tool call (BR-5.4)."""

    @staticmethod
    def log(tool: str, args: dict, status: str, duration_ms: float) -> None:
        digest = hashlib.sha256(
            json.dumps(args, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
        logger.info(
            "mcp_tool_call tool=%s args_digest=%s status=%s duration_ms=%.0f",
            tool, digest, status, duration_ms,
        )


def build_server(settings: MCPSettings, stdio: bool = False) -> tuple[FastMCP, MiStudioClient]:
    """Create the FastMCP server with only the enabled tool categories registered."""
    if not settings.auth_token and not (settings.allow_anonymous or stdio):
        raise SystemExit(
            "MCP_AUTH_TOKEN is required — the MCP port is LAN-reachable by default. "
            "Set a token, or set MCP_ALLOW_ANONYMOUS=true for local stdio development only."
        )

    categories = settings.enabled_categories()
    mcp = FastMCP(
        "mistudio",
        instructions=SERVER_INSTRUCTIONS,
        host=settings.host,
        port=settings.port,
        stateless_http=True,
        json_response=True,
    )
    client = MiStudioClient(settings.api_url)

    registered = []
    for category, modules in CATEGORY_MODULES.items():
        if category not in categories:
            continue
        for module in modules:
            module.register(mcp, client, settings)
        registered.append(category)

    # miLLM categories (Unified MCP, Feature 9): wired against MiLLMClient +
    # HealthGate. Unset MILLM_API_URL skips them at REGISTRATION (log once) —
    # a running-but-unreachable miLLM instead keeps tools registered and the
    # gate returns structured unavailable per call (contract §3).
    from .health_gate import HealthGate

    gate = HealthGate(settings.millm_api_url, mistudio_url=settings.api_url)

    requested_millm = [c for c in MILLM_CATEGORY_MODULES if c in categories]
    if requested_millm:
        if settings.millm_api_url:
            from .millm_client import MiLLMClient

            millm_client = MiLLMClient(settings.millm_api_url)
            for category in requested_millm:
                for module in MILLM_CATEGORY_MODULES[category]:
                    module.register(mcp, millm_client, gate)
                registered.append(category)
        else:
            logger.warning(
                "millm categories requested but MILLM_API_URL is unset — "
                f"skipping: {sorted(requested_millm)}"
            )
    logger.info(f"MCP tool categories enabled: {sorted(registered)}")

    @mcp.custom_route("/health", methods=["GET"])
    async def health(request: Request) -> JSONResponse:
        # Per-product availability (US-9.4) — NEVER blocks on live probes:
        # this route answers orchestrator readiness/liveness checks (default
        # timeout 1 s), and a hung backend must degrade its TOOLS, not take
        # the MCP server out of rotation or restart-loop it (009 R2).
        # snapshot() serves the last known state and refreshes in the
        # background. Reasons are coarse categories — the route is
        # unauthenticated and detailed reasons carry internal URLs.
        products = {}
        product_names = ["mistudio"]
        if settings.millm_api_url or requested_millm:
            product_names.append("millm")
        for product in product_names:
            available, reason = gate.snapshot(product)
            products[product] = {
                "available": available,
                "reason": gate.public_reason(reason),
            }
        return JSONResponse(
            {"status": "ok", "service": "mistudio-mcp",
             "categories": sorted(registered), "products": products}
        )

    # Shutdown hooks (009 R2: aclose existed with zero callers). __main__
    # and tests can close everything via mcp — build_server's 2-tuple
    # return shape is load-bearing for existing callers.
    closables = [client, gate]
    if settings.millm_api_url and requested_millm:
        closables.append(millm_client)

    async def _close_backend_clients() -> None:
        for closable in closables:
            try:
                close = getattr(closable, "aclose", None) or getattr(
                    closable, "close", None)
                if close is not None:
                    await close()
            except Exception:  # noqa: BLE001 — best-effort shutdown
                pass

    mcp.close_backend_clients = _close_backend_clients  # type: ignore[attr-defined]

    return mcp, client


def wrap_tool_with_audit(mcp: FastMCP) -> None:
    """Wrap registered tools with the audit logger."""
    # FastMCP stores tools in the tool manager; wrap their run for audit logging.
    manager = mcp._tool_manager  # noqa: SLF001 — no public hook for this in the SDK
    for tool in manager.list_tools():
        original = tool.fn

        def make_wrapper(fn, tool_name):
            async def wrapper(*args, **kwargs):
                start = time.monotonic()
                try:
                    result = await fn(*args, **kwargs)
                    AuditToolLogger.log(tool_name, kwargs, "ok", (time.monotonic() - start) * 1000)
                    return result
                except Exception as e:
                    AuditToolLogger.log(tool_name, kwargs, f"error:{type(e).__name__}",
                                        (time.monotonic() - start) * 1000)
                    raise
            return wrapper

        tool.fn = make_wrapper(original, tool.name)
