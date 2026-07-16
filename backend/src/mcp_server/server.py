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
miStudio MCP server — agentic access to SAE feature analysis and steering.

Typical loop: list_extractions → compute_feature_groups → get_feature_groups →
get_feature_group_members → get_feature_examples/token-analysis/logit-lens →
steer_sweep (causal validation) → update_feature_label with evidence notes.

Long-running operations return task ids — poll get_task_status /
get_steering_result. Steering is GPU-heavy: enter_steering_mode first, exit
when done, and respect the guardrail messages.

Cross-product (when millm_* tools are present): miLLM is the always-on
serving runtime. Move a tuned cluster into production with
export_cluster_definition → millm_import_cluster(definition=…, activate=true),
then millm_set_intensity to dial it and millm_sensing_enable /
millm_sensing_events to watch its members co-fire on live traffic.
millm_status answers "what is steering right now" in one call. When miLLM is
down, its tools return {"unavailable": "millm", "reason": …} — report the
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
        # Per-product availability (US-9.4). Probes are TTL-cached (10 s),
        # so orchestrator readiness checks don't hammer the backends; the
        # server itself is "ok" regardless — a down product degrades its
        # tools to structured-unavailable, never the whole server.
        products = {}
        product_names = ["mistudio"] + (
            ["millm"] if settings.millm_api_url else []
        )
        for product in product_names:
            available, reason = await gate.check(product)
            products[product] = {"available": available, "reason": reason}
        return JSONResponse(
            {"status": "ok", "service": "mistudio-mcp",
             "categories": sorted(registered), "products": products}
        )

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
