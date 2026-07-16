"""
miLLM sensing tools (category: millm_sensing) — Feature 9, Unified MCP.

Cluster co-activation observation: when did the active cluster's members
fire together, on what tokens, with what context.
"""

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from ..health_gate import HealthGate, gated
from ..millm_client import MiLLMClient


def register(mcp: FastMCP, millm: MiLLMClient, gate: HealthGate) -> None:
    @mcp.tool()
    @gated(gate, "millm")
    async def millm_sensing_status() -> Any:
        """Sensing runtime status: armed cluster, quorum/threshold mode,
        per-request overhead, retention limits, and which clusters have the
        persistent sensing toggle enabled (distinct from armed)."""
        return await millm.get("/api/sensing/status")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_sensing_events(profile_id: Optional[str] = None,
                                   limit: int = 50,
                                   since: Optional[str] = None) -> Any:
        """Co-activation events newest-first. Each row carries the span,
        fired members with peak activations, score, human summary, the
        ±K token context window (context_text/context_token_ids) when the
        cluster captures context, and ambient_fired_count — the
        alone-vs-within signal (how many features across the WHOLE SAE
        fired; null when full-width monitoring wasn't co-running — never
        estimated). `since` (ISO-8601 timestamp) time-bounds
        polling so agents fetch only new events. MUST carry an explicit
        UTC offset (e.g. 2026-07-16T12:00:00Z) — naive timestamps are
        rejected because the server would silently assume UTC."""
        if since is not None:
            from datetime import datetime

            try:
                parsed = datetime.fromisoformat(since.replace("Z", "+00:00"))
            except ValueError:
                return {"error": f"`since` is not ISO-8601: {since!r}"}
            if parsed.tzinfo is None:
                return {"error": "`since` must carry a UTC offset "
                                 "(e.g. ...T12:00:00Z) — naive timestamps "
                                 "shift the polling window silently"}
        return await millm.get("/api/sensing/events",
                               profile_id=profile_id, limit=limit,
                               since=since)

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_sensing_enable(profile_id: str) -> Any:
        """Enable sensing for a cluster (persists; arms live when that
        cluster is active with an SAE attached)."""
        return await millm.post(f"/api/sensing/{profile_id}/enable")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_sensing_disable(profile_id: str) -> Any:
        """Disable sensing for a cluster (disarms live if armed)."""
        return await millm.post(f"/api/sensing/{profile_id}/disable")
