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
                                   limit: int = 50) -> Any:
        """Co-activation events newest-first (span, fired members, score,
        summary). Event DETAIL (±K token context) is in each event's
        context fields when fetched individually via the REST detail route;
        list rows carry the human summary."""
        return await millm.get("/api/sensing/events",
                               profile_id=profile_id, limit=limit)

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
