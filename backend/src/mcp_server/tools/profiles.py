"""Cluster-profile tools (category: profiles) — Feature 014, IDL-30.

Read/save/export access to durable cluster profiles and the portable
`mistudio.cluster-definition/v1` interchange format. The agent's loop:
discover a promising cluster (groups tools) → tune it (compute_cluster_allocation
+ steer_combined) → save_cluster_profile with narrative → export_cluster_definition
to move it toward MILLM / other consumers.
"""

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def list_cluster_profiles(
        sae_id: Optional[str] = None, search: Optional[str] = None
    ) -> Any:
        """List saved cluster profiles (newest first), optionally filtered by SAE
        or a name/display-token substring. Profiles are durable snapshots —
        they survive grouping-index recomputes."""
        params: dict[str, Any] = {}
        if sae_id:
            params["sae_id"] = sae_id
        if search:
            params["search"] = search
        return await client.get("/cluster-profiles", **params)

    @mcp.tool()
    async def get_cluster_profile(profile_id: str) -> Any:
        """Get one cluster profile: members with tuned strengths, budget
        snapshot (013 allocation), narrative, provenance."""
        return await client.get(f"/cluster-profiles/{profile_id}")

    @mcp.tool()
    async def save_cluster_profile(
        name: str,
        members: list[dict[str, Any]],
        sae_id: Optional[str] = None,
        narrative: Optional[str] = None,
        display_token: Optional[str] = None,
        source_group_id: Optional[str] = None,
        budget: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Save a cluster profile (durable, decoupled from recomputable groups).

        members: [{feature_idx, strength, sign?, similarity?, activation_frequency?,
        label?, pinned?}] — max 20, strengths are the TUNED values you validated.
        budget: optional 013 allocation snapshot {B, B_dir, G, formula_id, constants,
        intensity}. Include your steering evidence in narrative (markdown OK)."""
        return await client.post(
            "/cluster-profiles",
            json_body={
                "name": name,
                "members": members,
                "sae_id": sae_id,
                "narrative": narrative,
                "display_token": display_token,
                "source_group_id": source_group_id,
                "budget": budget,
            },
        )

    @mcp.tool()
    async def export_cluster_definition(profile_id: str) -> Any:
        """Export a profile as portable `mistudio.cluster-definition/v1` JSON —
        the consumer-neutral artifact (never contains secrets or local paths)."""
        return await client.get(f"/cluster-profiles/{profile_id}/export")
