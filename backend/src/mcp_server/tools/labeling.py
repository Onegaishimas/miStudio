"""Labeling write-back tools (category: labeling) — Feature 010 provenance rules."""

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def update_feature_label(
        feature_id: str,
        name: Optional[str] = None,
        category: Optional[str] = None,
        description: Optional[str] = None,
        notes: Optional[str] = None,
        override_protected: bool = False,
    ) -> Any:
        """Update a feature's label. Writes carry label_source='mcp_agent' provenance.

        Aqua-starred features hold protected (completed enhanced) labels: editing
        their name/category/description returns 409 PROTECTED_LABEL unless
        override_protected=true — only override with strong steering evidence.
        Convention: append evidence to notes as
        '[MCP <date>] evidence: experiment <id> — <one-line summary>'.
        """
        body: dict[str, Any] = {"label_source": "mcp_agent", "override_protected": override_protected}
        if name is not None:
            body["name"] = name
        if category is not None:
            body["category"] = category
        if description is not None:
            body["description"] = description
        if notes is not None:
            body["notes"] = notes
        return await client.patch(f"/features/{feature_id}", json_body=body)

    @mcp.tool()
    async def run_enhanced_labeling(feature_id: str) -> Any:
        """Trigger two-pass enhanced LLM labeling for one feature (background job;
        uses the labeling backend configured in Settings). Poll get_enhanced_label."""
        return await client.post(f"/features/{feature_id}/label/enhanced", json_body={})

    @mcp.tool()
    async def get_enhanced_label(feature_id: str) -> Any:
        """Latest enhanced-labeling job + synthesized label for a feature."""
        return await client.get(f"/features/{feature_id}/label/enhanced/latest")
