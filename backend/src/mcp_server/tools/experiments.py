"""Saved-experiment tools (category: experiments) — persist steering evidence."""

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def save_experiment(
        name: str,
        comparison_id: str,
        result: dict[str, Any],
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> Any:
        """Save a steering result as a persistent experiment record.

        comparison_id: unique id of the run being saved (e.g. the sweep_id or
        comparison_id from the steering result) — duplicates are rejected (409).
        result: the full steering result payload plus your findings/conclusion.
        Reference the returned experiment id in feature notes as evidence:
        '[MCP <date>] evidence: experiment <id> — <summary>'."""
        return await client.post(
            "/steering/experiments",
            json_body={
                "name": name,
                "comparison_id": comparison_id,
                "result": result,
                "description": description,
                "tags": tags,
            },
        )

    @mcp.tool()
    async def list_experiments(limit: int = 50, offset: int = 0) -> Any:
        """List saved steering experiments (paginated, limit ≤ 100)."""
        return await client.get("/steering/experiments", limit=min(limit, 100), offset=offset)

    @mcp.tool()
    async def get_experiment(experiment_id: str) -> Any:
        """Get one saved experiment with its full stored data."""
        return await client.get(f"/steering/experiments/{experiment_id}")
