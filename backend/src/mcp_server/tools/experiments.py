"""Saved-experiment tools (category: experiments) — persist steering evidence."""

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def save_experiment(
        name: str,
        experiment_type: str,
        data: dict[str, Any],
        description: Optional[str] = None,
    ) -> Any:
        """Save a steering result as a persistent experiment record. Reference its
        id in feature notes as evidence: '[MCP <date>] evidence: experiment <id> — <summary>'."""
        return await client.post(
            "/steering/experiments",
            json_body={
                "name": name,
                "experiment_type": experiment_type,
                "description": description,
                "data": data,
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
