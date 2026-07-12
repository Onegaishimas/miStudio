"""Discovery tools (category: read) — extractions and trainings."""

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def list_extractions(
        status_filter: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> Any:
        """List feature-extraction jobs (the entry point for feature analysis).

        Each extraction harvested features from one SAE. Use its id with
        search_features, feature-group tools, and by-token queries.
        status_filter: comma-separated statuses (e.g. 'completed'). limit ≤ 100.
        """
        return await client.get(
            "/extractions", status_filter=status_filter, limit=min(limit, 100), offset=offset
        )

    @mcp.tool()
    async def get_extraction_summary(extraction_id: str) -> Any:
        """Get one extraction job's detail: SAE, status, feature counts, config."""
        return await client.get(f"/extractions/{extraction_id}")

    @mcp.tool()
    async def list_trainings(limit: int = 50, offset: int = 0) -> Any:
        """List SAE training runs (id, model, status, hyperparameters, metrics)."""
        return await client.get("/trainings", limit=min(limit, 100), offset=offset)
