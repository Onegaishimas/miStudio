"""Job-tracking tools (category: jobs) — the async-202 polling half (BR-1.5)."""

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def get_task_status(task_queue_id: Optional[str] = None, list_active: bool = False) -> Any:
        """Poll background jobs. With task_queue_id: that job's record (status,
        progress, error). With list_active=true: all currently running/queued
        jobs. Task ids are durable — they survive reconnects."""
        if task_queue_id:
            return await client.get(f"/task-queue/{task_queue_id}")
        if list_active:
            return await client.get("/task-queue/active")
        return await client.get("/task-queue", limit=25)
