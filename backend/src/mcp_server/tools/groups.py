"""Cross-feature grouping tools (category: groups) — Feature 010's new capability."""

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def compute_feature_groups(
        extraction_id: str,
        similarity_threshold: float = 0.35,
        min_group_size: int = 2,
        force: bool = False,
    ) -> Any:
        """Start the grouping precompute job for an extraction (background job).

        Builds a token→feature index and splits shared-token buckets into
        context-similarity subgroups. Poll get_task_status or
        get_feature_groups afterwards. Idempotent unless force=true.
        Returns 409 detail if a run is already computing.
        """
        return await client.post(
            f"/extractions/{extraction_id}/feature-groups/compute",
            json_body={
                "params": {
                    "similarity_threshold": similarity_threshold,
                    "min_group_size": min_group_size,
                },
                "force": force,
            },
        )

    @mcp.tool()
    async def get_grouping_status(extraction_id: str) -> Any:
        """State of the grouping index: none | pending | computing | completed | failed,
        with progress, params, and counts."""
        return await client.get(f"/extractions/{extraction_id}/feature-groups/status")

    @mcp.tool()
    async def get_feature_groups(
        extraction_id: str,
        token: Optional[str] = None,
        search: Optional[str] = None,
        min_group_size: int = 2,
        sort_by: str = "size",
        limit: int = 50,
        offset: int = 0,
    ) -> Any:
        """List feature groups (features sharing a top activating token with similar
        context). token = exact normalized match; search = substring.
        sort_by: size | cohesion | token. Paginated, limit ≤ 100."""
        return await client.get(
            f"/extractions/{extraction_id}/feature-groups",
            token=token,
            search=search,
            min_group_size=min_group_size,
            sort_by=sort_by,
            limit=min(limit, 100),
            offset=offset,
        )

    @mcp.tool()
    async def get_feature_group_members(
        extraction_id: str,
        group_id: str,
        category: Optional[str] = None,
        has_label: Optional[bool] = None,
        star_color: Optional[str] = None,
    ) -> Any:
        """Members of one group with current labels, stars, stats, and a context
        snippet each (labels are live — never stale)."""
        return await client.get(
            f"/extractions/{extraction_id}/feature-groups/{group_id}",
            category=category,
            has_label=has_label,
            star_color=star_color,
        )

    @mcp.tool()
    async def find_features_by_token(
        extraction_id: str,
        token: str,
        match: str = "normalized",
        limit: int = 50,
        offset: int = 0,
    ) -> Any:
        """Features whose top activating token matches. match: exact (raw surface
        form) | normalized (case/BPE-marker-insensitive) | prefix. Requires the
        grouping index (compute_feature_groups first; 409 NO_INDEX otherwise)."""
        return await client.get(
            f"/extractions/{extraction_id}/features/by-token",
            token=token,
            match=match,
            limit=min(limit, 100),
            offset=offset,
        )

    @mcp.tool()
    async def find_related_features(
        feature_id: str, min_similarity: float = 0.2, limit: int = 50
    ) -> Any:
        """Features related to a seed feature via shared tokens, context overlap,
        and cached correlations. Each result carries link_types explaining the
        connection."""
        return await client.get(
            f"/features/{feature_id}/related",
            min_similarity=min_similarity,
            limit=min(limit, 100),
        )
