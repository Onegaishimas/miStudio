"""Feature query tools (category: read) — search, detail, per-feature analysis."""

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def search_features(
        extraction_id: str,
        search: Optional[str] = None,
        category: Optional[str] = None,
        is_favorite: Optional[bool] = None,
        sort_by: str = "activation_freq",
        sort_order: str = "desc",
        limit: int = 50,
        offset: int = 0,
    ) -> Any:
        """Search/filter features within an extraction.

        search matches name/description. sort_by: activation_freq | max_activation |
        feature_id | name | category. Results are paginated (limit ≤ 100) —
        never request the full feature set; narrow with filters instead.
        """
        return await client.get(
            f"/extractions/{extraction_id}/features",
            search=search,
            category=category,
            is_favorite=is_favorite,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=min(limit, 100),
            offset=offset,
        )

    @mcp.tool()
    async def get_feature(feature_id: str) -> Any:
        """Full feature detail: label, category, description, notes, statistics, star state."""
        return await client.get(f"/features/{feature_id}")

    @mcp.tool()
    async def get_feature_examples(feature_id: str, limit: int = 20) -> Any:
        """Top activating examples with per-token activations — the primary
        evidence for what a feature detects. limit ≤ 100."""
        return await client.get(f"/features/{feature_id}/examples", limit=min(limit, 100))

    @mcp.tool()
    async def get_feature_token_analysis(feature_id: str) -> Any:
        """Aggregated token statistics for a feature (ranked token frequencies)."""
        return await client.get(f"/features/{feature_id}/token-analysis")

    @mcp.tool()
    async def get_feature_logit_lens(feature_id: str) -> Any:
        """Logit-lens: vocabulary tokens this feature promotes/suppresses when active."""
        return await client.get(f"/features/{feature_id}/logit-lens")

    @mcp.tool()
    async def get_feature_correlations(feature_id: str) -> Any:
        """Features correlated with this one (token overlap + activation-stat similarity).
        May take a few seconds on first call; results are cached server-side."""
        return await client.get(f"/features/{feature_id}/correlations")

    @mcp.tool()
    async def get_feature_ablation(feature_id: str) -> Any:
        """Ablation analysis for a feature."""
        return await client.get(f"/features/{feature_id}/ablation")

    @mcp.tool()
    async def get_feature_nlp_analysis(feature_id: str) -> Any:
        """Stored NLP analysis (prime-token distribution, POS tags, context patterns,
        semantic clusters), if it has been computed for this feature."""
        return await client.get(f"/features/{feature_id}/nlp-analysis")
