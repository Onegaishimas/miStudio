"""
miLLM cluster tools (category: millm_clusters) — Feature 9, Unified MCP.

The cross-product loop this enables (SERVER_INSTRUCTIONS reference):
miStudio export_cluster_definition → millm_import_cluster(activate=True) →
millm_set_intensity / millm_sensing_enable.
"""

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from ..health_gate import HealthGate, gated
from ..millm_client import MiLLMClient


def register(mcp: FastMCP, millm: MiLLMClient, gate: HealthGate) -> None:
    @mcp.tool()
    @gated(gate, "millm")
    async def millm_list_clusters() -> Any:
        """List clusters imported into miLLM: bound state, warnings, current
        lambda intensity, active flag."""
        return await millm.get("/api/clusters")

    @mcp.tool()
    async def millm_import_cluster(
        definition: Optional[dict] = None,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        revision: Optional[str] = None,
        activate: bool = False,
        on_conflict: Optional[str] = None,
    ) -> Any:
        """Import a mistudio.cluster-definition/v1 into miLLM.

        Provide EXACTLY ONE source: `definition` (the inline document — e.g.
        the result of miStudio's export_cluster_definition) OR
        `repo_id`+`filename` (a public Hugging Face cluster pack).
        Incompatible definitions import as UNBOUND and refuse only at
        activation. Caps: 1 MB, 20 members/definition. on_conflict:
        'rename' (default — dedupes the name) or 'fail'."""
        # Presence checks, not truthiness: an empty-dict definition should
        # reach miLLM's validator for a real contract error (009 R1).
        if (definition is not None) == (repo_id is not None):
            return {"error": "provide exactly one source: `definition` OR "
                             "`repo_id`+`filename`"}
        if repo_id is not None and not filename:
            return {"error": "`filename` is required with `repo_id`"}
        if on_conflict not in (None, "rename", "fail"):
            return {"error": "`on_conflict` must be 'rename' or 'fail'"}
        ok, reason = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason}
        if definition is not None:
            return await millm.post("/api/clusters/import",
                                    json_body=definition,
                                    activate=str(bool(activate)).lower(),
                                    on_conflict=on_conflict)
        body: dict[str, Any] = {"repo_id": repo_id, "filename": filename,
                                "activate": bool(activate)}
        if revision:
            body["revision"] = revision
        if on_conflict:
            # Supported by miLLM's hub route since the 009 R2 contract
            # amendment — silently dropping it ignored agents' dedupe guard.
            body["on_conflict"] = on_conflict
        return await millm.post("/api/clusters/hub/import", json_body=body)

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_hub_search(q: Optional[str] = None,
                               base_model: Optional[str] = None,
                               limit: int = 20) -> Any:
        """Search public Hugging Face cluster packs (repos tagged
        mistudio-cluster-definition), optionally narrowed to a base model."""
        return await millm.get("/api/clusters/hub/search",
                               q=q, base_model=base_model, limit=limit)

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_activate_cluster(cluster_id: str) -> Any:
        """Activate an imported cluster (applies all members at
        sign x strength x lambda; hard compatibility gate server-side)."""
        return await millm.post(f"/api/clusters/{cluster_id}/activate")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_deactivate_cluster(cluster_id: str) -> Any:
        """Deactivate a cluster (clears its live steering)."""
        return await millm.post(f"/api/clusters/{cluster_id}/deactivate")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_export_cluster(cluster_id: str) -> Any:
        """Re-export a cluster's lossless original v1 document (byte-honest —
        survives unknown additive fields)."""
        return await millm.raw_get(f"/api/clusters/{cluster_id}/export")
