"""Circuit tools (category: circuits) — Feature 018, IDL-33/IDL-35.

Rung-aware access to circuits: every returned circuit/edge carries its
evidence rung AND the server-rendered rung_language string — agents MUST use
that language verbatim and never describe rung-0/1 artifacts with causal
words (the evidence-ladder discipline). The agent loop: run discovery (016
tools, when present) → review candidates → create/promote circuits here →
steer them (steer_combined with per-member sae_id) → export definitions or
per-layer v1 slices for today's single-SAE consumers.
"""

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def list_circuits(
        promoted: Optional[bool] = None,
        min_rung: Optional[int] = None,
        granularity: Optional[str] = None,
    ) -> Any:
        """List circuits with rung + rung_language on every row. min_rung
        filters by evidence rung (0 mined, 1 attribution-supported,
        2 causally validated, 3 faithfulness-tested)."""
        params: dict[str, Any] = {}
        if promoted is not None:
            params["promoted"] = promoted
        if min_rung is not None:
            params["min_rung"] = min_rung
        if granularity:
            params["granularity"] = granularity
        return await client.get("/circuits", **params)

    @mcp.tool()
    async def get_circuit(circuit_id: str) -> Any:
        """One circuit: members by layer, typed edges with full evidence
        (statistics, attribution, validation manifest refs), budget,
        faithfulness, rung + rung_language + what-moves-it-up."""
        return await client.get(f"/circuits/{circuit_id}")

    @mcp.tool()
    async def create_circuit(
        name: str,
        saes: list,
        members: list,
        edges: Optional[list] = None,
        narrative: Optional[str] = None,
        budget: Optional[dict] = None,
        granularity: str = "feature",
        discovery_run_id: Optional[str] = None,
    ) -> Any:
        """Create a circuit from discovery candidates or hand assembly.
        Contract rules enforced server-side: per-layer member caps (20 PER
        LAYER), edges must ascend layers and reference members. Rejections
        return the exact violation."""
        return await client.post("/circuits", json_body={
            "name": name, "saes": saes, "members": members,
            "edges": edges or [], "narrative": narrative, "budget": budget,
            "granularity": granularity, "discovery_run_id": discovery_run_id,
        })

    @mcp.tool()
    async def promote_circuit(circuit_id: str) -> Any:
        """Promote a circuit into a loadable multi-layer steering profile.
        Promotion is a BADGE, not a gate — unvalidated circuits promote and
        carry their rung visibly everywhere."""
        return await client.post(f"/circuits/{circuit_id}/promote")

    @mcp.tool()
    async def export_circuit_definition(circuit_id: str) -> Any:
        """Export mistudio.circuit-definition/v1 JSON (lossless: rungs, edge
        types, attribution, manifest refs, provenance all travel)."""
        return await client.get(f"/circuits/{circuit_id}/export")

    @mcp.tool()
    async def export_circuit_slices(circuit_id: str) -> Any:
        """Export per-layer cluster-definition/v1 slices (BR-014) for today's
        single-SAE consumers (miLLM). Each slice is a PARTIAL RENDERING —
        parent rung travels in the response and in each slice's provenance;
        never present a slice as the validated whole."""
        return await client.post(f"/circuits/{circuit_id}/export-slices")
