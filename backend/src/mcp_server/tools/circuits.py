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
        discovery: Optional[dict] = None,
        faithfulness: Optional[dict] = None,
        model_id: Optional[str] = None,
        model_hf_id: Optional[str] = None,
    ) -> Any:
        """Create a circuit from discovery candidates or hand assembly.
        Contract rules enforced server-side: per-layer member caps (20 PER
        LAYER), edges must ascend layers and reference members. Rejections
        return the exact violation. model_hf_id (HF repo id) is the
        cross-instance-stable model provenance carried by exports."""
        return await client.post("/circuits", json_body={
            "name": name, "saes": saes, "members": members,
            "edges": edges or [], "narrative": narrative, "budget": budget,
            "granularity": granularity, "discovery_run_id": discovery_run_id,
            "discovery": discovery, "faithfulness": faithfulness,
            "model_id": model_id, "model_hf_id": model_hf_id,
        })

    @mcp.tool()
    async def promote_circuit(circuit_id: str, promoted: bool = True) -> Any:
        """Promote a circuit into a loadable multi-layer steering profile —
        or unpromote it (promoted=false). Promotion is a BADGE, not a gate —
        unvalidated circuits promote and carry their rung visibly everywhere."""
        return await client.post(f"/circuits/{circuit_id}/promote",
                                 json_body={"promoted": promoted})

    @mcp.tool()
    async def update_circuit(
        circuit_id: str,
        name: Optional[str] = None,
        narrative: Optional[str] = None,
        members: Optional[list] = None,
        edges: Optional[list] = None,
        budget: Optional[dict] = None,
        granularity: Optional[str] = None,
    ) -> Any:
        """Edit a circuit (rename, fix a narrative, drop a bad edge, adjust
        members, switch granularity) — the agent review loop is not
        create-only. Structural contract rules re-validate on every update."""
        body = {k: v for k, v in {
            "name": name, "narrative": narrative, "members": members,
            "edges": edges, "budget": budget,
            "granularity": granularity}.items() if v is not None}
        return await client.patch(f"/circuits/{circuit_id}", json_body=body)

    @mcp.tool()
    async def delete_circuit(circuit_id: str) -> Any:
        """Delete a circuit permanently (its manifests survive — they are
        first-class records)."""
        return await client.delete(f"/circuits/{circuit_id}")

    @mcp.tool()
    async def import_circuit_definition(definition: dict) -> Any:
        """Import a mistudio.circuit-definition/v1 document (the BR-013
        round-trip). Unknown kinds are rejected explicitly."""
        return await client.post("/circuits/import", json_body=definition)

    @mcp.tool()
    async def export_circuit_definition(circuit_id: str) -> Any:
        """Export mistudio.circuit-definition/v1 JSON (lossless: rungs, edge
        types, attribution, manifest refs, provenance all travel). This is
        the raw contract document — for the human-readable rung_language
        string, use get_circuit/list_circuits."""
        return await client.get(f"/circuits/{circuit_id}/export")

    @mcp.tool()
    async def export_circuit_slices(circuit_id: str) -> Any:
        """Export per-layer cluster-definition/v1 slices (BR-014) for today's
        single-SAE consumers (miLLM). Each slice is a PARTIAL RENDERING —
        parent rung travels in the response and in each slice's provenance;
        never present a slice as the validated whole."""
        return await client.post(f"/circuits/{circuit_id}/export-slices")

    # ── Feature 016: capture → discovery → attribution ───────────────────

    @mcp.tool()
    async def start_circuit_capture(
        dataset_id: str,
        layers: list,
        model_id: Optional[str] = None,
        epsilon: float = 0.1,
        sample_cap: int = 2000,
        attention_capture: Optional[dict] = None,
        confirm: bool = False,
    ) -> Any:
        """Start a circuit-capture run (per-token multi-layer SAE activations
        with FIRST-CLASS positions + error norms). layers = [{layer, sae_id}].
        confirm=false returns a cost ESTIMATE only (probe); call again with
        confirm=true — or start_circuit_capture_confirm — to launch the full
        capture. Managed GPU task: poll get_task_status."""
        return await client.post("/circuit-capture", json_body={
            "dataset_id": dataset_id, "model_id": model_id, "layers": layers,
            "epsilon": epsilon, "sample_cap": sample_cap,
            "attention_capture": attention_capture, "confirm": confirm,
        })

    @mcp.tool()
    async def list_circuit_captures() -> Any:
        """List capture runs (status, corpus, layers, split, size, stale flag)."""
        return await client.get("/circuit-capture")

    @mcp.tool()
    async def run_circuit_discovery(
        capture_run_id: str,
        granularity: str = "feature",
        mode: str = "open",
        seed_refs: Optional[list] = None,
        s_min: int = 20,
        null_shuffles: int = 100,
        fdr_q: float = 0.05,
        force: bool = False,
    ) -> Any:
        """Mine a completed capture store for candidate cross-layer edges.
        granularity: feature | cluster (supernodes over curated profiles —
        recommended for seeded mode). mode: seeded (seed_refs = [{layer,
        feature_idx}|{layer, cluster_profile_id}]) | open. Statistically
        disciplined: PMI over raw counts, within-document circular-shift null,
        pooled-standardized BH-FDR, held-out replication. ALL co-activation is
        lag-0 (same token position) — attention-mediated structure is not
        found here (Tier-2.5). Returns a run id; get_discovery_results carries
        the first-class report."""
        return await client.post("/circuit-discovery", json_body={
            "capture_run_id": capture_run_id, "granularity": granularity,
            "mode": mode, "seed_refs": seed_refs, "s_min": s_min,
            "null_shuffles": null_shuffles, "fdr_q": fdr_q, "force": force,
        })

    @mcp.tool()
    async def get_discovery_results(run_id: str,
                                    include_candidates: bool = True) -> Any:
        """A discovery run + its report (null-model summary, FDR discipline,
        held-out replication RATE, stage counts, caps, uncovered seeds, lag-0
        disclosure) + ranked candidates (PMI, support, null percentile,
        attribution when the pass has run, both orderings). The report is the
        trust surface — read it before trusting the list."""
        return await client.get(f"/circuit-discovery/{run_id}",
                                include_candidates=include_candidates)

    @mcp.tool()
    async def run_attribution_pass(run_id: str,
                                   prompt_limit: Optional[int] = None) -> Any:
        """Tier-2 gradient-attribution pass over a discovery run's candidates:
        re-ranks the shortlist before 017's causal validation and gates rung-1
        (attribution_supported) by sign-agreement + magnitude. This is
        attribution, NOT causal proof — rung stays ≤1 until 017 intervenes.
        Both orderings are preserved so 017 can report survival-rate uplift.
        GPU task: poll get_task_status, then get_discovery_results."""
        return await client.post(f"/circuit-discovery/{run_id}/attribution",
                                 json_body={"prompt_limit": prompt_limit})
