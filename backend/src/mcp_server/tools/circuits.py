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

        PASS `discovery_run_id` IF YOU HAVE ONE. Omitting it PERMANENTLY
        forfeits `run_circuit_faithfulness`, which draws its prompts from that
        capture store and 409s without it. The failure surfaces later, at a
        different tool, and the only remedy is recreating the circuit. Any
        hand-assembled circuit therefore cannot be faithfulness-tested.

        Contract rules enforced server-side: per-layer member caps (20 PER
        LAYER), edges must ascend layers and reference members. Rejections
        return the exact violation. model_hf_id (HF repo id) is the
        cross-instance-stable model provenance carried by exports.

        STRENGTHS live on members (`member.feature.strength`) and are what
        actually steers — edges carry evidence only. Measured envelope: about
        0.15 x the feature's REAL max_activation per member, total ~3 across
        two layers; a circuit at ~50x that emitted pure token soup in
        production. See mistudio_howto('strength_calibration')."""
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
        first-class records).

        IRREVERSIBLE, and this tool does NOT check whether the circuit is
        serving in miLLM — unlike its sibling `millm_delete_circuit`, which
        refuses a serving circuit without `acknowledge_serving`. Deleting one
        that miLLM has imported does not stop it serving there, but you lose
        the definition. Export first (`export_circuit_definition`) if you may
        want it back."""
        return await client.delete(f"/circuits/{circuit_id}")

    @mcp.tool()
    async def import_circuit_definition(definition: dict) -> Any:
        """Import a mistudio.circuit-definition/v1 document (the BR-013
        round-trip). Unknown kinds are rejected explicitly."""
        return await client.post("/circuits/import", json_body=definition)

    @mcp.tool()
    async def export_circuit_definition(circuit_id: str) -> Any:
        """Export the portable circuit-definition JSON (lossless: rungs, edge
        types, attribution, manifest refs, provenance all travel). This is
        the raw contract document — for the human-readable rung_language
        string, use get_circuit/list_circuits.

        FEEDING THIS TO miLLM: the document's `saes[].mistudio_sae_id` must be
        the id MILLM knows (e.g.
        `mistudio--sae-<model>--layer_12--width_8k--res`), NOT miStudio's
        internal `sae_xxxxxxxx` row id. They are DIFFERENT NAMESPACES for the
        same SAE and no tool translates between them — read miLLM's ids from
        `millm_status`. A mismatched id imports fine and only fails at
        activation, as an unbound SAE.

        The `kind` field is `mistudio.circuit-definition` with NO `/v1`
        suffix, despite the schema file and docs saying v1."""
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
        capture. Managed GPU task.

        POLL `list_circuit_captures` (NOT get_task_status). This endpoint
        returns a raw Celery task id and creates no task-queue row, so
        get_task_status cannot see it — status lives on the capture run.

        Serialised against ALL other circuit GPU work (attribution,
        validation, faithfulness) by one advisory lock, ACROSS UNRELATED
        circuits: a 409 may name a run you do not own. Back off and poll; do
        not retry immediately."""
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
        GPU task.

        POLL `get_discovery_results` and read `attribution_status` (NOT
        get_task_status — this returns a raw Celery id with no task-queue
        row). Terminal when it leaves pending/running.

        Holds the shared circuit-GPU lock (see start_circuit_capture); a 409
        may name unrelated work."""
        return await client.post(f"/circuit-discovery/{run_id}/attribution",
                                 json_body={"prompt_limit": prompt_limit})

    # ── Feature 017: validation + faithfulness + manifests ───────────────

    @mcp.tool()
    async def validate_circuit_edges(
        run_id: str,
        ordering: str = "coact",
        k: int = 20,
        prompts_per_edge: int = 8,
        null_samples: int = 20,
        sign_frac: float = 0.8,
        baseline: str = "zero",
    ) -> Any:
        """Causally validate the top-K edges of a discovery run (rung 2):
        suppress the upstream feature, run the model, measure the downstream
        effect size vs a shuffled-non-edge null; rung-2 iff |ES| beats the null
        percentile AND is sign-consistent. This is the REAL causal tier (rung
        2) — the tier where causal language is earned. ordering: coact | attr
        (attr needs an attribution pass first). Returns a run id; poll
        get_task_status then get_discovery_results for the per-edge verdicts."""
        return await client.post(
            f"/circuit-discovery/{run_id}/validate", json_body={
                "ordering": ordering, "k": k, "prompts_per_edge": prompts_per_edge,
                "null_samples": null_samples, "sign_frac": sign_frac,
                "baseline": baseline})

    @mcp.tool()
    async def run_circuit_faithfulness(
        circuit_id: str,
        mode: str = "both",
        k_nonmembers: int = 256,
        ablate_all_n: int = 1024,
        n_prompts: int = 16,
        seed: int = 0,
    ) -> Any:
        """Faithfulness-test a circuit (rung 3 — the HIGHEST tier): suppress
        its members and measure how much of the behavior they drive is
        NECESSARY (ablating them collapses it) and, with mode='both',
        SUFFICIENT (ablating only non-members leaves it standing) — vs a
        per-layer top-N 'ablate all' proxy (N recorded). Behavior metric v1 =
        the summed activation of the circuit's downstream-most members over
        prompts from the circuit's own capture store (SAME tokenization; never
        re-decoded). mode='necessity' skips the sufficiency probe (marked
        untested, never silently omitted). Rung 3 sits above rung 2 causal
        validation, so the score is a real faithfulness measurement — the
        manifest records the exact metric so the number is never trusted blind.
        Returns a task
        id.

        POLL `get_circuit` and read `faithfulness_status`, then
        `circuit.faithfulness` for the result (NOT get_task_status — raw
        Celery id, no task-queue row).

        REQUIRES the circuit to have a `discovery_run_id`: its prompts come
        from that capture store. A circuit created WITHOUT one — any
        hand-authored circuit — is refused 409 here and can never be
        faithfulness-tested. That is decided at create_circuit, not here.

        Holds the shared circuit-GPU lock; a 409 may name unrelated work."""
        return await client.post(
            f"/circuits/{circuit_id}/faithfulness", json_body={
                "mode": mode, "k_nonmembers": k_nonmembers,
                "ablate_all_n": ablate_all_n, "n_prompts": n_prompts,
                "seed": seed})

    @mcp.tool()
    async def get_validation_manifest(manifest_id: str) -> Any:
        """A validation manifest — the SELF-CONTAINED, reproducible record of a
        validation run (intervention config, baseline, prompts, seeds, null
        summary, per-edge effect sizes). The evidence behind a rung-2 claim."""
        return await client.get(f"/validation-manifests/{manifest_id}")

    @mcp.tool()
    async def list_validation_manifests(
        discovery_run_id: Optional[str] = None,
        circuit_id: Optional[str] = None,
    ) -> Any:
        """List validation manifests for a discovery run or a circuit."""
        params: dict[str, Any] = {}
        if discovery_run_id:
            params["discovery_run_id"] = discovery_run_id
        if circuit_id:
            params["circuit_id"] = circuit_id
        return await client.get("/validation-manifests", **params)

    @mcp.tool()
    async def reproduce_validation(manifest_id: str) -> Any:
        """Re-execute an edge_batch manifest from its payload and compare —
        the test that a rung-2 claim is reproducible, not a one-off. Returns a
        task id; the reproduction manifest carries per-edge deltas + a
        within-tolerance verdict."""
        return await client.post(
            f"/validation-manifests/{manifest_id}/reproduce")
