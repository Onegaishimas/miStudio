"""J-space lens tools (category: jlens) — BR-027 full MCP parity.

Every J-space capability reachable in the workbench must be reachable by an
agent, and the tools ship WITH the feature that creates them rather than being
batched into a final one. That sequencing is not a preference: this server once
shipped 16 tools that were fully implemented, unit-tested and documented in the
contract while registered nowhere, and every test passed by importing the module
directly. `tests/unit/test_reachability.py` is the harness that now guards it.

Scope here is what EXISTS as a route. A tool calling a route that does not
exist is the same defect in a new place, so tools land as their endpoints do.
"""

from typing import Annotated, Any, Dict, List, Optional

from pydantic import Field
from mcp.server.fastmcp import FastMCP

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def list_jlens_artifacts() -> Any:
        """List J-lens artifacts present in the mounted registry.

        PRESENCE, NOT VALIDITY. An artifact appearing here has not been
        validated — run validate_jlens_artifact before trusting one. The
        consumer's lens loading fails at request time WITHOUT RAISING, so an
        unvalidated artifact presents as a feature that quietly returns
        nothing rather than as an error.
        """
        return await client.get("/jlens/artifacts")

    @mcp.tool()
    async def validate_jlens_artifact(
        slug: Annotated[str, Field(description="Artifact slug from list_jlens_artifacts")],
        d_model: Annotated[int, Field(description="Model hidden size the artifact was fitted for")],
        n_layers: Annotated[int, Field(description="Layer count the artifact should cover")],
        n_vocab: Annotated[int, Field(description="Model vocabulary size — the envelope bound is derived from it, so a wrong value makes the check meaningless")],
    ) -> Any:
        """Run the BR-030 validation suite against one artifact.

        Reports all six classes individually. `passed` is FAIL-CLOSED: the
        three checks needing a loaded model or a running consumer report
        NOT_RUN from here, so `passed` is False and that is the honest answer,
        not a defect — "we did not check" must never read like "we checked and
        it was fine".

        The model's dimensions are required rather than looked up because the
        envelope bound must come from the model the artifact was fitted for.
        The required-vs-materialised ratio scales with vocabulary, so a bound
        derived from the wrong model passes while missing a real
        materialisation.
        """
        return await client.post(
            f"/jlens/artifacts/{slug}/validate",
            d_model=d_model,
            n_layers=n_layers,
            n_vocab=n_vocab,
        )

    @mcp.tool()
    async def get_jlens_band_report(
        slug: Annotated[str, Field(description="Artifact slug from list_jlens_artifacts")],
    ) -> Any:
        """This model's OWN sensory / workspace / motor boundaries, or null.

        A null result means no band report has been computed for this model,
        and NOTHING should be inferred about where its bands lie. The published
        boundaries in the literature were measured on one specific model and do
        not transfer — miStudio has no default and will not supply one.

        The report also carries the per-layer profile, including next-token
        agreement. That figure is DESCRIPTIVE. Do not rank or gate on it: the
        J-lens is deliberately worse than the logit lens on agreement through
        most of the network (BR-004).
        """
        return await client.get(f"/jlens/artifacts/{slug}/band-report")

    @mcp.tool()
    async def get_jlens_gate(
        slug: Annotated[str, Field(description="Artifact slug from list_jlens_artifacts")],
    ) -> Any:
        """The recorded Phase-0 GO / NO-GO / GO-AT-LARGER-SCALE decision, or null.

        NO_GO is a complete, publishable outcome rather than a failure — it
        means the full workspace claim set did not replicate at this scale, and
        it BLOCKS product surface beyond the readout viewer (BR-003).

        Null means no decision has been recorded yet, which is not the same as
        GO and must not be read as one.
        """
        return await client.get(f"/jlens/artifacts/{slug}/gate")

    @mcp.tool()
    async def jlens_readout(
        model_id: Annotated[str, Field(description="miStudio model id (m_xxxxxxxx)")],
        prompt: Annotated[str, Field(description="Text to read out, max 8000 characters")],
        types: Annotated[Optional[List[str]], Field(description="LOGIT_LENS and/or JACOBIAN_LENS. Defaults to LOGIT_LENS, which needs no artifact")] = None,
        layers: Annotated[Optional[List[int]], Field(description="Absolute layer indices; omit for every layer")] = None,
        top_n: Annotated[int, Field(description="Readout depth per cell")] = 8,
        artifact_id: Annotated[Optional[str], Field(description="Required for JACOBIAN_LENS; must be the artifact fitted for THIS model's weights")] = None,
    ) -> Any:
        """QUEUE a readout of what a model is poised to say per layer and position.

        Returns a TASK ID, not a readout. Poll `get_jlens_readout(task_id)`.
        The readout is asynchronous because it needs the whole model resident
        for a forward pass — bound synchronously it exceeded the ingress
        timeout on a real model — so the first readout for a model takes about
        a minute and subsequent ones are fast.

        RUNG 0. A concept appearing in a readout is NOT a causal claim — it says
        the direction was present, not that the model used it. Raising the rung
        takes a coordinate swap with a matched control.

        Three limits worth stating before interpreting a result: readouts only
        surface concepts with SINGLE-TOKEN names; a readout that resists
        interpretation is not a null result; and absence of a signal is not
        evidence that the computation did not occur.

        The logit lens needs no artifact and works on any downloaded model.
        JACOBIAN_LENS requires a validated artifact fitted for these exact
        weights and is REFUSED without one — it is never silently answered with
        logit data under a Jacobian label.
        """
        body: dict[str, Any] = {"model_id": model_id, "prompt": prompt, "top_n": top_n}
        if types:
            body["types"] = types
        if layers:
            body["layers"] = layers
        if artifact_id:
            body["artifact_id"] = artifact_id
        return await client.post("/jlens/readout", json_body=body)

    @mcp.tool()
    async def get_jlens_readout(
        task_id: Annotated[str, Field(description="Task id returned by jlens_readout")],
    ) -> Any:
        """Poll a queued readout.

        `readout` is null until `status` is SUCCESS — a PENDING or PROGRESS
        task is NOT an empty readout, and reading it as one is exactly the
        confusion this feature exists to prevent. A FAILURE reports its reason.
        """
        return await client.get(f"/jlens/readout/{task_id}")

    @mcp.tool()
    async def annotate_jlens_feature(
        model_id: Annotated[str, Field(description="miStudio model id")],
        sae_id: Annotated[str, Field(description="SAE the feature belongs to")],
        feature_id: Annotated[str, Field(description="Feature id being annotated")],
        layer: Annotated[int, Field(description="Layer the feature lives at")],
        direction: Annotated[List[float], Field(description="The feature's decoder direction, d_model long")],
        label_tokens: Annotated[Optional[List[str]], Field(description="The feature's existing label, to compare the readout against. Omit and no disagreement is computed")] = None,
        top_k: Annotated[int, Field(description="How many readout tokens to return")] = 8,
    ) -> Any:
        """Describe an SAE feature in J-space: what it pushes TOWARD.

        TWO INDEPENDENT FIELDS, and the second one matters. `lens_kurtosis` is
        geometric; `workspace_class` is behavioural. High kurtosis ALONE is not
        workspace alignment — a MOTOR feature is sharp too, so classifying on
        kurtosis would call every motor feature a workspace feature.

        `workspace_class` is UNKNOWN unless a band report exists for this
        model. That is a real answer, not a failure: without boundaries
        measured here there is no principled middle of the stack.

        RUNG 0. An annotation is an observation about a direction, not a claim
        that the feature causes anything.
        """
        body: dict[str, Any] = {
            "model_id": model_id,
            "sae_id": sae_id,
            "feature_id": feature_id,
            "layer": layer,
            "direction": direction,
            "top_k": top_k,
        }
        if label_tokens:
            body["label_tokens"] = label_tokens
        return await client.post("/jlens/annotate", json_body=body)

    @mcp.tool()
    async def create_jlens_watchlist(
        name: Annotated[str, Field(description="Watchlist name")],
        artifact_ref: Annotated[str, Field(description="The artifact its directions live in. Lens coordinates are artifact-specific and mean nothing elsewhere")],
        scoring_definition: Annotated[str, Field(description="HOW the score is computed. REQUIRED: a threshold applied to a differently computed score is a different detector, and the consumer cannot notice")],
        concepts: Annotated[List[Dict[str, Any]], Field(description="[{token, threshold}] pairs")],
        control_set: Annotated[Optional[List[str]], Field(description="Unrelated concrete nouns the score is measured against")] = None,
    ) -> Any:
        """Author a watchlist for miLLM to evaluate per token at inference.

        miStudio EMITS; runtime evaluation is miLLM's plane.

        A watchlist is a detector definition, not a list of words: directions,
        thresholds and the scoring definition travel together or none of them
        mean anything. Missing either the scoring definition or the artifact
        reference is refused rather than exported and discovered later.
        """
        body: dict[str, Any] = {
            "name": name,
            "artifact_ref": artifact_ref,
            "scoring_definition": scoring_definition,
            "concepts": concepts,
        }
        if control_set:
            body["control_set"] = control_set
        return await client.post("/jlens/watchlists", json_body=body)

    @mcp.tool()
    async def jlens_cost_estimate(
        operation: Annotated[str, Field(description="artifact_construction | readout | decomposition | annotation_sweep | intervention_run | template_lens_build")],
        d_model: Annotated[int, Field(description="Model hidden size")],
        n_layers: Annotated[int, Field(description="Layer count")],
        n_positions: Annotated[int, Field(description="Prompt length in tokens")] = 1,
        n_prompts: Annotated[int, Field(description="Corpus size, for a fit")] = 1,
        n_features: Annotated[int, Field(description="Dictionary size, for a sweep")] = 1,
    ) -> Any:
        """Estimate an operation's cost BEFORE committing to it.

        CALL THIS FIRST for anything larger than a single readout. An
        annotation sweep over a 32k-feature dictionary and one readout differ by
        orders of magnitude, and there is no way to tell them apart from the
        request alone.

        Estimates are ORDER-OF-MAGNITUDE and carry their basis. An unknown
        operation is an error rather than a cheap default — a small number would
        invite exactly the run it should warn about.
        """
        return await client.get(
            "/jlens/cost-estimate",
            operation=operation,
            d_model=d_model,
            n_layers=n_layers,
            n_positions=n_positions,
            n_prompts=n_prompts,
            n_features=n_features,
        )

    @mcp.tool()
    async def get_jlens_replication_report(
        slug: Annotated[str, Field(description="Artifact slug")],
    ) -> Any:
        """The recorded replication report, or null (BR-001).

        Published whether favourable or not. A partial run reports as partial —
        `complete: false` with the missing evaluation sets named — rather than
        as a clean table over whatever happened to finish.
        """
        return await client.get(f"/jlens/reports/replication?slug={slug}")

    @mcp.tool()
    async def fit_jlens_artifact(
        model_id: Annotated[str, Field(description="miStudio model id to fit a lens for")],
        prompts: Annotated[List[str], Field(description="Fitting corpus. The fitter REFUSES fewer than 100 — an under-fitted lens is indistinguishable from a fitted one by inspection")],
        layers: Annotated[Optional[List[int]], Field(description="Absolute layer indices; omit for every layer")] = None,
        freeze_qk: Annotated[bool, Field(description="Freeze Q/K as well as norms. INAPPLICABLE on layers that do not attend, and recorded per layer rather than claimed wholesale")] = True,
        corpus_name: Annotated[str, Field(description="Recorded in the artifact's recipe (BR-007) — name the corpus, do not leave it unspecified")] = "unspecified",
    ) -> Any:
        """Queue a J-lens fit. GPU-bound and long-running; poll get_task_status.

        Fitting is the PRIMARY path, not a fallback: pre-fitted lenses exist
        for a limited model set and most models this workbench runs are not in
        it.

        The result carries a per-check validation report. An artifact that is
        `serviceable` can be read out locally; `passed` additionally requires
        the two consumer-interop checks, which need a live external consumer
        and are deferred until handover.
        """
        body: dict[str, Any] = {
            "model_id": model_id,
            "prompts": prompts,
            "freeze_qk": freeze_qk,
            "corpus_name": corpus_name,
        }
        if layers:
            body["layers"] = layers
        return await client.post("/jlens/fit", json_body=body)
