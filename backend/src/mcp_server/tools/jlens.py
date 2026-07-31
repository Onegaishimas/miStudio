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

from typing import Annotated, Any, List, Optional

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
    async def jlens_readout(
        model_id: Annotated[str, Field(description="miStudio model id (m_xxxxxxxx)")],
        prompt: Annotated[str, Field(description="Text to read out, max 8000 characters")],
        types: Annotated[Optional[List[str]], Field(description="LOGIT_LENS and/or JACOBIAN_LENS. Defaults to LOGIT_LENS, which needs no artifact")] = None,
        layers: Annotated[Optional[List[int]], Field(description="Absolute layer indices; omit for every layer")] = None,
        top_n: Annotated[int, Field(description="Readout depth per cell")] = 8,
        artifact_id: Annotated[Optional[str], Field(description="Required for JACOBIAN_LENS; must be the artifact fitted for THIS model's weights")] = None,
    ) -> Any:
        """Read out what a model is poised to say at every layer and position.

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
