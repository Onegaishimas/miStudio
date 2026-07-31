"""J-space lens tools (category: jlens) — BR-027 full MCP parity.

Every J-space capability reachable in the workbench must be reachable by an
agent, and the tools ship WITH the feature that creates them rather than being
batched into a final one. That sequencing is not a preference: this server once
shipped 16 tools that were fully implemented, unit-tested and documented in the
contract while registered nowhere, and every test passed by importing the module
directly. `tests/unit/test_reachability.py` is the harness that now guards it.

Scope here is what exists: the artifact registry and its validation suite.
Readout, band-report and gate tools land as their endpoints do — a tool calling
a route that does not exist is the same defect in a new place.
"""

from typing import Annotated, Any

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
