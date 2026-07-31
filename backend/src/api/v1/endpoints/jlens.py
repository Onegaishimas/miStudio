"""
J-space readout endpoints.

Emits the upstream lens wire format verbatim (BR-029, PADR IDL-45) so a
miStudio stream and a Neuronpedia stream are interchangeable at the client and
the readout panel is driven by either with no adaptation layer.

The LOGIT lens needs no artifact and is the default (BR-005). Requesting
JACOBIAN_LENS without an artifact is refused at the schema, not silently served
as logit data under a Jacobian label — that would breach rung discipline
(BR-019).
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ....core.config import settings
from ....schemas.jlens import (
    LensDoneMessage,
    LensMetaMessage,
    LensTokenMessage,
    ProbeRequest,
    ProbeScore,
    ReadoutRequest,
)
from ....services.jlens_artifact_service import JLensArtifactService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jlens", tags=["jlens"])


def _service() -> JLensArtifactService:
    return JLensArtifactService(settings.jlens_artifacts_dir)


class ArtifactSummary(BaseModel):
    """One artifact as it exists ON DISK — presence, not validity.

    `validated` is deliberately absent from this shape: an artifact's validity
    is the outcome of running the suite, not a property of the file, and a
    field here would be read as a verdict the listing never computed.
    """

    slug: str
    directory: str
    lens_file: str
    size_bytes: int
    has_config: bool


class CheckOutcome(BaseModel):
    check: str
    status: str
    detail: str
    evidence: Dict[str, Any] = {}


class ValidationResponse(BaseModel):
    """The suite's verdict, with every class reported individually.

    `passed` is FAIL-CLOSED: a class that could not run is not a pass. The
    three live checks need a loaded model or a running consumer, so a
    validation performed from here reports them NOT_RUN and `passed` is False
    — which is the honest answer, not a defect.
    """

    slug: str
    passed: bool
    summary: str
    results: List[CheckOutcome]


@router.get(
    "/artifacts",
    response_model=List[ArtifactSummary],
    summary="J-lens artifacts present in the mounted registry",
)
async def list_artifacts() -> List[ArtifactSummary]:
    """List conformant artifact directories.

    Staging directories are excluded — an artifact still being written is not
    an artifact, and the whole point of staging is that it is invisible until
    it commits.
    """
    return [
        ArtifactSummary(
            slug=ref.slug,
            directory=str(ref.directory),
            lens_file=ref.lens_path.name,
            size_bytes=ref.size_bytes,
            has_config=ref.config_path is not None,
        )
        for ref in _service().list_artifacts()
    ]


@router.post(
    "/artifacts/{slug}/validate",
    response_model=ValidationResponse,
    summary="Run the artifact validation suite (BR-030)",
)
async def validate_artifact(
    slug: str,
    d_model: int,
    n_layers: int,
    n_vocab: int,
) -> ValidationResponse:
    """Run every check that does not require a loaded model or a live consumer.

    The model's dimensions are REQUIRED parameters rather than looked up,
    because the envelope bound must come from the model the artifact was fitted
    for. Defaulting them would produce a bound derived from nothing, and a
    wrong envelope bound passes on one model while missing a real
    materialisation on another.
    """
    service = _service()
    ref = next((a for a in service.list_artifacts() if a.slug == slug), None)
    if ref is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No conformant J-lens artifact directory named {slug!r}",
        )

    report = service.validate(
        ref, d_model=d_model, expected_layers=range(n_layers), n_vocab=n_vocab
    )
    return ValidationResponse(
        slug=slug,
        passed=report.passed,
        summary=report.summary(),
        results=[
            CheckOutcome(
                check=r.check.value,
                status=r.status.value,
                detail=r.detail,
                evidence=r.evidence,
            )
            for r in report.results
        ],
    )


class ReadoutResponse(BaseModel):
    """Non-streaming envelope: the meta message plus its token messages.

    CONTAINS a meta message rather than INHERITING one. Subclassing
    LensMetaMessage carried its `kind: "meta"` discriminator onto the envelope,
    so the response announced itself as a meta message while also carrying a
    `tokens` array — a client dispatching on `kind` would mis-handle it, in the
    one format this feature exists to conform to.

    A streaming transport (SSE/WebSocket) can be added later without changing
    the message shapes, which is the point of adopting the upstream format.
    """

    meta: LensMetaMessage
    tokens: List[LensTokenMessage]


@router.post(
    "/readout",
    response_model=ReadoutResponse,
    summary="Position x layer lens readout",
)
async def readout(request: ReadoutRequest) -> ReadoutResponse:
    """Return a position x layer readout in the upstream wire format.

    NOTE the model-resolution and service-construction wiring lands with
    feature 021 (artifact lifecycle), which owns model loading for J-space.
    Until then this endpoint validates the request and reports that no readout
    backend is bound, rather than returning a fabricated stream — a plausible
    empty readout would be indistinguishable from a real one with no content.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Readout backend is not bound yet. The service and wire format are "
            "implemented and tested (feature 022); model resolution and "
            "artifact loading land with feature 021. Returning an empty "
            "readout here would be indistinguishable from a real readout with "
            "no content."
        ),
    )


@router.post(
    "/probe",
    response_model=List[ProbeScore],
    summary="Score named directions without ranking the vocabulary",
)
async def probe(request: ProbeRequest) -> List[ProbeScore]:
    """Probe mode (BR-008).

    Distinct from the full ranked readout: the two can disagree because ranking
    applies a data-dependent normalisation this does not, so which mode is
    canonical must be recorded per analysis.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Probe backend is not bound yet; see /jlens/readout.",
    )
