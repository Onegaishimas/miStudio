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
from typing import List

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ....schemas.jlens import (
    LensDoneMessage,
    LensMetaMessage,
    LensTokenMessage,
    ProbeRequest,
    ProbeScore,
    ReadoutRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jlens", tags=["jlens"])


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
