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

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.config import settings
from ....core.deps import get_db
from ....schemas.jlens import (
    LensDoneMessage,
    LensMetaMessage,
    LensTokenMessage,
    ProbeRequest,
    ProbeScore,
    ReadoutRequest,
)
from ....services.jlens_artifact_service import (
    ArtifactNotValidated,
    JLensArtifactService,
)
from ....services.jlens_model_registry import ModelNotAvailable, load_for_readout

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


class FitRequest(BaseModel):
    """Start a fit. Prompts are supplied rather than sampled server-side.

    The corpus is part of the recipe (BR-007), so it is the caller's choice and
    is recorded in `config.yaml`. A server-chosen default corpus would produce
    artifacts whose provenance says nothing.
    """

    model_id: str
    prompts: List[str]
    layers: Optional[List[int]] = None
    freeze_qk: bool = True
    corpus_name: str = "unspecified"


class FitAccepted(BaseModel):
    task_id: str
    model_id: str
    queue: str


@router.post(
    "/fit",
    response_model=FitAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Fit a J-lens artifact for a model",
)
async def fit(request: FitRequest, db: AsyncSession = Depends(get_db)) -> FitAccepted:
    """Queue a fit. GPU-bound and long-running, so it never runs inline.

    The prompt floor (Appendix A.2) is enforced by the fitter itself and
    REFUSED rather than warned about: an under-fitted lens is indistinguishable
    from a fitted one by inspection.
    """
    from ....models.model import Model
    from ....workers.jlens_fit_tasks import fit_jlens_artifact

    result = await db.execute(select(Model).where(Model.id == request.model_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model with id {request.model_id!r}",
        )

    task = fit_jlens_artifact.delay(
        model_id=request.model_id,
        prompts=request.prompts,
        layers=request.layers,
        freeze_qk=request.freeze_qk,
        corpus_name=request.corpus_name,
    )
    return FitAccepted(task_id=task.id, model_id=request.model_id, queue="extraction")


class BandReportResponse(BaseModel):
    """A model's measured profile and the boundaries it does or does not support.

    `boundaries` is nullable and a null is the HONEST answer, not a missing
    value: bands are drawn only from a report computed for this model, and
    there is no default anywhere in the product (BR-002). The client renders
    nothing when this is null.
    """

    model_id: str
    has_bands: bool
    boundaries: Optional[Dict[str, int]]
    derivation: str
    control_seed: Optional[int]
    profiles: List[Dict[str, Any]]


class GateResponse(BaseModel):
    model_id: str
    decision: str
    rationale: str
    blocking: bool
    has_bands: bool


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


def _validated_report(loaded: Any, artifact_id: Optional[str]):
    """Locate and validate the artifact for THIS model, or refuse.

    WEIGHT IDENTITY IS PART OF THE CHECK (BR-031, FPRD §3.4). The artifact slug
    is derived from the repo id, so an artifact fitted for a base model has a
    different slug from its instruction-tuned variant. Accepting an
    `artifact_id` that does not match this model's own slug would serve a lens
    fitted for DIFFERENT WEIGHTS — which produces a complete, plausible readout
    and is undetectable downstream. Checking the model NAME alone is what makes
    that mistake easy, so the comparison is on the slug the fit would produce.

    The SEMANTIC check runs here because it needs the loaded model; the two
    consumer-interop classes cannot run without a live external consumer and are
    reported NOT_RUN, which is why serving is gated on `serviceable` rather than
    `passed`.
    """
    from ....services.jlens_artifact_service import slug_for

    service = _service()
    expected = slug_for(loaded.name)
    if artifact_id and artifact_id != expected:
        raise ArtifactNotValidated(
            f"artifact {artifact_id!r} was not fitted for {loaded.name} "
            f"(expected slug {expected!r}). A lens fitted for different weights "
            "produces a complete, plausible readout that is wrong."
        )

    ref = service.find(loaded.name)
    if ref is None:
        raise FileNotFoundError(
            f"No J-lens artifact for {loaded.name}. The logit lens needs none; "
            "the Jacobian lens does — fit and validate one first."
        )

    return service.validate(
        ref,
        d_model=loaded.d_model,
        expected_layers=range(loaded.n_layers),
        n_vocab=loaded.n_vocab,
        semantic_result=_semantic_check(loaded, ref),
    )


def _semantic_check(loaded: Any, ref: Any):
    """Does the artifact recover a known UNSPOKEN intermediate?

    Structure can be perfect while content is absent — a shuffled or
    zero-initialised J is the right shape and the right size and passes every
    other local class. The fixture's answer deliberately appears in neither the
    prompt nor the output, because a token present in the prompt is recoverable
    by an artifact that encodes nothing.
    """
    from ....services.jlens_readout_service import JacobianTransport, ReadoutService
    from ....services.jlens_validation import check_semantic

    service = _service()
    payload = service._load_payload(ref)  # noqa: SLF001 - same package concern
    if payload is None:
        from ....services.jlens_validation import CheckClass, CheckResult, CheckStatus

        return CheckResult(
            CheckClass.SEMANTIC, CheckStatus.FAIL, "artifact did not deserialize"
        )

    jacobians = {int(k): v for k, v in payload.items()}
    readout = ReadoutService(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        structure=loaded.structure,
        unembedding=loaded.unembedding,
        model_name=loaded.name,
    )

    def top_at(prompt: str, layer: int, top_k: int):
        transport = JacobianTransport(jacobians)
        for message in readout.stream(prompt, [transport], layers=[layer], top_n=top_k):
            if isinstance(message, LensTokenMessage):
                last = message
        return last.results[0].top_tokens[0]

    # Mid-band by position in the stack, not by a band constant: no band report
    # is required to say "about two thirds of the way up", and BR-002 forbids a
    # boundary constant anywhere.
    mid = max(0, int(loaded.n_layers * 2 / 3) - 1)
    return check_semantic(
        top_at,
        prompt=SEMANTIC_FIXTURE_PROMPT,
        layer=mid,
        expected_intermediate=SEMANTIC_FIXTURE_ANSWER,
    )


# The intermediate appears in NEITHER the prompt nor the expected output, so
# recovering it cannot be explained by the artifact encoding nothing.
SEMANTIC_FIXTURE_PROMPT = "The number of legs on the animal that spins webs is"
SEMANTIC_FIXTURE_ANSWER = "spider"


@router.post(
    "/readout",
    response_model=ReadoutResponse,
    summary="Position x layer lens readout",
)
async def readout(
    request: ReadoutRequest, db: AsyncSession = Depends(get_db)
) -> ReadoutResponse:
    """Return a position x layer readout in the upstream wire format (BR-029).

    LOGIT NEEDS NO ARTIFACT (BR-005) and is the default. A JACOBIAN_LENS
    request requires a VALIDATED artifact: the schema already refuses one
    without an `artifact_id`, and this handler additionally refuses an artifact
    that has not passed the suite. Falling back to identity in either case
    would serve logit data under a Jacobian label — a lower evidence rung in a
    higher rung's clothing (BR-019).
    """
    from ....models.model import Model
    from ....services.jlens_readout_service import (
        IdentityTransport,
        JacobianTransport,
        LensTransport,
        ReadoutService,
        ReadoutTooLarge,
    )

    result = await db.execute(select(Model).where(Model.id == request.model_id))
    model_record = result.scalar_one_or_none()
    if model_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model with id {request.model_id!r}",
        )

    try:
        loaded = load_for_readout(model_record)
    except ModelNotAvailable as exc:
        # 409, not 500: the request is well-formed and the server is healthy —
        # the model simply is not in a state that can serve a readout, and the
        # user can act on that.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    transports: List[LensTransport] = []
    for lens_type in request.types:
        if lens_type == "LOGIT_LENS":
            transports.append(IdentityTransport())
            continue
        try:
            jacobians = _service().load_for_readout(
                loaded.name, report=_validated_report(loaded, request.artifact_id)
            )
        except (ArtifactNotValidated, FileNotFoundError) as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
        transports.append(JacobianTransport(jacobians))

    service = ReadoutService(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        structure=loaded.structure,
        unembedding=loaded.unembedding,
        model_name=loaded.name,
    )

    meta = None
    tokens: List[LensTokenMessage] = []
    try:
        for message in service.stream(
            request.prompt,
            transports=transports,
            layers=request.layers,
            top_n=request.top_n,
        ):
            if isinstance(message, LensMetaMessage):
                meta = message
            elif isinstance(message, LensTokenMessage):
                tokens.append(message)
    except ReadoutTooLarge as exc:
        # 413, not 400: the request is valid and simply too expensive, and the
        # message carries the numbers so the caller can shrink it.
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    if meta is None:
        # Never returned as an empty success: an empty readout is
        # indistinguishable from a real one with no content, which is the
        # failure mode this whole feature is built to avoid.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Readout produced no meta message",
        )

    return ReadoutResponse(meta=meta, tokens=tokens)


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
