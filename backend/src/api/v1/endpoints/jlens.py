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
    #: Fixture for the SEMANTIC validation class: {prompt, expected_intermediate,
    #: layer?, top_k?}. Optional, and its absence FAILS CLOSED — the artifact is
    #: fitted and then discarded unpublished, because publishing on a check that
    #: never ran is the failure this suite exists to prevent. The intermediate
    #: must not appear in the prompt: a token already present is recovered by an
    #: artifact encoding nothing at all.
    semantic_probe: Optional[Dict[str, Any]] = None


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
        semantic_probe=request.semantic_probe,
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


@router.get(
    "/artifacts/{slug}/band-report",
    response_model=Optional[BandReportResponse],
    summary="This model's own sensory / workspace / motor boundaries",
)
async def band_report(slug: str) -> Optional[BandReportResponse]:
    """Return the stored band report, or NULL when there is none.

    A null body is the honest answer and the client draws no bands (BR-002).
    It is not a 404: the artifact exists, it simply has no report yet, and
    those are different facts. Boundaries measured on another model are never
    substituted — there is no default anywhere in the product.
    """
    from ....services.jlens_band_service import load_band_report

    ref = next((a for a in _service().list_artifacts() if a.slug == slug), None)
    if ref is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No conformant J-lens artifact directory named {slug!r}",
        )

    stored = load_band_report(ref.directory)
    if stored is None:
        return None

    return BandReportResponse(
        model_id=stored.get("model_id", slug),
        has_bands=stored.get("boundaries") is not None,
        boundaries=stored.get("boundaries"),
        derivation=stored.get("derivation", ""),
        control_seed=stored.get("control_seed"),
        profiles=stored.get("profiles", []),
    )


@router.get(
    "/artifacts/{slug}/gate",
    response_model=Optional[GateResponse],
    summary="The recorded Phase-0 GO / NO-GO decision",
)
async def gate(slug: str) -> Optional[GateResponse]:
    """Return the recorded gate decision, or NULL when none has been made.

    NO_GO reads back exactly like GO and is a complete, publishable outcome
    (BR-003) — not an error state and not an absence.
    """
    from ....services.jlens_band_service import load_gate

    ref = next((a for a in _service().list_artifacts() if a.slug == slug), None)
    if ref is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No conformant J-lens artifact directory named {slug!r}",
        )

    stored = load_gate(ref.directory)
    if stored is None:
        return None
    return GateResponse(
        model_id=stored.get("model_id", slug),
        decision=stored["decision"],
        rationale=stored.get("rationale", ""),
        blocking=stored.get("blocking", True),
        has_bands=stored.get("has_bands", False),
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


# Validation is cached per (slug, mtime, size). Without this, EVERY Jacobian
# readout re-runs the suite — including the SEMANTIC check, which is itself a
# full readout — so each request paid for two readouts plus a revalidation. The
# key includes mtime and size so a replaced artifact is revalidated rather than
# served on a stale verdict.
_VALIDATION_CACHE: Dict[tuple, Any] = {}


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

    stat = ref.lens_path.stat()
    key = (ref.slug, stat.st_mtime_ns, stat.st_size, loaded.d_model, loaded.n_layers)
    cached = _VALIDATION_CACHE.get(key)
    if cached is not None:
        return cached

    report = service.validate(
        ref,
        d_model=loaded.d_model,
        expected_layers=range(loaded.n_layers),
        n_vocab=loaded.n_vocab,
        semantic_result=_semantic_check(loaded, ref),
    )
    _VALIDATION_CACHE[key] = report
    return report


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
    # Built ONCE. JacobianTransport casts every matrix to the compute dtype in
    # its constructor — deliberately, so `apply` does not copy a d_model^2
    # matrix per call — and constructing it inside the closure moved that cost
    # back to per-invocation, over the whole artifact.
    transport = JacobianTransport(jacobians)

    def top_at(prompt: str, layer: int, top_k: int):
        last = None
        for message in readout.stream(prompt, [transport], layers=[layer], top_n=top_k):
            if isinstance(message, LensTokenMessage):
                last = message
        if last is None:
            # An empty stream is a FAILED semantic check, not a NameError and
            # not an empty pass — the distinction this feature exists for.
            raise ValueError("readout produced no token messages")
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


class ReadoutAccepted(BaseModel):
    """A queued readout. The result arrives via the task, not this response.

    202, not 200: the readout has been ACCEPTED and not performed. Returning a
    body that looked like a readout would be the same lie the 501 refused to
    tell.
    """

    task_id: str
    model_id: str
    status: str = "queued"


@router.post(
    "/readout",
    response_model=ReadoutAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a position x layer lens readout",
)
async def readout(
    request: ReadoutRequest, db: AsyncSession = Depends(get_db)
) -> ReadoutAccepted:
    """Queue a readout and return its task id. Poll `/jlens/readout/{task_id}`.

    ASYNCHRONOUS BECAUSE IT MEASURABLY HAD TO BE. Bound synchronously, this
    endpoint 502'd at the ingress twice on a real model — 64.9s and 54.0s
    against nginx's 60s ceiling — because a J-space readout needs the whole
    model resident for its forward pass and loading it takes about a minute on
    CPU. Raising the proxy timeout would not bound it: readout cost is
    O(positions x layers x top_n) ON TOP of the load.

    Queueing also puts the readout in the process that can CACHE the loaded
    model across requests. A cache in the API process cannot help the worker
    and vice versa, so this is what makes the first-load cost payable once.

    Every other model-bound operation here already works this way.
    """
    from ....models.model import Model
    from ....workers.jlens_readout_tasks import compute_readout

    result = await db.execute(select(Model).where(Model.id == request.model_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model with id {request.model_id!r}",
        )

    task = compute_readout.delay(
        model_id=request.model_id,
        prompt=request.prompt,
        types=list(request.types),
        layers=request.layers,
        top_n=request.top_n,
        artifact_id=request.artifact_id,
    )
    return ReadoutAccepted(task_id=task.id, model_id=request.model_id)


class ReadoutResult(BaseModel):
    """A readout task's state, and its payload once ready.

    `readout` is null until the task succeeds. A caller that treats a pending
    task as an empty readout reproduces exactly the confusion this feature
    exists to prevent, so `status` is always present and always authoritative.
    """

    task_id: str
    status: str
    stage: Optional[str] = None
    readout: Optional[ReadoutResponse] = None
    error: Optional[str] = None


@router.get(
    "/readout/{task_id}",
    response_model=ReadoutResult,
    summary="Poll a queued readout",
)
async def readout_result(task_id: str) -> ReadoutResult:
    """Report a readout task's state.

    A FAILED task reports its reason rather than an empty readout — the
    distinction the 501 was protecting and that survives here.
    """
    import asyncio

    from ....core.celery_app import celery_app

    def _read():
        async_result = celery_app.AsyncResult(task_id)
        return async_result.state, async_result.info

    state, info = await asyncio.to_thread(_read)

    if state == "SUCCESS":
        return ReadoutResult(
            task_id=task_id, status=state, readout=ReadoutResponse(**info)
        )
    if state == "FAILURE":
        return ReadoutResult(task_id=task_id, status=state, error=str(info))
    return ReadoutResult(
        task_id=task_id,
        status=state,
        stage=(info or {}).get("stage") if isinstance(info, dict) else None,
    )


class ProbeAccepted(BaseModel):
    """Queued probe. Same two-step contract as the readout, for the same reason."""

    task_id: str
    model_id: str
    status: str = "queued"


class ProbeResult(BaseModel):
    """A polled probe. `scores` is null until `status` is SUCCESS."""

    task_id: str
    status: str
    stage: Optional[str] = None
    scores: Optional[List[ProbeScore]] = None
    #: Which mode produced these numbers. Probe and full-ranking scores can
    #: disagree, so an analysis that does not say which it used cannot be
    #: compared against one that does (BR-008).
    mode: Optional[str] = None
    lens_type: Optional[str] = None
    error: Optional[str] = None


@router.post(
    "/probe",
    response_model=ProbeAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Score named directions without ranking the vocabulary",
)
async def probe(
    request: ProbeRequest, db: AsyncSession = Depends(get_db)
) -> ProbeAccepted:
    """Probe mode (BR-008).

    Distinct from the full ranked readout: the two can disagree because ranking
    applies a data-dependent normalisation this does not, so which mode is
    canonical is RECORDED on the result rather than left to the caller.

    Queued rather than inline for the readout's measured reason — a real model
    takes about a minute to load, and nginx gives up at 60s.
    """
    from ....models.model import Model
    from ....workers.jlens_probe_tasks import compute_probe

    result = await db.execute(select(Model).where(Model.id == request.model_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model with id {request.model_id!r}",
        )

    task = compute_probe.delay(
        model_id=request.model_id,
        prompt=request.prompt,
        tokens=request.tokens,
        layers=request.layers,
        artifact_id=request.artifact_id,
    )
    return ProbeAccepted(task_id=task.id, model_id=request.model_id)


@router.get(
    "/probe/{task_id}",
    response_model=ProbeResult,
    summary="Poll a queued probe",
)
async def probe_result(task_id: str) -> ProbeResult:
    """Poll a probe. A FAILURE carries its REASON, never an empty score list.

    An empty `scores` on a failed task is indistinguishable from a real probe
    that found nothing — the confusion this whole feature exists to prevent.
    """
    from celery.result import AsyncResult

    from ....core.celery_app import celery_app

    async_result = AsyncResult(task_id, app=celery_app)
    state = async_result.state

    if state == "SUCCESS":
        payload = async_result.result or {}
        return ProbeResult(
            task_id=task_id,
            status="SUCCESS",
            scores=[ProbeScore(**row) for row in payload.get("scores", [])],
            mode=payload.get("mode"),
            lens_type=payload.get("lens_type"),
        )
    if state == "FAILURE":
        return ProbeResult(
            task_id=task_id, status="FAILURE", error=str(async_result.info)
        )

    info = async_result.info if isinstance(async_result.info, dict) else {}
    return ProbeResult(task_id=task_id, status=state, stage=info.get("stage"))


# ── annotation, interventions, watchlists, replication ─────────────────────
#
# These four services were implemented, unit-tested and documented while NO
# user or agent could call any of them — the same shape as the 16 MCP tools
# this project once shipped registered with nothing. The reachability harness
# does not catch it, because a harness asserts the surfaces that EXIST.


class AnnotateRequest(BaseModel):
    """Annotate one SAE feature's decoder direction (BR-012..015)."""

    model_id: str
    sae_id: str
    feature_id: str
    layer: int
    direction: List[float]
    label_tokens: List[str] = []
    top_k: int = 8


class AnnotationResponse(BaseModel):
    feature_id: str
    layer: int
    lens_kurtosis: Optional[float]
    #: UNKNOWN when no band report exists for this model. That is a real
    #: answer, not a failure: without boundaries measured HERE there is no
    #: principled middle of the stack to classify against.
    workspace_class: str
    top_tokens: List[str]
    disagreement_score: Optional[float] = None
    has_disagreement: bool = False
    #: Rung 0. Carried so a caller cannot receive an annotation stripped of
    #: what it is (BR-019).
    evidence_rung: int = 0


@router.post(
    "/annotate",
    response_model=AnnotationResponse,
    summary="Annotate a weight-space direction through the lens",
)
async def annotate(
    request: AnnotateRequest, db: AsyncSession = Depends(get_db)
) -> AnnotationResponse:
    """Project a feature's decoder direction and describe it in J-space.

    TWO INDEPENDENT FIELDS (BR-012). The geometric field alone labels every
    MOTOR feature a workspace feature, because a motor direction is sharp too —
    so `workspace_class` is reported separately and is UNKNOWN without a band
    report rather than guessed.
    """
    import torch

    from ....models.model import Model
    from ....services.jlens_annotation import annotate_direction, label_disagreement
    from ....services.jlens_band_service import load_band_report
    from ....services.jlens_readout_service import IdentityTransport

    result = await db.execute(select(Model).where(Model.id == request.model_id))
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail=f"No model {request.model_id!r}")

    try:
        loaded = load_for_readout(record)
    except ModelNotAvailable as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    direction = torch.tensor(request.direction, dtype=torch.float32)
    if direction.numel() != loaded.d_model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"direction has {direction.numel()} entries but this model's "
                f"d_model is {loaded.d_model}"
            ),
        )

    ref = _service().find(loaded.name)
    band_report = _band_report_object(ref.directory) if ref else None

    annotation = annotate_direction(
        direction,
        IdentityTransport(),
        loaded.unembedding,
        layer=request.layer,
        feature_id=request.feature_id,
        decode=lambda ids: loaded.tokenizer.convert_ids_to_tokens(ids),
        top_k=request.top_k,
        band_report=band_report,
    )

    score = None
    if request.label_tokens:
        score = label_disagreement(request.label_tokens, annotation.top_tokens)

    return AnnotationResponse(
        feature_id=annotation.feature_id,
        layer=annotation.layer,
        lens_kurtosis=annotation.lens_kurtosis,
        workspace_class=annotation.workspace_class.value,
        top_tokens=annotation.top_tokens,
        disagreement_score=score,
        has_disagreement=bool(score is not None and score >= 0.8),
    )


def _band_report_object(directory):
    """Adapt a stored band report to the shape `classify_behaviour` reads.

    Returns None when there is none, so the behavioural field stays UNKNOWN
    rather than being classified against boundaries that do not exist.
    """
    from ....services.jlens_band_service import load_band_report

    stored = load_band_report(directory)
    if stored is None or stored.get("boundaries") is None:
        return None

    class _Bands:
        boundaries = stored["boundaries"]

    return _Bands()


class InterventionRequest(BaseModel):
    """Run an intervention. The control is NOT optional (BR-018)."""

    model_id: str
    primitive: str
    layers: List[int]
    positions: List[int]
    prompt: str
    direction: Optional[List[float]] = None
    strength: float = 1.0
    control_seed: int = 0
    control_k: int = 8


class InterventionResponse(BaseModel):
    primitive: str
    parameters: Dict[str, Any]
    intervened_outcome: float
    control_outcome: float
    #: THE finding. The raw intervened outcome is not one.
    excess_over_control: float
    control: Dict[str, Any]
    layers: List[int]
    positions: List[int]
    #: Rung 2. Reaching it REQUIRES the size-matched control above; without one
    #: there is no result here at all (BR-018).
    evidence_rung: int = 2


class WatchlistRequest(BaseModel):
    name: str
    artifact_ref: str
    scoring_definition: str
    concepts: List[Dict[str, Any]]
    control_set: List[str] = []


class WatchlistResponse(BaseModel):
    name: str
    artifact_ref: str
    scoring_definition: str
    concept_count: int


@router.post(
    "/watchlists",
    response_model=WatchlistResponse,
    summary="Author a watchlist for runtime handoff",
)
async def create_watchlist(request: WatchlistRequest) -> WatchlistResponse:
    """Validate and echo a watchlist definition (BR-025).

    A watchlist missing its scoring definition or its artifact reference is
    REFUSED here rather than exported and discovered later: a threshold applied
    to a differently computed score is a different detector, and the consumer
    has no way to notice.
    """
    from ....services.jlens_watchlist import WatchedConcept, Watchlist

    try:
        watchlist = Watchlist(
            name=request.name,
            concepts=[
                WatchedConcept(token=c["token"], threshold=float(c["threshold"]))
                for c in request.concepts
            ],
            scoring_definition=request.scoring_definition,
            artifact_ref=request.artifact_ref,
            control_set=request.control_set,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return WatchlistResponse(
        name=watchlist.name,
        artifact_ref=watchlist.artifact_ref,
        scoring_definition=watchlist.scoring_definition,
        concept_count=len(watchlist.concepts),
    )


class CostEstimateResponse(BaseModel):
    operation: str
    order_of_magnitude_seconds: float
    order_of_magnitude_peak_bytes: int
    basis: str
    is_estimate: bool = True


@router.get(
    "/cost-estimate",
    response_model=CostEstimateResponse,
    summary="Estimate an operation's cost BEFORE committing to it",
)
async def cost_estimate(
    operation: str,
    d_model: int,
    n_layers: int,
    n_positions: int = 1,
    n_prompts: int = 1,
    n_features: int = 1,
) -> CostEstimateResponse:
    """Order-of-magnitude cost for one J-space operation class (BR-028).

    An unknown class is a 400 rather than a cheap default: a small number
    invites exactly the run it should warn about, and a caller cannot tell
    "cheap" from "unmeasured".
    """
    from ....services.jlens_watchlist import OperationClass, estimate_cost

    try:
        op = OperationClass(operation)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"unknown operation {operation!r}. Known: "
                f"{[o.value for o in OperationClass]}"
            ),
        )

    est = estimate_cost(
        op,
        d_model=d_model,
        n_layers=n_layers,
        n_positions=n_positions,
        n_prompts=n_prompts,
        n_features=n_features,
    )
    return CostEstimateResponse(
        operation=est.operation.value,
        order_of_magnitude_seconds=est.order_of_magnitude_seconds,
        order_of_magnitude_peak_bytes=est.order_of_magnitude_peak_bytes,
        basis=est.basis,
    )


@router.get(
    "/reports/replication",
    response_model=Optional[Dict[str, Any]],
    summary="The recorded replication report (BR-001)",
)
async def replication_report(slug: str) -> Optional[Dict[str, Any]]:
    """Return the stored replication report, or null when none was recorded.

    Published whether favourable or not — there is no filter here, which is the
    structural half of BR-001.
    """
    from ....services.jlens_replication import load_replication_report

    ref = next((a for a in _service().list_artifacts() if a.slug == slug), None)
    if ref is None:
        raise HTTPException(status_code=404, detail=f"No artifact {slug!r}")
    return load_replication_report(ref.directory)

