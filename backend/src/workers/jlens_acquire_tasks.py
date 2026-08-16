"""
Adopt a J-lens someone else fitted, as a background task.

WHY A WORKER AND NOT A REQUEST. The artifact is hundreds of megabytes to several
gigabytes, and adopting it requires a real readout — `ValidationReport.serviceable`
demands SEMANTIC, and `load_for_readout` gates on `serviceable`, so an artifact
committed without one is unreadable by the very panel that would display it. That
means loading the model, which means the GPU.

ON THE `extraction` QUEUE, WITH A COST. That is the single-GPU queue, so a
multi-gigabyte download head-of-line-blocks fits and readouts while it runs. It
is still the right queue: the semantic check needs the model, and the readout
service's single-entry model cache lives here — routing the download elsewhere
would either duplicate the model load or hand the artifact to a worker that
cannot check it. The alternative is publishing an unvalidated lens, which is the
one outcome BR-030 exists to prevent.

THE TASK NAME IS FULLY QUALIFIED ON PURPOSE. `celery_app.task_routes` and
`autodiscover_tasks` both hold ENUMERATED entries per J-space module — there is
no `jlens_*` glob. A short name lands on the default queue silently, and a
missing autodiscover entry means the worker never imports this module at all, so
`.delay()` publishes a message nothing will ever consume. Both are asserted in
`tests/unit/test_jlens_reachable.py`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..core.celery_app import celery_app
from . import jlens_progress
from .task_heartbeat import beat

logger = logging.getLogger(__name__)


@celery_app.task(
    name="src.workers.jlens_acquire_tasks.acquire_jlens_artifact",
    bind=True,
    max_retries=0,
)
@jlens_progress.owns_its_failure
def acquire_jlens_artifact(
    self,
    model_id: str,
    repo_id: str,
    path_in_repo: str,
    revision: Optional[str] = None,
    access_token: Optional[str] = None,
    allow_coverage_loss: bool = False,
    allow_quality_regression: bool = False,
) -> Dict[str, Any]:
    """Download a published lens, describe it honestly, validate it, publish it.

    The measurement this performs is the SEMANTIC check — a real readout through
    the downloaded artifact, on the same fixture a local fit is held to. Nothing
    else about the transfer is taken on trust: weight identity comes from the
    publisher's own declaration, the layer convention and target from the
    tensors, and byte identity from a digest of what actually landed.
    """
    import torch

    from ..core.config import settings
    from ..core.database import get_sync_db
    from ..models.model import Model
    from ..services.jlens_acquire_service import (
        AcquisitionRefused,
        WeightIdentity,
        check_free_space,
        check_weight_identity,
        config_yaml_for_acquired,
        dtype_of,
        fetch_file,
        fetch_optional,
        file_digest,
        inspect_layers,
        parse_upstream_config,
        preview_repo,
        sibling_paths,
        write_acquisition_record,
    )
    from ..services.jlens_artifact_service import (
        ArtifactCoverageLoss,
        ArtifactQualityRegression,
        JLensArtifactService,
        normalise_payload,
    )
    from ..services.jlens_model_registry import ModelNotAvailable, load_for_readout
    from ..services.jlens_validation import defer_consumer_checks
    from ..services.huggingface_sae_service import resolve_hf_token

    jlens_progress.update_row(self.request.id, status="running", progress=1.0)

    with get_sync_db() as db:
        record = db.query(Model).filter(Model.id == model_id).first()
        if record is None:
            raise ValueError(f"No model with id {model_id!r}")
        repo_of_model = getattr(record, "repo_id", None)
        if not repo_of_model:
            raise ValueError(f"Model {model_id!r} has no repo_id to attach a lens to")

    token = resolve_hf_token(access_token)

    # ------------------------------------------------------------ the source
    self.update_state(state="PROGRESS", meta=beat({"stage": "resolving_source"}))
    preview = preview_repo(repo_id, revision=revision, token=token)
    remote = next((c for c in preview.candidates if c.path == path_in_repo), None)
    if remote is None:
        raise AcquisitionRefused(
            f"{path_in_repo!r} is not a downloadable lens candidate in "
            f"{repo_id}@{preview.revision[:8]}. Candidates: "
            f"{[c.path for c in preview.candidates][:5]}"
        )

    # REFUSED BEFORE A BYTE MOVES. A download that cannot fit fails halfway and
    # leaves the volume full — and this volume also holds every model, dataset
    # and checkpoint.
    needed = int(remote.size_bytes or 0)
    check_free_space(
        settings.jlens_artifacts_dir,
        settings.data_dir,
        needed_bytes=needed,
    )

    self.update_state(
        state="PROGRESS", meta=beat({"stage": "downloading", "bytes": needed})
    )
    jlens_progress.update_row(self.request.id, progress=10.0)

    # PINNED TO THE REVISION THE PREVIEW RESOLVED, so the file that was
    # inspected is the file that arrives, and "acquired from X@Y" is a
    # statement someone else can check.
    lens_file = fetch_file(repo_id, path_in_repo, preview.revision, token)
    siblings = sibling_paths(path_in_repo)
    config_file = fetch_optional(repo_id, siblings["config"], preview.revision, token)
    convergence_file = fetch_optional(
        repo_id, siblings["convergence"], preview.revision, token
    )
    upstream_config = (
        parse_upstream_config(config_file.read_text(encoding="utf-8"))
        if config_file
        else None
    )

    jlens_progress.update_row(self.request.id, progress=40.0)
    self.update_state(state="PROGRESS", meta=beat({"stage": "inspecting"}))

    payload = normalise_payload(
        torch.load(lens_file, map_location="cpu", weights_only=True)
    )

    # ---------------------------------------------------------- the model
    # AFTER the download, because the endpoint already refused synchronously
    # when the weights are absent; this load is belt-and-braces rather than the
    # primary guard, and doing it first would pay a model load to discover a
    # bad path.
    device = "cuda" if torch.cuda.is_available() else None
    try:
        loaded = load_for_readout(record, capture_device=device)
    except ModelNotAvailable as exc:
        raise AcquisitionRefused(str(exc)) from exc

    try:
        identity = check_weight_identity(upstream_config, repo_of_model)
        if identity.state is WeightIdentity.MISMATCH:
            # A REFUSAL, NOT A BADGE. The readout would be complete, plausible
            # and wrong, and nothing downstream can tell that from a good one.
            raise AcquisitionRefused(identity.detail)

        layers = inspect_layers(
            payload, n_layers=int(loaded.n_layers), d_model=int(loaded.d_model)
        )
        config_yaml = config_yaml_for_acquired(
            repo_id=repo_of_model,
            layers=layers,
            n_vocab=int(loaded.n_vocab),
            n_layers=int(loaded.n_layers),
            dtype=dtype_of(payload),
            upstream_config=upstream_config,
        )

        service = JLensArtifactService(settings.jlens_artifacts_dir)
        sidecars = {}
        if convergence_file is not None:
            sidecars[f"{loaded.name.split('/')[-1].lower()}_convergence.csv"] = (
                convergence_file
            )
        ref = service.stage_from_file(
            repo_of_model, lens_file, config_yaml, sidecars=sidecars or None
        )

        write_acquisition_record(
            ref.directory,
            source_repo=repo_id,
            source_path=path_in_repo,
            revision=preview.revision,
            upstream_sha256=remote.sha256,
            local_sha256=file_digest(ref.lens_path),
            identity=identity,
            layers=layers,
            upstream_config=upstream_config,
        )

        jlens_progress.update_row(self.request.id, progress=65.0)
        self.update_state(state="PROGRESS", meta=beat({"stage": "validating"}))

        # THE SAME FIXTURE A LOCAL FIT IS HELD TO, through the same helper. A
        # separate probe here would let an acquired artifact be published on an
        # easier test than one miStudio fitted itself.
        from .jlens_fit_tasks import _run_semantic_check
        from ..api.v1.endpoints.jlens import (
            SEMANTIC_FIXTURE_ANSWER,
            SEMANTIC_FIXTURE_CONTROL,
            SEMANTIC_FIXTURE_PROMPT,
        )

        semantic_result = _run_semantic_check(
            service=service,
            ref=ref,
            loaded=loaded,
            probe={
                "prompt": SEMANTIC_FIXTURE_PROMPT,
                "expected_intermediate": SEMANTIC_FIXTURE_ANSWER,
                "control_prompt": SEMANTIC_FIXTURE_CONTROL,
            },
            fitted_layers=layers.fitted,
        )

        report = service.validate(
            ref,
            d_model=int(loaded.d_model),
            expected_layers=layers.fitted,
            n_vocab=int(loaded.n_vocab),
            semantic_result=semantic_result,
        )

        jlens_progress.update_row(self.request.id, progress=85.0)

        published = False
        unpublished_reason: Optional[str] = None
        displaced: Optional[Dict[str, Any]] = None
        try:
            incumbent = service.find(repo_of_model)
            if incumbent is not None:
                displaced = service._recipe_summary(incumbent)  # noqa: SLF001
            service.commit(
                repo_of_model,
                # ITS OWN WORDING. Copying the fit worker's sentence would put
                # "requires a live external consumer" over an artifact nobody
                # fitted here, implying a local fit was performed.
                defer_consumer_checks(report),
                allow_coverage_loss=allow_coverage_loss,
                allow_quality_regression=allow_quality_regression,
            )
            published = True
        except (
            ArtifactCoverageLoss,
            ArtifactQualityRegression,
        ) as exc:
            # THE STAGED ARTIFACT SURVIVES. It is a completed download; throwing
            # it away because the gate refused means paying the bandwidth again
            # to make the same decision with a flag set.
            unpublished_reason = str(exc)
            logger.warning("Acquired lens staged but not published: %s", exc)

        jlens_progress.update_row(
            self.request.id, status="completed", progress=100.0
        )
        return {
            "model": repo_of_model,
            "source": {
                "repo": repo_id,
                "path": path_in_repo,
                "revision": preview.revision,
            },
            "published": published,
            "unpublished_reason": unpublished_reason,
            # WHAT WAS ARCHIVED, NAMED. `_quality_regression` cannot refuse an
            # acquired lens carrying no `converged` key over a converged local
            # fit — `True == None` is False, so the rule does not fire. The
            # backstop is `.superseded` plus `restore_superseded`, and the
            # caller has to be TOLD, or the backstop is one nobody reaches for.
            "displaced": displaced,
            "weight_identity": identity.state.value,
            "weight_identity_detail": identity.detail,
            "bytes_identical": bool(remote.sha256)
            and remote.sha256 == file_digest(ref.lens_path),
            "fitted_layers": layers.fitted,
            "target_layer": layers.target_layer,
            "degenerate_layers": layers.degenerate,
            "validation": {
                "passed": report.passed,
                "serviceable": report.serviceable,
                # FALSE, AND CORRECTLY SO. No interop harness has run against
                # this or any other artifact this project holds.
                "cleared_for_handover": report.cleared_for_handover,
                "results": [
                    {"check": r.check.value, "status": r.status.value, "detail": r.detail}
                    for r in report.results
                ],
            },
            "evidence_rung": 0,
            "caveat": (
                "Adopting a lens checks that it is conformant and that it "
                "discriminates on our fixture. It does not reproduce the fit. "
                "Weight identity rests on the publisher's own declaration, and "
                "is recorded as unverified when they made none."
            ),
        }
    finally:
        if device == "cuda":
            from ..services.jlens_model_registry import clear_cache

            # Same release the intervention task performs, and for the same
            # reason: `clear_cache` runs gc then `empty_cache`, so a live
            # reference in this frame keeps every block allocated.
            loaded = None  # noqa: F841 - the assignment IS the release
            clear_cache()
            logger.info("Released the acquisition model from GPU memory")
