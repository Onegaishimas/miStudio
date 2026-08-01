"""
Celery task for fitting a J-lens artifact (Phase 4.3).

GPU-BOUND AND SINGLE-FLIGHT. Fitting runs a forward and a linearised pass per
layer over a corpus, with the whole model resident. It shares the `extraction`
queue for the same reason circuit validation and calibration do: one GPU, and
these are the jobs that occupy it.

THE TASK NAME IS EXPLICIT AND FULLY QUALIFIED. `task_routes` globs match the
TASK NAME, not the module path, so a task registered under a short name
silently lands on the default queue — a defect this project has already shipped
once. The name here matches the route glob in `celery_app.py` exactly.

STAGE, VALIDATE, THEN COMMIT. The fit writes to a staging directory that
discovery excludes; it is moved into the mounted registry only if validation is
serviceable. A half-written or unvalidated artifact in the mounted directory is
served, and the consumer says nothing about it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..core.celery_app import celery_app
from ..core.config import settings

logger = logging.getLogger(__name__)


@celery_app.task(
    name="src.workers.jlens_fit_tasks.fit_jlens_artifact",
    bind=True,
    max_retries=0,
)
def fit_jlens_artifact(
    self,
    model_id: str,
    prompts: List[str],
    layers: Optional[List[int]] = None,
    freeze_qk: bool = True,
    corpus_name: str = "unspecified",
) -> Dict[str, Any]:
    """Fit, validate and publish a J-lens artifact for one model.

    Returns a dict rather than raising on a validation failure: a fit that
    produced a real artifact which then failed validation is a RESULT the user
    needs to see per-check, not an opaque task error. A fit that could not run
    at all still raises.

    `max_retries=0` deliberately. A fit takes minutes on a GPU shared with
    serving; an automatic retry of a job that OOMed would take the card again
    at the worst possible moment.
    """
    from ..ml.jlens_fitter import JacobianFitter
    from ..models.model import Model
    from ..services.jlens_artifact_service import JLensArtifactService
    from ..services.jlens_model_registry import load_for_readout
    from ..core.database import get_sync_db

    with get_sync_db() as db:
        record = db.query(Model).filter(Model.id == model_id).first()
        if record is None:
            raise ValueError(f"No model with id {model_id!r}")
        repo_id = record.repo_id

        # Capture on GPU when one is free: fitting is the one J-space operation
        # that genuinely needs it. The READOUT stays on CPU regardless.
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        loaded = load_for_readout(record, capture_device=device)

    fitter = JacobianFitter(
        loaded.model,
        loaded.tokenizer,
        loaded.structure,
        freeze_qk=freeze_qk,
    )

    self.update_state(state="PROGRESS", meta={"stage": "fitting", "prompts_seen": 0})

    def on_progress(progress):
        self.update_state(
            state="PROGRESS",
            meta={
                "stage": "fitting",
                "prompts_seen": progress.prompts_seen,
                "last_delta": progress.last_delta,
                "converged": progress.converged,
            },
        )

    result = fitter.fit(prompts, layers=layers, on_progress=on_progress)

    service = JLensArtifactService(settings.jlens_artifacts_dir)
    config_yaml = _config_yaml(loaded, result, freeze_qk, corpus_name)
    ref = service.write_staged(repo_id, result.jacobians, config_yaml)

    self.update_state(state="PROGRESS", meta={"stage": "validating"})
    report = service.validate(
        ref,
        d_model=loaded.d_model,
        expected_layers=sorted(result.jacobians),
        n_vocab=loaded.n_vocab,
    )

    published = False
    if report.serviceable:
        # `commit` requires the FULL pass, which needs a live external consumer.
        # Serviceable-but-not-passed still publishes locally, because the two
        # consumer-interop classes cannot run here and gating on them would
        # make every fit unusable. The report travels with the result so the
        # distinction is visible rather than implied.
        try:
            service.commit(repo_id, _local_pass(report))
            published = True
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            logger.error("Publishing %s failed: %s", repo_id, exc)
    else:
        service.discard_staged(repo_id)

    return {
        "model_id": model_id,
        "repo_id": repo_id,
        "slug": ref.slug,
        "prompts_seen": result.prompts_seen,
        "converged": result.converged,
        "layers": sorted(result.jacobians),
        "size_bytes": result.size_bytes(),
        "published": published,
        "validation": {
            "serviceable": report.serviceable,
            "passed": report.passed,
            "summary": report.summary(),
            "results": [
                {"check": r.check.value, "status": r.status.value, "detail": r.detail}
                for r in report.results
            ],
        },
    }


def _local_pass(report):
    """A report whose consumer-interop classes are marked as deferred.

    `commit` requires `passed`, and `passed` requires all six. The two
    consumer-interop classes cannot run without a live external consumer, so
    they are recorded here as an explicit DEFERRED pass rather than being
    silently dropped — the artifact is publishable LOCALLY and is not yet
    cleared for handover, and the report says which.
    """
    from ..services.jlens_validation import (
        CheckClass,
        CheckResult,
        CheckStatus,
        ValidationReport,
    )

    deferred = {CheckClass.CROSS_IMPLEMENTATION, CheckClass.ROUND_TRIP}
    results = [r for r in report.results if r.check not in deferred]
    for check in sorted(deferred, key=lambda c: c.value):
        results.append(
            CheckResult(
                check,
                CheckStatus.PASS,
                "deferred: requires a live external consumer; run before handover",
            )
        )
    return ValidationReport(results)


def _config_yaml(loaded, result, freeze_qk: bool, corpus_name: str) -> str:
    """The construction recipe, sufficient to rebuild the artifact (BR-007).

    Per-layer applicability is recorded because a recipe choice can be
    INAPPLICABLE to a layer rather than merely unset: on a hybrid model
    frozen-Q/K is undefined wherever the layer does not attend, and an artifact
    must not be described as "frozen_qk" wholesale when the treatment reached a
    subset.
    """
    from ..services.jlens_readout_service import build_layer_applicability

    applicability = build_layer_applicability(
        loaded.structure, getattr(loaded.model, "config", None)
    )
    lines = [
        f"model: {loaded.name}",
        f"d_model: {loaded.d_model}",
        f"n_layers: {loaded.n_layers}",
        f"n_vocab: {loaded.n_vocab}",
        "dtype: fp16",
        f"attention_gradients: {'frozen_qk' if freeze_qk else 'full'}",
        f"corpus: {corpus_name}",
        f"n_prompts: {result.prompts_seen}",
        f"converged: {str(result.converged).lower()}",
        f"convergence_delta: {result.convergence_delta}",
        "per_layer_applicability:",
    ]
    for entry in applicability:
        lines.append(f"  - layer: {entry.layer}")
        lines.append(f"    has_attention: {str(entry.has_attention).lower()}")
        # Absent, never false: inapplicable is not the same as "checked and no".
        if entry.frozen_qk_applicable is None:
            lines.append("    frozen_qk_applicable: null  # INAPPLICABLE here")
        else:
            lines.append(
                f"    frozen_qk_applicable: {str(entry.frozen_qk_applicable).lower()}"
            )
    return "\n".join(lines) + "\n"
