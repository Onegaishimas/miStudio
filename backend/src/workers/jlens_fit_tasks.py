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
from typing import Any, Dict, List, Optional, Sequence

from ..core.celery_app import celery_app
from .task_heartbeat import beat
from . import jlens_progress
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
    # FULL BACKWARD IS THE STANDARD RECIPE (D4). Frozen Q/K is an
    # ABLATION — useful, and not the default. This defaulted to True, so
    # every fit started from the API or MCP silently ran the ablation while
    # the fitter class it constructs defaults to False; the aligned recipe
    # was unreachable without naming the flag explicitly, and the artifact
    # published on 2026-08-03 records `frozen_qk` for exactly this reason.
    freeze_qk: bool = False,
    corpus_name: str = "unspecified",
    semantic_probe: Optional[Dict[str, Any]] = None,
    allow_coverage_loss: bool = False,
    allow_quality_regression: bool = False,
    freeze_norms: bool = False,
    target_layer: str = "penultimate",
    convergence_delta: Optional[float] = None,
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
    from ..services.jlens_artifact_service import (
        ArtifactCoverageLoss,
        ArtifactQualityRegression,
        JLensArtifactService,
    )
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

    # RELEASE THE CARD WHEN THE FIT IS DONE. `load_for_readout` caches the model
    # so a fit does not reload it per prompt, and NOTHING dropped it afterwards:
    # `clear_cache` existed with zero callers. An LFM2 fit left 4.0 GB resident
    # on the shared 3090 at 0% utilisation from the moment it finished until the
    # pod was next restarted — on a card miLLM serves from.
    #
    # `try/finally` around everything after the load, because the release must
    # happen on the failure paths too: an OOM or a validation raise is exactly
    # when the card most needs to come back.
    try:
        return _fit_and_publish(
            self,
            loaded=loaded,
            model_id=model_id,
            repo_id=repo_id,
            prompts=prompts,
            layers=layers,
            freeze_qk=freeze_qk,
            corpus_name=corpus_name,
            semantic_probe=semantic_probe,
            allow_coverage_loss=allow_coverage_loss,
            allow_quality_regression=allow_quality_regression,
            freeze_norms=freeze_norms,
            target_layer=target_layer,
            convergence_delta=convergence_delta,
        )
    finally:
        if device == "cuda":
            from ..services.jlens_model_registry import clear_cache

            # DROP OUR OWN REFERENCE FIRST. `clear_cache` nulls the CACHE's
            # entry and then runs gc + `torch.cuda.empty_cache()` — but this
            # frame still held `loaded`, so there was nothing for gc to collect
            # and `empty_cache` had no free blocks to return. The first version
            # of this release looked correct and recovered only the activation
            # workspace: 7706 MiB fell to 2608 MiB, which is LFM2's fp16 weights
            # still sitting on a card miLLM serves from.
            #
            # The three tests written with it all passed, because they asserted
            # `clear_cache` was CALLED. Being called is not being effective.
            loaded = None  # noqa: F841 - the assignment IS the release
            clear_cache()
            logger.info("Released the fitted model from GPU memory")


def _fit_and_publish(
    self,
    loaded,
    model_id: str,
    repo_id: str,
    prompts: List[str],
    layers: Optional[List[int]],
    freeze_qk: bool,
    corpus_name: str,
    semantic_probe: Optional[Dict[str, Any]],
    allow_coverage_loss: bool,
    allow_quality_regression: bool,
    freeze_norms: bool,
    target_layer: str,
    convergence_delta: Optional[float],
) -> Dict[str, Any]:
    """The fit itself, split out so the caller can guarantee the GPU release."""
    from ..ml.jlens_fitter import DEFAULT_CONVERGENCE_DELTA, JacobianFitter
    from ..services.jlens_artifact_service import (
        ArtifactCoverageLoss,
        JLensArtifactService,
    )

    # None means the fitter's default. Passed explicitly rather than defaulted
    # in the signature so the value that was ACTUALLY used is the one written
    # into config.yaml below — a recipe naming a threshold the fit did not use
    # is worse than one naming none.
    fitter = JacobianFitter(
        loaded.model,
        loaded.tokenizer,
        loaded.structure,
        freeze_qk=freeze_qk,
        freeze_norms=freeze_norms,
        target_layer=target_layer,
        convergence_delta=(
            convergence_delta
            if convergence_delta is not None
            else DEFAULT_CONVERGENCE_DELTA
        ),
    )

    self.update_state(state="PROGRESS", meta={"stage": "fitting", "prompts_seen": 0})

    total_prompts = max(len(prompts), 1)

    def on_progress(progress):
        # The SAME numbers the heartbeat carries, written where Active
        # Operations and the J-Lens panel can see them. Without this a fit is
        # visible only to the browser tab that started it.
        jlens_progress.update_row(
            self.request.id,
            status="running",
            progress=100.0 * progress.prompts_seen / total_prompts,
        )
        self.update_state(
            state="PROGRESS",
            meta=beat({
                "stage": "fitting",
                "prompts_seen": progress.prompts_seen,
                "last_delta": progress.last_delta,
                "converged": progress.converged,
            }),
        )

    result = fitter.fit(prompts, layers=layers, on_progress=on_progress)

    service = JLensArtifactService(settings.jlens_artifacts_dir)
    config_yaml = _config_yaml(
        loaded,
        result,
        freeze_qk,
        corpus_name,
        freeze_norms=freeze_norms,
        target_layer=target_layer,
    )
    ref = service.write_staged(repo_id, result.jacobians, config_yaml)

    self.update_state(state="PROGRESS", meta={"stage": "validating"})
    # SEMANTIC runs HERE or nowhere. The check needs a loaded model, and this
    # task is the one place in the system that has one alongside a freshly
    # written artifact — so leaving it NOT_RUN made `serviceable` false on every
    # successful fit, and the artifact was discarded seconds after being built.
    # It still needs a FIXTURE, which cannot be invented: the intermediate must
    # be one this model would plausibly reach and must not appear in the prompt.
    # So it is the caller's to supply, and its absence fails closed with a
    # stated reason rather than publishing on an unrun check.
    semantic_result = None
    if semantic_probe:
        semantic_result = _run_semantic_check(
            service=service,
            ref=ref,
            loaded=loaded,
            probe=semantic_probe,
            fitted_layers=sorted(result.jacobians),
        )

    report = service.validate(
        ref,
        d_model=loaded.d_model,
        expected_layers=sorted(result.jacobians),
        n_vocab=loaded.n_vocab,
        semantic_result=semantic_result,
    )

    published = False
    coverage_refusal = None
    if report.serviceable:
        # `commit` requires the FULL pass, which needs a live external consumer.
        # Serviceable-but-not-passed still publishes locally, because the two
        # consumer-interop classes cannot run here and gating on them would
        # make every fit unusable. The report travels with the result so the
        # distinction is visible rather than implied.
        try:
            service.commit(
                repo_id,
                _local_pass(report),
                allow_coverage_loss=allow_coverage_loss,
                allow_quality_regression=allow_quality_regression,
            )
            published = True
        except ArtifactQualityRegression as exc:
            # SAME SHAPE AS A COVERAGE REFUSAL, and for the same reason: the fit
            # succeeded and publishing was refused to protect something the user
            # already has. The staged fit survives so it can be published
            # deliberately rather than refitted.
            logger.warning("Refused to publish %s: %s", repo_id, exc)
            coverage_refusal = str(exc)
        except ArtifactCoverageLoss as exc:
            # NOT an error the user should have to read in a log. Publishing was
            # refused to protect layers they already paid GPU time for, and the
            # staged fit is kept so they can publish it deliberately.
            logger.warning("Refused to publish %s: %s", repo_id, exc)
            coverage_refusal = str(exc)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            logger.error("Publishing %s failed: %s", repo_id, exc)
    else:
        # THE STAGED FIT SURVIVES A FAILED VALIDATION. This used to be
        # `service.discard_staged(repo_id)`, which destroyed a converged
        # 15-layer LFM2 artifact — 754 seconds of GPU time — because one
        # fixture token did not appear at one layer. The lens was fine; the
        # fixture was wrong, and there was then nothing left to re-validate
        # against, so proving that required paying for the whole fit again.
        #
        # Keeping it is safe: staging is excluded from discovery, so nothing
        # serves it, and `write_staged` clears the directory before the next
        # fit, so it cannot accumulate beyond one per model.
        logger.warning(
            "Not publishing %s: %s. The staged fit is kept at %s so it can be "
            "re-validated without refitting.",
            repo_id,
            report.summary(),
            service.staging_dir(repo_id),
        )

    # Say WHY nothing was published when the cause is a missing fixture rather
    # than a bad lens. Without this the result reads "semantic=not_run" and the
    # user is left to infer that the fit failed, when in fact it succeeded and
    # was discarded for want of one prompt.
    unpublished_reason = None
    if not published:
        if coverage_refusal is not None:
            unpublished_reason = coverage_refusal
        elif semantic_result is None:
            unpublished_reason = (
                "No semantic_probe was supplied, so the SEMANTIC check could not "
                "run and the artifact was not published. Supply "
                "{prompt, expected_intermediate, layer} — an intermediate the "
                "model should reach that does NOT appear in the prompt."
            )
        elif not report.serviceable:
            unpublished_reason = (
                "The artifact failed a local validation class; see `validation`."
            )

    jlens_progress.update_row(self.request.id, status="completed", progress=100.0)
    return {
        "model_id": model_id,
        "repo_id": repo_id,
        "slug": ref.slug,
        "prompts_seen": result.prompts_seen,
        "converged": result.converged,
        "layers": sorted(result.jacobians),
        "size_bytes": result.size_bytes(),
        "published": published,
        "unpublished_reason": unpublished_reason,
        "validation": {
            "serviceable": report.serviceable,
            "passed": report.passed,
            "summary": report.summary(),
            "results": [
                {
                    "check": r.check.value,
                    "status": r.status.value,
                    "detail": r.detail,
                    # WHAT THE CHECK ACTUALLY SAW. Dropped until now, which
                    # made a failed SEMANTIC check unactionable: it said the
                    # intermediate was absent from the top-k and never said
                    # what WAS there, so the only way to learn was to publish
                    # the artifact the check had just refused.
                    "evidence": r.evidence,
                }
                for r in report.results
            ],
        },
    }


def _run_semantic_check(service, ref, loaded, probe: Dict[str, Any], fitted_layers):
    """Read out the STAGED artifact and check for a known unspoken intermediate.

    Deliberately reads the file that was just written rather than the tensors
    still in memory. The in-memory ones are known good — they came straight out
    of the fitter — so checking them would confirm the fit and prove nothing
    about the artifact anyone else will load. A truncated or mis-keyed write is
    only visible on the way back in.

    A layer outside the fitted set is a fixture error, not a lens failure, and
    is reported as such: reading out at an unfitted layer has no Jacobian to
    apply and would fail for a reason that has nothing to do with the artifact.
    """
    from ..services.jlens_readout_service import JacobianTransport, ReadoutService
    from ..services.jlens_validation import (
        CheckClass,
        CheckResult,
        CheckStatus,
        check_semantic,
    )
    from ..schemas.jlens import LensTokenMessage

    prompt = str(probe.get("prompt", ""))
    expected = str(probe.get("expected_intermediate", ""))
    top_k = int(probe.get("top_k", 8))
    control_prompt = probe.get("control_prompt")
    if control_prompt is not None:
        control_prompt = str(control_prompt)
    layer = probe.get("layer")
    if layer is not None:
        # An EXPLICIT layer is honoured exactly. A caller naming a layer is
        # making a claim about that layer, and quietly scanning around it would
        # answer a question they did not ask.
        scan: Sequence[int] = [int(layer)]
    else:
        # EVERY FITTED LAYER. Two earlier defaults were both wrong for the same
        # reason: they asserted WHERE an unspoken intermediate must appear. The
        # top of the stack was wrong because the model has moved on to the
        # answer by then; "two thirds up" was wrong because that is a band
        # constant, and BR-002 forbids this project assuming a band it has not
        # measured for the model in front of it.
        #
        # Observed on hardware: the aligned LFM2 lens reads
        # ' tourism'/' located'/' geography' at L9 for an Eiffel-Tower fixture —
        # the concept field, correct and useful — and a single-layer check
        # discarded a converged 15-layer artifact over it.
        #
        # Scanning is weaker than a single layer, which is why the fixture's
        # control prompt matters; see `check_semantic`.
        scan = list(fitted_layers)

    unfitted = [c for c in scan if c not in fitted_layers]
    if unfitted:
        return CheckResult(
            CheckClass.SEMANTIC,
            CheckStatus.FAIL,
            (
                f"probe layer(s) {unfitted} were not fitted (fitted: "
                f"{list(fitted_layers)}); there is no Jacobian to read out through"
            ),
        )

    payload = service._load_payload(ref)
    if payload is None:
        return CheckResult(
            CheckClass.SEMANTIC,
            CheckStatus.FAIL,
            "the staged artifact did not deserialize, so it cannot be read out",
        )

    # The STAGED artifact's scales, read from the config written beside it.
    transport = JacobianTransport(
        {int(k): v for k, v in payload.items()},
        scales=service.layer_scales(ref),
    )
    readout_service = ReadoutService(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        structure=loaded.structure,
        unembedding=loaded.unembedding,
        model_name=loaded.name,
    )

    def readout(text: str, at_layers: Sequence[int], k: int) -> Dict[int, List[str]]:
        """Top-k at the LAST position, for EVERY requested layer, in one pass.

        The last position is where the next token is being formed. One stream
        call covers the whole layer list: `stream` captures residuals once and
        reads each requested layer off them, so asking per layer would re-run
        the model once per layer.
        """
        ordered = list(at_layers)
        last = None
        for message in readout_service.stream(
            text, [transport], layers=ordered, top_n=k
        ):
            if isinstance(message, LensTokenMessage):
                last = message
        if last is None:
            raise ValueError("readout produced no tokens")
        rows = last.results[0].top_tokens
        # `top_tokens` is indexed by POSITION IN THE REQUESTED LIST, not by
        # absolute layer number. Zipping against `ordered` is what keeps a
        # partial artifact from reading the wrong layer's row.
        return {layer: rows[i] for i, layer in enumerate(ordered) if i < len(rows)}

    return check_semantic(
        readout, prompt, scan, expected, top_k=top_k, control_prompt=control_prompt
    )


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


def _config_yaml(
    loaded,
    result,
    freeze_qk: bool,
    corpus_name: str,
    freeze_norms: bool = False,
    target_layer: str = "penultimate",
) -> str:
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
    attended = [e.layer for e in applicability if e.has_attention]
    treatment = "frozen_qk" if freeze_qk else "full"

    lines = [
        f"model: {loaded.name}",
        f"d_model: {loaded.d_model}",
        f"n_layers: {loaded.n_layers}",
        f"n_vocab: {loaded.n_vocab}",
        "dtype: fp16",
        # THE RECIPE'S OWN VOCABULARY (BR-007). `JLensArtifactRecipe` declares
        # these fields and this writer emitted none of them, so the schema was a
        # contract nothing honoured and the provenance said nothing about how the
        # lens was built.
        #
        # THE TARGET IS THE CALLER'S CHOICE AND MUST BE REPORTED AS RUN. This
        # line was the literal string "final" — correct when the fitter always
        # ran to the last block, and a LIE the moment the target became
        # selectable. The parameter was added to this function's signature and
        # then never read, so a 15-of-16-layer penultimate fit published a
        # recipe claiming it targeted the final block. Nothing objected: the
        # value was threaded the whole way here and dropped on the last line.
        f"target_layer: {target_layer}",
        # THE PAPER'S DEFINITION, recorded as it was run: an expectation over
        # SOURCE positions of the summed effect on all subsequent target
        # positions. The previous fitter recorded `self_only_isolated` because
        # that is what it did — one source position, length-1 sub-network.
        "target_position_scope: all_subsequent",
        "source_position_aggregation: mean_over_all_positions",
        "differentiation_mode: reverse",
        "aggregation: mean",
        f"seq_len: {getattr(result, 'mean_seq_len', 0.0):.1f}",
        # PER-LAYER, NOT WHOLESALE. Describing a hybrid model's lens as
        # "frozen_qk" when the treatment reached 6 of 16 layers is the exact
        # overstatement this file's own docstring warns about. The requested
        # treatment and where it actually applied are separate facts.
        f"attention_gradients_requested: {treatment}",
        f"norm_statistics: {'frozen' if freeze_norms else 'differentiated'}",
        f"attention_gradients_applied_to_layers: {attended}",
        # THE LAYERS THIS ARTIFACT ACTUALLY COVERS, stated once and cheaply.
        # Everything else that needs them had to deserialise the whole tensor
        # file — 276 MB to answer "which layers?" — which is why the artifact
        # listing never carried the fact and a partial fit looked identical to
        # a full one right up until the readout refused.
        f"fitted_layers: {sorted(result.jacobians)}",
        # LAYERS WHERE THE LENS IS THE LOGIT LENS, exactly. The last decoder
        # layer has no blocks after it, so its sub-network is the identity and
        # J = I by construction. A Diff there is empty because the two lenses
        # ARE the same lens — not because they happen to agree — and an empty
        # top row read without that context looks like a finding.
        f"degenerate_layers: {result.degenerate_layers}",
        f"corpus: {corpus_name}",
        f"n_prompts: {result.prompts_seen}",
        f"converged: {str(result.converged).lower()}",
        f"convergence_delta: {result.convergence_delta}",
        # THE SCALE, WITHOUT WHICH THE ARTIFACT IS WRONG. `_to_storage_dtype`
        # divides each matrix down so the fp16 cast cannot saturate, and its
        # docstring has always said the factor is recorded here — it was not.
        # Ranked readouts never noticed, because the model's final norm divides
        # a positive scalar straight back out. Everything that does NOT
        # normalise did: probe scores and intervention magnitudes were off by
        # an unrecorded per-layer factor, so they were not comparable across
        # layers, and an external consumer multiplying by W_U got the wrong
        # magnitudes — the exact case that docstring names.
        "layer_scales:",
    ]
    for layer in sorted(result.scales):
        lines.append(f"  {layer}: {result.scales[layer]!r}")

    # HOW LOCAL THE LENS IS, over the whole corpus. Both figures, because the
    # mean says what is typical and the max says how bad it gets — and a lens
    # is judged on the second. This used to be a single number taken from
    # whichever prompt happened to be last.
    if result.residual_mean:
        lines.append("linearisation_residual_mean:")
        for layer in sorted(result.residual_mean):
            lines.append(f"  {layer}: {result.residual_mean[layer]:.6g}")
    if result.residual_max:
        lines.append("linearisation_residual_max:")
        for layer in sorted(result.residual_max):
            lines.append(f"  {layer}: {result.residual_max[layer]:.6g}")

    lines += [
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
