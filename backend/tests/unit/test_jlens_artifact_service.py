"""
Artifact lifecycle: stage, validate, commit, serve.

The property under test throughout is that NOTHING REACHES A CONSUMER WITHOUT A
FULL PASS. The consumer's lens loading is best-effort and fails at request time
without raising, so every shortcut here — publishing early, serving without a
report, leaving a half-written directory mounted — surfaces as a feature that
quietly returns nothing.

MUTATION CONTROLS (each must turn this file red):
  * commit without checking report.passed          -> "refuses to publish" fails
  * treat a NOT_RUN class as a pass                 -> "not run blocks publish" fails
  * load_for_readout defaulting report=None to ok   -> "serve refuses" fails
  * include staging dirs in list_artifacts          -> "staging is invisible" fails
  * accept a directory with two lens files          -> "ambiguous" fails
  * torch.load without weights_only                 -> "weights only" fails
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from src.services.jlens_artifact_service import (
    ArtifactNotValidated,
    JLensArtifactService,
    slug_for,
)
from src.services.jlens_validation import (
    CheckClass,
    CheckResult,
    CheckStatus,
    ValidationReport,
)

D_MODEL = 8
LAYERS = [0, 1, 2]
N_VOCAB = 512


def jacobians():
    return {i: torch.eye(D_MODEL, dtype=torch.float16) for i in LAYERS}


def live_passes():
    """The three checks that need a loaded model or a running consumer."""
    return {
        "semantic_result": CheckResult(CheckClass.SEMANTIC, CheckStatus.PASS, "ok"),
        "cross_impl_result": CheckResult(
            CheckClass.CROSS_IMPLEMENTATION, CheckStatus.PASS, "ok"
        ),
        "round_trip_result": CheckResult(CheckClass.ROUND_TRIP, CheckStatus.PASS, "ok"),
    }


def service(tmp_path: Path) -> JLensArtifactService:
    return JLensArtifactService(tmp_path / "artifacts")


# ------------------------------------------------------------------- slug


def test_slug_matches_the_consumer_convention():
    assert slug_for("LiquidAI/LFM2.5-1.2B-Instruct") == "lfm2.5-1.2b-instruct"
    assert slug_for("google/gemma-2-2b-it") == "gemma-2-2b-it"


def test_an_unslugabble_id_is_refused_not_silently_emptied():
    with pytest.raises(ValueError):
        slug_for("///")


# ---------------------------------------------------------------- staging


def test_a_staged_artifact_is_invisible_to_discovery(tmp_path: Path):
    """Half-written artifacts must not be servable.

    The consumer mounts the directory and reads whatever is there, without
    reporting what it found.
    """
    svc = service(tmp_path)
    svc.write_staged("org/model", jacobians(), "recipe: test")

    assert svc.list_artifacts() == []
    assert svc.find("org/model") is None


def test_committing_makes_it_discoverable(tmp_path: Path):
    svc = service(tmp_path)
    ref = svc.write_staged("org/model", jacobians(), "recipe: test")
    report = svc.validate(ref, D_MODEL, LAYERS, N_VOCAB, **live_passes())

    published = svc.commit("org/model", report)
    assert published.lens_path.exists()
    assert [a.slug for a in svc.list_artifacts()] == ["model"]
    assert svc.find("org/model") is not None
    assert not svc.staging_dir("org/model").exists()


def test_restaging_replaces_a_previous_stage(tmp_path: Path):
    svc = service(tmp_path)
    svc.write_staged("org/model", jacobians(), "first")
    ref = svc.write_staged("org/model", jacobians(), "second")
    assert ref.config_path.read_text() == "second"


# -------------------------------------------------------------- publishing


def test_commit_refuses_to_publish_a_failing_report(tmp_path: Path):
    """The last point at which a bad artifact can be stopped by anything."""
    svc = service(tmp_path)
    svc.write_staged("org/model", jacobians(), "recipe: test")
    failing = ValidationReport(
        [CheckResult(c, CheckStatus.FAIL, "no") for c in CheckClass]
    )

    with pytest.raises(ArtifactNotValidated, match="fails silently|refusing"):
        svc.commit("org/model", failing)
    assert svc.find("org/model") is None


def test_a_not_run_class_blocks_publication(tmp_path: Path):
    """"We did not check" must never publish like "we checked and it was fine".

    This is the default path: the three live checks are absent unless supplied,
    so an artifact validated without a model or a consumer cannot be published
    at all.
    """
    svc = service(tmp_path)
    ref = svc.write_staged("org/model", jacobians(), "recipe: test")
    report = svc.validate(ref, D_MODEL, LAYERS, N_VOCAB)  # no live checks

    assert report.passed is False
    assert set(report.missing) == set()
    statuses = {r.check: r.status for r in report.results}
    assert statuses[CheckClass.SEMANTIC] is CheckStatus.NOT_RUN
    assert statuses[CheckClass.ROUND_TRIP] is CheckStatus.NOT_RUN

    with pytest.raises(ArtifactNotValidated):
        svc.commit("org/model", report)


def test_commit_without_a_stage_is_an_error_not_a_silent_noop(tmp_path: Path):
    svc = service(tmp_path)
    passing = ValidationReport([CheckResult(c, CheckStatus.PASS, "ok") for c in CheckClass])
    with pytest.raises(FileNotFoundError):
        svc.commit("org/model", passing)


# -------------------------------------------------------------- validation


def test_validation_catches_a_wrong_d_model(tmp_path: Path):
    svc = service(tmp_path)
    wrong = {i: torch.eye(4, dtype=torch.float16) for i in LAYERS}
    ref = svc.write_staged("org/model", wrong, "recipe: test")
    report = svc.validate(ref, D_MODEL, LAYERS, N_VOCAB, **live_passes())

    assert report.passed is False
    structural = next(r for r in report.results if r.check is CheckClass.STRUCTURAL)
    assert "d_model is 8" in structural.detail


def test_validation_catches_a_missing_layer(tmp_path: Path):
    svc = service(tmp_path)
    ref = svc.write_staged("org/model", {0: torch.eye(D_MODEL)}, "recipe: test")
    report = svc.validate(ref, D_MODEL, LAYERS, N_VOCAB, **live_passes())
    assert report.passed is False


def test_a_corrupt_artifact_fails_structurally_rather_than_raising(tmp_path: Path):
    """A file that does not deserialize is a FAIL, not an exception at serve."""
    svc = service(tmp_path)
    ref = svc.write_staged("org/model", jacobians(), "recipe: test")
    ref.lens_path.write_bytes(b"not a torch file")

    report = svc.validate(ref, D_MODEL, LAYERS, N_VOCAB, **live_passes())
    assert report.passed is False
    structural = next(r for r in report.results if r.check is CheckClass.STRUCTURAL)
    assert structural.status is CheckStatus.FAIL


def test_artifacts_are_loaded_weights_only():
    """An artifact is an untrusted file this process is about to load.

    The unrestricted loader executes pickled code, so this is a security
    property rather than a preference — asserted on the source because a
    behavioural test would need a malicious pickle.
    """
    source = inspect.getsource(JLensArtifactService._load_payload)
    assert "weights_only=True" in source


# ----------------------------------------------------------------- serving


def test_serving_refuses_without_a_passing_report(tmp_path: Path):
    """Serving an unvalidated artifact is the failure BR-030 exists for."""
    svc = service(tmp_path)
    ref = svc.write_staged("org/model", jacobians(), "recipe: test")
    report = svc.validate(ref, D_MODEL, LAYERS, N_VOCAB, **live_passes())
    svc.commit("org/model", report)

    with pytest.raises(ArtifactNotValidated):
        svc.load_for_readout("org/model", report=None)

    failing = ValidationReport([CheckResult(c, CheckStatus.FAIL, "no") for c in CheckClass])
    with pytest.raises(ArtifactNotValidated):
        svc.load_for_readout("org/model", report=failing)


def test_serving_returns_tensors_keyed_by_int_layer(tmp_path: Path):
    svc = service(tmp_path)
    ref = svc.write_staged("org/model", jacobians(), "recipe: test")
    report = svc.validate(ref, D_MODEL, LAYERS, N_VOCAB, **live_passes())
    svc.commit("org/model", report)

    loaded = svc.load_for_readout("org/model", report=report)
    assert sorted(loaded) == LAYERS
    assert all(isinstance(k, int) for k in loaded)
    assert loaded[0].shape == (D_MODEL, D_MODEL)


def test_serving_an_absent_artifact_raises(tmp_path: Path):
    svc = service(tmp_path)
    passing = ValidationReport([CheckResult(c, CheckStatus.PASS, "ok") for c in CheckClass])
    with pytest.raises(FileNotFoundError):
        svc.load_for_readout("org/absent", report=passing)


# --------------------------------------------------------------- ambiguity


def test_a_directory_with_two_lens_files_is_not_an_artifact(tmp_path: Path):
    """The consumer picks among them without saying which."""
    svc = service(tmp_path)
    ref = svc.write_staged("org/model", jacobians(), "recipe: test")
    report = svc.validate(ref, D_MODEL, LAYERS, N_VOCAB, **live_passes())
    published = svc.commit("org/model", report)

    (published.directory / "other_jacobian_lens.pt").write_bytes(b"x")
    assert svc.find("org/model") is None
    assert svc.list_artifacts() == []


# ---------------------------------------------------------------------------
# The verdict is recorded with the artifact, and identity-checked
#
# A published, semantically-valid partial artifact was refused at read time.
# The fit had validated it with a fixture the caller chose for the layers they
# fitted; that verdict was discarded and the readout re-validated with its own
# hard-coded fixture, which targets mid-stack — a question a top-of-stack fit
# was never fitted to answer. It failed a different test than the one it passed.
#
# MUTATION CONTROLS (each must turn this section red):
#   * commit stops writing validation.json      -> "records the verdict" fails
#   * stored_report skips the identity check     -> "a swapped file" fails
#   * the refusal drops the failing detail       -> "names the failing class" fails
# ---------------------------------------------------------------------------


def _passing_report():
    from src.services.jlens_validation import (
        CheckClass,
        CheckResult,
        CheckStatus,
        ValidationReport,
    )

    return ValidationReport(
        [
            CheckResult(c, CheckStatus.PASS, f"{c.value} ok")
            for c in CheckClass
        ]
    )


def _staged(tmp_path, service, layers=(24, 25)):
    import torch

    return service.write_staged(
        "org/model",
        {l: torch.randn(4, 4) for l in layers},
        "corpus: test\n",
    )


def test_commit_records_the_verdict_beside_the_artifact(tmp_path):
    from src.services.jlens_artifact_service import (
        VALIDATION_FILE,
        JLensArtifactService,
    )

    service = JLensArtifactService(tmp_path)
    _staged(tmp_path, service)
    ref = service.commit("org/model", _passing_report())

    assert (ref.directory / VALIDATION_FILE).is_file(), (
        "commit published without recording the verdict, so the readout must "
        "re-derive it — and re-derives it with a different fixture"
    )
    stored = service.stored_report(ref)
    assert stored is not None and stored["serviceable"] is True


def test_a_swapped_lens_file_invalidates_the_recorded_verdict(tmp_path):
    """Serving a lens fitted for other weights is what this gate prevents."""
    import torch

    from src.services.jlens_artifact_service import JLensArtifactService

    service = JLensArtifactService(tmp_path)
    _staged(tmp_path, service)
    ref = service.commit("org/model", _passing_report())
    assert service.stored_report(ref) is not None

    # Replace the weights, leaving the verdict beside them untouched.
    torch.save({24: torch.randn(4, 4), 25: torch.randn(4, 4)}, ref.lens_path)

    assert service.stored_report(ref) is None, (
        "a replaced lens file was still served on the OLD verdict; that verdict "
        "describes different weights, which produces a complete, plausible "
        "readout that is wrong"
    )


def test_the_refusal_names_the_failing_class(tmp_path):
    """A refusal the user cannot act on is only half a guard."""
    import pytest

    from src.api.v1.endpoints.jlens import _StoredReport
    from src.services.jlens_artifact_service import (
        ArtifactNotValidated,
        JLensArtifactService,
    )

    service = JLensArtifactService(tmp_path)
    failed = _StoredReport(
        {
            "passed": False,
            "serviceable": False,
            "summary": "semantic=fail",
            "results": [
                {
                    "check": "semantic",
                    "status": "fail",
                    "detail": "'spider' absent from the top-8 at layer 24",
                }
            ],
        }
    )

    with pytest.raises(ArtifactNotValidated) as excinfo:
        service.load_for_readout("org/model", report=failed)

    message = str(excinfo.value)
    assert "semantic" in message, f"the refusal names no failing class: {message}"
    assert "spider" in message, (
        f"the refusal drops the check's own detail: {message}. Without it the "
        "user cannot tell a missing report from a failed check"
    )


# ---------------------------------------------------------------------------
# A refit must not silently destroy coverage
#
# FROM THE CLUSTER LOGS, not from review:
#   2026-08-01 12:06:41  published lfm2.5-1.2b-instruct, layers 0..15, 134 MB
#   2026-08-02 12:57:52  published lfm2.5-1.2b-instruct, layers [1,2,3,10..15]
#
# The second `shutil.rmtree`'d the first. Nine minutes of GPU and the reference
# model's only FULL-STACK lens, gone with no warning, no backup and no way back
# — and the replacement does not dominate it (16 layers/120 prompts vs
# 9 layers/400 prompts, neither strictly better).
#
# MUTATION CONTROLS (each must turn this section red):
#   * commit rmtree's the old dir instead of archiving  -> "archives" fails
#   * the coverage guard is removed                      -> "refuses" fails
#   * .superseded is not excluded from discovery         -> "hidden" fails
#   * allow_coverage_loss is ignored                     -> "override" fails
# ---------------------------------------------------------------------------


def _commit_layers(tmp_path, layers, allow_loss=False):
    """Stage and commit an artifact covering exactly `layers`."""
    import torch

    from src.services.jlens_artifact_service import JLensArtifactService

    service = JLensArtifactService(tmp_path)
    service.write_staged(
        "org/model", {l: torch.randn(4, 4) for l in layers}, "corpus: t\n"
    )
    return service, service.commit(
        "org/model", _passing_report(), allow_coverage_loss=allow_loss
    )


def test_a_refit_that_loses_layers_is_REFUSED(tmp_path):
    from src.services.jlens_artifact_service import ArtifactCoverageLoss

    service, _ = _commit_layers(tmp_path, range(16))

    import torch

    service.write_staged(
        "org/model",
        {l: torch.randn(4, 4) for l in [1, 2, 3, 10, 11, 12, 13, 14, 15]},
        "corpus: t\n",
    )

    with pytest.raises(ArtifactCoverageLoss) as excinfo:
        service.commit("org/model", _passing_report())

    message = str(excinfo.value)
    # The refusal must NAME the layers at risk — "coverage would be reduced"
    # sends the user back to the logs to work out what they nearly lost.
    for layer in (0, 4, 5, 6, 7, 8, 9):
        assert str(layer) in message, f"layer {layer} not named in: {message}"

    # And the existing artifact must be untouched by a refused commit.
    payload = service._load_payload(service.find("org/model"))
    assert sorted(int(k) for k in payload) == list(range(16))


def test_the_refusal_can_be_overridden_deliberately(tmp_path):
    """Losing coverage is allowed — it just has to be a DECISION."""
    import torch

    service, _ = _commit_layers(tmp_path, range(16))
    service.write_staged("org/model", {l: torch.randn(4, 4) for l in [1, 2]}, "c: t\n")

    ref = service.commit("org/model", _passing_report(), allow_coverage_loss=True)
    payload = service._load_payload(ref)
    assert sorted(int(k) for k in payload) == [1, 2]


def test_a_superseded_artifact_is_ARCHIVED_not_deleted(tmp_path):
    """The replaced artifact survives one generation, so a mistake is undoable."""
    import torch

    from src.services.jlens_artifact_service import SUPERSEDED_SUFFIX

    service, _ = _commit_layers(tmp_path, range(16))
    service.write_staged("org/model", {l: torch.randn(4, 4) for l in [1, 2]}, "c: t\n")
    service.commit("org/model", _passing_report(), allow_coverage_loss=True)

    archive = tmp_path / f"model{SUPERSEDED_SUFFIX}"
    assert archive.is_dir(), (
        "the replaced artifact was deleted outright; nine minutes of GPU and a "
        "full-stack lens went with it the first time this happened"
    )
    recovered = torch.load(
        next(archive.glob("*_jacobian_lens.pt")), weights_only=True
    )
    assert sorted(int(k) for k in recovered) == list(range(16))


def test_the_archive_is_hidden_from_discovery(tmp_path):
    """Two directories for one model would let the consumer pick either."""
    import torch

    service, _ = _commit_layers(tmp_path, range(16))
    service.write_staged("org/model", {l: torch.randn(4, 4) for l in [1, 2]}, "c: t\n")
    service.commit("org/model", _passing_report(), allow_coverage_loss=True)

    slugs = [a.slug for a in service.list_artifacts()]
    assert slugs == ["model"], (
        f"discovery returned {slugs}; a superseded artifact must not be "
        "servable, or a stale lens is one directory listing away from being used"
    )


def test_a_refit_that_ADDS_layers_is_not_refused(tmp_path):
    """Negative control: the guard must not block a genuine upgrade."""
    import torch

    service, _ = _commit_layers(tmp_path, [1, 2])
    service.write_staged(
        "org/model", {l: torch.randn(4, 4) for l in range(16)}, "c: t\n"
    )
    ref = service.commit("org/model", _passing_report())
    payload = service._load_payload(ref)
    assert sorted(int(k) for k in payload) == list(range(16))


# ---------------------------------------------------------------------------
# The fp16 storage scale must SURVIVE to the readout (F2)
#
# `_to_storage_dtype` divides each matrix down so the fp16 cast cannot saturate
# — GPT-2's layer-6 Jacobian peaks around 1.7e7 against fp16's 65504 ceiling —
# and its docstring has always said "The scale is stored in the artifact's
# config.yaml". It was not. `FitResult.scales` was computed, returned, and
# dropped on the floor.
#
# Ranked readouts never noticed: the model's final norm divides a positive
# scalar straight back out, so `softmax(W_U @ norm(alpha * J @ h))` is exactly
# `softmax(W_U @ norm(J @ h))`. Everything that does NOT normalise did notice —
# probe scores and intervention magnitudes came out scaled by an unrecorded
# per-layer alpha and were not comparable across layers.
#
# MUTATION CONTROLS (each must turn this section red):
#   * stop writing layer_scales into config.yaml -> "round trip" fails
#   * JacobianTransport ignores `scales`         -> "unscales" fails
# ---------------------------------------------------------------------------


def test_layer_scales_round_trip_through_the_written_config(tmp_path):
    """Write a fit's scales, read them back off disk."""
    import torch

    from src.services.jlens_artifact_service import JLensArtifactService
    from src.workers.jlens_fit_tasks import _config_yaml

    class _Result:
        scales = {24: 1.0, 25: 259.4}
        prompts_seen = 100
        converged = False
        convergence_delta = 1e-3
        residual_mean = {24: 0.01, 25: 0.02}
        residual_max = {24: 0.03, 25: 0.04}

    class _Loaded:
        name = "org/model"
        d_model = 4
        n_layers = 26
        n_vocab = 256
        model = None
        structure = type("S", (), {"num_layers": 26, "attention_module": None})()

    service = JLensArtifactService(tmp_path)
    ref = service.write_staged(
        "org/model",
        {24: torch.randn(4, 4), 25: torch.randn(4, 4)},
        _config_yaml(_Loaded(), _Result(), freeze_qk=True, corpus_name="t"),
    )

    recovered = service.layer_scales(ref)
    assert recovered == {24: 1.0, 25: 259.4}, (
        f"scales did not survive the write/read round trip: {recovered}. An "
        "unrecorded scale makes every probe and intervention magnitude wrong "
        "by a per-layer factor, and the artifact unreconstructible"
    )


def test_the_transport_undoes_the_storage_scale():
    """A scaled matrix read back unscaled is the stored J, not the fitted J."""
    import torch

    from src.services.jlens_readout_service import JacobianTransport

    true_j = torch.eye(4) * 3.0
    alpha = 100.0
    stored = true_j / alpha  # what _to_storage_dtype would have written

    unscaled = JacobianTransport({7: stored}, scales={7: alpha})
    h = torch.ones(4)
    assert torch.allclose(unscaled.apply(h, 7), true_j @ h, atol=1e-4), (
        "the transport did not undo the storage scale; probe scores and "
        "intervention magnitudes are off by that factor"
    )

    # And an artifact with no recorded scale is read as-is rather than refused,
    # so lenses fitted before the scale was written stay usable.
    plain = JacobianTransport({7: stored})
    assert torch.allclose(plain.apply(h, 7), stored @ h, atol=1e-6)


def test_ranking_is_invariant_to_the_scale_but_probing_is_not():
    """Why this went unnoticed, pinned so the reasoning is not re-derived."""
    import torch

    from src.services.jlens_readout_service import JacobianTransport

    j = torch.randn(6, 6)
    h = torch.randn(6)
    alpha = 250.0

    scaled = JacobianTransport({0: j / alpha}, scales={0: alpha}).apply(h, 0)
    unscaled = JacobianTransport({0: j / alpha}).apply(h, 0)

    # RMS-normalised, the two are identical — which is exactly why every ranked
    # readout looked correct while the magnitudes were wrong.
    def rms_norm(x):
        return x / x.pow(2).mean().sqrt().clamp_min(1e-6)

    assert torch.allclose(rms_norm(scaled), rms_norm(unscaled), atol=1e-4)
    # Unnormalised — a probe score — they differ by the factor.
    assert not torch.allclose(scaled, unscaled, atol=1e-3)
