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
