"""CircuitCalibrationService orchestration (IDL-37), covered without GPU/LLM.

build_band() is the tested seam: it runs probes → search → band + manifest with
injected generation/judge/divergence, so everything except the GPU generation
loop is unit-covered. Also pins config validation and the manifest's
self-containment.
"""

import pytest

from src.services.circuit_calibration_service import (
    CalibrationConfigError, CircuitCalibrationService, cosine_text_divergence)


class _Circuit:
    """Minimal stand-in with the attributes the service reads."""
    def __init__(self, labels, intensity_range=None):
        self.members = [{"feature": {"label": lb}} for lb in labels]
        self.budget = {"intensity_range": intensity_range} if intensity_range else {}


# Synthetic world: cliff at 0.6, fluent throughout (so only the judge finds it).
def _gen_at(dial, prompt):
    return f"dial={dial}|{prompt}"


def _baseline_at(prompt):
    return f"dial=0.0|{prompt}"


def _judge(text, expected):
    dial = float(text.split("|", 1)[0].split("=")[1])
    return "correct" if dial <= 0.6 else "broken"


def _divergence(a, b):
    da = float(a.split("|", 1)[0].split("=")[1])
    db_ = float(b.split("|", 1)[0].split("=")[1])
    return abs(da - db_)


def _fixed_probes_llm(prompt, max_tokens):
    import json
    return json.dumps([{"prompt": "What is the capital of France?", "expected": "Paris"}])


class TestBuildBand:
    def test_produces_an_ordered_provisional_band_with_a_manifest(self):
        c = _Circuit(["parody", "humor"], intensity_range=[0.0, 1.0])
        out = CircuitCalibrationService.build_band(
            c, gen_at=_gen_at, baseline_at=_baseline_at, judge=_judge,
            divergence=_divergence, llm_call=_fixed_probes_llm,
            config={"step_budget": 12})
        band = out["band"]
        assert band["onset"] <= band["sweet_spot"] <= band["cliff"]
        assert abs(band["cliff"] - 0.6) <= 0.06
        assert band["provisional"] is True            # generated probes
        assert band["judge_metric_id"]
        assert band["probe_set"]                       # travels in the contract

    def test_manifest_is_self_contained_and_valid(self):
        from src.services.manifest_service import validate_payload
        c = _Circuit(["parody"], intensity_range=[0.0, 1.0])
        out = CircuitCalibrationService.build_band(
            c, gen_at=_gen_at, baseline_at=_baseline_at, judge=_judge,
            divergence=_divergence, llm_call=_fixed_probes_llm)
        # Must satisfy the calibration manifest kind's required keys.
        validate_payload("calibration", out["manifest_payload"])  # no raise
        assert out["manifest_payload"]["trace"]        # every judged step recorded

    def test_search_stays_within_the_authored_intensity_range(self):
        # A narrow authored range caps where calibration can look.
        c = _Circuit(["parody"], intensity_range=[0.0, 0.5])
        out = CircuitCalibrationService.build_band(
            c, gen_at=_gen_at, baseline_at=_baseline_at, judge=_judge,
            divergence=_divergence, llm_call=_fixed_probes_llm)
        # Cliff is 0.6 in the judge, but the search cannot exceed hi=0.5.
        assert out["band"]["cliff"] <= 0.5


class TestConfig:
    def test_rejects_tiny_step_budget(self):
        with pytest.raises(CalibrationConfigError):
            CircuitCalibrationService.create_config({"step_budget": 1})

    def test_rejects_zero_probes(self):
        with pytest.raises(CalibrationConfigError):
            CircuitCalibrationService.create_config({"probe_count": 0})

    def test_rejects_out_of_range_margin(self):
        with pytest.raises(CalibrationConfigError):
            CircuitCalibrationService.create_config({"margin": 1.5})


class _FakeCircuit:
    """In-memory circuit for the sync write-back, matching the attributes
    _write_calibration reads/writes without a DB session."""
    def __init__(self):
        self.id = "crc_test"
        self.name = "calib"
        self.narrative = None
        self.saes = [{"mistudio_sae_id": "sae_l12", "layer": 12}]
        self.members = [{"layer": 12, "member_kind": "feature_ref",
                         "feature": {"feature_idx": 1, "strength": 1.0,
                                     "label": "parody"}}]
        self.edges = []
        self.budget = {"intensity_range": [0.0, 1.0], "intensity": 1.0,
                       "layers": {"12": {"B": 1.0}}}
        self.faithfulness = None
        self.calibration = None
        self.version = 3
        self.calibration_status = "running"


class _FakeSyncDB:
    def __init__(self, circuit):
        self._c = circuit
        self.committed = False

    def query(self, _model):
        db = self

        class _Q:
            def filter(self, *a, **k):
                return self

            def first(self_inner):
                return db._c
        return _Q()

    def commit(self):
        self.committed = True


class TestWriteCalibrationClampsThroughTheContract:
    """The sync write-back (used by the GPU task) clamps the dial AND writes the
    band through the CircuitDefinitionV1 contract — never a raw JSONB mutation,
    mirroring _write_circuit_faithfulness."""

    _BAND = {"onset": 0.2, "sweet_spot": 0.5, "cliff": 0.6, "provisional": True,
             "probe_set": [{"prompt": "Capital of France?", "expected": "Paris"}],
             "judge_metric_id": "x/v1", "step_budget": 10, "non_monotone": False,
             "manifest_ref": "vman_x"}

    def test_it_clamps_the_dial_and_stores_the_band(self):
        c = _FakeCircuit()
        db = _FakeSyncDB(c)
        CircuitCalibrationService._write_calibration(db, c.id, self._BAND)
        assert c.budget["intensity_range"] == [0.2, 0.6]
        assert c.budget["intensity"] == 0.5
        assert c.budget["layers"]["12"]["B"] == 1.0   # per-layer budget preserved
        assert c.calibration["sweet_spot"] == 0.5
        assert c.calibration_status == "completed"
        assert c.version == 4                          # bumped
        assert db.committed

    def test_it_refuses_an_inverted_band(self):
        from src.services.circuit_calibration_service import CalibrationRunError
        c = _FakeCircuit()
        db = _FakeSyncDB(c)
        with pytest.raises(CalibrationRunError):
            CircuitCalibrationService._write_calibration(
                db, c.id, {**self._BAND, "onset": 0.7, "sweet_spot": 0.5, "cliff": 0.6})
        assert c.calibration is None                   # nothing written on refusal


class TestDivergence:
    def test_embedder_cosine(self):
        div = cosine_text_divergence(embed=lambda t: [1.0, 0.0] if "a" in t else [0.0, 1.0])
        assert div("a", "a") == pytest.approx(0.0)
        assert div("a", "b") == pytest.approx(1.0)

    def test_jaccard_fallback_when_no_embedder(self):
        div = cosine_text_divergence(embed=None)
        assert div("the cat sat", "the cat sat") == pytest.approx(0.0)
        assert div("cat", "dog") == pytest.approx(1.0)
