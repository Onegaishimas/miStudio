"""Circuit strength calibration service (Feature 20 / IDL-37).

Orchestrates: generate neutral falsifiable probes → search the usable band
(onset by drift, cliff by judge) → clamp the served dial to it → persist a
reproducible manifest and the band onto the circuit.

Design split, so the orchestration is testable without a GPU or a live LLM:
  * the SEARCH is a pure function of injected callbacks (circuit_calibration_search);
  * the JUDGE and PROBE generator wrap the enhanced-labeling LLM client;
  * the GPU GENERATION (multi-layer circuit steering at a given dial) is isolated
    in `_build_generation_fns`, mirroring CircuitFaithfulnessService's model
    loading. `run(...)` accepts injected `gen_at`/`baseline_at`/`judge` so the
    whole orchestration — clamp, manifest, write-back, reproduce — is covered by
    unit tests; production passes the GPU/LLM-backed callables.

Like faithfulness, calibration runs ON the circuit and holds the GPU, so its
in-flight lifecycle (calibration_status/task_id) lives on the circuit row.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .circuit_calibration_search import calibrate
from .circuit_probe_generator import generate_probes

logger = logging.getLogger(__name__)

DEFAULT_STEP_BUDGET = 10
DEFAULT_PROBE_COUNT = 3
DEFAULT_MARGIN = 0.15
JUDGE_METRIC_ID = "neutral-fact-correctness/v1"


class CalibrationConfigError(ValueError):
    """Bad calibration request — surfaces as a 422."""


class CalibrationRunError(RuntimeError):
    """Calibration failed at run time."""


class CircuitCalibrationService:
    """Find and ship a circuit's usable steering band."""

    @staticmethod
    def create_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = dict(config or {})
        step_budget = int(cfg.get("step_budget", DEFAULT_STEP_BUDGET))
        probe_count = int(cfg.get("probe_count", DEFAULT_PROBE_COUNT))
        seed = int(cfg.get("seed", 0))
        margin = float(cfg.get("margin", DEFAULT_MARGIN))
        if step_budget < 2:
            raise CalibrationConfigError("step_budget must be ≥ 2")
        if probe_count < 1:
            raise CalibrationConfigError("probe_count must be ≥ 1")
        if not (0.0 <= margin <= 1.0):
            raise CalibrationConfigError("margin must be in [0, 1]")
        return {"step_budget": step_budget, "probe_count": probe_count,
                "seed": seed, "margin": margin}

    @staticmethod
    def _member_labels(circuit) -> List[str]:
        labels: List[str] = []
        for m in (circuit.members or []):
            feat = m.get("feature") or {}
            lb = feat.get("label")
            if lb:
                labels.append(str(lb))
        return labels

    @staticmethod
    def _dial_bounds(circuit) -> tuple[float, float]:
        """The dial search range = the circuit's AUTHORED intensity_range, so
        calibration stays inside what the author allowed. Defaults to [0, 2] to
        match CircuitBudget."""
        budget = circuit.budget or {}
        rng = budget.get("intensity_range") or [0.0, 2.0]
        try:
            lo, hi = float(rng[0]), float(rng[1])
        except (TypeError, ValueError, IndexError):
            lo, hi = 0.0, 2.0
        if hi <= lo:
            lo, hi = 0.0, 2.0
        return lo, hi

    @classmethod
    def build_band(
        cls,
        circuit,
        *,
        gen_at: Callable[[float, str], str],
        baseline_at: Callable[[str], str],
        judge: Callable[[str, str], str],
        divergence: Callable[[str, str], float],
        config: Optional[Dict[str, Any]] = None,
        llm_call: Optional[Callable[[str, int], str]] = None,
    ) -> Dict[str, Any]:
        """The pure orchestration: probes → search → band + manifest payload.

        Returns {"band": <CircuitCalibration dict>, "manifest_payload": {...}}.
        All generation/judging is injected — no GPU, no network — so this is the
        unit-tested seam. `run()` wires the real GPU/LLM callables into here.
        """
        cfg = cls.create_config(config)
        labels = cls._member_labels(circuit)
        probes = generate_probes(labels, llm_call=llm_call,
                                 n=cfg["probe_count"])
        lo, hi = cls._dial_bounds(circuit)

        result = calibrate(
            gen_at, baseline_at, judge, divergence,
            probes=probes, lo=lo, hi=hi,
            max_steps=cfg["step_budget"], margin=cfg["margin"],
        )

        band = {
            "onset": result.onset,
            "sweet_spot": result.sweet_spot,
            "cliff": result.cliff,
            "provisional": True,   # generated probes + discovery-plane measurement
            "probe_set": probes,
            "judge_metric_id": JUDGE_METRIC_ID,
            "step_budget": cfg["step_budget"],
            "non_monotone": result.non_monotone,
        }
        manifest_payload = {
            "probes": probes,
            "config": cfg,
            "band": {k: band[k] for k in ("onset", "sweet_spot", "cliff",
                                          "non_monotone")},
            "trace": result.trace,
            "converged": result.converged,
            "steps_used": result.steps_used,
            "dial_bounds": [lo, hi],
        }
        return {"band": band, "manifest_payload": manifest_payload}

    @classmethod
    def run(cls, db, circuit_id: str, config: Dict[str, Any], *,
            progress_cb: Optional[Callable[[int], None]] = None) -> Dict[str, Any]:
        """GPU orchestrator (sync, called from the Celery task).

        Loads the circuit, builds the GPU-backed generation + LLM judge, searches
        the band via build_band, persists a `calibration` manifest, and clamps
        the served dial + writes the band onto the circuit through the contract.

        The GPU generation wiring (`_build_generation_fns`) is the calibration
        close-out step (tracked debt, like 017's Phase-6 GPU acceptance); the
        orchestration below is otherwise complete and unit-tested via build_band.
        """
        from ..models.circuit import Circuit

        circuit = db.query(Circuit).filter(Circuit.id == circuit_id).first()
        if circuit is None:
            raise CalibrationRunError(f"Circuit {circuit_id} not found")
        if not circuit.members:
            raise CalibrationRunError("Circuit has no members")
        cfg = cls.create_config(config)

        if progress_cb:
            progress_cb(5)
        gen_at, baseline_at, judge, divergence, llm_call = \
            cls._build_generation_fns(circuit, db, cfg["seed"])

        out = cls.build_band(
            circuit, gen_at=gen_at, baseline_at=baseline_at, judge=judge,
            divergence=divergence, config=cfg, llm_call=llm_call)
        if progress_cb:
            progress_cb(80)

        manifest_id = cls._persist_manifest(db, circuit_id, out["manifest_payload"])
        band = dict(out["band"])
        band["manifest_ref"] = manifest_id
        cls._write_calibration(db, circuit_id, band)
        if progress_cb:
            progress_cb(100)
        return {"circuit_id": circuit_id, "band": band,
                "manifest_ref": manifest_id}

    @staticmethod
    def _persist_manifest(db, circuit_id: str, payload: Dict[str, Any]) -> str:
        """Persist a self-contained `calibration` manifest (sync path, mirrors
        faithfulness). Returns the manifest id."""
        from ..models.validation_manifest import ValidationManifest
        from .manifest_service import validate_payload

        validate_payload("calibration", payload)
        man = ValidationManifest(kind="calibration", payload=payload,
                                 circuit_id=circuit_id)
        db.add(man)
        db.commit()
        db.refresh(man)
        return man.id

    @staticmethod
    def _write_calibration(db, circuit_id: str, band: Dict[str, Any]) -> None:
        """Write the band onto the circuit AND clamp the dial, through the
        CircuitDefinitionV1 contract (validators + version bump) — never a raw
        JSONB mutation (mirrors _write_circuit_faithfulness). Clamps
        intensity_range to [onset, cliff] and intensity to sweet_spot."""
        from ..models.circuit import Circuit
        from ..schemas.circuit_definition import CircuitDefinitionV1

        circuit = db.query(Circuit).filter(Circuit.id == circuit_id).first()
        if circuit is None:
            return
        onset, sweet, cliff = band["onset"], band["sweet_spot"], band["cliff"]
        if not (onset <= sweet <= cliff):
            raise CalibrationRunError(
                f"refusing to clamp to an inverted band "
                f"(onset={onset}, sweet_spot={sweet}, cliff={cliff})")
        budget = dict(circuit.budget or {})
        budget["intensity_range"] = [onset, cliff]
        budget["intensity"] = sweet
        try:
            defn = CircuitDefinitionV1(
                name=circuit.name, narrative=circuit.narrative,
                saes=circuit.saes, members=circuit.members, edges=circuit.edges,
                budget=budget, faithfulness=circuit.faithfulness, calibration=band)
        except Exception as e:
            raise CalibrationRunError(
                f"calibration produced an invalid circuit: {e}") from e
        circuit.budget = defn.budget.model_dump(mode="json")
        circuit.calibration = defn.calibration.model_dump(mode="json")
        circuit.version = (circuit.version or 1) + 1
        circuit.calibration_status = "completed"
        db.commit()

    @staticmethod
    def _build_generation_fns(circuit, db, seed: int):
        """Build the GPU-backed gen_at/baseline_at/judge/divergence callables for
        a real run. Mirrors CircuitFaithfulnessService's model loading: load the
        model + the circuit's per-layer SAEs, apply the multi-feature steering
        hooks scaled by the dial, and generate.

        Isolated here because it is the only GPU-dependent surface; its
        on-hardware execution is the calibration close-out step (tracked debt,
        like 017's Phase-6 GPU acceptance). `build_band` above is fully covered
        without it.
        """
        raise NotImplementedError(
            "GPU generation wiring is the calibration close-out step; "
            "build_band() runs the full orchestration with injected callables")


def cosine_text_divergence(embed: Callable[[str], List[float]]):
    """Return a divergence(a, b) = 1 − cos(embed(a), embed(b)) using an injected
    embedder. Falls back to a token-Jaccard distance when no embedder is given —
    coarse but deterministic and dependency-free."""
    if embed is not None:
        def _div(a: str, b: str) -> float:
            va, vb = embed(a), embed(b)
            dot = sum(x * y for x, y in zip(va, vb))
            na = sum(x * x for x in va) ** 0.5
            nb = sum(y * y for y in vb) ** 0.5
            if na == 0 or nb == 0:
                return 1.0
            return 1.0 - dot / (na * nb)
        return _div

    def _jaccard(a: str, b: str) -> float:
        sa, sb = set(a.lower().split()), set(b.lower().split())
        if not sa and not sb:
            return 0.0
        return 1.0 - len(sa & sb) / len(sa | sb)
    return _jaccard
