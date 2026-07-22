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

from ..core.config import settings
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
        # The judge + probe-generation LLM endpoint/model, carried on the request
        # like an enhanced-labeling job (labeling config is per-request, not a
        # global setting). Optional: without them the search still runs with the
        # neutral static probes but cannot judge a real cliff — the endpoint
        # gates whether run() can do a real GPU pass.
        return {"step_budget": step_budget, "probe_count": probe_count,
                "seed": seed, "margin": margin,
                "judge_endpoint": cfg.get("judge_endpoint"),
                "judge_model": cfg.get("judge_model"),
                "judge_api_key": cfg.get("judge_api_key")}

    @staticmethod
    def _member_labels(circuit) -> List[str]:
        labels: List[str] = []
        for m in (circuit.members or []):
            feat = m.get("feature") or {}
            lb = feat.get("label")
            if lb:
                labels.append(str(lb))
        return labels

    #: CircuitBudget.intensity is bounded le=2.0, so a served dial (and thus a
    #: calibrated cliff/sweet_spot) can never exceed this. The search must not
    #: look above it, or the clamp would produce a budget the contract rejects.
    MAX_DIAL = 2.0

    @classmethod
    def _dial_bounds(cls, circuit) -> tuple[float, float]:
        """The dial search range = the circuit's AUTHORED intensity_range,
        CAPPED at MAX_DIAL (the servable ceiling). Defaults to [0, 2]."""
        budget = circuit.budget or {}
        rng = budget.get("intensity_range") or [0.0, cls.MAX_DIAL]
        try:
            lo, hi = float(rng[0]), float(rng[1])
        except (TypeError, ValueError, IndexError):
            lo, hi = 0.0, cls.MAX_DIAL
        if hi <= lo:
            lo, hi = 0.0, cls.MAX_DIAL
        # Never search (or clamp) above what the budget can hold — an authored
        # range like [0, 3.0] would otherwise yield a cliff the contract rejects.
        hi = min(hi, cls.MAX_DIAL)
        lo = max(0.0, min(lo, hi))
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
            "usable_band": result.usable_band,
            "trace": result.trace,
            "converged": result.converged,
            "steps_used": result.steps_used,
            "floor": result.floor,
            "dial_bounds": [lo, hi],
        }
        # `usable_band` is an ORCHESTRATION signal, not a contract field: when
        # False the search found no correct dial above onset, so the caller must
        # NOT clamp the served dial to this degenerate band (badge, not gate —
        # a failed measurement must not disable the circuit).
        return {"band": band, "usable_band": result.usable_band,
                "manifest_payload": manifest_payload}

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
            cls._build_generation_fns(circuit, db, cfg)

        out = cls.build_band(
            circuit, gen_at=gen_at, baseline_at=baseline_at, judge=judge,
            divergence=divergence, config=cfg, llm_call=llm_call)
        if progress_cb:
            progress_cb(80)

        # Persist the manifest FIRST so the measurement is never lost — but the
        # order below (manifest → write) is safe because _write_calibration
        # validates the band and can only fail on an inverted band, which
        # build_band cannot produce. The no-usable-band case is NOT a failure:
        # we record the manifest and mark the run completed WITHOUT clamping.
        manifest_id = cls._persist_manifest(db, circuit_id, out["manifest_payload"])

        if not out["usable_band"]:
            # No correct dial above onset — do NOT clamp (badge, not gate: a
            # failed measurement must not disable the circuit). Record the
            # outcome so an agent can see WHY there is no band.
            cls._mark_no_usable_band(db, circuit_id)
            if progress_cb:
                progress_cb(100)
            return {"circuit_id": circuit_id, "band": None,
                    "usable_band": False, "manifest_ref": manifest_id,
                    "reason": "no dial above onset was judged correct"}

        band = dict(out["band"])
        band["manifest_ref"] = manifest_id
        cls._write_calibration(db, circuit_id, band)
        if progress_cb:
            progress_cb(100)
        return {"circuit_id": circuit_id, "band": band,
                "usable_band": True, "manifest_ref": manifest_id}

    @classmethod
    def reproduce(cls, db, manifest_id: str, *,
                  progress_cb: Optional[Callable[[int], None]] = None
                  ) -> Dict[str, Any]:
        """Re-run calibration from a stored manifest's config + probes and store
        a `reproduction` manifest with the band-delta verdict (FPRD §8.5). Uses
        the SAME probes and seed as the original, so a reproducible measurement
        lands within tolerance. Does NOT clamp the circuit — reproduction only
        checks the number, it doesn't re-ship the band.
        """
        from ..models.circuit import Circuit
        from ..models.validation_manifest import ValidationManifest
        from .manifest_service import ManifestService

        original = db.query(ValidationManifest).filter(
            ValidationManifest.id == manifest_id).first()
        if original is None:
            raise CalibrationRunError(f"Manifest {manifest_id} not found")
        if original.kind != "calibration":
            raise CalibrationRunError(
                f"Manifest {manifest_id} is {original.kind!r}, not calibration")
        payload = original.payload or {}
        circuit = db.query(Circuit).filter(
            Circuit.id == original.circuit_id).first()
        if circuit is None:
            raise CalibrationRunError(
                "Original circuit is gone — cannot reproduce")

        cfg = cls.create_config(payload.get("config") or {})
        stored_probes = payload.get("probes") or []
        if progress_cb:
            progress_cb(5)
        gen_at, baseline_at, judge, divergence, _llm = \
            cls._build_generation_fns(circuit, db, cfg)

        # Reuse the EXACT stored probes — reproduction must not re-generate them.
        lo, hi = cls._dial_bounds(circuit)
        result = calibrate(
            gen_at, baseline_at, judge, divergence,
            probes=stored_probes, lo=lo, hi=hi,
            max_steps=cfg["step_budget"], margin=cfg["margin"])
        if progress_cb:
            progress_cb(80)

        repro_payload = {
            "probes": stored_probes,
            "config": cfg,
            "band": {"onset": result.onset, "sweet_spot": result.sweet_spot,
                     "cliff": result.cliff, "non_monotone": result.non_monotone},
            "usable_band": result.usable_band,
            "trace": result.trace,
        }
        verdict = ManifestService.calibration_reproduction_verdict(
            payload, repro_payload)
        repro_payload["reproduction_verdict"] = verdict
        repro_payload["reproduces"] = manifest_id

        from .manifest_service import validate_payload
        validate_payload("reproduction", repro_payload)  # path-safety
        man = ValidationManifest(kind="reproduction", payload=repro_payload,
                                 circuit_id=circuit.id,
                                 parent_manifest_id=manifest_id)
        db.add(man)
        db.commit()
        db.refresh(man)
        if progress_cb:
            progress_cb(100)
        return {"reproduces": manifest_id, "reproduction_manifest": man.id,
                "verdict": verdict}

    @staticmethod
    def _mark_no_usable_band(db, circuit_id: str) -> None:
        """A completed run that found no usable band: clear the in-flight marker
        WITHOUT clamping the dial or writing a calibration block. The circuit
        serves exactly as before."""
        from ..models.circuit import Circuit
        circuit = db.query(Circuit).filter(Circuit.id == circuit_id).first()
        if circuit is not None:
            circuit.calibration_status = "completed"
            db.commit()

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

    #: A short neutral prompt the model completes for onset/cliff generation.
    #: The probe's own prompt is what actually gets generated; this is the
    #: system-style lead-in kept minimal so the circuit's tint, not the prompt,
    #: drives divergence.
    GEN_MAX_TOKENS = 80

    @classmethod
    def _build_generation_fns(cls, circuit, db, cfg: Dict[str, Any]):
        """Build the GPU-backed (gen_at, baseline_at, judge, divergence, llm_call)
        for a real run. Mirrors CircuitFaithfulnessService's model loading; the
        hook is ADDITIVE steering (dial × strength × W_dec on the residual), not
        suppression.

        Executes only on a GPU host with the model + SAEs present — the same
        envelope as faithfulness's run() (which is likewise exercised on hardware,
        not in unit tests). The orchestration that consumes these callables
        (build_band, the clamp, the manifest) is fully unit-tested via injection.
        """
        import torch

        seed = int(cfg.get("seed", 0))
        judge_endpoint = cfg.get("judge_endpoint")
        judge_model = cfg.get("judge_model")
        if not judge_endpoint or not judge_model:
            raise CalibrationRunError(
                "calibration needs a judge LLM: pass judge_endpoint + judge_model "
                "(an OpenAI-compatible endpoint, e.g. the miLLM instance). Without "
                "a judge the correctness cliff cannot be found.")

        from ..ml.layer_discovery import (discover_transformer_structure,
                                          get_hookable_module)
        from ..ml.model_loader import load_model_from_hf
        from ..models.external_sae import ExternalSAE
        from ..models.model import Model, QuantizationFormat
        from .circuit_capture_service import _load_sae_sync
        from .enhanced_labeling_service import EnhancedLabelingService
        from .steering_service import resolve_decoder_weight

        # Resolve model + per-layer SAEs + decoder weights + member strengths.
        model_rec = db.query(Model).filter(Model.id == circuit.model_id).first()
        if model_rec is None:
            raise CalibrationRunError(
                f"Circuit's model {circuit.model_id} not found — cannot generate")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved = (settings.resolve_data_path(model_rec.file_path)
                    if model_rec.file_path else None)
        model, tokenizer, _c, _m = load_model_from_hf(
            repo_id=model_rec.repo_id,
            quant_format=QuantizationFormat(model_rec.quantization),
            cache_dir=resolved, device_map=device,
            local_files_only=bool(resolved and resolved.exists()))
        model.eval()
        structure = discover_transformer_structure(model)

        sae_id_by_layer = {int(s["layer"]): s.get("mistudio_sae_id")
                           for s in (circuit.saes or []) if s.get("layer") is not None}
        # Per member: (layer, feature_idx, strength). Steering is additive:
        # residual += dial × strength × W_dec[:, idx].
        steer = []  # (layer, idx, strength, decoder_weight)
        wdec_by_layer = {}
        for L, sid in sae_id_by_layer.items():
            sae_rec = db.query(ExternalSAE).filter(ExternalSAE.id == sid).first()
            if sae_rec is None:
                raise CalibrationRunError(
                    f"SAE {sid} for layer {L} not found — cannot steer")
            sae = _load_sae_sync(sae_rec, device)
            wdec_by_layer[L] = resolve_decoder_weight(sae)
        for m in (circuit.members or []):
            feat = m.get("feature") or {}
            L = m.get("layer")
            idx = feat.get("feature_idx")
            strength = float(feat.get("strength") or 0.0)
            if L in wdec_by_layer and idx is not None:
                steer.append((int(L), int(idx), strength, wdec_by_layer[L]))

        # Correct signature: get_hookable_module(layer_module, hook_type, structure)
        # — mirror faithfulness/steering. The residual is where additive steering
        # is applied. (A wrong arg order silently returns None and crashes at the
        # first hook registration — Feature 20 R2.)
        hook_layers = {}
        for L in wdec_by_layer:
            layer_mod = structure.layers_module[L]
            hook_layers[L] = get_hookable_module(layer_mod, "residual", structure)
            if hook_layers[L] is None:
                raise CalibrationRunError(
                    f"No hookable residual module for layer {L} on this model")

        # Gemma-2 and other hook-hostile architectures corrupt output under a
        # forward hook unless the KV cache is disabled — reuse the steering
        # service's marker check so calibration doesn't measure a garbage band.
        _marker_hay = f"{model_rec.repo_id} {type(model).__name__}".lower()
        from .steering_service import _CACHE_INCOMPATIBLE_MARKERS
        disable_cache = any(mk in _marker_hay
                            for mk in _CACHE_INCOMPATIBLE_MARKERS)

        def _make_hook(dial):
            layer_members = {}
            for (L, idx, strength, wdec) in steer:
                layer_members.setdefault(L, []).append((idx, strength, wdec))

            def _hook_for(L):
                def hook(module, inp, output):
                    is_tuple = isinstance(output, tuple)
                    hidden = output[0] if is_tuple else output
                    if hidden.dim() != 3:
                        return output
                    with torch.no_grad():
                        for (idx, strength, wdec) in layer_members[L]:
                            vec = wdec[:, idx].to(hidden.dtype)
                            hidden.add_(dial * strength * vec)
                    return output
                return hook
            return {L: _hook_for(L) for L in layer_members}

        def _generate(prompt: str, dial: float, gseed: int) -> str:
            torch.manual_seed(gseed)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            handles = []
            if dial > 0:
                for L, hook in _make_hook(dial).items():
                    handles.append(hook_layers[L].register_forward_hook(hook))
            gen_kwargs = dict(max_new_tokens=cls.GEN_MAX_TOKENS,
                              do_sample=True, temperature=0.7)
            if disable_cache:
                gen_kwargs["use_cache"] = False
            try:
                with torch.no_grad():
                    out = model.generate(**inputs, **gen_kwargs)
            finally:
                for h in handles:
                    h.remove()
            text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            return text

        # onset compares a STEERED draw against a baseline draw; both must use the
        # SAME sampling seed (0) so the divergence reflects the circuit's tint, not
        # a seed-induced sampling difference (Feature 20 R2). The cliff judge reads
        # correctness, not divergence, so its dial-seed choice doesn't bias it; we
        # use seed 0 throughout for reproducibility.
        def gen_at(dial, prompt):
            return _generate(prompt, dial, 0)

        def baseline_at(prompt, s):
            return _generate(prompt, 0.0, s)

        # Judge + probe-generation LLM: the enhanced-labeling client, same
        # endpoint/key config as labeling (reasoning-model quirks handled there).
        labeler = EnhancedLabelingService(
            endpoint=judge_endpoint, model=judge_model,
            api_key=cfg.get("judge_api_key"))

        def llm_call(prompt, max_tokens):
            return labeler._call_llm(prompt, max_tokens)

        def judge(generation, expected):
            raw = labeler._call_llm(
                "You are grading whether a model response is factually CORRECT.\n"
                f"Question context: the correct answer is: {expected}\n"
                f"Model response: {generation}\n"
                "Reply with EXACTLY one word: CORRECT (the facts are right), "
                "DEGRADING (partly wrong/rambling), or BROKEN (confidently false "
                "or incoherent).", 8)
            return _parse_verdict(raw)

        divergence = cosine_text_divergence(embed=None)  # Jaccard; no embed dep

        return gen_at, baseline_at, judge, divergence, llm_call


def _parse_verdict(raw: str) -> str:
    """Map a judge LLM reply to correct | degrading | broken, robustly.

    Guards against negation ("not broken", "correct, not broken") which a naive
    `"broken" in text` misclassifies as broken and truncates the band (Feature 20
    R2). Strategy: find the FIRST verdict WORD (whole-token), ignoring one leading
    negation, and prefer an explicit CORRECT/DEGRADING before falling to BROKEN.
    """
    import re

    text = (raw or "").strip().lower()
    tokens = re.split(r"[^a-z]+", text)
    # Drop a "not"/"n't"-style negator immediately before a verdict word so
    # "not broken" reads as CORRECT, not broken.
    verdict_words = {"correct", "degrading", "degrade", "degraded", "broken"}
    for i, tok in enumerate(tokens):
        if tok in verdict_words:
            negated = i > 0 and tokens[i - 1] in ("not", "isnt", "arent", "no")
            if tok == "correct":
                return "correct"
            if tok.startswith("degrad"):
                return "degrading"
            # tok == "broken"
            return "correct" if negated else "broken"
    # No recognizable verdict word: default to the CONSERVATIVE reading so an
    # unparseable judge reply does not silently pass a dial as usable.
    return "broken"


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
