"""Steered-transcript recorder (Steered Transcript Recorder increment).

Records `(dial, prompt, unsteered_output, steered_output)` transcripts for ANY
steerable artifact — a circuit, a cluster profile, or an ad-hoc feature set — on
the GPU, and persists them as a `steering_samples` manifest. The transcripts are
the raw material for a SEPARATE, post-run LLM meaning-analysis pass (hand them to
Opus 4.8): "what did steering this artifact semantically DO to the outputs across
the dial?". This is NOT a correctness judge — the path is judge-free.

Built on the unified `steering_core` (the same generation core calibration uses),
so a recorded transcript matches the calibrated band (same hook target, greedy
generation, dial-owned-here multiply).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .steering_core import (SteeringCoreError, build_steer_generator,
                            load_model_and_structure, resolve_circuit_members,
                            resolve_cluster_members, resolve_feature_members)

logger = logging.getLogger(__name__)

# Caps that bound the dials × prompts × tokens product on the single GPU.
MAX_RECORD_DIALS = 8
MAX_RECORD_PROMPTS = 8
MAX_RECORD_TOKENS = 200
MAX_RECORD_GENERATIONS = 64   # hard ceiling on prompts × (1 + dials)
DEFAULT_RECORD_TOKENS = 80
MAX_DIAL = 2.0                # servable ceiling (matches calibration MAX_DIAL)


class RecordConfigError(ValueError):
    """Bad record request — surfaces as a 422."""


class RecordRunError(RuntimeError):
    """Recording failed at run time."""


class SteeringRecorderService:
    """Generate + record steered transcripts for a circuit / cluster / features."""

    @staticmethod
    def create_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = dict(config or {})
        artifact = cfg.get("artifact")
        if not isinstance(artifact, dict) or artifact.get("kind") not in (
                "circuit", "cluster", "features"):
            raise RecordConfigError(
                "artifact must be {kind: 'circuit'|'cluster'|'features', ...}")

        dials = cfg.get("dials") or []
        prompts = cfg.get("prompts") or []
        if not isinstance(dials, list) or not dials:
            raise RecordConfigError("dials must be a non-empty list")
        if not isinstance(prompts, list) or not prompts:
            raise RecordConfigError("prompts must be a non-empty list")
        # Dedupe dials preserving order; validate range.
        seen, uniq = set(), []
        for d in dials:
            fd = float(d)
            if not (0.0 <= fd <= MAX_DIAL):
                raise RecordConfigError(
                    f"dial {fd} outside [0, {MAX_DIAL}] (the servable ceiling)")
            if fd not in seen:
                seen.add(fd)
                uniq.append(fd)
        dials = uniq
        prompts = [str(p) for p in prompts]
        if any(not p.strip() for p in prompts):
            raise RecordConfigError("prompts must be non-empty strings")
        if len(dials) > MAX_RECORD_DIALS:
            raise RecordConfigError(f"at most {MAX_RECORD_DIALS} dials")
        if len(prompts) > MAX_RECORD_PROMPTS:
            raise RecordConfigError(f"at most {MAX_RECORD_PROMPTS} prompts")

        max_tokens = int(cfg.get("max_tokens") or DEFAULT_RECORD_TOKENS)
        if not (1 <= max_tokens <= MAX_RECORD_TOKENS):
            raise RecordConfigError(f"max_tokens must be in [1, {MAX_RECORD_TOKENS}]")

        # Hard product cap: baseline (1) + one per dial, per prompt.
        generations = len(prompts) * (1 + len(dials))
        if generations > MAX_RECORD_GENERATIONS:
            raise RecordConfigError(
                f"{generations} generations requested (prompts × (1+dials)) "
                f"exceeds the {MAX_RECORD_GENERATIONS} ceiling; reduce dials or "
                "prompts")

        return {"artifact": artifact, "dials": dials, "prompts": prompts,
                "max_tokens": max_tokens, "seed": int(cfg.get("seed", 0))}

    @classmethod
    def _resolve(cls, artifact: Dict[str, Any], db, device):
        """Dispatch to the per-type resolver → (model_id, resolved_members)."""
        kind = artifact["kind"]
        if kind == "circuit":
            from ..models.circuit import Circuit
            circ = db.query(Circuit).filter(
                Circuit.id == artifact["circuit_id"]).first()
            if circ is None:
                raise RecordRunError(f"Circuit {artifact['circuit_id']} not found")
            return resolve_circuit_members(circ, db, device)
        if kind == "cluster":
            return resolve_cluster_members(artifact["cluster_profile_id"], db, device)
        if kind == "features":
            return resolve_feature_members(
                artifact["features"], artifact["model_id"], db, device)
        raise RecordRunError(f"unknown artifact kind {kind!r}")

    @classmethod
    def record_samples(cls, db, config: Dict[str, Any], *,
                       progress_cb: Optional[Callable[[int], None]] = None
                       ) -> Dict[str, Any]:
        """GPU orchestrator (sync, called from the Celery task). Loads the model
        once, resolves the artifact's members, generates the baseline + each dial
        per prompt, and persists a `steering_samples` manifest with the text."""
        cfg = cls.create_config(config)
        artifact = cfg["artifact"]

        if progress_cb:
            progress_cb(5)
        try:
            # Model id from the artifact → load the model → resolve members on
            # the model's real device (W_dec + hidden must share a device).
            model_id = cls._artifact_model_id(artifact, db)
            model, tokenizer, structure, disable_cache, device = \
                load_model_and_structure(model_id, db)
            _mid, resolved = cls._resolve(artifact, db, device)
            gen_at, baseline_at = build_steer_generator(
                model, tokenizer, structure, resolved,
                disable_cache=disable_cache, max_tokens=cfg["max_tokens"])
        except SteeringCoreError as e:
            raise RecordRunError(str(e)) from e

        transcripts: List[Dict[str, Any]] = []
        total = len(cfg["prompts"]) * (1 + len(cfg["dials"]))
        done = 0
        for pi, prompt in enumerate(cfg["prompts"]):
            unsteered = baseline_at(prompt, cfg["seed"])
            done += 1
            if progress_cb:
                progress_cb(5 + int(90 * done / total))
            samples = []
            for dial in cfg["dials"]:
                steered = (unsteered if dial == 0.0
                           else gen_at(dial, prompt))
                samples.append({"dial": dial, "steered_output": steered})
                done += 1
                if progress_cb:
                    progress_cb(5 + int(90 * done / total))
            transcripts.append({"prompt_index": pi, "prompt": prompt,
                                "unsteered_output": unsteered, "samples": samples})

        payload = {
            "artifact": {"kind": artifact["kind"], **cls._artifact_ref(artifact)},
            "dials": cfg["dials"], "prompts": cfg["prompts"],
            "config": {"max_tokens": cfg["max_tokens"], "seed": cfg["seed"],
                       "model_hf_id": cls._model_hf_id(model_id, db)},
            "transcripts": transcripts,
        }
        manifest_id = cls._persist(db, artifact, payload)
        if progress_cb:
            progress_cb(100)
        return {"manifest_ref": manifest_id, "artifact": payload["artifact"],
                "counts": {"prompts": len(cfg["prompts"]),
                           "dials": len(cfg["dials"]), "generations": total}}

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _artifact_model_id(artifact, db):
        kind = artifact["kind"]
        if kind == "features":
            return artifact["model_id"]
        if kind == "circuit":
            from ..models.circuit import Circuit
            c = db.query(Circuit).filter(
                Circuit.id == artifact["circuit_id"]).first()
            if c is None:
                raise RecordRunError(f"Circuit {artifact['circuit_id']} not found")
            return c.model_id
        if kind == "cluster":
            from ..models.cluster_profile import ClusterProfile
            p = db.query(ClusterProfile).filter(
                ClusterProfile.id == artifact["cluster_profile_id"]).first()
            if p is None:
                raise RecordRunError(
                    f"Cluster profile {artifact['cluster_profile_id']} not found")
            return getattr(p, "model_id", None) or getattr(p, "mistudio_model_id", None)
        raise RecordRunError(f"unknown artifact kind {kind!r}")

    @staticmethod
    def _artifact_ref(artifact) -> Dict[str, Any]:
        kind = artifact["kind"]
        if kind == "circuit":
            return {"circuit_id": artifact["circuit_id"]}
        if kind == "cluster":
            return {"cluster_profile_id": artifact["cluster_profile_id"]}
        # features: store the specs (no file paths) so the record is self-contained.
        return {"features": artifact["features"], "model_id": artifact["model_id"]}

    @staticmethod
    def _model_hf_id(model_id, db):
        from ..models.model import Model
        m = db.query(Model).filter(Model.id == model_id).first()
        return getattr(m, "repo_id", None) if m else None

    @staticmethod
    def _persist(db, artifact, payload) -> str:
        from ..models.validation_manifest import ValidationManifest
        from .manifest_service import validate_payload

        validate_payload("steering_samples", payload)
        # A circuit artifact links the manifest to its circuit for retrieval;
        # cluster/feature manifests carry the artifact in the payload only.
        circuit_id = (artifact["circuit_id"] if artifact["kind"] == "circuit"
                      else None)
        man = ValidationManifest(kind="steering_samples", payload=payload,
                                 circuit_id=circuit_id)
        db.add(man)
        db.commit()
        db.refresh(man)
        return man.id
