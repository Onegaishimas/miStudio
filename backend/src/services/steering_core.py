"""Unified GPU steering-generation core (Steered Transcript Recorder increment).

The circuit, cluster, and feature steering paths are the SAME additive
residual-stream math — `residual += dial · strength · W_dec[:, idx]` — differing
only in HOW members and decoder weights are resolved. This module is the one
shared core: three resolvers turn each artifact type into a common
`resolved_members` list, and `build_steer_generator` turns that plus a loaded
model into `(gen_at(dial, prompt), baseline_at(prompt, seed))` closures.

Two conventions, fixed here and shared by every caller (Feature 20 review +
recorder design):
  * HOOK TARGET = the discovered residual module
    (`get_hookable_module(layer, "residual", structure)`), NOT the whole layer
    output. This is what the calibrated band and miLLM serving measure against,
    so a recorded transcript matches the band.
  * The RECORDER owns the dial multiply: `effective = dial · base_strength`.
    (Feature/cluster strengths are otherwise terminal; only the circuit path
    threads a dial.)

The generation body is MOVED verbatim from
`CircuitCalibrationService._build_generation_fns` (Feature 20, 32-findings-
hardened): greedy/deterministic, Gemma-cache-aware, silent-drop-refusing. That
method now calls this core, so calibration behaviour is unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

#: One resolved circuit/cluster/feature member ready for the hook.
ResolvedMember = Tuple[int, int, float, Any]  # (layer, feature_idx, base_strength, W_dec)

#: Default greedy generation length (matches calibration's GEN_MAX_TOKENS).
DEFAULT_MAX_TOKENS = 80


class SteeringCoreError(RuntimeError):
    """A steering artifact could not be resolved or generated."""


# ── member resolvers (per artifact type) ─────────────────────────────────────

def _load_wdec_by_layer(sae_ids_by_layer: Dict[int, str], db, device) -> Dict[int, Any]:
    """Load each layer's SAE and resolve its decoder weight [d_model, d_sae]."""
    from ..models.external_sae import ExternalSAE
    from .circuit_capture_service import _load_sae_sync
    from .steering_service import resolve_decoder_weight

    wdec: Dict[int, Any] = {}
    for L, sid in sae_ids_by_layer.items():
        if not sid:
            raise SteeringCoreError(f"Layer {L} has no SAE id — cannot steer it")
        sae_rec = db.query(ExternalSAE).filter(ExternalSAE.id == sid).first()
        if sae_rec is None:
            raise SteeringCoreError(f"SAE {sid} for layer {L} not found")
        wdec[L] = resolve_decoder_weight(_load_sae_sync(sae_rec, device))
    return wdec


def resolve_circuit_members(circuit, db, device) -> Tuple[str, List[ResolvedMember]]:
    """Resolve a circuit's members to (layer, idx, strength, W_dec).

    Members nest their strength at `member.feature.strength`; SAE refs come from
    `circuit.saes`. Refuses a member whose layer has no SAE (silent-drop class,
    Feature 20 R3) — a partial resolution would record a weaker circuit than is
    served.
    """
    sae_ids_by_layer = {int(s["layer"]): s.get("mistudio_sae_id")
                        for s in (circuit.saes or []) if s.get("layer") is not None}
    wdec = _load_wdec_by_layer(sae_ids_by_layer, db, device)
    members: List[ResolvedMember] = []
    for m in (circuit.members or []):
        feat = m.get("feature") or {}
        L, idx = m.get("layer"), feat.get("feature_idx")
        strength = float(feat.get("strength") or 0.0)
        if idx is None:
            continue
        if L not in wdec:
            raise SteeringCoreError(
                f"Member at layer {L} has no SAE decoder weight — the circuit "
                f"cannot be steered as it would be served (add an SAE ref for "
                f"layer {L}).")
        members.append((int(L), int(idx), strength, wdec[L]))
    if not members:
        raise SteeringCoreError("Circuit resolved to no steerable members")
    return circuit.model_id, members


def resolve_feature_members(feature_specs: List[Dict[str, Any]], model_id: str,
                            db, device) -> Tuple[str, List[ResolvedMember]]:
    """Resolve an ad-hoc feature set: [{layer, feature_idx, strength, sae_id}].

    One SAE id per layer is required (all specs at a layer must name the same
    SAE — mixing SAEs at one layer is rejected as ambiguous).
    """
    sae_ids_by_layer: Dict[int, str] = {}
    for spec in feature_specs:
        L = int(spec["layer"])
        sid = spec.get("sae_id")
        if L in sae_ids_by_layer and sae_ids_by_layer[L] != sid:
            raise SteeringCoreError(
                f"Layer {L} names two different SAEs ({sae_ids_by_layer[L]} vs "
                f"{sid}); one SAE per layer.")
        sae_ids_by_layer[L] = sid
    wdec = _load_wdec_by_layer(sae_ids_by_layer, db, device)
    members: List[ResolvedMember] = []
    for spec in feature_specs:
        L, idx = int(spec["layer"]), spec.get("feature_idx")
        if idx is None:
            continue
        members.append((L, int(idx), float(spec.get("strength") or 0.0), wdec[L]))
    if not members:
        raise SteeringCoreError("Feature set resolved to no steerable members")
    return model_id, members


def resolve_cluster_members(cluster_profile_id: str, db, device
                            ) -> Tuple[str, List[ResolvedMember]]:
    """Resolve a cluster profile's PERSISTED tuned members to
    (layer, idx, sign·strength, W_dec). Does NOT re-run allocation — the profile
    already stores the strengths that were validated (CLUSTERS arc IDL-30)."""
    from ..models.cluster_profile import ClusterProfile

    prof = db.query(ClusterProfile).filter(
        ClusterProfile.id == cluster_profile_id).first()
    if prof is None:
        raise SteeringCoreError(f"Cluster profile {cluster_profile_id} not found")
    members_raw = prof.members or []
    # A cluster member carries feature_idx, strength, sign, and a layer — resolve
    # the SAE per layer from the profile's sae refs.
    sae_ids_by_layer = _cluster_sae_ids_by_layer(prof)
    wdec = _load_wdec_by_layer(sae_ids_by_layer, db, device)
    members: List[ResolvedMember] = []
    for m in members_raw:
        L, idx = m.get("layer"), m.get("feature_idx")
        if L is None or idx is None:
            continue
        if L not in wdec:
            raise SteeringCoreError(
                f"Cluster member at layer {L} has no SAE decoder weight")
        sign = int(m.get("sign", 1) or 1)
        strength = sign * float(m.get("strength") or 0.0)
        members.append((int(L), int(idx), strength, wdec[L]))
    if not members:
        raise SteeringCoreError("Cluster profile resolved to no steerable members")
    return _cluster_model_id(prof), members


def _cluster_sae_ids_by_layer(prof) -> Dict[int, str]:
    """Per-layer SAE ids for a cluster profile. Profiles may carry a single
    `mistudio_sae_id` (single-layer) or per-layer refs; support both."""
    # Per-layer saes list, if present (multi-SAE clusters).
    saes = getattr(prof, "saes", None) or []
    by_layer = {int(s["layer"]): s.get("mistudio_sae_id")
                for s in saes if isinstance(s, dict) and s.get("layer") is not None}
    if by_layer:
        return by_layer
    # Fall back to a single sae id applied to whatever layers the members use.
    single = getattr(prof, "mistudio_sae_id", None) or getattr(prof, "sae_id", None)
    layers = {int(m["layer"]) for m in (prof.members or [])
              if m.get("layer") is not None}
    return {L: single for L in layers}


def _cluster_model_id(prof):
    return getattr(prof, "model_id", None) or getattr(prof, "mistudio_model_id", None)


# ── the generation core (moved verbatim from calibration) ────────────────────

def build_steer_generator(model, tokenizer, structure, resolved_members,
                          *, disable_cache: bool, max_tokens: int):
    """Return (gen_at, baseline_at) closures over a loaded model.

    `gen_at(dial, prompt) -> str` applies `dial · base_strength · W_dec[:, idx]`
    additively at each member's residual module and generates greedily.
    `baseline_at(prompt, seed) -> str` generates unsteered.

    Body moved verbatim from CircuitCalibrationService._build_generation_fns
    (Feature 20). Hook target is the residual module; greedy + Gemma-cache-aware.
    """
    import torch

    from ..ml.layer_discovery import get_hookable_module

    # Group members by layer and resolve each layer's residual hook target once.
    by_layer: Dict[int, List[Tuple[int, float, Any]]] = {}
    for (L, idx, strength, wdec) in resolved_members:
        by_layer.setdefault(L, []).append((idx, strength, wdec))

    hook_layers: Dict[int, Any] = {}
    for L in by_layer:
        layer_mod = structure.layers_module[L]
        target = get_hookable_module(layer_mod, "residual", structure)
        if target is None:
            raise SteeringCoreError(
                f"No hookable residual module for layer {L} on this model")
        hook_layers[L] = target

    def _make_hook(dial):
        def _hook_for(L):
            def hook(module, inp, output):
                is_tuple = isinstance(output, tuple)
                hidden = output[0] if is_tuple else output
                if hidden.dim() != 3:
                    return output
                with torch.no_grad():
                    for (idx, strength, wdec) in by_layer[L]:
                        # Match hidden's dtype AND device — the shared core can
                        # be handed W_dec resolved on a different device than the
                        # active hidden state (defensive; calibration always
                        # matches, but the recorder's resolvers may not).
                        vec = wdec[:, idx].to(dtype=hidden.dtype,
                                              device=hidden.device)
                        hidden.add_(dial * strength * vec)
                return output
            return hook
        return {L: _hook_for(L) for L in by_layer}

    def _generate(prompt: str, dial: float, gseed: int) -> str:
        torch.manual_seed(gseed)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        handles = []
        if dial > 0:
            for L, hook in _make_hook(dial).items():
                handles.append(hook_layers[L].register_forward_hook(hook))
        gen_kwargs = dict(max_new_tokens=max_tokens, do_sample=False)
        if disable_cache:
            gen_kwargs["use_cache"] = False
        try:
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
        finally:
            for h in handles:
                h.remove()
        return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)

    def gen_at(dial, prompt):
        return _generate(prompt, dial, 0)

    def baseline_at(prompt, s):
        return _generate(prompt, 0.0, s)

    return gen_at, baseline_at


def load_model_and_structure(model_id: str, db):
    """Load a model by miStudio Model id and discover its structure. Returns
    (model, tokenizer, structure, disable_cache, device). Shared by calibration
    and the recorder."""
    import torch

    from ..core.config import settings
    from ..ml.layer_discovery import discover_transformer_structure
    from ..ml.model_loader import load_model_from_hf
    from ..models.model import Model, QuantizationFormat
    from .steering_service import _CACHE_INCOMPATIBLE_MARKERS

    model_rec = db.query(Model).filter(Model.id == model_id).first()
    if model_rec is None:
        raise SteeringCoreError(f"Model {model_id} not found — cannot generate")
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
    _marker_hay = f"{model_rec.repo_id} {type(model).__name__}".lower()
    disable_cache = any(mk in _marker_hay for mk in _CACHE_INCOMPATIBLE_MARKERS)
    return model, tokenizer, structure, disable_cache, device
