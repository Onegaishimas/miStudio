"""
Cross-layer steering hazards (Feature 015, BR-004/BR-024 — IDL-32/IDL-35).

When a multi-layer circuit steers an UPSTREAM feature that drives a
DOWNSTREAM steered feature, their influences COMPOUND (or, with opposite
signs, CANCEL). We surface this — WARNED, never silently corrected (BR-004).

Two evidence sources, in priority order:
  PRIMARY   — a stored circuit edge at rung >= 2 (017-validated): the warning
              is QUANTIFIED from the edge's measured effect size
              ("validated edge, ES=X — combined influence ≈ Y× the naive sum").
  FALLBACK  — the IDL-32 weight prior cos(W_dec(Lᵢ)[:,i], W_enc(Lⱼ)[j,:])
              above a threshold; EVERY such warning is labeled `heuristic`
              per the evidence-ladder language rules (IDL-35) — never causal.

Pure functions over weight tensors + edge dicts — no GPU orchestration, no
DB — so the whole hazard matrix is exhaustively unit-testable.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_PRIOR_THRESHOLD = 0.5  # config steering_hazard_prior_threshold


@dataclass
class Hazard:
    type: str            # "compounding" | "cancellation"
    up: Dict[str, int]   # {layer, feature_idx}
    down: Dict[str, int]
    evidence: str        # source label — "validated:ES=…" | "heuristic:weight_prior=…"
    rung: int            # the edge's rung (0 for a pure heuristic pair)
    quantified_effect: Optional[float] = None  # ES for validated edges

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "up": self.up, "down": self.down,
                "evidence": self.evidence, "rung": self.rung,
                "quantified_effect": self.quantified_effect}


def weight_prior(up_decoder, up_idx: int, down_encoder, down_idx: int) -> float:
    """cos(W_dec(Lᵢ)[:, i], W_enc(Lⱼ)[j, :]) — the IDL-32 prior. Orientation:
    decoder is [d_model, d_sae] (column i), encoder is [d_sae, d_model] (row j)
    — both project the same d_model space, so the cosine is well-defined.
    Reuses the resolve_*_weight conventions (add resolve_encoder_weight beside
    resolve_decoder_weight). Out-of-range indices ⇒ 0.0 (no hazard), never an
    IndexError — detect_hazards is a public pure function (R1 #5)."""
    import torch

    if not (0 <= up_idx < up_decoder.shape[1]) or \
            not (0 <= down_idx < down_encoder.shape[0]):
        return 0.0
    d_i = up_decoder[:, up_idx]        # [d_model]
    e_j = down_encoder[down_idx, :]    # [d_model]
    return float(torch.nn.functional.cosine_similarity(
        d_i.float(), e_j.float(), dim=0))


def _edge_key(e: Dict[str, Any]) -> Tuple:
    up, down = e.get("up", {}), e.get("down", {})
    return (up.get("layer"), up.get("feature_idx"),
            down.get("layer"), down.get("feature_idx"))


def detect_hazards(
    steered: List[Dict[str, int]],
    *,
    circuit_edges: Optional[List[Dict[str, Any]]] = None,
    decoders: Optional[Dict[int, Any]] = None,
    encoders: Optional[Dict[int, Any]] = None,
    prior_threshold: float = DEFAULT_PRIOR_THRESHOLD,
) -> List[Hazard]:
    """Surface compounding/cancellation across co-steered members.

    `steered` = [{layer, feature_idx, strength}] — the members being steered.
    For each upstream→downstream pair (up.layer < down.layer):
      • if a stored edge (up→down) at rung >= 2 exists, QUANTIFY from its ES;
      • else, if the weight prior >= threshold, warn labeled `heuristic`.
    Sign of the pair's steering strengths decides compounding vs cancellation.
    """
    edges_by_key = {}
    for e in (circuit_edges or []):
        k = _edge_key(e)
        # Skip cluster-ref edges (feature_idx None — R1 QA-2): the steered list
        # is feature-level, so a (layer, None, …) key can never match and its
        # rung/ES aren't per-feature. Feature-level hazards only, in v1.
        if None in k:
            continue
        edges_by_key[k] = e

    # Feature-level members only (a cluster-ref member has no feature_idx).
    feats = [m for m in steered if m.get("feature_idx") is not None]

    hazards: List[Hazard] = []
    seen_pairs = set()  # dedup (R1 #6): duplicate members must not double-warn
    for up in feats:
        for down in feats:
            if up["layer"] >= down["layer"]:
                continue
            pair = (up["layer"], up["feature_idx"], down["layer"], down["feature_idx"])
            if pair in seen_pairs:
                continue
            up_ref = {"layer": up["layer"], "feature_idx": up["feature_idx"]}
            down_ref = {"layer": down["layer"], "feature_idx": down["feature_idx"]}
            # co-steered sign: same sign ⇒ compounding, opposite ⇒ cancellation
            same_sign = (up.get("strength", 0) >= 0) == (down.get("strength", 0) >= 0)
            key = pair

            edge = edges_by_key.get(key)
            if edge is not None and int(edge.get("rung", 0)) >= 2 \
                    and edge.get("effect_size") is not None:
                es = float(edge["effect_size"])
                # a validated NEGATIVE edge flips the compounding/cancellation
                edge_positive = es >= 0
                compounding = same_sign == edge_positive
                hazards.append(Hazard(
                    type="compounding" if compounding else "cancellation",
                    up=up_ref, down=down_ref,
                    evidence=f"validated:ES={es:.3f}",
                    rung=int(edge.get("rung", 2)),
                    quantified_effect=es))
                seen_pairs.add(pair)
                continue

            # heuristic fallback — weight prior (labeled, never causal)
            if decoders is not None and encoders is not None:
                dec = decoders.get(up["layer"])
                enc = encoders.get(down["layer"])
                if dec is not None and enc is not None:
                    prior = weight_prior(dec, up["feature_idx"], enc,
                                         down["feature_idx"])
                    if abs(prior) >= prior_threshold:
                        # prior sign + steering sign → compounding/cancellation
                        prior_positive = prior >= 0
                        compounding = same_sign == prior_positive
                        hazards.append(Hazard(
                            type="compounding" if compounding else "cancellation",
                            up=up_ref, down=down_ref,
                            evidence=f"heuristic:weight_prior={prior:.3f}",
                            rung=0, quantified_effect=None))
                        seen_pairs.add(pair)
    return hazards
