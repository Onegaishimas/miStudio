"""
Circuit faithfulness service (Feature 017, IDL-34 — A.7 normative).

necessity   = (B_clean − B_ablate_M)          / (B_clean − B_ablate_all)
sufficiency = (B_ablate_topk_nonmembers − B_ablate_all) / (B_clean − B_ablate_all)

where M = the circuit's members (cluster_ref members expand to features),
B = a behavior metric (v1 = the compare-workflow output-shift, imported — the
metric identity is ALWAYS recorded in the manifest so the open question stays
visible). "Ablate all at layers" is intractable literally → per-layer top-N
proxy (N recorded). Suppression of a member LIST is one hook per layer
subtracting the SUM (the FTID pitfall: N separate hooks reorder float ops).

The math lives in circuit_validation_math (necessity/sufficiency); this
service supplies the four behavior measurements from real forward passes and
persists scores on the circuit record (018) + a faithfulness manifest.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.config import settings
from . import circuit_validation_math as vmath

logger = logging.getLogger(__name__)

DEFAULT_TOPK_NONMEMBERS = 256   # per layer, disclosed
DEFAULT_ABLATE_ALL_N = 1024     # per-layer top-N proxy for "ablate all", recorded


class FaithfulnessConfigError(ValueError):
    """Invalid faithfulness config — surfaces as a 422."""


def expand_members(circuit_members: List[Dict[str, Any]],
                   resolve_cluster) -> Dict[int, List[int]]:
    """circuit members → {layer: [feature_idx]}. cluster_ref members expand to
    their profile's features via `resolve_cluster(profile_id) -> [idx]`."""
    by_layer: Dict[int, List[int]] = {}
    for m in circuit_members:
        layer = m["layer"]
        if m.get("member_kind") == "cluster_ref" or m.get("cluster_profile_id"):
            for idx in resolve_cluster(m.get("cluster_profile_id")):
                by_layer.setdefault(layer, []).append(int(idx))
        else:
            feat = m.get("feature") or {}
            if feat.get("feature_idx") is not None:
                by_layer.setdefault(layer, []).append(int(feat["feature_idx"]))
    return by_layer


def scores_from_behaviors(b_clean: float, b_ablate_m: float,
                          b_ablate_all: float,
                          b_ablate_nonmembers: Optional[float],
                          *, mode: str) -> Dict[str, Any]:
    """Assemble necessity (+ sufficiency when mode='both') from the four
    behavior measurements. sufficiency is marked untested in necessity-only
    mode (never silently omitted)."""
    nec = vmath.necessity(b_clean, b_ablate_m, b_ablate_all)
    out = {"necessity": nec}
    if mode == "both" and b_ablate_nonmembers is not None:
        out["sufficiency"] = vmath.sufficiency(
            b_ablate_nonmembers, b_ablate_all, b_clean)
    else:
        out["sufficiency"] = None
        out["sufficiency_status"] = "untested (necessity-only mode)"
    return out


class CircuitFaithfulnessService:
    @staticmethod
    def create_config(config: Dict[str, Any]) -> Dict[str, Any]:
        mode = config.get("mode", "both")
        if mode not in ("necessity", "both"):
            raise FaithfulnessConfigError("mode must be necessity|both")
        return {
            "mode": mode,
            "k_nonmembers": int(config.get("k_nonmembers", DEFAULT_TOPK_NONMEMBERS)),
            "ablate_all_n": int(config.get("ablate_all_n", DEFAULT_ABLATE_ALL_N)),
            "metric_id": config.get("metric_id", "compare_output_shift/v1"),
            "n_prompts": int(config.get("n_prompts", 16)),
            "seed": int(config.get("seed", 0)),
        }

    @staticmethod
    def build_manifest_payload(circuit_id: str, config: Dict[str, Any],
                               scores: Dict[str, Any],
                               behaviors: Dict[str, float]) -> Dict[str, Any]:
        """Self-contained faithfulness manifest (metric identity ALWAYS
        recorded — the open question stays visible)."""
        return {
            "intervention": {"kind": "member_sum_suppression"},
            "config": config,
            "seeds": [config["seed"]],
            "circuit_id": circuit_id,
            "metric_id": config["metric_id"],
            "behaviors": behaviors,
            "scores": scores,
            "ablate_all_proxy": {"per_layer_top_n": config["ablate_all_n"],
                                 "note": "literal ablate-all intractable; "
                                         "per-layer top-N proxy (N recorded)"},
        }
