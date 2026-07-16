"""
Cluster strength allocation (Feature 013, IDL-29).

Computes a principled starting strength allocation for steering a CLUSTER of
SAE features, replacing guessed uniform strengths. Design (pressure-tested; see
0xcc/tdds/013_FTDD|Cluster_Strength_Model.md for the derivation):

    weights      w_i   = s~_i / Σ s~_j          (similarity-normalized; equal sims ⇒ equal shares)
    frequency    f_eff = Σ w_i f_i / Σ w_i      (over members with known f)
    dir. budget  B_dir = clamp(a − b·f_eff, m, M)   (the empirically-fit solo law at cluster level)
    gain         G     = ‖Σ σ_i w_i d_i‖₂       (exact resultant norm of the injected direction)
    total budget B     = min( B_dir / max(G, G_FLOOR), Σ b*(f_i) )
    strengths    s_i   = σ_i · B · w_i          (rounded to 0.1; residual → largest member)

The hook injects v = Σ strength_i · d_i, so ‖v‖ = B·G; setting B = B_dir/G makes
the *injected vector* match the validated solo magnitude. Identical members ⇒
G=1 ⇒ B=B_dir (the constant-budget heuristic MCP experiments discovered);
orthogonal members ⇒ B = B_dir·√N. Small G means members nearly cancel — that
degrades direction QUALITY, so guards are coherence flags, not magnitude caps.

The math core is pure (no I/O) for unit testing; decoder access goes through
steering_service.resolve_decoder_weight so the gain sees exactly the directions
the hook will inject.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

FORMULA_ID = "freq-budget/sim-alloc@1"
G_FLOOR = 0.05
ROUND_GRAIN = 0.1

# Defaults from the IDL-27 solo-law fit (experiment c4a273f1). Overridable per
# SAE via settings.steering_cluster_constants_json so MCP calibration can write
# results without a code change.
DEFAULT_CONSTANTS: Dict[str, float] = {
    "a": 2.9,       # intercept
    "b": 2.6,       # slope on effective frequency
    "m": 1.0,       # budget floor
    "M": 3.0,       # budget ceiling
    "cohesion_gate": 0.5,  # below this group cohesion → recommend solo baselines
}

# Cluster-level default when NO member has a known frequency: the clamp
# midpoint. Deliberately NOT the solo path's DEFAULT_STRENGTH=10 — that is a
# per-feature legacy fallback; a whole cluster of unknowns starts conservative.
DEFAULT_B_DIR = 2.0
# Per-member cap contribution when the member's frequency is unknown.
DEFAULT_MEMBER_CAP = 2.0


@dataclass
class AllocationMember:
    feature_idx: int
    layer: int
    similarity: Optional[float] = None
    activation_frequency: Optional[float] = None
    sign: int = 1  # +1 boost, -1 suppress


@dataclass
class AllocationResult:
    B: float
    B_dir: float
    G: float
    f_eff: Optional[float]
    weights: List[float]
    strengths: List[float]
    flags: List[str] = field(default_factory=list)
    cancellation_pair: Optional[Tuple[int, int]] = None  # feature idxs of worst pair
    constants_used: Dict[str, float] = field(default_factory=dict)
    formula_id: str = FORMULA_ID
    approximate: bool = False  # True when decoder unavailable (G := 1)


def resolve_constants(settings_json: Optional[str], sae_id: Optional[str]) -> Dict[str, float]:
    """Merge DEFAULT_CONSTANTS ← config default ← per-SAE override."""
    constants = dict(DEFAULT_CONSTANTS)
    if settings_json:
        try:
            cfg = json.loads(settings_json)
            constants.update(cfg.get("default", {}) or {})
            if sae_id:
                constants.update((cfg.get("per_sae", {}) or {}).get(sae_id, {}) or {})
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"[ClusterAllocation] Bad steering_cluster_constants_json ({e}); using defaults")
    return constants


def _solo_cap(f: Optional[float], c: Dict[str, float]) -> float:
    """b*(f): the member's solo-optimal contribution to the budget cap."""
    if f is None or not (0.0 <= f <= 1.0) or math.isnan(f):
        return DEFAULT_MEMBER_CAP
    return max(c["m"], min(c["M"], c["a"] - c["b"] * f))


def compute_allocation(
    members: List[AllocationMember],
    decoder=None,  # Optional[torch.Tensor] [d_model, d_sae]; None ⇒ approximate G=1
    constants: Optional[Dict[str, float]] = None,
    group_cohesion: Optional[float] = None,
) -> AllocationResult:
    """
    Pure-math allocation (FTDD §2 steps 1–7 + §3 edge table). No I/O.

    Raises ValueError on an empty member list or mixed layers (the caller maps
    mixed-layer to a refusal + solo-baseline fallback; N=1 is the caller's
    solo-path short-circuit but is still computed correctly here).
    """
    c = dict(constants or DEFAULT_CONSTANTS)
    if not members:
        raise ValueError("members must be non-empty")
    layers = {m.layer for m in members}
    if len(layers) > 1:
        raise ValueError(f"mixed-layer member set {sorted(layers)}: cluster budget model requires a single layer")

    n = len(members)
    flags: List[str] = []

    # -- Step 1: similarity-normalized weights -------------------------------
    valid_sims = [m.similarity for m in members
                  if m.similarity is not None and not math.isnan(m.similarity) and m.similarity > 0]
    mean_sim = (sum(valid_sims) / len(valid_sims)) if valid_sims else 1.0
    s_tilde: List[float] = []
    for m in members:
        s = m.similarity
        if s is None or math.isnan(s):
            s = mean_sim  # neutral imputation
        s_tilde.append(max(0.0, s))
    total_s = sum(s_tilde)
    if total_s <= 0:
        weights = [1.0 / n] * n  # all missing/zero → uniform
        flags.append("uniform_weights")
    else:
        weights = [s / total_s for s in s_tilde]
        if any(w == 0 for w in weights):
            flags.append("inactive_member")  # zero-sim member gets zero strength

    # -- Step 2: effective frequency ------------------------------------------
    known = [(w, m.activation_frequency) for w, m in zip(weights, members)
             if m.activation_frequency is not None
             and not math.isnan(m.activation_frequency)
             and 0.0 <= m.activation_frequency <= 1.0]
    if known:
        wsum = sum(w for w, _ in known)
        f_eff: Optional[float] = (sum(w * f for w, f in known) / wsum) if wsum > 0 else None
    else:
        f_eff = None

    # -- Step 3: direction budget ---------------------------------------------
    if f_eff is None:
        b_dir = DEFAULT_B_DIR
        flags.append("default_budget")  # no frequencies known
    else:
        b_dir = max(c["m"], min(c["M"], c["a"] - c["b"] * f_eff))

    # -- Step 4: gain (exact resultant norm, signed weights) -------------------
    approximate = False
    cancellation_pair: Optional[Tuple[int, int]] = None
    if decoder is not None:
        import torch  # local import: math core stays importable without torch

        idxs = [m.feature_idx for m in members]
        d_sae = decoder.shape[1]
        bad = [i for i in idxs if i < 0 or i >= d_sae]
        if bad:
            raise ValueError(f"feature indices out of bounds for SAE with {d_sae} features: {bad}")

        with torch.no_grad():
            cols = decoder[:, idxs].to(torch.float32)          # [d_model, n]
            norms = torch.linalg.vector_norm(cols, dim=0)      # defensive: normalize
            norms = torch.clamp(norms, min=1e-8)
            unit = cols / norms
            sw = torch.tensor([m.sign * w for m, w in zip(members, weights)], dtype=torch.float32)
            resultant = unit @ sw                              # [d_model]
            g = float(torch.linalg.vector_norm(resultant))

            # Cancellation check only when it can trip (positive-sign cluster,
            # worse-than-orthogonal gain) — N² cosines on demand only.
            if n > 1 and all(m.sign == 1 for m in members) and g < 1.0 / math.sqrt(n):
                flags.append("cancellation")
                cos = (unit.t() @ unit)                        # [n, n]
                cos.fill_diagonal_(1.0)
                flat = torch.argmin(cos)
                i, j = int(flat) // n, int(flat) % n
                cancellation_pair = (members[i].feature_idx, members[j].feature_idx)
    else:
        g = 1.0  # constant-budget conservative fallback (errs weak)
        approximate = True
        flags.append("approximate")

    # -- Step 5: total budget ---------------------------------------------------
    cap = sum(_solo_cap(m.activation_frequency, c) for m in members)
    b = min(b_dir / max(g, G_FLOOR), cap)
    if b == cap and n > 1:
        flags.append("cap_bound")

    # -- Cohesion gate ------------------------------------------------------------
    if group_cohesion is not None and group_cohesion < c.get("cohesion_gate", 0.5):
        flags.append("low_cohesion")

    # -- Step 6: allocation with rounding-residual fold ---------------------------
    raw = [m.sign * b * w for m, w in zip(members, weights)]
    strengths = [round(x / ROUND_GRAIN) * ROUND_GRAIN for x in raw]
    # Fold the rounding residual into the largest-|weight| member so Σ|s| tracks B.
    residual = b - sum(abs(s) for s in strengths)
    if abs(residual) >= ROUND_GRAIN / 2 and n > 0:
        k = max(range(n), key=lambda i: weights[i])
        strengths[k] = round((strengths[k] + members[k].sign * residual) / ROUND_GRAIN) * ROUND_GRAIN
    strengths = [round(s, 1) for s in strengths]

    return AllocationResult(
        B=round(b, 4),
        B_dir=round(b_dir, 4),
        G=round(g, 4),
        f_eff=round(f_eff, 4) if f_eff is not None else None,
        weights=[round(w, 4) for w in weights],
        strengths=strengths,
        flags=flags,
        cancellation_pair=cancellation_pair,
        constants_used={k: c[k] for k in ("a", "b", "m", "M", "cohesion_gate") if k in c},
        approximate=approximate,
    )


def rebalance(
    strengths: List[float],
    weights: List[float],
    signs: List[int],
    pinned: List[bool],
    total_budget: float,
) -> Tuple[List[float], List[str]]:
    """
    Budget-preserving rebalance (FTDD step 8) — reference implementation kept
    server-side for parity tests; the frontend mirrors this arithmetic.

    Pinned members keep their values; the remaining budget R = B − Σ|pinned|
    redistributes across unpinned members by renormalized weights. R < 0 ⇒
    warn + unpinned to 0. Never silently rescales pins.
    """
    flags: List[str] = []
    n = len(strengths)
    pinned_total = sum(abs(s) for s, p in zip(strengths, pinned) if p)
    r = total_budget - pinned_total
    out = list(strengths)
    unpinned = [i for i in range(n) if not pinned[i]]
    if not unpinned:
        return out, flags
    if r < 0:
        flags.append("over_budget")
        for i in unpinned:
            out[i] = 0.0
        return out, flags
    wsum = sum(weights[i] for i in unpinned)
    for i in unpinned:
        share = (weights[i] / wsum) if wsum > 0 else 1.0 / len(unpinned)
        out[i] = round(signs[i] * r * share, 1)
    return out, flags
