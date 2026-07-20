"""
Circuit discovery service (Feature 016, BR-007/BR-015/BR-016 — FTDD §2).

Orchestrates the A.3 pipeline over a completed capture store:

  unit selection (feature | cluster supernodes; seeded | open)
  → pair generation (up.layer < down.layer; support floors)
  → PMI + circular-shift null + BH-FDR (circuit_stats_service — the math
    lives there, pinned; this service never re-implements it)
  → held-out replication on the CAPTURE-TIME split
  → report + candidates persisted (cap DEFAULT_CANDIDATE_CAP, truncation
    NOTED in the report — no silent caps, ever)

Discovery emits raw candidates; 018's edge-type classifier annotates and
017's validation ladder promotes. All co-activation is lag-0 same-position —
disclosed in the report and every downstream surface (IDL-35).
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.config import settings
from ..models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun
from ..models.cluster_profile import ClusterProfile
from . import circuit_stats_service as stats
from .circuit_capture_store import EventReader, layer_files_exist

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_CAP = 2000
DEFAULT_MAX_UNITS_PER_LAYER = 500     # top-by-support; recorded in report
DEFAULT_MAX_NULL_TESTED = 2000        # null-test only the support+PMI shortlist
DEFAULT_COHESION_FLOOR = 0.3


class DiscoveryConfigError(ValueError):
    """Invalid discovery configuration — surfaces as a 422."""


class DiscoveryConflictError(RuntimeError):
    """A discovery is already active on this store — surfaces as a 409."""


class CircuitDiscoveryService:
    @staticmethod
    def create_run(db, config: Dict[str, Any]) -> CircuitDiscoveryRun:
        capture = db.query(CircuitCaptureRun).filter(
            CircuitCaptureRun.id == config.get("capture_run_id")).first()
        if capture is None:
            raise DiscoveryConfigError(
                f"Capture run {config.get('capture_run_id')} not found")
        if capture.status != "completed":
            raise DiscoveryConfigError(
                f"Capture run {capture.id} is {capture.status}, not completed")
        if capture.stale and not config.get("force"):
            raise DiscoveryConfigError(
                f"Capture store {capture.id} is STALE (a referenced SAE "
                f"changed since capture) — pass force=true to mine anyway")
        granularity = config.get("granularity", "feature")
        if granularity not in ("feature", "cluster"):
            raise DiscoveryConfigError(f"Unknown granularity {granularity!r}")
        mode = config.get("mode", "open")
        if mode not in ("seeded", "open"):
            raise DiscoveryConfigError(f"Unknown mode {mode!r}")
        if mode == "seeded" and not config.get("seed_refs"):
            raise DiscoveryConfigError("seeded mode requires seed_refs")
        # 409-on-concurrent: don't race two mines on the same store's readers
        # (R1 QA-P1). Same transaction as the insert below.
        active = db.query(CircuitDiscoveryRun).filter(
            CircuitDiscoveryRun.capture_run_id == capture.id,
            CircuitDiscoveryRun.status.in_(("pending", "running"))).first()
        if active is not None:
            raise DiscoveryConflictError(
                f"Discovery {active.id} is already {active.status} on this "
                f"capture store — wait or cancel it")
        params = {
            "capture_run_id": capture.id,
            "granularity": granularity,
            "mode": mode,
            "seed_refs": config.get("seed_refs") or [],
            "s_min": int(config.get("s_min", stats.DEFAULT_S_MIN)),
            "null_shuffles": int(config.get("null_shuffles",
                                            stats.DEFAULT_NULL_SHUFFLES)),
            "null_percentile": float(config.get("null_percentile",
                                                stats.DEFAULT_NULL_PERCENTILE)),
            "fdr_q": float(config.get("fdr_q", stats.DEFAULT_FDR_Q)),
            "cohesion_floor": float(config.get("cohesion_floor",
                                               DEFAULT_COHESION_FLOOR)),
            "max_units_per_layer": int(config.get("max_units_per_layer",
                                                  DEFAULT_MAX_UNITS_PER_LAYER)),
            "max_null_tested": int(config.get("max_null_tested",
                                              DEFAULT_MAX_NULL_TESTED)),
            "seed": int(config.get("seed", 0)),
        }
        run = CircuitDiscoveryRun(capture_run_id=capture.id, status="pending",
                                  params=params)
        db.add(run)
        db.commit()
        db.refresh(run)
        return run

    # ── worker body ──────────────────────────────────────────────────────

    @staticmethod
    def run(db, run_id: str, *, cancel_check: Optional[Callable] = None,
            progress_cb: Optional[Callable] = None) -> Dict[str, Any]:
        run = db.query(CircuitDiscoveryRun).filter(
            CircuitDiscoveryRun.id == run_id).first()
        if run is None:
            raise ValueError(f"Discovery run {run_id} not found")
        capture = db.query(CircuitCaptureRun).filter(
            CircuitCaptureRun.id == run.capture_run_id).first()
        if capture is None or not capture.store_path:
            raise DiscoveryConfigError("Capture store missing")
        p = run.params
        manifest = capture.manifest or {}
        store_dir = settings.resolve_data_path(capture.store_path)
        doc_lengths = {int(k): int(v)
                       for k, v in (manifest.get("doc_lengths") or {}).items()}
        heldout = np.array(manifest.get("split", {}).get("heldout_docs", []),
                           dtype=np.uint32)
        all_docs = np.arange(manifest.get("counts", {}).get("documents", 0),
                             dtype=np.uint32)
        discovery_docs = all_docs[~np.isin(all_docs, heldout)]
        n_tokens_discovery = int(sum(doc_lengths.get(int(d), 0)
                                     for d in discovery_docs))

        run.status = "running"
        run.progress = 0.0
        db.commit()
        t0 = time.monotonic()

        # ── units per layer ──────────────────────────────────────────────
        layers = sorted(e["layer"] for e in manifest.get("layers", []))
        readers = {L: EventReader(store_dir, L) for L in layers
                   if layer_files_exist(store_dir, L)}
        sae_by_layer = {e["layer"]: e["sae_id"]
                        for e in manifest.get("layers", [])}
        units_by_layer: Dict[int, List[Dict[str, Any]]] = {}
        uncovered_seeds: List[Dict[str, Any]] = []
        capped_layers: List[int] = []
        for L, reader in readers.items():
            if p["granularity"] == "feature":
                # Seed features survive the per-layer unit cap: the cap trims
                # the OPEN pool by support; seeds are the point of the run.
                seed_feature_ids = {
                    int(r["feature_idx"]) for r in p.get("seed_refs", [])
                    if r.get("layer") == L and r.get("feature_idx") is not None
                } if p["mode"] == "seeded" else set()
                units, capped = _feature_units(
                    reader, discovery_docs, p["s_min"],
                    p["max_units_per_layer"],
                    always_keep=seed_feature_ids)
            else:
                units, capped = _cluster_units(
                    db, reader, sae_by_layer.get(L), L, discovery_docs,
                    p["cohesion_floor"], p["s_min"])
            if capped:
                capped_layers.append(L)
            units_by_layer[L] = units

        # seeded restriction: the UP side of every pair must match a seed ref
        # (downstream side keeps all units — we mine the seeds' partners).
        seed_keys = _seed_key_set(p) if p["mode"] == "seeded" else None
        if seed_keys is not None:
            for L, units in units_by_layer.items():
                for u in units:
                    u["is_seed"] = u["seed_key"] in seed_keys
                seen = {u["seed_key"] for u in units}
                for sk in seed_keys:
                    if sk[0] == L and sk not in seen:
                        uncovered_seeds.append(
                            {"layer": sk[0], "ref": sk[1],
                             "reason": "below support floor or absent from store"})

        # ── pair generation + PMI shortlist ─────────────────────────────
        shortlist: List[Dict[str, Any]] = []
        layer_list = sorted(units_by_layer.keys())
        total_pairs = 0
        for i, Lu in enumerate(layer_list):
            for Ld in layer_list[i + 1:]:
                ups = [u for u in units_by_layer[Lu]
                       if seed_keys is None or u.get("is_seed")]
                for up in ups:
                    if cancel_check is not None and cancel_check():
                        run.status = "cancelled"
                        db.commit()
                        return {"status": "cancelled"}
                    for down in units_by_layer[Ld]:
                        total_pairs += 1
                        ps = stats.pair_stats(
                            up["keys"], down["keys"], n_tokens_discovery,
                            up.get("acts"), down.get("acts"))
                        if ps.n_ud < p["s_min"]:
                            continue
                        shortlist.append({"up": up, "down": down, "stats": ps})
            if progress_cb is not None:
                progress_cb(min(50.0, (i + 1) / max(len(layer_list), 1) * 50))

        # PMI-ranked shortlist for null testing (cap recorded, not silent)
        shortlist.sort(key=lambda c: c["stats"].pmi, reverse=True)
        null_capped = len(shortlist) > p["max_null_tested"]
        tested = shortlist[: p["max_null_tested"]]

        # ── null + FDR ───────────────────────────────────────────────────
        null_results = []
        for j, cand in enumerate(tested):
            # The null loop is the dominant cost — poll cancellation here too,
            # not only in pair-generation (R1 CR#5).
            if cancel_check is not None and j % 25 == 0 and cancel_check():
                run.status = "cancelled"
                db.commit()
                return {"status": "cancelled"}
            null_results.append(stats.null_test(
                cand["up"]["keys"], cand["down"]["keys"], doc_lengths,
                k_shuffles=p["null_shuffles"],
                percentile=p["null_percentile"],
                seed=p["seed"] + j))
            if progress_cb is not None and j % 50 == 0:
                progress_cb(50.0 + j / max(len(tested), 1) * 35)
        # Pooled standardized p-values — per-pair empirical p floors at
        # 1/(K+1), which BH over m pairs can never clear (see stats module).
        pooled_p = stats.pooled_null_pvalues(null_results)
        keep = stats.bh_fdr(pooled_p, q=p["fdr_q"])
        survivors = [dict(c, null=n, pooled_p=float(pp)) for c, n, pp, k in
                     zip(tested, null_results, pooled_p, keep) if k]

        # ── held-out replication ────────────────────────────────────────
        repl, flags = stats.heldout_replication(
            [(c["up"]["keys_all"], c["down"]["keys_all"]) for c in survivors],
            heldout, doc_lengths,
            k_shuffles=p["null_shuffles"], percentile=p["null_percentile"],
            seed=p["seed"] + 10_000)

        # ── candidates + report ─────────────────────────────────────────
        truncated = len(survivors) > DEFAULT_CANDIDATE_CAP
        candidates = []
        for rank, (c, replicated) in enumerate(
                zip(survivors[:DEFAULT_CANDIDATE_CAP],
                    flags[:DEFAULT_CANDIDATE_CAP])):
            ps: stats.PairStats = c["stats"]
            null: stats.NullResult = c["null"]
            candidates.append({
                "up": _unit_ref(c["up"], p["granularity"]),
                "down": _unit_ref(c["down"], p["granularity"]),
                "granularity": p["granularity"],
                "stats": {
                    "pmi": round(ps.pmi, 4), "lift": round(ps.lift, 4),
                    "support": ps.n_ud,
                    "spearman": (round(ps.spearman, 4)
                                 if ps.spearman is not None else None),
                    "null_pct": round(null.null_percentile, 2),
                    "p_value": round(null.p_value, 5),        # per-pair (floor 1/(K+1))
                    "pooled_p": round(c["pooled_p"], 6),      # FDR input
                },
                "replicated_heldout": bool(replicated),
                "attribution": None,   # Tier-2 pass fills
                "orderings": {"coact_rank": rank, "attr_rank": None},
            })
        report = {
            "granularity": p["granularity"],
            "mode": p["mode"],
            "supernode_activation": "max" if p["granularity"] == "cluster" else None,
            "lag0_disclosure": (
                "All co-activation is lag-0 (same token position); "
                "attention-mediated (lagged) structure requires Tier-2.5."),
            "null_summary": {
                "method": "within_document_circular_shift",
                "shuffles": p["null_shuffles"],
                "percentile": p["null_percentile"],
            },
            "fdr": {"discipline": "benjamini_hochberg",
                    "p_source": "pooled_standardized_empirical_null",
                    "p_resolution": round(
                        1.0 / (p["null_shuffles"] * max(len(tested), 1) + 1), 8),
                    "q": p["fdr_q"],
                    "tested": len(tested), "passed": int(keep.sum())},
            "replication": {"tested": repl.tested,
                            "replicated": repl.replicated,
                            "rate": repl.rate},
            "counts_by_stage": {
                "pairs_considered": total_pairs,
                "post_support": len(shortlist),
                "null_tested": len(tested),
                "post_fdr": len(survivors),
                "candidates_persisted": len(candidates),
            },
            "caps": {
                "units_per_layer": p["max_units_per_layer"],
                "unit_cap_hit_layers": capped_layers,
                "null_tested_cap": p["max_null_tested"],
                "null_cap_hit": null_capped,
                "candidate_cap": DEFAULT_CANDIDATE_CAP,
                "candidates_truncated": truncated,
            },
            "uncovered_seeds": uncovered_seeds,
            "attribution": None,        # Tier-2 pass fills envelope + method
            "uplift": None,             # 017 fills survival-rate uplift
            "echo_filter": None,        # 018 classifier feeds counts back
            "wall_clock_seconds": round(time.monotonic() - t0, 1),
        }
        # Last-writer race: don't clobber a cancel that landed during the
        # final stages (R1 CR#6).
        db.refresh(run)
        if run.status == "cancelled":
            return {"status": "cancelled"}
        run.candidates = candidates
        run.report = report
        run.status = "completed"
        run.progress = 100.0
        db.commit()
        return {"status": "completed",
                "candidates": len(candidates),
                "replication_rate": repl.rate}


# ── unit builders ────────────────────────────────────────────────────────

def _feature_units(reader: EventReader, discovery_docs: np.ndarray,
                   s_min: int, max_units: int,
                   always_keep: Optional[set] = None) -> Tuple[List[Dict], bool]:
    always_keep = always_keep or set()
    units = []
    for fid in reader.feature_ids:
        ev = reader.feature_events(fid)
        keys_all = (ev["doc_id"].astype(np.uint64) << np.uint64(16)) | \
            ev["token_pos"].astype(np.uint64)
        mask = np.isin((keys_all >> np.uint64(16)).astype(np.uint32),
                       discovery_docs)
        keys = keys_all[mask]
        if len(keys) < s_min:
            continue
        units.append({
            "kind": "feature", "layer": reader.layer, "feature_idx": int(fid),
            "keys": keys, "keys_all": keys_all,
            "acts": ev["act"][mask].astype(np.float64),
            "support": len(keys),
            "seed_key": (reader.layer, f"feature:{int(fid)}"),
        })
    capped = len(units) > max_units
    units.sort(key=lambda u: u["support"], reverse=True)
    kept = units[:max_units]
    kept_ids = {u["feature_idx"] for u in kept}
    kept.extend(u for u in units[max_units:]
                if u["feature_idx"] in always_keep - kept_ids)
    return kept, capped


def _cluster_units(db, reader: EventReader, sae_id: Optional[str], layer: int,
                   discovery_docs: np.ndarray, cohesion_floor: float,
                   s_min: int) -> Tuple[List[Dict], bool]:
    """Cluster supernodes: A_C(t) = max over members ⇔ any-member event
    (events already θ-thresholded). Cohesion eligibility from the profile
    budget's G (013's cohesion gain)."""
    profiles = db.query(ClusterProfile).filter(
        ClusterProfile.sae_id == sae_id).all() if sae_id else []
    units = []
    for prof in profiles:
        g = (prof.budget or {}).get("G")
        if g is not None and float(g) < cohesion_floor:
            continue
        member_sets_all, member_sets, acts_parts = [], [], []
        for m in (prof.members or []):
            ev = reader.feature_events(int(m.get("feature_idx", -1)))
            if not len(ev):
                continue
            keys_all = (ev["doc_id"].astype(np.uint64) << np.uint64(16)) | \
                ev["token_pos"].astype(np.uint64)
            member_sets_all.append(keys_all)
            mask = np.isin((keys_all >> np.uint64(16)).astype(np.uint32),
                           discovery_docs)
            member_sets.append(keys_all[mask])
            acts_parts.append((keys_all[mask], ev["act"][mask].astype(np.float64)))
        keys = stats.supernode_keys(member_sets)
        if len(keys) < s_min:
            continue
        acts = _max_acts_per_key(keys, acts_parts)
        units.append({
            "kind": "cluster", "layer": layer,
            "cluster_profile_id": prof.id, "cluster_name": prof.name,
            "keys": keys, "keys_all": stats.supernode_keys(member_sets_all),
            "acts": acts, "support": len(keys),
            "seed_key": (layer, f"cluster:{prof.id}"),
        })
    return units, False  # profiles are few — no unit cap needed


def _max_acts_per_key(keys: np.ndarray,
                      parts: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """A_C per key = max over member activations at that key."""
    out = np.zeros(len(keys), dtype=np.float64)
    order = np.argsort(keys)
    sorted_keys = keys[order]
    for part_keys, part_acts in parts:
        idx = np.searchsorted(sorted_keys, part_keys)
        valid = (idx < len(sorted_keys)) & (sorted_keys[np.clip(idx, 0, len(sorted_keys) - 1)] == part_keys)
        np.maximum.at(out, order[idx[valid]], part_acts[valid])
    return out


def _seed_key_set(params: Dict[str, Any]):
    """seed_refs → {(layer, 'feature:<idx>'|'cluster:<id>')}.

    Discriminate on VALUE, not key presence: the typed DiscoverySeedRef
    model_dump()s BOTH keys with the unset one None, so `"feature_idx" in ref`
    is always true — the R1 typed-model fix relocated the int(None) crash to
    the cluster-seed path, which is the DEFAULT seeded flow (R2 B1)."""
    keys = set()
    for ref in params.get("seed_refs", []):
        layer = ref.get("layer")
        if ref.get("feature_idx") is not None:
            keys.add((layer, f"feature:{int(ref['feature_idx'])}"))
        elif ref.get("cluster_profile_id") is not None:
            keys.add((layer, f"cluster:{ref['cluster_profile_id']}"))
    return keys


def _unit_ref(unit: Dict[str, Any], granularity: str) -> Dict[str, Any]:
    if unit["kind"] == "feature":
        return {"layer": unit["layer"], "feature_idx": unit["feature_idx"]}
    return {"layer": unit["layer"],
            "cluster_profile_id": unit["cluster_profile_id"],
            "cluster_name": unit["cluster_name"]}
