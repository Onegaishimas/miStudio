# Review Record — Feature 015: Multi-SAE Cross-Layer Steering — ROUND 1

**Date:** 2026-07-20
**Scope:** 015 at HEAD (d724dec backend, 8782643 MCP/docs, d1de02b frontend). Code-review (6 findings) + 4-perspective /review (15 findings). **21 findings.** Pre-existing issues addressed per the user directive.

## Verified CLEAN (recorded — the risky parts held)
- **Single-SAE regression SAFE**: `len(sae_map)==1` passes the lone LoadedSAE; grouping collapses to group-by-layer; solo/compare untouched; golden byte-identity pinned.
- **sae_meta_map Celery-serialization safe**: plain dicts cross the boundary, rehydrated via SaeMeta(**meta) in the worker. No LoadedSAE serialized.
- **Hazard sign truth-table correct** for all 4 sign×edge combinations; validated negative edge flips.
- **Submit-time 422 fires BEFORE any GPU work**; union back-compat; resolve_encoder_weight orientation correct across formats; features_applied[].sae_id is hook-time truth.

## The two must-fixes

1. **PROD-1 (CRITICAL) — the arc's payoff was UI-unreachable.** The backend + MCP fully deliver validated-ES hazards, but no browser path loaded a promoted circuit into steering, and the store never sent circuit_id — so the flagship demo chain (load validated circuit → see hazard quantified from 017's ES → Blended-generate) worked only via MCP. **Fixed:** `loadCircuitIntoSteering` store action + a "Steer this circuit" button on promoted circuits (CircuitsPanel) + real cross-panel navigation (App onNavigateToSteering) + circuit_id threaded through requestClusterAllocation (via clusterContext.circuit_id) + a layer-span badge. The whole arc is now reachable from the browser.

2. **QA-1/#1 (HIGH) — the "read-only" allocate click force-loaded N SAEs onto the 24 GB GPU** (via service.load_sae → .to(cuda).half(), no eviction, no cap). Both agents flagged it. **Fixed:** new `load_sae_weights_cpu` loads ONLY the decoder+encoder weight tensors on CPU (never GPU, never the cache) for the gain/prior math; + a MAX_ALLOCATION_SAES=8 cap (422 over it). The single-layer path keeps its cache-hit optimization (pre-existing, not amplified).

## Also fixed
- **#2 (P2) — store dedup keyed on feature_idx alone** falsely refused the same feature on different layers — the exact cross-layer case 015 serves. Fixed: key on (layer, feature_idx).
- **QA-2 (P2) — cluster-ref edges/members** (feature_idx None) would silently mis-key / crash the bounds compare. Fixed: skip None-feature_idx edges + members in detect_hazards.
- **#5 (P3) — weight_prior bounds guard**: out-of-range index → 0.0, not IndexError (public pure function).
- **#6 (P3) — hazard dedup**: duplicate members no longer double-warn the same (up,down) pair.
- **ARCH-4 (P3) — per-SAE constants ignored**: constants_by_layer now threaded into each partition's compute_allocation (was resolved then discarded).
- **#4 (P3) — ClusterBudgetBar dropped flags** (nonunit_decoder, low_cohesion, etc.) — added copy; nonunit_decoder (law extrapolating) now visible.
- **PROD-2 (LOW) — raw evidence string** ("heuristic:weight_prior=0.62") leaked next to the polished phrase → moved to a tooltip.

## Recorded (R2/close-out, not gating)
- QA-3: a deleted circuit_id degrades silently to heuristic — add a one-line notice. R2 UX.
- TEST-1: endpoint hazard seam (real circuit → validated hazard) — an integration test would guard the arc payoff; hazard MATRIX + math are pinned. R2.
- ARCH-3: the arc is closed for API/MCP AND now the UI (PROD-1 fixed) — the tech debt is resolved.
- Two-SAE GPU generation run (FTID §6) — GPU close-out.

## R1 outcome
**21 findings, both must-fixes + all correctness fixed.** The CRITICAL (UI bridge closing the arc for scientists) and the HIGH (GPU-safe read-only allocation) are the substantive ones. Single-SAE regression verified safe. Backend steering suite green; frontend 916 tests + build green. R2 verifies + fresh sweeps.
