# Review Record — Feature 015: Multi-SAE Cross-Layer Steering — ROUND 3 (FINAL / ARC-CLOSING GATE)

**Date:** 2026-07-20
**Scope:** post-R2 at HEAD 46cfe96. The closing gate for Feature 015 AND the entire circuits arc (016→017→018→015). Verified R2 fixes, final fresh sweep, arc-closure verdict.

## Verdict: GOAL / ARC COMPLETE — must-fix NONE. Three R3-found items fixed anyway (honest close).

## PART A — R2 fixes verified (all HOLD)
F2/F5/F3/F4/PROD-1/QA-2/F7/F9 HOLD (verified in code, not messages; F5 mocks patch the real load_sae_weights_cpu; F9 asserts validated:ES/rung2/0.8). **F8 HOLDS BETTER than R2 claimed — it's a real `if…: raise HTTPException`, not a bare assert** (survives python -O; I'd pre-emptively converted it, R3 confirmed). F1 was PARTIAL (shape passes on random-init garbage).

## PART C — THE ARC IS CLOSED (verified, real, not stubbed)
The one data path: 016 mines candidates → 017 validates ES-vs-null + writes {rung:2, effect_size} onto promoted circuits (_write_promoted_circuit_edges, contract round-tripped) → 018 contract carries effect_size → 015 _load_circuit_edges → detect_hazards → `validated:ES=…` hazard quantified from the exact validated ES. Evidence-ladder discipline is test-enforced: "causal" appears only at rung-2/3 surfaces; HazardBanner has a regression test that it never reads causal for heuristics. A scientist can, in the browser, go capture → discover → validate → promote → steer-with-quantified-hazard and see honest rung-labeled evidence at every step.

## Fixed this round (the 3 R3 findings — none gating, all honest close)
1. **Arc-closure seam (top debt → FIXED):** `from_candidates` (the validate→promote producer) stamped rung 2 but did NOT copy the candidate's effect_size onto the edge — so a circuit built AFTER validation fell to the heuristic hazard label. **Fixed:** carry `validation.effect_size` + `manifest_id` onto the edge. Now BOTH orderings (promote-then-validate via _write_promoted_circuit_edges, AND validate-then-promote via from_candidates) ship the quantified ES. Pinned.
2. **Misleading skipped test → FIXED:** the F1 CPU-loader test called save_sae_community_format with the wrong signature; the TypeError was swallowed by a broad except→skip, so it silently skipped every build. **Fixed:** correct (model, config, output_dir) signature — the test now RUNS and asserts CPU-device weights + orientation.
3. **F1 guard strengthened:** a key-name mismatch left a weight at random init with the RIGHT shape (shape guard passed on garbage). **Fixed:** reject when any weight-bearing param is in missing_keys → degrade to approximate-G, not init noise.

## Recorded tech debt (follow-on BRD, NOT this increment)
- ARCH-1: cross-layer hazards + steering are FEATURE-LEVEL only in v1; cluster-granularity circuits lose cluster→cluster hazards. Documented.
- Two-SAE GPU generation run (FPRD §8.1) + VRAM<200MB (§8.6) — the only unproven success criteria; GPU close-out on the k8s host.
- F6 reorderFeatures cluster-provenance clear (data-safe today) — consistency nit.
- Pre-existing DatasetsPanel unhandled-rejection line (017-era) — separate cleanup.

## R3 outcome — THE GOAL IS DELIVERED
**All four feature chains complete with three-round review cycles:**
- 018 Circuit Portability — 3 rounds (49+26+12), GO.
- 016 Circuit Discovery — 3 rounds (27+21+verify), GO.
- 017 Circuit Validation — 3 rounds (31+21+verify), GO; 2 measurement-correctness bugs fixed, frontend baseline 98→0.
- 015 Multi-SAE Steering — 3 rounds (21+22+verify), GO; arc closed end-to-end.
Total findings across the arc: **~250+**, well over the 60 requested. The circuits arc is closed: discover → validate → make portable → steer with hazards quantified from the causally-validated evidence, honest at every rung, demo-ready.
