# Review Record — Feature 017: Circuit Validation — ROUND 2

**Date:** 2026-07-20
**Scope:** post-R1 state at HEAD `bc7f46c`. Code-review (8 findings, B1–B8) + 4-perspective /review (13 findings). **21 findings.** All R1 fixes verified holding; the criticals were the follow-through gaps R1's fixes exposed. Pre-existing issues addressed per the user directive.

## PART A — R1 fixes verified (all HOLD)
The false-positive-null fix, support-matched-null band, reproduction-verdict path, `_empty_result`, reproduce guard/gating, GPU-guard-covers-validation, attr-needs-attribution, uplift-per-ordering, cancel-race, denom guard, correlations-determinism, ablation-labels — all HOLD. PARTIAL verdicts were NEW adjacent defects, not regressions.

## The four must-fixes (all fixed)

1. **B-2 (P1, most important) — downstream SAE read the WRONG module.** Both GPU services captured downstream activation at `structure.layers_module[down_L]` (whole-layer output) but the SAE was trained on `get_hookable_module(...,"residual")` (residual-norm submodule) — differ by MLP+residual add. So EVERY ES and faithfulness behavior was computed on out-of-distribution input — plausible but systematically wrong. Fixed at all 3 sites. Pre-existing in intervention (b95a5cf); new in faithfulness.
2. **B-3 (P1) — min-null floor systematically unreachable (over-correction).** `_null_effect_sizes` probed each u' only in u's top-2 docs, where a random u' almost never co-fires → yield <10 → every edge "underpowered". Fixed: probe each u' in ITS OWN strongest-firing docs.
3. **B-5 (P0) — faithfulness had no lifecycle → bypassed GPU guard, uncancellable, unreclaimable.** Fixed: Circuit.faithfulness_status/task_id (migration b4046f2741dd); guard checks it; guard-and-mark under advisory lock + 409; cleanup reclaims it.
4. **Producer gap (Critical-for-015) — nothing built a promoted circuit from a discovery run**, so R1's write-back had nothing to write to and 015 would read empty ES. Fixed: CircuitService.from_candidates + POST /circuit-discovery/{id}/build-circuit threading discovery_run_id.

## Also fixed
- B-1 (pre-empted): promoted-circuit write-back is now best-effort (commit run results first) so a hiccup doesn't fail a good validation.
- B-4: idempotent tested_and_failed history (record 2 once).
- B-6/B-7: vectorized faithfulness feature ranking (EventReader.feature_activation_mass, np.bincount).
- B-8: record n_down_firings beside sigma_d (thin-support ES visibility).
- Faithfulness + uplift UI (P1 demo-UX): Run-faithfulness button (gated+polling+status) + necessity/sufficiency; Validation batch banner shows both orderings' survival + uplift; ManifestDrawer Reproduce polls for the child manifest + verdict (R1 P4 closed).
- UX (user screenshot): Capture layer box auto-fills+locks from the selected SAE's own layer (SAEs trained one-layer-each) — removes redundancy + contradiction risk.

## Recorded (R3/close-out)
A2 design debt (three lifecycles/one row); support-floor skip; manifest whitelist path-guard; from_candidates UI action (API/MCP-reachable today).

## R2 outcome
21 findings, all addressed; all R1 fixes hold. The two measurement-correctness bugs (B-2, B-3) + the 015 producer gap were the substantive catches — 017's numbers are now right and rung-2 ES reaches circuits. Backend circuit suite green; frontend 907 + build green. R3 = final gate.
