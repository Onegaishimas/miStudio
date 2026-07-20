# Review Record — Feature 017: Circuit Validation — ROUND 3 (FINAL GATE before 015)

**Date:** 2026-07-20
**Scope:** HEAD `f165d93`. Verified R2's fixes hold against current code (not messages), fresh sweep of `bc7f46c..HEAD`, and a definitive GO/NO-GO for Feature 015 (Multi-SAE Steering). Pre-existing issues in scope per the user directive.

**Verdict: GO for 015. Must-fix-before-015 list is EMPTY.**

---

## PART A — R2 fixes verified against HEAD

1. **B-2 wrong-module — HOLDS.** All THREE downstream reads now resolve `get_hookable_module(structure.layers_module[down_L], "residual", structure)`:
   - `circuit_intervention_service._validate_edge` (line 300-301)
   - `circuit_intervention_service._null_effect_sizes` (line 438-439)
   - `circuit_faithfulness_service._behavior/_downstream_sum` (line 407-408)
   The suppression side (`up_module`) matches the capture residual module at both intervention sites (line 294-295, 432-433) and in faithfulness (`hook_layers`, line 410-412). No remaining bare `structure.layers_module[down_L]` for a capture/encode. Grep of tests confirms no test pins `layers_module`/`get_hookable_module` (these GPU paths need a real model), so the fix changes no test expectation.

2. **B-3 null-yield — HOLDS, and the null is CORRECT not merely non-empty.** `_null_effect_sizes` now probes each u′ in ITS OWN strongest-firing docs (line 452-454), symmetric with how the real edge measures u where u fires. Assessed the "spurious null" risk raised in the brief: measuring u′ at its own support and reading Δ on d is exactly the right null — for a random non-edge u′→d, suppressing u′ barely moves d ⇒ small Δ ⇒ small null ES; if d isn't active there, a_d≈baseline ⇒ Δ≈0 ⇒ still small. A spurious null would be one that inflates the threshold (rejecting real edges) or stays empty (the old bug); neither occurs. The `MIN_NULL_SAMPLES=10` floor now realistically fills; `edge_verdict` still FAILS-closed below it (vmath line 68-72). Not too permissive.

3. **B-5 faithfulness lifecycle — HOLDS, single head.** `Circuit.faithfulness_status/task_id` added by migration `b4046f2741dd` (down_revision `1c3ac72efd47`, both columns, `IF NOT EXISTS`/`IF EXISTS`). **Verified sole alembic head, no duplicate revisions, no merge-tuple weirdness.** `assert_no_active_gpu_run` checks faithfulness (capture_service line 116-123). `start_faithfulness` guard-and-marks under the transaction-scoped advisory lock (`assert_no_active_gpu_run` takes `pg_advisory_xact_lock` then sets `pending` and commits in the SAME `_run_sync` txn) + 409. Run sets `running`→`completed`; task sets `failed`/`cancelled`; cleanup reclaims (`cleanup_stuck_circuit_runs` line 77-84). All wired.

4. **Producer chain (from_candidates) — HOLDS, traced end-to-end + tested.**
   - `CircuitService.from_candidates` + `POST /circuit-discovery/{id}/build-circuit` thread `discovery_run_id`, build members (union of endpoints by (layer,idx)), edges, SAE refs from the capture manifest, create UNPROMOTED at the candidates' min rung. (`test_builds_circuit_carrying_discovery_run_id` green: discovery_run_id set, promoted=False, 2 members, edge rung=1, SAE layers {13,14}.)
   - Full chain: build-circuit → `set_promoted(True)` → validate → `_write_promoted_circuit_edges` fires when `Circuit.discovery_run_id == run_id AND promoted == True`, matches edges by (up.layer, up.feature_idx, down.layer, down.feature_idx), writes `{rung:2, effect_size, validation_manifest_ref}`, round-trips through `CircuitDefinitionV1`, bumps rung+version. (`test_validated_edge_lifts_promoted_circuit_rung` green: rung 2, effect_size 0.9, manifest_ref, version+1.) **Rung-2 ES lands on `circuit.edges[].effect_size` — confirmed.**

5. **B-1 best-effort — HOLDS.** Propagation is after the run's own `db.commit()` (intervention line 220), inside try/except with `db.rollback()` on failure (line 224-231); a hiccup logs and does NOT fail the validation or lose the persisted results. **B-4 idempotent** (records rung 2 once via `if 2 not in hist`, line 536-537). **B-6/B-7 vectorized** (`feature_activation_mass`, `np.bincount` in faithfulness `_select_prompts`/`_top_features`/`_top_nonmembers`). **B-8 n_down_firings** recorded beside sigma_d on full results (intervention line 359).

6. **UI — HOLDS.** Faithfulness Run button + mode toggle + status polling + necessity/sufficiency display (CircuitsPanel line 104-337); uplift + both-orderings banner (line 1349-1373); ManifestDrawer reproduce child-manifest polling + three-state verdict (line 72-84, 195-206); Capture layer-lock auto-fills from SAE and `readOnly` when the SAE has a layer (line 664-691).

---

## PART B — fresh sweep (bc7f46c..HEAD)

- **B-2 fix vs tests:** touches only GPU-service module resolution; the toy-model attribution/ε pins are pure hand-computation/graph tests using their own toy setup — no `layers_module` reference. They still pass. No test expectation changed.
- **from_candidates edge-rung derivation:** `validated_rung==2 → 2`, else `attribution.rung1_gate → 1`, else `0` — correct. Member union by (layer,idx) via `members_by_key.setdefault` — correct dedup (test asserts 2 members from a shared-endpoint pair). **Cluster candidates:** endpoints with `feature_idx=None` are skipped as MEMBERS (line 113) but STILL emitted as an edge with `feature_idx:None` (line 121-125) → `CircuitNodeRef` (default kind="feature") rejects it → `CircuitValidationError` → 422. So a cluster-granularity run can't be built into a circuit; it 422s rather than filtering/handling. Consistent with v1 being feature-only (the intervention `run` already skips cluster candidates, line 169-170) and with 015 consuming feature circuits — but the 422 is opaque. **Ride-along**, not gating.
- **Commit reordering in run():** non-reproduce commits the run's results at line 220 BEFORE the best-effort propagation; reproduce branch skips line 220 (it's inside the `else`) and `_persist_reproduction` commits its own manifest; the final `completed`-state commit (line 261) happens for BOTH branches; report/by_ordering write is gated behind `not reproduce_of` (line 243) so a reproduce doesn't stomp the report. No double-commit, no stale-run: run is re-queried (line 232-233) and `db.refresh`ed (line 237) after the best-effort block.
- **Faithfulness guard-and-mark serialization:** the mark (`row.faithfulness_status = "pending"`) is INSIDE the advisory-locked `_run_sync` txn (circuits.py line 299-303); two concurrent POSTs serialize on `pg_advisory_xact_lock` and the second sees `pending` → 409. Serializes correctly.
- **Migration:** single head `b4046f2741dd`, no duplicate revisions.
- **None/empty:** `_empty_result` (intervention line 489-493) omits `n_down_firings` — but the ManifestDrawer only reads `n_prompts`/`sigma_d` (both present) and the TS `ValidationEdge` type doesn't require `n_down_firings`; no crash, no 015 impact. Trivial consistency nit (ride-along).

**Tests:** `test_circuit_service.py` 14 passed in isolation (incl. TestFromCandidates + TestPromotedCircuitValidationWriteBack); full circuit unit set green when run without the shared-DB enum-creation race. The only errors seen are the documented `export_status already exists` / advisory-lock enum-isolation flakiness (MEMORY.md), never assertion failures.

---

## PART C — GO/NO-GO for 015

**Definitive: the validated-ES seam 015 consumes is intact.**

- **What 015 reads (015_FTID §2.3):** `detect_hazards(members_by_layer, sae_map, circuit_edges=None)` — PRIMARY evidence = stored edges at **rung ≥2** with `quantified_effect` **from the edge's measured ES**. i.e. it reads `edge.rung` and `edge.effect_size` off a promoted circuit.
- **What 017 writes:** `_write_promoted_circuit_edges` sets `edge.rung=2` + `edge.effect_size=<measured ES>` + `validation_manifest_ref`, round-tripped through `CircuitEdge` whose schema declares `rung: EvidenceRung` and `effect_size: Optional[float]  # measured ES when rung >= 2 (hazard-v2 consumes)`. **Exact field match.** No adapter needed.
- **Encoder/weight-prior seam (015 needs it; 016 Task 0.1 pin):** BOTH `resolve_decoder_weight` and `resolve_encoder_weight` exist in `steering_service.py` (line 311, 331), correctly oriented ([d_model,d_sae] and [d_sae,d_model]), documented as the single orientation source for IDL-32's cross-layer weight prior and 015 hazards. Intact.
- **Anything 015 trips on immediately:** none found. The producer exists (build-circuit), promotion is a badge, validation writes ES onto promoted circuits, and 015's read is field-identical. 015 also (per its own §3.4 item 14) ships heuristic-labeled and consumes stored rung-≥2 edges when present — which is precisely the state 017 leaves.

### Must-fix-before-015
*(empty)*

### Ride-along (do in 015 or a close-out, not gating)
- `from_candidates` on a cluster-granularity run 422s opaquely instead of filtering cluster edges or returning a clear "cluster build unsupported in v1" message. (Matches R2-recorded `from_candidates` debt.)
- `_empty_result` omits `n_down_firings` (harmless; add for shape consistency).
- Faithfulness task passes `cancel_check=None` (circuit_validation_tasks line 71) — mid-run faithfulness cancel isn't wired though the service supports it; cleanup/guard still reclaim a stuck run, so non-blocking.
- A2 design debt (three lifecycles on one discovery row) still recorded; consider the child `circuit_validation_runs` table before a further lifecycle is added.

---

## R3 outcome
All six R2 fix families HOLD at HEAD. The B-2 module correction, B-3 null correctness, B-5 faithfulness lifecycle, and the from_candidates producer are verified end-to-end with green tests. The validated rung-2 ES reaches `circuit.edges[].effect_size` in exactly the shape 015's `detect_hazards` reads, and the encoder-weight seam is present. **GO for 015 with an empty must-fix list.**
