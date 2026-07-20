# Task List: Intervention, Validation & Faithfulness

**Document ID:** 017_FTASKS|Circuit_Validation
**Version:** 1.0
**Status:** ⏳ IMPLEMENTING — Task 3.0 + Ph1-5 backend + MCP + remediation + manual DONE; Validation-tab UI in progress; then 3-round review. Phase 6 GPU-E2E at close-out.
**Source:** 017_FPRD · 017_FTDD · 017_FTID · IDL-34 · CIRCUITS-002 A.5/A.7 normative

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Intervention primitives | 1 task | ⏳ |
| Phase 2: Manifests | 2 tasks | ⏳ |
| Phase 3: Edge validation | 2 tasks | ⏳ |
| Phase 4: Faithfulness | 1 task | ⏳ |
| Phase 5: Remediation + MCP + UI | 3 tasks | ⏳ |
| Phase 6: Verification + acceptance | 2 tasks | ⏳ |

---

## Phase 1: Intervention primitives

### Task 1.1: Suppression hook
- [x] DONE (22aa850: circuit_intervention_hooks — directional, never re-decode; ε-invariance + subtraction-exactness + feature-list-sum pins) `_create_suppression_hook(sae, feature_idx, a_base, positions?)` (steering-hook variant; feature-LIST support for faithfulness) · same-pass encode for a_u(t) · **ε-invariance pin (never re-decode) + subtraction-exactness pin on toy tensors** · baseline variants (zero | corpus-mean from 016 store)

## Phase 2: Manifests

### Task 2.1: Table + service
- [x] DONE (22aa850: validation_manifests migration 85f73dbda900; manifest_service persist/get/list + payload self-containment validation) `validation_manifests` (`vman_`, kind edge_batch|faithfulness|reproduction, self-contained payload JSONB, parent refs) + Alembic (single-head, up+down) · persist/get/list-by-parent

### Task 2.2: Reproduce
- [x] DONE (ac6c64f: POST /validation-manifests/{id}/reproduce; reproduction_verdict per-edge ES delta + tolerance; pinned) `POST /validation-manifests/{id}/reproduce` re-executes from payload → reproduction manifest with deltas + tolerance verdict · automated reproduction test

## Phase 3: Edge validation

### Task 3.0: 018 hand-off preconditions for the circuit writer (MUST land before 3.1/3.2 write results — escalates to P1 once the writer exists; 018 R2-Q2/R2-A5, R3-B5)
- [x] DONE (Circuit.version col, migration 9c8683b365f6; CircuitService.update expected_version→409; PATCH threads it; frontend inline-edit participates). **Optimistic concurrency on circuit writes:** add a version/updated_at precondition to `CircuitService.update` + PATCH (reject stale writes with 409) — 017's validation writer and a user editing in the panel will race without it.
- [x] DONE (CircuitService.write_edge_validation — the ONLY edge-write path, routes through update() so validators+rung-recompute+version-bump run; pinned: validation write lifts rung + bumps version + respects expected_version). **`update()`-only edge-write rule:** validation results write edges exclusively through `CircuitService.update` (never raw JSONB mutation) so contract validators + rung recompute always run; document in the service docstring and pin with a test that a validation write triggers rung recomputation.

### Task 3.1: Intervention service + task
- [x] DONE (b95a5cf: circuit_intervention_service — prompt windows from 016 store, matched passes, ES=mean/σ_d (σ_d from SAME store), shuffled-non-edge null, sign gate, tested_and_failed history via 018 enum) Prompt windows from 016 store + SAME tokenization (id asserted) · matched greedy passes (fixed seeds) · Δ_p over clean-fire tokens · ES = mean/σ_d (σ_d from the SAME store) · shuffled-NON-edge support-matched null · sign-consistency gate (default 8/10, config) · tested_and_failed recording (rung history via 018's shared enum — import, never redefine)

### Task 3.2: Batch + uplift
- [x] DONE (b95a5cf: top-K per ordering, survival per ordering, uplift(attr−coact at equal K) in math module; write-back to discovery candidates). Circuit-edge write via CircuitService.write_edge_validation (Task 3.0) Per-ordering scopes at EQUAL K · survival per tier per ordering · **uplift number into batch result + 016's discovery-run report** · results into discovery candidates + promoted-circuit edges · WS progress

## Phase 4: Faithfulness

### Task 4.1: Faithfulness service + task
- [x] DONE (b95a5cf: member expansion, per-layer SUM suppression, necessity/sufficiency (necessity-only marks sufficiency untested), metric_id + ablate-all top-N proxy recorded). GPU measurement wiring at Phase-6 close-out Member expansion (cluster_ref → features) · per-layer SUM suppression hook · necessity + sufficiency (top-k non-members, k disclosed, default 256/layer; ablate-all proxy top-N recorded) · behavior metric = compare-workflow output-shift (imported, metric_id recorded) · necessity-only mode w/ sufficiency marked untested · scores on circuit record + manifest

## Phase 5: Remediation + MCP + UI

### Task 5.1: Heuristic remediation + copy audit
- [x] DONE (ac6c64f: calculate_ablation relabeled method='statistical_estimate' + honest docstring; MCP get_feature_ablation redirects to validate_circuit_edges; shared causal-audit extended to 017 surfaces) `calculate_ablation` docstring + `method: "statistical_estimate"` field · FeatureDetailModal retitle + caveat · MCP `get_feature_ablation` docstring · NEW shared `test_causal_language_audit.py` ("causal" forbidden below rung 2 across 015–018 surfaces)

### Task 5.2: MCP tools
- [x] DONE (ac6c64f: validate_circuit_edges/get_validation_manifest/list_validation_manifests/reproduce_validation — rung-aware; smoke-pinned) `validate_circuit_edges`, `get_validation_manifest`, `reproduce_validation`, `run_circuit_faithfulness` (extend tools/circuits.py, same category) · rung-aware docstrings

### Task 5.3: Validation tab UI
- [ ] Scope picker (ordering + K) + config · per-edge results (ES vs threshold, status chip, manifest link) · batch banner (survival per ordering + uplift) · ManifestDrawer (payload + Reproduce + verdict) · faithfulness display hooks for 018's circuit cards

## Phase 6: Verification + acceptance

### Task 6.1: GPU integration + E2E
- [ ] Planted-obvious edge validates (forward passes asserted; repeat-deterministic) · reproduction within tolerance · one necessity run on a small circuit · E2E: validate top-K of a real seeded run → manifests → faithfulness → scores displayed · cap `0xcc/caps/miStudio_Circuit_Validation_<date>.png`

### Task 6.2: Acceptance (per instruct 007)
- [ ] FPRD §8 criteria 1–6 verified (incl. ε pin, reproduction test, copy audit) · suites green · manual sections · CLAUDE.md + PPRD row 18 status update

---

## Relevant Files

| File | Purpose |
|------|---------|
| `backend/src/services/circuit_intervention_service.py` (NEW) + hooks | suppression + ES + null |
| `backend/src/services/circuit_faithfulness_service.py` (NEW) | necessity/sufficiency |
| `backend/src/services/manifest_service.py` + `models/validation_manifest.py` (NEW) + migration | manifests + reproduce |
| `backend/src/workers/circuit_validation_tasks.py` (NEW) | GPU tasks |
| `backend/src/api/v1/endpoints/circuit_validation.py`, `circuit_faithfulness.py`, `validation_manifests.py` (NEW) | REST + WS |
| `backend/src/services/analysis_service.py` · FeatureDetailModal · MCP analysis tool | remediation |
| `backend/tests/unit/test_causal_language_audit.py` (NEW, shared) | ladder language discipline |
| `frontend/src/components/panels/CircuitsPanel.tsx` (Validation tab) + `ManifestDrawer.tsx` (NEW) | UI |
| `manual/docs/**` | docs |

## Coverage audit (instruct 007)
- Data ✅ (Ph2, migration both directions) · API ✅ (Ph2-3) · MCP ✅ (Ph5) · UI/State ✅ (Ph5) ·
  Tests ✅ (ε pin, null, reproduction, GPU integration, copy audit, E2E) · Docs ✅ (Ph5-6) ·
  Acceptance ✅ (Ph6). Security: hostile-input tests; manifests contain no secrets/paths (payload
  validator).
