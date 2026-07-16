# Task List: Cluster Strength Budget Model

**Document ID:** 013_FTASKS|Cluster_Strength_Model
**Version:** 1.0
**Status:** Not started
**Source:** 013_FPRD · 013_FTDD · 013_FTID · IDL-29

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Backend allocation service + endpoint | 4 tasks | ⬜ Not started |
| Phase 2: MCP tool + config | 2 tasks | ⬜ Not started |
| Phase 3: Frontend store + rebalance + intensity | 3 tasks | ⬜ Not started |
| Phase 4: UI surfaces | 3 tasks | ⬜ Not started |
| Phase 5: Empirical validation (shipping gate) | 2 tasks | ⬜ Not started |
| Phase 6: Feature acceptance | 1 task | ⬜ Not started |

---

## Phase 1: Backend allocation service + endpoint

### Task 1.1: Pure-math core
- [ ] `cluster_allocation_service.py::compute_allocation` — FTDD §2 steps 1–7, I/O-free
- [ ] Unit fixtures: identical members ⇒ B=B_dir; synthetic-orthogonal ⇒ B=B_dir·√N; N=1 canary; signed member

### Task 1.2: Edge-case table
- [ ] Implement + unit-test all 14 FTDD §3 rows (missing f/s, zero sims, floor, cap, mixed-layer refusal, …)

### Task 1.3: Decoder gain
- [ ] Decoder access via `load_sae` cache; orientation canary test (single member ⇒ G=1); on-demand pairwise-min-cosine for cancellation flag; `approximate` fallback (G=1) when decoder unavailable

### Task 1.4: Endpoint + schemas
- [ ] `POST /steering/cluster-allocation` + `ClusterAllocationRequest/Response`; single-layer + idx-bounds validation; NO steering-mode/Celery dependency; API contract tests

## Phase 2: MCP tool + config

### Task 2.1: Constants config
- [ ] `steering_cluster_constants {default, per_sae}` storage + resolution + echo in response (mirror existing settings mechanism)

### Task 2.2: MCP tool
- [ ] `compute_cluster_allocation` (read-only, steering category) calling the same service; smoke test

## Phase 3: Frontend store + rebalance + intensity

### Task 3.1: Types + allocation request
- [ ] `ClusterAllocation` type; `strengthSource += 'cluster'`; `SelectedFeature.pinned?`
- [ ] `requestClusterAllocation` on cluster hand-off (progressive: solo baselines first, upgrade on response; stale-response guard)

### Task 3.2: Rebalance
- [ ] `rebalance`/`togglePin` per FTDD step 8; property test (random edit sequences hold Σ within 0.05·N)
- [ ] Presets in cluster mode exit cluster mode (documented behavior)

### Task 3.3: Intensity dial state
- [ ] `intensity` λ applied ONCE in request builders; excluded (with clusterBudget) from persist partialize; unit tests

## Phase 4: UI surfaces

### Task 4.1: ClusterBudgetBar
- [ ] B used/total + G + flags; amber over-budget; renders only in cluster mode

### Task 4.2: Tiles
- [ ] `cluster` badge; pin icon + pinned ring; slider edits route to rebalance in cluster mode

### Task 4.3: Dial + warnings
- [ ] Intensity slider beside Blended|Compare; banners for cancellation/low-cohesion/mixed-layer/approximate

## Phase 5: Empirical validation (shipping gate — BR-005)

### Task 5.1: Runbook + execution
- [ ] `0xcc/docs/cluster-strength-validation.md`: ≥3 coherent clusters (vary N) + 1 low-cohesion + sim-vs-uniform comparison, via MCP sweeps around predicted B

### Task 5.2: Acceptance + constants
- [ ] Hard gate: best empirical total within ±30% of predicted B AND non-degenerate at predicted; low-cohesion correctly flagged
- [ ] Write fitted constants to `per_sae.<sae_id>`; record results; adjust law/constants on failure BEFORE default-on

## Phase 6: Feature acceptance

### Task 6.1: Acceptance (per instruct 007)
- [ ] Verify FPRD §8 criteria 1–6 (first-run usability on ≥2 live clusters; sanity equalities unit-proven; rebalance invariant; validation recorded; downgrade path visible; p50 < 500 ms warm)
- [ ] Full suites green (pytest, vitest, type-check, build); E2E + caps screenshot
- [ ] Update CLAUDE.md inventory + statuses (PPRD row 14 → Complete)

---

## Relevant Files

| File | Purpose |
|------|---------|
| `backend/src/services/cluster_allocation_service.py` (+tests, NEW) | formula core |
| `backend/src/schemas/steering.py` | allocation schemas |
| `backend/src/api/v1/endpoints/steering.py` | endpoint |
| `backend/src/core/config.py` / settings service | constants namespace |
| `backend/src/mcp_server/tools/steering.py` | MCP tool |
| `frontend/src/stores/steeringStore.ts` (+test) | request/rebalance/intensity/pins |
| `frontend/src/types/steering.ts`, `utils/steeringStrength.ts` | types; solo path untouched |
| `frontend/src/components/steering/ClusterBudgetBar.tsx` (NEW), `SelectedFeatureCard.tsx`, `SteeringPanel.tsx`, `FeatureSelector.tsx` | UI |
| `0xcc/docs/cluster-strength-validation.md` (NEW) | validation runbook + results |

## Coverage audit (instruct 007)
- API ✅ (Ph1) · Data/config ✅ (Ph2) · State ✅ (Ph3) · UI ✅ (Ph4) · Tests ✅ (every phase + property test) ·
  Docs ✅ (Ph5 runbook; manual page updates deferred to 014 which documents the full cluster workflow) ·
  Acceptance ✅ (Ph6). No DB migration needed (config-only persistence).
