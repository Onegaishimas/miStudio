# Task List: Cluster Strength Budget Model

**Document ID:** 013_FTASKS|Cluster_Strength_Model
**Version:** 1.1
**Status:** ✅ COMPLETE — implemented, 3× reviewed (28/28 fixed), validation gate PASSED (γ=0 fitted + deployed), E2E-verified (2026-07-16)
**Source:** 013_FPRD · 013_FTDD · 013_FTID · IDL-29

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Backend allocation service + endpoint | 4 tasks | ✅ Complete |
| Phase 2: MCP tool + config | 2 tasks | ✅ Complete |
| Phase 3: Frontend store + rebalance + intensity | 3 tasks | ✅ Complete |
| Phase 4: UI surfaces | 3 tasks | ✅ Complete |
| Phase 5: Empirical validation (shipping gate) | 2 tasks | ✅ EXECUTED 2026-07-16 — gate PASSED (γ=0 fitted) |
| Phase 6: Feature acceptance | 1 task | ✅ Complete (2026-07-16) |

---

## Phase 1: Backend allocation service + endpoint

### Task 1.1: Pure-math core
- [x] `cluster_allocation_service.py::compute_allocation` — FTDD §2 steps 1–7, I/O-free
- [x] Unit fixtures: identical members ⇒ B=B_dir; synthetic-orthogonal ⇒ B=B_dir·√N; N=1 canary; signed member

### Task 1.2: Edge-case table
- [x] Implement + unit-test all 14 FTDD §3 rows (missing f/s, zero sims, floor, cap, mixed-layer refusal, …)

### Task 1.3: Decoder gain
- [x] Decoder access via `load_sae` cache; orientation canary test (single member ⇒ G=1); on-demand pairwise-min-cosine for cancellation flag; `approximate` fallback (G=1) when decoder unavailable

### Task 1.4: Endpoint + schemas
- [x] `POST /steering/cluster-allocation` + `ClusterAllocationRequest/Response`; single-layer + idx-bounds validation; NO steering-mode/Celery dependency; API contract tests

## Phase 2: MCP tool + config

### Task 2.1: Constants config
- [x] `steering_cluster_constants {default, per_sae}` storage + resolution + echo in response (mirror existing settings mechanism)

### Task 2.2: MCP tool
- [x] `compute_cluster_allocation` (read-only, steering category) calling the same service; smoke test

## Phase 3: Frontend store + rebalance + intensity

### Task 3.1: Types + allocation request
- [x] `ClusterAllocation` type; `strengthSource += 'cluster'`; `SelectedFeature.pinned?`
- [x] `requestClusterAllocation` on cluster hand-off (progressive: solo baselines first, upgrade on response; stale-response guard)

### Task 3.2: Rebalance
- [x] `rebalance`/`togglePin` per FTDD step 8; property test (random edit sequences hold Σ within 0.05·N)
- [x] Presets in cluster mode exit cluster mode (documented behavior)

### Task 3.3: Intensity dial state
- [x] `intensity` λ applied ONCE in request builders; excluded (with clusterBudget) from persist partialize; unit tests

## Phase 4: UI surfaces

### Task 4.1: ClusterBudgetBar
- [x] B used/total + G + flags; amber over-budget; renders only in cluster mode

### Task 4.2: Tiles
- [x] `cluster` badge; pin icon + pinned ring; slider edits route to rebalance in cluster mode

### Task 4.3: Dial + warnings
- [x] Intensity slider beside Blended|Compare; banners for cancellation/low-cohesion/mixed-layer/approximate

## Phase 5: Empirical validation (shipping gate — BR-005)

### Task 5.1: Runbook + execution
- [x] `0xcc/docs/cluster-strength-validation.md`: 3 clusters (N=4/15/5) + low-cohesion gate + sim-vs-uniform, 48 sweep + 9 confirmation generations via async steering API (seed 42)

### Task 5.2: Acceptance + constants
- [x] Hard gate: PASSED with fitted γ=0 (Δ = +8.3% / −28.9% / −6.0%; non-degenerate at predicted B on all prompts; C4 flagged+downgraded). Round 1 (γ=1) FAILED uniformly (+~100%) — the 1/G boost overdrives; members saturate on Σ|s|
- [x] Fitted constant committed as `DEFAULT_CONSTANTS.gamma = 0.0` (formula-level property, consistent across all clusters; per-SAE override remains); results recorded in runbook; law adjusted BEFORE default-on ✓

## Phase 6: Feature acceptance

### Task 6.1: Acceptance (per instruct 007)
- [x] FPRD §8 verified: live budget bar + allocation on real clusters (subscribe-19, galaxies-5, occasional-4); sanity equalities unit-proven (identical⇒B_dir, γ-override √N); rebalance invariant property-tested + backend-parity vectors; validation recorded (gate PASSED, γ=0 fitted); downgrade path E2E'd (low-cohesion notice screenshot); allocation endpoint <1s warm
- [x] Suites green (backend 100%; frontend steering suites green — pre-existing unrelated panel failures noted); E2E + caps `miStudio_Steering_Panel-ClusterBudget_20260716.png`
- [x] CLAUDE.md + PPRD v3.7 row 14 → ✅ Complete (2026-07-16)

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

## Review iterations (goal requirement: 3× /code-review + /review, ≥10 findings)

- **Iteration 1 (multi-angle /code-review): 27 findings — ALL FIXED 2026-07-16.** Highlights:
  - *Math-critical (proven with regression tests):* normalized-vs-raw gain mismatch (G now computed on RAW
    decoder columns — hook parity); greedy fold sign-flip at n=20 (Σ|s| was 53% over budget).
  - *Backend:* mean-over-present-sims imputation; f_eff unweighted fallback; `_default_b_dir=(m+M)/2`;
    solo cap; positive-subset cancellation check; endpoint SAE-layer cross-check (400); decoder passed
    as-is (no full CPU copy); `resolve_decoder_weight` extracted + shared with the hook; MCP FLAGS CONTRACT.
  - *Frontend:* allocation mapped per-instance from REQUEST order (dup/reorder/sign-safe);
    duplicate-idx refusal; low-cohesion gate now yields `clusterNotice` and NO governing budget
    (rebalance/λ/bar cannot engage); λ applied in Compare builders too (parity with Blended);
    persist partialize strips `pinned` + demotes `cluster`→`manual`; pinned badge is click-to-unpin.
  - *Tests added:* 32 backend unit + 6 endpoint contract; frontend 63-test store suite incl. reorder-safety,
    dup refusal, λ-in-Compare, partialize, backend-parity rebalance vectors; `ClusterBudgetBar.test.tsx` (7).
- **Iteration 2 (post-fix verification):** 1 new finding — stale "pinned" badge after budget-clearing
  mutations → badge now gated on cluster mode (onTogglePin passed only when clusterBudget active). Fixed.
- **Iteration 3 (/review, 4 perspectives):** SHIP-WITH-NOTES —
  `.claude/context/sessions/review_feature013_cluster_strength_2026-07-16.md`. Ledger: 28 found / 28 fixed,
  0 open P0/P1, 2 recorded P2 debts (constants duplication; profile-budget recompute rule for 014).
  Remaining before close: empirical validation on deployed env (hard gate) + Playwright E2E.
- *Recorded debt:* allocation constants (a,b,m,M) duplicated between `steeringStrength.ts` (solo path) and
  `cluster_allocation_service.py` defaults — consolidate when constants become per-SAE-calibrated (Ph5).
