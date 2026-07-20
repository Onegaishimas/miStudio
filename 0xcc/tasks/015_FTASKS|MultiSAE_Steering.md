# Task List: Multi-SAE Cross-Layer Steering

**Document ID:** 015_FTASKS|MultiSAE_Steering
**Version:** 1.0
**Status:** ⏳ IMPLEMENTED (backend d724dec + MCP/docs 8782643 + frontend d1de02b) — Ph1-5 done; 3-round review next; Ph6 GPU-E2E at close-out
**Source:** 015_FPRD · 015_FTDD · 015_FTID · IDL-31, IDL-35 (hazard labels) · BR-024 via CIRCUITS-002

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Request threading + validation | 2 tasks | ⏳ |
| Phase 2: Multi-SAE hooks | 1 task | ⏳ |
| Phase 3: Per-layer allocation + hazards | 2 tasks | ⏳ |
| Phase 4: Frontend | 3 tasks | ⏳ |
| Phase 5: MCP + docs | 1 task | ⏳ |
| Phase 6: Verification + acceptance | 2 tasks | ⏳ |

---

## Phase 1: Request threading + validation

### Task 1.1: Schemas
- [x] DONE `SteerFeature.sae_id?` / `AppliedFeature.sae_id`; allocation request/response union (single-layer shape preserved FIRST)

### Task 1.2: Submit-time validation
- [x] DONE (422 listing offenders at submit) Per-feature SAE existence/READY check + layer-mismatch 422 (structured, lists offenders) at the endpoint — not in the worker

## Phase 2: Multi-SAE hooks

### Task 2.1: sae_map through the service
- [x] DONE (resolve_sae_map + group-by-(sae_id,layer); single-SAE regression guarded + golden-pinned) `resolve_sae_map` (defaults, cache reuse) · `_register_steering_hooks(model, sae_map, configs)` grouped by (sae_id, layer) · `features_applied[].sae_id` from hook-time config · unit pin: each hook's SAE.layer == group layer · single-SAE paths untouched (solo/compare regression)

## Phase 3: Per-layer allocation + hazards

### Task 3.1: compute_multi_layer_allocation
- [x] DONE (partition→reuse 013 compute_allocation; byte-identical single-layer golden) Partition by layer → reuse existing `compute_allocation` per partition (NO formula fork) → assemble `layers` map + flattened strengths · formula id `freq-budget/sim-alloc/per-layer@1` · golden test: single-layer response byte-identical to 013 fixture

### Task 3.2: steering_hazards.py
- [x] DONE (validated-ES primary + weight-prior heuristic labeled; full matrix + copy audit) `resolve_encoder_weight` (both orientations tested) · `weight_prior` cosine · `detect_hazards` v2 (rung≥2 edges primary w/ quantified ES; prior fallback labeled `heuristic` per IDL-35 — copy-audit test; sign-cancellation via negative edges) — pure function, exhaustive unit matrix · NOTE: ships heuristic-labeled; validated-ES path activates on 017/018 data

## Phase 4: Frontend

### Task 4.1: Types + store
- [x] DONE `SelectedFeature.sae_id?` set on add/load · `layerBudgets` map (single-layer mirrors `clusterBudget` for existing consumers — grep first) · within-layer `rebalance(layer, …)` · λ applies to all layers · 014 hydration guard extended not forked

### Task 4.2: Budget bars + hazard banner
- [x] DONE ClusterBudgetBar per-layer rows (layer chip) · NEW HazardBanner (amber, pairs+evidence, dismiss-per-run, reset on selection mutation)

### Task 4.3: Tiles + applied summary
- [x] DONE `L{n}` chip tooltip gains SAE name · AppliedFeaturesSummary grouped by layer

## Phase 5: MCP + docs

### Task 5.1: MCP schemas + manual
- [x] DONE `steer_combined`/`compute_cluster_allocation` member schema += `sae_id?` + own-layer/envelope docstrings · manual steering page "Multi-layer circuits" section · mcp-server page schemas

## Phase 6: Verification + acceptance

### Task 6.1: GPU integration + E2E
- [ ] Two-SAE (L13+L14) combined run: both hooks verified fired, `features_applied` correct, VRAM delta measured (<200 MB) · E2E: two-layer circuit load → Blended → grouped applied summary → circuit-titled result · cap `0xcc/caps/miStudio_MultiSAE_Steering_<date>.png`

### Task 6.2: Acceptance (per instruct 007)
- [ ] FPRD §8 criteria 1–6 verified · suites green · CLAUDE.md + PPRD row 16 status update

---

## Relevant Files

| File | Purpose |
|------|---------|
| `backend/src/schemas/steering*.py` | sae_id threading, allocation union |
| `backend/src/services/steering_service.py` | sae_map, hooks, per-layer allocation |
| `backend/src/services/steering_hazards.py` (NEW) | weight prior + hazard detection |
| `backend/src/api/v1/endpoints/steering.py` | submit validation, allocation route |
| `backend/src/mcp_server/tools/steering.py` | MCP schema updates |
| `frontend/src/types/steering.ts` · `stores/steeringStore.ts` | sae_id, layerBudgets, rebalance |
| `frontend/src/components/steering/ClusterBudgetBar.tsx`, `HazardBanner.tsx` (NEW), `SelectedFeatureCard.tsx`, `AppliedFeaturesSummary.tsx` | UI |
| `manual/docs/**` | docs |

## Coverage audit (instruct 007)
- API ✅ (Ph1) · Service ✅ (Ph2-3) · UI/State ✅ (Ph4) · MCP+Docs ✅ (Ph5) ·
  Tests ✅ (unit matrix, golden regression, GPU integration, E2E) · Acceptance ✅ (Ph6).
  Data: N/A — no migrations (request-level feature). Security: N/A — no new surfaces beyond
  existing authenticated steering routes (submit validation tightens input).
