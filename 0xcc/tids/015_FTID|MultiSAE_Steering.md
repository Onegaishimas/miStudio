# Technical Implementation Document: Multi-SAE Cross-Layer Steering

**Document ID:** 015_FTID|MultiSAE_Steering
**Version:** 1.0
**Status:** Planned
**Related:** 015_FPRD · 015_FTDD · IDL-31, IDL-35

---

## 1. Implementation Order

1. Backend request/response types (`sae_id` threading) + validation.
2. `resolve_sae_map` + `_register_steering_hooks(sae_map, …)` + unit pins.
3. Per-layer allocation endpoint (partition + reuse) + golden single-layer regression.
4. Hazard computation (weight prior, heuristic-labeled; rung≥2 edge source activates on 017/018 data).
5. Frontend types/store (`sae_id`, `layerBudgets`, within-layer rebalance).
6. UI (per-layer budget bars, hazard banner, layer/SAE chips, applied-summary grouping).
7. MCP schema updates + integration/E2E on GPU.

## 2. File-by-file

### 2.1 `backend/src/schemas/steering.py` (or the request models' home)
- `SteerFeature.sae_id: Optional[str] = None`; `AppliedFeature.sae_id: str`.
- Allocation request members gain `sae_id: Optional[str]`; response union: single-layer (unchanged 013
  shape) | multi-layer (`layers` map + `hazards` + flattened `strengths`). Keep the single-layer shape
  FIRST in the union so old clients deserialize identically.

### 2.2 `backend/src/services/steering_service.py`
- `resolve_sae_map(request) -> Dict[str, LoadedSAE]`: distinct per-feature `sae_id ?? request.sae_id`,
  each through the existing `load_sae` (cache hit when already resident). Validate
  `feature.layer == sae.layer` per member; raise structured 422 listing offenders.
- `_register_steering_hooks(model, sae_map, feature_configs)`: group by `(sae_id, layer)`; call
  `_create_steering_hook(sae_map[sid], group)` per group. **Pin in tests:** each created hook received
  the SAE whose `layer` equals the group's layer.
- `generate_combined`: replace the single `sae` local with the map; baseline path unchanged; cleanup
  clears all map entries' hooks (`_clear_all_model_hooks` already model-wide — no change).
- Per-layer allocation: `compute_multi_layer_allocation(members)` partitions by layer and calls the
  EXISTING `compute_allocation` per partition with that layer's decoder; assemble response; compute
  hazards (§2.3). Do not fork the IDL-29 math.

### 2.3 Hazards (new module `backend/src/services/steering_hazards.py`)
- `weight_prior(up_sae, up_idx, down_sae, down_idx) -> float`: cosine of `W_dec(Lᵢ)[:,i]` vs
  `W_enc(Lⱼ)[:,j]` (mind encoder orientation — reuse `resolve_decoder_weight` conventions; add
  `resolve_encoder_weight` beside it).
- `detect_hazards(members_by_layer, sae_map, circuit_edges=None)`: pairs (upstream layer < downstream
  layer); PRIMARY evidence = stored edges at rung ≥2 (when `circuit_id` provided — 018) with
  `quantified_effect` from the edge's measured ES; fallback = prior ≥ threshold (config
  `steering_hazard_prior_threshold`, default 0.5) labeled `heuristic` (IDL-35 language rules —
  copy-audit-tested); cancellation via negative validated edges or opposite-signed heuristic pairs.
  Pure function — exhaustively unit-testable without GPU.

### 2.4 `backend/src/api/v1/endpoints/steering.py`
- Thread `sae_id`s through async submit validation (the per-feature SAE must exist and be READY —
  reuse the existing SAE lookup used for the request-level id).
- `cluster-allocation` route: accept mixed layers; branch single vs multi shape.

### 2.5 MCP `backend/src/mcp_server/tools/steering.py`
- `steer_combined`/`compute_cluster_allocation`: member schema += `sae_id?`; docstrings note the
  own-layer rule + envelope. No new tools.

### 2.6 Frontend
- `types/steering.ts`: `SelectedFeature.sae_id?`, `LayerBudgets`, `Hazard`.
- `stores/steeringStore.ts`: set `sae_id` on add (selected SAE) and on load (member's); `layerBudgets`
  map (single-layer mirrors into `clusterBudget` for compat); `rebalance(layer, …)`; λ unchanged.
  **Pitfall:** the loader must keep 014's explicit-strength hydration (`isHydratingProfile`) — extend,
  don't fork.
- `components/steering/ClusterBudgetBar.tsx`: per-layer rows (layer chip + B/λ as today).
- NEW `components/steering/HazardBanner.tsx`: amber, pairs + evidence, dismiss-per-run
  (state resets on selection mutation — reuse clusterContext clearing rules).
- `SelectedFeatureCard`: the `L{n}` chip tooltip gains the SAE name.
- `AppliedFeaturesSummary`: group rows by layer.

### 2.7 Manual
- `manual/docs/core-workflow/steering.md`: "Multi-layer circuits" section (own-layer rule, per-layer
  budgets, λ, hazards); mcp-server page: updated tool schemas.

## 3. Pitfalls

- **Encoder orientation** differs across SAE formats — `resolve_encoder_weight` must normalize like
  `resolve_decoder_weight` does (community_standard vs internal orientation); test both formats.
- The allocation response union must keep old single-layer clients working — golden-test the JSON.
- steeringStore has many consumers of `clusterBudget` — keep it populated for single-layer; only
  multi-layer flows read `layerBudgets` (grep consumers before renaming anything).
- GPU integration test needs TWO real SAEs resident — use the L13/L14 8k SAEs already on the k8s host;
  keep max_new_tokens tiny (≤32).
- 422 validation must run at SUBMIT time (fast feedback), not inside the worker (which would burn a
  GPU slot to fail).

## 4. Testing

- Unit: sae-map resolution (defaults, distinct, missing SAE), layer-mismatch 422, hook/SAE pairing pin,
  per-layer partition math vs 013 single calls, hazard matrix (edge source, prior source, sign
  cancellation, threshold edge cases), encoder orientation both formats.
- Golden: single-layer allocation response byte-identical to 013 fixture.
- Integration (GPU): two-SAE combined run — assert both hooks fired (hook-side counters) and
  `features_applied[].sae_id` correct.
- Frontend: store add/load sae_id propagation; layerBudgets single/multi; within-layer rebalance;
  hazard banner render/dismiss/reset.
- E2E: load two-layer circuit (018 fixture or hand-built) → Blended → applied summary grouped by layer →
  title from circuit; screenshot `0xcc/caps/miStudio_MultiSAE_Steering_<date>.png`.
