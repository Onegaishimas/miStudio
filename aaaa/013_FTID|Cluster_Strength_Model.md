# Technical Implementation Document: Cluster Strength Budget Model

**Document ID:** 013_FTID|Cluster_Strength_Model
**Version:** 1.0
**Status:** Planned
**Related:** 013_FPRD · 013_FTDD · IDL-29

---

## 1. Implementation Order

1. Backend service + schemas + endpoint (formula steps 1–7) with unit fixtures.
2. Config constants namespace + resolution.
3. MCP tool `compute_cluster_allocation`.
4. Frontend types + store (allocation request, rebalance, intensity, pins).
5. UI (budget bar, dial, pins, badges, warnings).
6. Validation protocol runbook + execution + constants commit.
7. Tests/E2E/caps.

## 2. File-by-file

### 2.1 `backend/src/services/cluster_allocation_service.py` (NEW)
- Pure-math core `compute_allocation(members, decoder: Optional[Tensor], constants) -> AllocationResult`
  implementing FTDD §2 steps 1–7 + §3 edge table. Keep math free of I/O for unit testing.
- Decoder access: reuse `steering_service.load_sae` cache (`_loaded_saes`) — call via the service singleton
  or a shared loader util; slice columns `[feature_idx]`, fp32, `torch.linalg.vector_norm`. NO model load.
- Pairwise-min-cosine only computed when the cancellation flag trips (N² only on demand).

### 2.2 `backend/src/schemas/steering.py`
- `ClusterAllocationMember {feature_idx: int, layer: int, similarity: float|None, activation_frequency:
  float|None, sign: Literal[1,-1] = 1}`; request `{sae_id, members: list[...] (1..20)}`;
  response `{B, B_dir, G, f_eff, weights, strengths, flags: list[str], cancellation_pair, constants_used,
  formula_id, approximate}`.

### 2.3 `backend/src/api/v1/endpoints/steering.py`
- `POST /cluster-allocation` — resolve SAE record (path/layer/n_features like the submit endpoints do),
  validate single-layer + idx < n_features, call service. No steering-mode requirement, no Celery — direct
  (CPU-fast). Auth/pattern per sibling endpoints.

### 2.4 `backend/src/core/config.py` (+ settings service if DB-backed)
- `steering_cluster_constants` default per FTDD §2; resolution: `per_sae[sae_id] or default`. Echo in
  response. (Check how existing labeling defaults are stored — mirror that mechanism.)

### 2.5 `backend/src/mcp_server/tools/steering.py`
- `compute_cluster_allocation(sae_id, members)` → calls the service; read-only category (no approval).

### 2.6 `frontend/src/types/steering.ts`
- `ClusterAllocation` mirror of response; `SelectedFeature += pinned?: boolean`;
  `StrengthSource` (utils) += `'cluster'`.

### 2.7 `frontend/src/stores/steeringStore.ts`
- State: `clusterBudget: ClusterAllocation | null`, `intensity: number (1)`.
- `requestClusterAllocation()`: fires when 012's cluster context is set post-hand-off; applies solo
  baselines immediately, upgrades strengths + `strengthSource:'cluster'` when the response lands
  (progressive; guard: ignore stale responses if selection changed — compare member sets).
- `rebalance(instanceId, newValue)`: FTDD step 8 (pin edited member; redistribute; rounding residual to
  largest unpinned). Replaces `updateFeatureStrength` ONLY in cluster mode; solo mode keeps direct set.
- `setIntensity(λ)`: multiplies at request-build time (`generateCombined`/batch build `selected_features`
  with `strength: λ·s` — apply λ in ONE place, the request builder, so tiles show pre-λ values).
- `togglePin(instanceId)`; membership/sign changes ⇒ re-request allocation.

### 2.8 UI components
- `frontend/src/components/steering/ClusterBudgetBar.tsx` (NEW): B used/total, G, flags, amber over-budget.
- `SelectedFeatureCard.tsx`: `cluster` badge (like auto/default), pin icon (Pin/PinOff lucide), pinned ring.
- `SteeringPanel.tsx`: intensity dial next to Blended|Compare toggle (visible only with clusterBudget);
  warning banners from flags.
- `FeatureSelector.tsx`: "Auto" preset in cluster mode re-requests allocation (not solo baselines).

### 2.9 `0xcc/docs/cluster-strength-validation.md` (NEW, runbook + results)
- Protocol per FTDD §5; table of predicted vs empirical; fitted constants; date + SAE id.

## 3. Pitfalls

- **Apply λ once** (request builder), or tiles/rebalance/export triple-scale. Pins stored pre-λ.
- Stale allocation race: response arriving after user changed selection must be dropped (compare sorted
  member idx lists).
- `steeringStrength.ts` MUST remain untouched for solo path; N=1 cluster short-circuits BEFORE the endpoint
  call (frontend routes to solo path).
- Decoder slice: ensure decoder orientation (d_model × n_features vs transposed) matches `load_sae_auto_detect`
  output — unit-test G=1 for a single member as the orientation canary.
- Rounding: strengths round to 0.1 but Σ must stay within 0.05·N — fold residual, don't renormalize.
- Persist: exclude `clusterBudget`/`intensity`? Keep intensity persisted per session only — exclude both
  from partialize (stale budget across reload is a lie).
- The endpoint must not require steering mode (no `_ensure_steering_worker_running`) — it's CPU math.
- Existing `applyStrengthPreset` (Subtle/Moderate/Strong) in cluster mode: presets set uniform values ⇒
  they EXIT cluster mode (clear clusterBudget + badges revert to manual). Document in UI copy.

## 4. Testing

- Backend unit (`test_cluster_allocation.py`): hand-computed fixtures — identical members (G=1, B=B_dir),
  orthogonal synthetic decoder (B=B_dir√N), every edge-table row, constants override, mixed-layer refusal,
  cap binding, floor, signed member.
- Backend API: contract test + n_features bound + approximate path (mock loader failure).
- Frontend: rebalance property test (100 random pin/edit sequences ⇒ Σ|strength| within 0.05·N of B);
  intensity single-application; stale-response guard; N=1 solo routing.
- E2E: hand-off → budget bar → edit → rebalance → dial 0.5 → Blended run completes; caps screenshot.
- MCP: `compute_cluster_allocation` smoke via server tool list + one call.
