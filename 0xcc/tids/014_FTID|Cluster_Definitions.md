# Technical Implementation Document: Cluster Authoring & Portable Definitions

**Document ID:** 014_FTID|Cluster_Definitions
**Version:** 1.0
**Status:** Planned
**Related:** 014_FPRD · 014_FTDD · IDL-30

---

## 1. Implementation Order

1. Model + migration + schemas (incl. Definition/Bundle validators + JSON Schema file).
2. Service + REST endpoints + serializer round-trip tests.
3. MCP profile tools.
4. Frontend store/api/types.
5. UI (save dialog, profiles list, import picker, Clusters-panel action).
6. Steering integration (load hydration, label tier 1).
7. Integration/E2E (recompute-survival, round-trip) + manual page + caps.

## 2. File-by-file

### 2.1 `backend/src/models/cluster_profile.py` (NEW) + Alembic migration
- Table per FTDD §1. Follow house model conventions (`models/feature_grouping.py` as the pattern; id
  prefix `clp_`). Single-head migration — check `alembic heads` first (multi-head merge history exists).
- FK `sae_id → saes.id` with RESTRICT; add relationship so SAE delete path can surface profile count.

### 2.2 `backend/src/schemas/cluster_profile.py` (NEW)
- `ClusterProfileCreate/Update/Out`; `ClusterDefinitionV1` + `ClusterBundleV1` (strict: kind literal,
  schema_version literal "1", members 1..20, strengths |≤300|, name ≤120, narrative ≤10k, bundle ≤50,
  file ≤1 MB enforced at endpoint).
- Also emit `docs/schemas/cluster-definition-v1.json` (generated from the Pydantic model via a small
  script or checked-in static — keep in sync task in tests: model.schema() ≈ file).

### 2.3 `backend/src/services/cluster_profile_service.py` (NEW)
- CRUD; `to_definition(profile)` / `from_definition(defn, bind_sae_id|None)`; compatibility matrix
  (FTDD §3) as a pure function returning `{action: bind|warn_bind|block|unbound, warnings[]}` —
  unit-test the matrix exhaustively.
- Member-bounds validation vs bound SAE `n_features` on save/load/bind.

### 2.4 `backend/src/api/v1/endpoints/cluster_profiles.py` (NEW) + router registration
- Routes per FPRD §5. Export sets `Content-Disposition` filename `<name>.cluster.json` /
  `<date>.cluster-bundle.json`. Import accepts JSON body (frontend reads the file client-side —
  avoids multipart; cap 1 MB). Error style per house pattern (structured detail).

### 2.5 `backend/src/mcp_server/tools/profiles.py` (NEW)
- `list_cluster_profiles(sae_id?)`, `get_cluster_profile(id)`, `save_cluster_profile(...)`,
  `export_cluster_profile(id)` → definition JSON as tool result. Category: new `profiles` (read/write
  split consistent with existing category gating in `mcp_server/server.py`); register + docs.

### 2.6 `frontend/src/types/clusterProfile.ts` · `api/clusterProfiles.ts` · `stores/clusterProfilesStore.ts` (NEW)
- Store: `profiles[]`, `loading`, `saveFromSteering()` (snapshot: selectedFeatures + pins + sign +
  intensity + clusterBudget + clusterContext/display_token), `loadIntoSteering(id)` (hydrate steeringStore:
  clearFeatures → addFeature loop with explicit strengths (manual add path — strengths are authoritative,
  NOT auto-baseline) → set pins/intensity/clusterBudget snapshot → set `activeProfile {id,name}`),
  `importFiles(File[])`, `exportOne/exportBundle`.
- Pitfall: `addFeature` computes baselines when strength omitted — ALWAYS pass explicit `strength` on load.

### 2.7 `frontend/src/stores/steeringStore.ts`
- `activeProfile: {id, name} | null` — feeds `blendedTitle` tier 1 (012); cleared by the same mutation
  rules as `clusterContext` (any selection mutation) AND on profile load replaced.
- Expose a `snapshotClusterConfig()` selector for the profiles store (avoid reaching into internals).

### 2.8 UI
- `components/steering/SaveProfileDialog.tsx` (NEW): name (required) + narrative (markdown textarea);
  validation; save → toast.
- `components/steering/ProfilesMenu.tsx` (NEW): per-SAE list (load/export/delete/import button); badges
  `imported`/`unbound`; markdown popover for narrative (react-markdown + remark-gfm, house pattern).
- `SteeringPanel.tsx`: "Save cluster profile" (visible when clusterBudget or activeProfile set) +
  ProfilesMenu near Recent.
- `FeatureGroupsPanel.tsx`: row action "Save as profile…" on expanded cluster → requests 013 allocation →
  opens SaveProfileDialog pre-filled with display_token.

### 2.9 Manual
- Extend `manual/docs/core-workflow/feature-groups.md` (Clusters page after 012) with a "Cluster profiles
  & portable definitions" section + the JSON schema reference; note the format is the MILLM-bound
  interchange contract (future).

## 3. Pitfalls

- **Load must bypass auto-baseline**: explicit strengths on every addFeature; then set pins/intensity;
  ordering matters (context/profile set LAST so mutation-clearing rules don't wipe it — mirror 012's
  "set after loop" pattern).
- 013's `requestClusterAllocation` must NOT fire on profile load (loaded strengths are authoritative) —
  guard: skip when hydrating (`isHydratingProfile` flag).
- Export from STORED profile only (save-then-export UX) — never serialize live UI state directly.
- Migration: verify single alembic head before adding; test both upgrade + downgrade.
- Bundle import partial failure: import valid definitions, report per-item warnings/errors — no
  all-or-nothing surprise (unless the file itself is invalid).
- RESTRICT FK will break existing SAE-delete flows — update `delete_sae` service to count profiles and
  return a structured 409 (mirrors the aqua-star 409 guard pattern from Feature 010).
- MCP category gating: adding a `profiles` category requires updating `MCP_TOOL_CATEGORIES` parsing +
  k8s/compose env defaults + manual page (mcp-server.md).

## 4. Testing

- Serializer property/round-trip: profile → definition → profile identical (incl. 0.1-grain strengths,
  signs, pins, intensity); bundle of 3.
- Compatibility matrix: unit-test every FTDD §3 row.
- Integration: create profile → `compute_feature_groups(force)` recompute → profile intact & loadable;
  SAE delete blocked with profile count.
- API: size caps, hostile JSON (extra fields ignored/stripped, path-like strings never touched as paths).
- Frontend: save snapshot completeness; load hydration (strengths authoritative, no allocation request);
  activeProfile title tier; import warning toasts.
- E2E: tune → save → recompute → load → Blended (title = profile name) → export → re-import → identical
  config; screenshot `0xcc/caps/miStudio_Cluster_Profiles_<date>.png`.
- Schema-sync test: Pydantic-generated schema matches `docs/schemas/cluster-definition-v1.json`.
