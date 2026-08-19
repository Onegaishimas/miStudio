# Technical Implementation Document: Clusters UX & Trustworthy Blended Results

**Document ID:** 012_FTID|Clusters_UX
**Version:** 1.0
**Status:** Planned
**Related:** 012_FPRD · 012_FTDD · IDL-28

---

## 1. Implementation Order

1. Rename copy (frontend) + NLP relabel — zero-risk, independently commitable.
2. Cluster context: steeringStore field + clear rules → hand-off set in FeatureGroupsPanel.
3. Label chain in `combinedToComparison` + `blendedTitle` helper.
4. `AppliedFeaturesSummary` component + wire into result cards.
5. Manual copy + copy-audit script.
6. Tests + E2E + caps screenshot.

## 2. File-by-file

### 2.1 `frontend/src/components/layout/Sidebar.tsx`
- `:30` `label: 'Feature Groups'` → `'Clusters'`. Do NOT touch `id: 'feature-groups'` or `ActivePanel`.

### 2.2 `frontend/src/components/panels/FeatureGroupsPanel.tsx`
- `:104` `<h1>Feature Groups</h1>` → `Clusters`; `:117-121` intro copy ("Clusters of features that fire…").
- `handleSteerSelected` (~`:59-93`): after `selectSAE(sae)` and before the addFeature loop, compute
  membership: `const allInGroup = groupDetail && [...selection.keys()].every(id => groupDetail.members.some(m => m.feature_id === id))`.
  After a successful loop (`added === selection.size`), call the store's context setter with
  `{group_id: groupDetail.group_id, display_token: groupDetail.display_token, selected_count: added}` when
  `allInGroup`, else ensure context is null. NOTE: `selectSAE` clears prior selection state — set context
  AFTER the loop, not before.

### 2.3 `frontend/src/components/featureGroups/GroupList.tsx`
- `:64` "{groupsTotal} groups" → "clusters"; `:71,75` loading/empty strings; sort label copy `:52-60`.

### 2.4 `frontend/src/components/featureGroups/ComputeIndexBanner.tsx`
- `:21,39,56,58-59,65,75,83` — "grouping index"/"groups" strings → "cluster index"/"clusters".

### 2.5 `frontend/src/components/features/NLPAnalysisView.tsx`
- `:384` section title → `Semantic Token Clusters (analysis)` (id `clusters` at `:382` stays).

### 2.6 `frontend/src/stores/steeringStore.ts`
- Add `clusterContext: ClusterContext | null` (NOT in the persist partialize — check the `persist` config's
  partialize list and exclude it).
- Clear in: `selectSAE`, `clearFeatures`, `addFeature` (any call when context non-null), `removeFeature`.
- `blendedTitle(ctx, n, profileName?)` helper near `combinedToComparison` (~`:240` region).
- `combinedToComparison`: title from `blendedTitle(get().clusterContext, c.features_applied.length)`;
  attach `applied_features: c.features_applied` to the adapted response.
- `generateCombined` single-prompt path: `combinedResults` rendering in SteeringPanel uses the same title
  (pass through response or read context at render).

### 2.7 `frontend/src/types/steering.ts`
- `SteeringComparisonResponse` += `applied_features?: CombinedFeatureApplied[]` (optional, frontend-only
  enrichment; document that the server compare endpoint never returns it).
- Export `ClusterContext`.

### 2.8 `frontend/src/components/steering/AppliedFeaturesSummary.tsx` (NEW)
- Props `{applied: CombinedFeatureApplied[]}`; collapsed header `Applied features (N)`; expanded chip list
  `#idx label @ +strength` using the existing FEATURE_COLORS classes by member color.

### 2.9 `frontend/src/components/steering/ComparisonResults.tsx`
- Batch result card (`renderBatchResult` `:509+`): render `AppliedFeaturesSummary` when
  `result.comparison?.applied_features` present.
- Steered-output title paths (`:586-591`, `:718`) untouched for compare mode; the blended path's title
  already arrives via the adapter's synthetic `feature_config.label` — verify it uses `blendedTitle` output.

### 2.10 `frontend/src/components/panels/SteeringPanel.tsx`
- Cluster chip above tiles when `clusterContext` non-null: `display_token · N members`.
- `combinedResults` card (`:768,785` region): title via `blendedTitle`, add `AppliedFeaturesSummary`.

### 2.11 Manual (`manual/docs/`)
- `core-workflow/feature-groups.md`: front-matter title, H1, body copy → Clusters (keep slug).
- Cross-link copy in `reference/api/features-labeling.md:46-48`, `advanced/mcp-server.md:9,96`,
  `advanced/mcp-agent-instructions.md:67`, `reference/websocket-channels.md:34`.

### 2.12 `scripts/copy-audit-clusters.sh` (NEW, dev-only)
- Greps `dist/assets/*.js` (post-build) and `manual/docs/**/*.md` for `Feature Group`; exits non-zero on hits
  (allowlist: API reference pages that document the literal `feature-groups` route strings).

## 3. Pitfalls

- `steeringStore` uses `persist` — adding `clusterContext` without excluding it from partialize would
  resurrect stale context across reloads. Exclude it.
- `selectSAE` in `handleSteerSelected` clears selection state — read `groupDetail`/selection BEFORE, set
  context AFTER the loop.
- `addFeature` returning `false` (cap) mid-loop → `added < selection.size` → do NOT set context.
- The batch path adapts combined→comparison; `applied_features` must ride the adapter or batch cards lose
  the expander.
- Don't rename the manual slug — 3 pages cross-link `/core-workflow/feature-groups`.
- MCP tool descriptions live server-side; rewording them is optional and must not change tool names.

## 4. Testing

- `steeringStore.test.ts`: context set on clean single-cluster hand-off; cleared on addFeature/removeFeature/
  selectSAE/clearFeatures; `blendedTitle` chain (profileName > token > generic); adapter attaches
  `applied_features` + titled variation.
- `AppliedFeaturesSummary.test.tsx`: renders N chips from server data.
- E2E: Clusters nav label; hand-off → cluster chip; Blended run → titled results; expander count ==
  selection count; mixed selection → generic title. Screenshot `0xcc/caps/miStudio_Clusters_UX_<date>.png`.
- Copy audit script green.
