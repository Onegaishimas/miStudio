# Technical Design Document: Clusters UX & Trustworthy Blended Results

**Document ID:** 012_FTDD|Clusters_UX
**Version:** 1.0
**Status:** Planned
**Related:** 012_FPRD|Clusters_UX · IDL-28

---

## 1. Architecture

Frontend-only feature (plus manual copy). Three concerns:

1. **Copy rename** — literal string replacement at known sites; no identifier churn.
2. **Cluster context plumbing** — a small value threaded featureGroupsStore → FeatureGroupsPanel →
   steeringStore, with strict validity rules (single-cluster provenance only).
3. **Result labeling + verification surface** — `combinedToComparison` and `ComparisonResults` consume the
   context; the applied-features expander renders `features_applied` from the server response.

## 2. Cluster context design

```ts
// steeringStore
export interface ClusterContext {
  group_id: string;
  display_token: string;     // label tier 2 (tier 1 = authored profile name, Feature 014)
  selected_count: number;
}
clusterContext: ClusterContext | null;   // NOT persisted to localStorage
```

**Set** only in `FeatureGroupsPanel.handleSteerSelected` when
`selection.size > 0 && groupDetail != null && every selected feature_id ∈ groupDetail.members`.
(`groupDetail` carries `group_id`/`display_token` — `types/featureGroups.ts:57-60`.)

**Cleared** on: `selectSAE` (already clears selection), `clearFeatures`, any `addFeature` whose
`feature_id` is not among the context members' ids at hand-off time (simplest sound rule: any addFeature
call after hand-off clears the context), `removeFeature` (context count no longer matches → clear).
Rationale: the context must never survive a mutation that could falsify it; dropping it early is always
honest (falls back to "Blended (N features)").

## 3. Label chain

Single helper in `steeringStore`:

```ts
function blendedTitle(ctx: ClusterContext | null, n: number, profileName?: string | null): string {
  if (profileName) return `${profileName} — Blended (${n} features)`;      // Feature 014 tier
  if (ctx)        return `${ctx.display_token} — Blended (${n} features)`;
  return `Blended (${n} features)`;
}
```

Consumed by `combinedToComparison` (currently hardcodes `Blended (N features)`), so single-prompt combined,
batch-blended, and recent-results all inherit it with no renderer changes to titles. Compare-mode paths
untouched.

## 4. Applied-features verification surface

- `CombinedSteeringResponse.features_applied: CombinedFeatureApplied[]` already returns
  `{feature_idx, layer, strength, label, color}` per member (backend `schemas/steering.py`).
- `combinedToComparison` currently collapses this into ONE synthetic steered variation. Design: attach the
  full array onto the adapted comparison as `applied_features` (additive optional field on the frontend
  `SteeringComparisonResponse` type — server type untouched) so `ComparisonResults` can render the expander
  on batch results too.
- New small component `AppliedFeaturesSummary` ({applied}: chips of `#idx label @ strength`, collapsed
  count header), used by both the combined result card and batch result cards.
- Verification is **server-truth**: render only what came back, never the request state.

## 5. Rename surface (exact, from inventory)

| Site | Change |
|---|---|
| `components/layout/Sidebar.tsx:30` | label `Feature Groups` → `Clusters` (id `feature-groups` stays) |
| `components/panels/FeatureGroupsPanel.tsx:104,112,117-121,124,130` | title/intro/buttons copy |
| `components/featureGroups/GroupList.tsx:40-101` | "N groups"→"N clusters", empty/loading, sort labels |
| `components/featureGroups/ComputeIndexBanner.tsx:21-83` | "grouping index"→"cluster index" strings |
| `components/features/NLPAnalysisView.tsx:384` | "Semantic Clusters"→"Semantic Token Clusters (analysis)" |
| `manual/docs/core-workflow/feature-groups.md` | title/H1/body copy → Clusters (slug may stay; if slug changes add redirects) |
| `manual/docs/reference/api/features-labeling.md:46-48`, `manual/docs/advanced/mcp-server.md:9,96`, `manual/docs/advanced/mcp-agent-instructions.md:67`, `manual/docs/reference/websocket-channels.md:34` | copy + cross-links |

NOT renamed: `feature_groups*` tables, `feature-groups` REST paths, MCP tool names, `featureGroupsStore`,
`FeatureGroupsPanel`/component filenames, `ActivePanel` id, WS channel names. (IDL-28.)

## 6. Type changes

- `steeringStore`: `clusterContext` + actions `setClusterContext/clearClusterContext` (internalized into
  existing actions per §2 rules); `combinedToComparison(c, ctx?)`.
- Frontend `SteeringComparisonResponse` += optional `applied_features?: CombinedFeatureApplied[]`.
- `featureGroupsStore`: no shape change (groupDetail already has what we need); `handleSteerSelected` reads it.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Stale/false cluster label after selection mutations | Aggressive clear rules (§2): any post-hand-off mutation clears context |
| Rename misses a string (trust in audit) | Copy-audit script greps built bundle + manual for `Feature Group` as an acceptance task |
| "Clusters" vs NLP "Semantic Clusters" confusion | Relabel NLP section in the same PR as the nav rename |
| Manual slug change breaks inbound links | Keep slug `feature-groups` in v1; only copy changes |
| `features_applied` absent on old cached results | Expander renders only when field present |
