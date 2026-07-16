# Task List: Clusters UX & Trustworthy Blended Results

**Document ID:** 012_FTASKS|Clusters_UX
**Version:** 1.1
**Status:** Implemented + 3 review iterations done — E2E/acceptance pending deploy (2026-07-16)
**Source:** 012_FPRD|Clusters_UX · 012_FTDD · 012_FTID · IDL-28

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Rename copy | 3 tasks | ✅ Complete |
| Phase 2: Cluster context plumbing | 3 tasks | ✅ Complete |
| Phase 3: Labels + verification surface | 3 tasks | ✅ Complete (E2E hook pending deploy) |
| Phase 4: Manual + audit | 2 tasks | ✅ Complete |
| Phase 5: Feature acceptance | 1 task | 🔄 In progress (deploy + E2E + closeout) |

---

## Phase 1: Rename copy (frontend)

### Task 1.1: Core UI strings
- [x] `Sidebar.tsx:30` nav label → "Clusters" (id untouched)
- [x] `FeatureGroupsPanel.tsx` title/intro/buttons (`:104,112,117-121,124,130`)
- [x] `GroupList.tsx` counts/sort/empty/loading (`:40-101`); `ComputeIndexBanner.tsx` index strings (`:21-83`)

### Task 1.2: NLP disambiguation
- [x] `NLPAnalysisView.tsx:384` → "Semantic Token Clusters (analysis)"

### Task 1.3: Sweep for stragglers
- [x] Grep all frontend user-visible strings for `[Ff]eature [Gg]roup` / grouping-context "group(s)"; fix hits

## Phase 2: Cluster context plumbing

### Task 2.1: steeringStore field + rules
- [x] `clusterContext` (excluded from persist partialize); cleared on selectSAE/clearFeatures/addFeature/removeFeature
- [x] Store tests for every set/clear rule

### Task 2.2: Hand-off sets context
- [x] `FeatureGroupsPanel.handleSteerSelected`: membership check vs `groupDetail`; set context only when all selected ∈ group AND `added === selection.size`

### Task 2.3: Cluster chip
- [x] SteeringPanel chip (`display_token · N members`) when context active

## Phase 3: Labels + verification surface

### Task 3.1: blendedTitle chain
- [x] Helper (profileName → display_token → "Blended (N features)"); wired into `combinedToComparison` + single-prompt combined card title

### Task 3.2: AppliedFeaturesSummary
- [x] New component; adapter attaches `applied_features`; render on combined card + batch cards; type extension `SteeringComparisonResponse.applied_features?`

### Task 3.3: Verification E2E hook
- [ ] E2E assertion: `features_applied` length == selected member count on a Blended run

## Phase 4: Manual + audit

### Task 4.1: Manual copy
- [x] `feature-groups.md` (title/H1/body; slug unchanged) + cross-link copy in `features-labeling.md`, `mcp-server.md`, `mcp-agent-instructions.md`, `websocket-channels.md`

### Task 4.2: Copy-audit script
- [x] `scripts/copy-audit-clusters.sh` greps built bundle + manual; wire into acceptance

## Phase 5: Feature acceptance

### Task 5.1: Acceptance (per instruct 007)
- [ ] Verify every FPRD §8 success criterion (copy audit zero-hits; cluster-titled Blended results; mixed-selection generic title; E2E applied-count assertion; no backend identifier changed)
- [x] Full test suites green (backend pytest, frontend vitest, type-check, build) *(backend 100% green; frontend steering suites green — 98 pre-existing failures in unrelated dataset/model panels, pre-date 012/013)*
- [ ] Playwright E2E + screenshot to `0xcc/caps/`
- [ ] Update CLAUDE.md Document Inventory + statuses (FPRD/FTDD/FTID/FTASKS → ✅, PPRD row 13 → Complete)

---

## Relevant Files

| File | Purpose |
|------|---------|
| `frontend/src/components/layout/Sidebar.tsx` | nav label |
| `frontend/src/components/panels/FeatureGroupsPanel.tsx` | panel copy + hand-off context |
| `frontend/src/components/featureGroups/GroupList.tsx`, `ComputeIndexBanner.tsx` | list/banner copy |
| `frontend/src/components/features/NLPAnalysisView.tsx` | NLP relabel |
| `frontend/src/stores/steeringStore.ts` (+test) | clusterContext, blendedTitle, adapter enrichment |
| `frontend/src/types/steering.ts` | ClusterContext, applied_features |
| `frontend/src/components/steering/AppliedFeaturesSummary.tsx` (+test, NEW) | verification surface |
| `frontend/src/components/steering/ComparisonResults.tsx`, `panels/SteeringPanel.tsx` | render titles/expander/chip |
| `manual/docs/**` (5 pages) | manual copy |
| `scripts/copy-audit-clusters.sh` (NEW) | acceptance audit |

## Coverage audit (instruct 007)
- UI ✅ (Ph1-3) · State ✅ (Ph2) · API ⬜ n/a (no backend change) · Data ⬜ n/a · Tests ✅ (Ph2-3,5) ·
  Docs ✅ (Ph4) · Acceptance ✅ (Ph5). No backend/migration categories apply — UI-only feature by design (IDL-28).

## Review iterations (goal requirement: 3× /code-review + /review, ≥10 findings)

- **Iteration 1 (multi-angle /code-review):** 18 findings → all fixed (stale-title bake, batch ctx snapshot,
  combined-shape recovery guard, `.length>0` guards, audit-script content-level allowlist, …).
- **Iteration 2 (post-fix verification):** fixes verified; regression tests added
  (`steeringStore.test.ts` cluster-context suite, `AppliedFeaturesSummary.test.tsx`,
  `ComparisonResults.blendedTitle.test.tsx`, `featureGroupsStore.test.ts`).
- **Iteration 3 (/review, 4 perspectives):** 10 findings; traceability table + SHIP-WITH-NOTES gate in
  `.claude/context/sessions/review_feature012_clusters_2026-07-16.md`. Total: 28 findings fixed/recorded.
