# Feature PRD: Clusters UX & Trustworthy Blended Results

**Document ID:** 012_FPRD|Clusters_UX
**Version:** 1.0
**Status:** Planned
**Related:** BRD-MIS-CLUSTERS-001 (BR-001, BR-002, BR-003, BR-011) · 000_PPRD|miStudio §3.13 · 000_PADR|miStudio IDL-28 · builds on 010 (Feature Groups) and 011 (Steering UX)

---

## 1. Overview

### 1.1 Purpose
Establish **Clusters** as the product's primary steering primitive in name and in trust: rename the
user-facing "Feature Groups" concept to "Clusters," carry a cluster's identity into steering, label
combined ("Blended") results by the cluster rather than a lone feature index, and make it verifiable that
every member of a steered cluster actually contributed.

### 1.2 User Problem
1. "Feature Groups" undersells the concept — users think of these co-firing sets as the unit of *meaning*
   they steer with, not an organizational bucket.
2. When steering a cluster in Blended mode, the result box is titled with the label/index of the **single
   top feature in the list** (`Feature #10091`-style). The product owner's words: *"people won't trust
   that."* There is no visible evidence that all 13 (or 16, or 20) selected members were combined.
3. The term "Clusters" collides with an existing per-feature NLP analysis section titled "Semantic
   Clusters" — without disambiguation the rename would create new confusion.

### 1.3 Solution
- UI-wide terminology change (copy only; zero backend/API renames).
- The Groups→Steering hand-off carries `{group_id, display_token}` when the selection comes from one
  cluster; steering results are titled by the cluster with a defined fallback chain.
- The combined-result card surfaces the applied-features summary the backend already returns
  (`features_applied` with per-member strengths) — trust by inspection.
- The NLP "Semantic Clusters" section is relabeled "Semantic Token Clusters (analysis)".

## 2. User Stories

- **US-1:** As a researcher, I see "Clusters" in the sidebar, panel titles, and copy everywhere the product
  previously said "Feature Groups," so the steering primitive has one clear name.
- **US-2:** As a researcher who selected 13 members of the cluster *"fear"* and ran Blended steering, I see
  each result box titled by the cluster (e.g. *fear — Blended (13 features)*), not "Feature #10091".
- **US-3:** As a skeptical user, I can expand an applied-features summary on any Blended result and see
  every member with its applied strength, confirming the combination actually happened.
- **US-4:** As a user reading a feature's NLP analysis, I can tell "Semantic Token Clusters (analysis)"
  apart from steering Clusters.
- **US-5:** As a user who hand-picked features from several different clusters, my results honestly say
  "Blended (N features)" — no cluster identity is falsely claimed.

## 3. Functional Requirements

### 3.1 Terminology (BR-001)
1. All user-visible copy replaces "Feature Group(s)"/"group(s)" (in the grouping context) with
   "Cluster(s)": sidebar nav label, panel title/intro, list controls ("N clusters", sort/filter labels,
   empty/loading states), index banner ("cluster index"), buttons.
2. Manual (Docusaurus) pages update titles/copy/cross-links; route slugs MAY remain (redirect optional).
3. Internal identifiers are NOT renamed: `feature_groups` tables, `feature-groups` REST paths, MCP tool
   names, store/type/component names. MCP tool *descriptions* may mention "clusters" for agent clarity.
4. `NLPAnalysisView` "Semantic Clusters" section → "Semantic Token Clusters (analysis)".

### 3.2 Cluster identity through the hand-off (BR-011 support)
5. When "Steer selected" fires and **every selected member belongs to the currently expanded cluster**, the
   hand-off passes cluster context `{group_id, display_token, member_count_selected}` to the steering store.
6. Mixed-cluster or manual selections pass no cluster context.
7. Cluster context is cleared when the selection is cleared, the SAE changes, or features are added from
   outside the cluster (context becomes stale → drop it, never mislabel).

### 3.3 Trustworthy result labels (BR-003)
8. Combined-result titles use the fallback chain: authored profile name (Feature 014, when present) →
   cluster `display_token` → "Blended (N features)". Never a single member's label/index.
9. Batch-mode prompt boxes use the same chain (each prompt's result is titled by the cluster).
10. Compare-mode per-feature outputs keep per-feature labels (they ARE per-feature — correct today).

### 3.4 Verifiable combination (BR-002)
11. Every Blended result card exposes an expandable "Applied features (N)" summary listing each member's
    index, label, and applied strength, sourced from `CombinedSteeringResponse.features_applied`.
12. The summary must reflect what the backend applied (server-returned data), not the client's request.
13. An automated E2E check asserts `features_applied` length equals the selected member count for a
    Blended run (guards the owner's "are we actually combining?" concern).

## 4. User Interface

- **Sidebar:** "Clusters" nav item (icon unchanged).
- **Clusters panel:** title "Clusters"; intro copy reworded around clusters; "Steer selected (n)" unchanged
  in behavior.
- **Steering panel:** when cluster context is active, a small cluster chip (display token + member count)
  above the selected-feature tiles; result cards titled per §3.3 with the "Applied features (N)" expander
  collapsed by default.
- **Feature detail modal:** NLP section header relabeled.

## 5. API / Integration
- **No new endpoints.** `features_applied` already exists on the combined response; batch-blended results
  flow through the existing `combinedToComparison` adapter, which gains the cluster label.
- MCP: no tool signature changes; descriptions may be reworded (optional task).

## 6. Data / Types
- `featureGroupsStore`: selection hand-off extended with cluster context (see FTDD).
- `steeringStore`: new `clusterContext: {group_id, display_token, count} | null`; `combinedToComparison`
  labels from it; persists only for the session (not localStorage).
- No backend schema changes.

## 7. Dependencies
- Feature 010 (grouping data + panel), Feature 011 (Blended mode, hand-off stats carrying).
- Feature 014 will later prepend the authored-name tier to the label chain (chain is designed for it now).

## 8. Success Criteria
1. Copy audit: zero occurrences of "Feature Group(s)" in built frontend UI strings and manual page bodies.
2. Blended run from a cluster: every result box shows the cluster title; E2E asserts `features_applied`
   count == selection count.
3. Mixed selection shows "Blended (N features)" (no false cluster claim).
4. NLP section relabeled; both terms visible in the same session without ambiguity.
5. No backend/API identifier changed (diff audit).

## 9. Non-Goals
- Backend/API/data rename (future work, recorded in IDL-28).
- Authored cluster names/narratives (Feature 014).
- Strength computation changes (Feature 013).
- Manual route-slug migration (optional; links must not break either way).

## 10. Testing Requirements
- Store tests: cluster-context set/clear rules (§3.2 items 5–7), label fallback chain.
- Component test: applied-features summary renders server data.
- E2E (Playwright, LAN host): Clusters nav → select members → Steer → Blended → cluster-titled results +
  applied-features count assertion; screenshot to `0xcc/caps/`.
- Copy audit script (grep of built bundle + manual) wired as a task.

## 11. BRD Traceability

| BRD req | Covered by |
|---|---|
| BR-001 (rename) | §3.1 |
| BR-002 (verifiable combination) | §3.4 |
| BR-003 (cluster labels) | §3.3 |
| BR-011 (stats/identity through hand-off) | §3.2 (identity; stats shipped in 011) |
