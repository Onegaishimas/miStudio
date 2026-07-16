# Task List: Cluster Authoring & Portable Definitions

**Document ID:** 014_FTASKS|Cluster_Definitions
**Version:** 1.1
**Status:** Phases 1–6.1 implemented (2026-07-16); reviews + acceptance pending
**Source:** 014_FPRD · 014_FTDD · 014_FTID · IDL-30

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Data layer | 2 tasks | ✅ Complete |
| Phase 2: Service + REST + contract | 3 tasks | ✅ Complete |
| Phase 3: MCP tools | 1 task | ✅ Complete |
| Phase 4: Frontend store + UI | 4 tasks | ✅ Complete |
| Phase 5: Steering integration | 2 tasks | ✅ Complete |
| Phase 6: Docs + feature acceptance | 2 tasks | 🔄 Manual done; acceptance pending deploy + reviews |

---

## Phase 1: Data layer

### Task 1.1: Model + migration
- [x] `models/cluster_profile.py` per FTDD §1 (JSONB members/budget, soft `source_group_id`, FK sae RESTRICT)
- [x] Alembic migration (verify single head first; upgrade+downgrade tested)

### Task 1.2: SAE-delete guard
- [x] `delete_sae` counts profiles → structured 409 (aqua-star-guard pattern); UI message

## Phase 2: Service + REST + contract

### Task 2.1: Schemas + published JSON Schema
- [x] `schemas/cluster_profile.py` (CRUD + strict `ClusterDefinitionV1`/`ClusterBundleV1`)
- [x] `docs/schemas/cluster-definition-v1.json` checked in + schema-sync test

### Task 2.2: Service
- [x] CRUD; `to_definition`/`from_definition`; compatibility matrix as pure function (unit-test all 7 rows); member-bounds validation vs bound SAE

### Task 2.3: Endpoints
- [x] CRUD + export (Content-Disposition) + export-bundle + import (JSON body, 1 MB/50 defs/20 members caps, per-item results); router registration; API tests incl. hostile-JSON

## Phase 3: MCP tools

### Task 3.1: profiles tools
- [x] `tools/profiles.py` (list/get/save/export); new `profiles` category wired through gating + env defaults (compose/k8s) + smoke test

## Phase 4: Frontend store + UI

### Task 4.1: Types/api/store
- [x] `types/clusterProfile.ts`, `api/clusterProfiles.ts`, `clusterProfilesStore` (save snapshot / load hydration / import / export)

### Task 4.2: SaveProfileDialog
- [x] Name+narrative dialog with validation; markdown preview optional

### Task 4.3: ProfilesMenu
- [x] Per-SAE list, load/export/delete/import, `imported`/`unbound` badges, narrative popover (react-markdown)

### Task 4.4: Clusters-panel action
- [x] "Save as profile…" on expanded cluster (013 allocation → pre-filled dialog)

## Phase 5: Steering integration

### Task 5.1: Load hydration
- [x] `loadIntoSteering`: explicit strengths (bypass auto-baseline), pins/intensity/budget snapshot, `activeProfile` set last, `isHydratingProfile` guard (no 013 allocation request)

### Task 5.2: Label tier 1 + save affordance
- [x] `blendedTitle` consumes activeProfile name; "Save cluster profile" visibility rules; store tests

## Phase 6: Docs + feature acceptance

### Task 6.1: Manual
- [x] Clusters manual page += profiles/export/import section + schema reference + MILLM-bound-contract note; mcp-server.md += profiles category

### Task 6.2: Acceptance (per instruct 007)
- [ ] Verify FPRD §8 criteria 1–6 (recompute-survival, round-trip equality, block/warn matrix, E2E titled run, schema validation + no secrets/paths, MCP end-to-end)
- [ ] Full suites green; E2E + caps screenshot
- [ ] Update CLAUDE.md inventory + statuses (PPRD row 15 → Complete) — closes the BRD-MIS-CLUSTERS-001 increment; follow-on BRD (MILLM import / unified MCP / Open WebUI) may start

---

## Relevant Files

| File | Purpose |
|------|---------|
| `backend/src/models/cluster_profile.py` + migration (NEW) | profile entity |
| `backend/src/schemas/cluster_profile.py` (NEW) · `docs/schemas/cluster-definition-v1.json` (NEW) | contract |
| `backend/src/services/cluster_profile_service.py` (+tests, NEW) | CRUD/serialize/compat |
| `backend/src/api/v1/endpoints/cluster_profiles.py` (NEW) | REST |
| `backend/src/services/sae_service.py` (delete guard) | RESTRICT UX |
| `backend/src/mcp_server/tools/profiles.py` (NEW) + server gating | MCP |
| `frontend/src/{types,api,stores}/clusterProfile*` (NEW) | client layer |
| `frontend/src/components/steering/SaveProfileDialog.tsx`, `ProfilesMenu.tsx` (NEW) | UI |
| `frontend/src/components/panels/SteeringPanel.tsx`, `FeatureGroupsPanel.tsx`, `stores/steeringStore.ts` | integration |
| `manual/docs/**` | docs |

## Coverage audit (instruct 007)
- Data ✅ (Ph1, migration both directions) · API ✅ (Ph2) · MCP ✅ (Ph3) · UI/State ✅ (Ph4-5) ·
  Tests ✅ (matrix, round-trip property, recompute-survival integration, E2E) · Docs ✅ (Ph6) ·
  Acceptance ✅ (Ph6). Security: import caps + no secrets/paths in exports (Ph2 tests).
