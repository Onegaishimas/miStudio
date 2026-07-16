# Feature PRD: Cluster Authoring & Portable Definitions

**Document ID:** 014_FPRD|Cluster_Definitions
**Version:** 1.0
**Status:** Planned
**Related:** BRD-MIS-CLUSTERS-001 (BR-007, BR-008, BR-009, BR-010) · 000_PPRD §3.15 · IDL-30 · builds on 012 (cluster identity) and 013 (tuned allocations)

---

## 1. Overview

### 1.1 Purpose
Turn a tuned cluster into a **first-class, mobile artifact**: name it, give it a narrative, persist its
tuned member strengths as a *profile*, and export/import it as standardized, versioned, consumer-neutral
JSON — the interchange contract the future MILLM / unified-MCP / Open WebUI arc consumes (that arc is a
separate BRD; nothing here touches MILLM).

### 1.2 User Problem
Hours of workbench tuning currently evaporate: there is no way to capture "this cluster, these strengths,
this meaning" — the grouping index is recomputable (recompute rebuilds `feature_groups`), groups have no
name/description fields, and nothing is exportable. The owner's vision — trade/share cluster definitions,
eventually dial them in live chat — requires a durable, portable artifact.

### 1.3 Solution
- **Cluster profiles**: a new persistent entity (name, narrative, members with tuned strengths, budget +
  formula parameters, intensity) decoupled from the recomputable grouping tables.
- **Portable JSON**: `mistudio.cluster-definition/v1` (single) and `mistudio.cluster-bundle/v1` (multi),
  self-describing with model/SAE references and provenance.
- **Round-trip fidelity**: import reproduces the identical steering configuration.

## 2. User Stories
- **US-1:** After tuning the *fear* cluster, I click "Save cluster profile," name it, write a short
  narrative of its meaning, and it persists — surviving grouping-index recomputes and restarts.
- **US-2:** I export the profile to a JSON file (or several profiles to one bundle) to keep, share, or
  later import into MILLM.
- **US-3:** I import a definition file on another miStudio instance with the same SAE and get the identical
  steering setup (members, strengths, intensity) in one click.
- **US-4:** Importing against a mismatched SAE warns me (model-name mismatch) or blocks me with a clear
  reason (n_features mismatch).
- **US-5:** On the Steering panel, I load a saved profile from a list and steer immediately; result titles
  use the profile's name (completing 012's label chain tier 1).

## 3. Functional Requirements

### 3.1 Cluster profiles (BR-007)
1. Create a profile from the current cluster steering configuration (members + strengths + pins + sign +
   intensity + budget/allocation metadata from 013) with **name** (required, ≤120 chars) and **narrative**
   (optional markdown, ≤10k chars).
2. Profiles persist in a new `cluster_profiles` table decoupled from grouping tables; recompute of the
   grouping index MUST NOT delete or alter profiles (`source_group_id` is a soft reference).
3. Profiles list: browse, search by name/token, open (loads into steering), update (re-save over), delete.
4. A profile records its source context: `sae_id`, `model_id`, `extraction_id?`, `display_token`,
   `formula_id` + constants + `{B, B_dir, G, f_eff}` snapshot (from 013's allocation response).

### 3.2 Export (BR-008)
5. Export one profile → `mistudio.cluster-definition/v1` JSON file (download).
6. Export selected/multiple profiles → `mistudio.cluster-bundle/v1` (array of definitions, one file).
7. The definition is **self-describing**: schema version/kind, full member records (idx, label, similarity,
   activation_frequency, max_activation, strength, sign, pinned), budget + formula id + constants,
   intensity + range, model refs (HF id + miStudio model_id), SAE refs (id, layer, hook, n_features,
   d_model, source repo/path hint), provenance (created_at, miStudio version, profile name/narrative).
8. Export never includes secrets, tokens, or absolute local paths.

### 3.3 Import (BR-009, BR-010)
9. Import accepts definition or bundle files; validates kind + schema version (unknown major ⇒ reject with
   message).
10. Compatibility: SAE `n_features` mismatch ⇒ **block** (indices meaningless); model-name or SAE-id
    mismatch ⇒ **warn + allow** (user selects the local SAE to bind to); missing local SAE ⇒ import as
    profile only (steerable once a compatible SAE exists).
11. Round-trip: export → import on the same instance reproduces member set, strengths, pins, sign, and
    intensity exactly (semantic equality).
12. Imported profiles are ordinary profiles (editable, re-exportable).

### 3.4 Steering integration
13. "Save cluster profile" appears on the Steering panel when a cluster configuration is active (012
    context or a loaded profile); "Load profile" lists profiles for the selected SAE.
14. Loading a profile sets 012's label context tier 1 (profile name) and 013's budget state (pins,
    intensity) exactly as saved.
15. The Clusters panel offers "Save as profile…" on an expanded cluster (pre-tuning capture uses 013's
    computed allocation as the strengths).

## 4. User Interface
- **Steering panel:** "Save cluster profile" button + name/narrative dialog; "Profiles" dropdown/list
  (per-SAE) with load/export/delete; import via file picker (accepts .json).
- **Clusters panel:** per-cluster "Save as profile…" action.
- **Profiles list entries:** name, token, N members, updated_at, badges (imported / SAE-unbound).
- Narrative renders as markdown (existing react-markdown stack) in a profile detail popover.

## 5. API / Integration
- `POST /api/v1/cluster-profiles` · `GET /api/v1/cluster-profiles?sae_id=` · `GET/PUT/DELETE
  /api/v1/cluster-profiles/{id}` · `GET /api/v1/cluster-profiles/{id}/export` ·
  `POST /api/v1/cluster-profiles/export-bundle {ids}` · `POST /api/v1/cluster-profiles/import` (multipart
  or JSON body; returns created profiles + warnings).
- MCP tools (read + write category `experiments` or new `profiles`): `list_cluster_profiles`,
  `get_cluster_profile`, `save_cluster_profile`, `export_cluster_profile` — lets agents capture validated
  clusters (natural continuation of the MCP experiment flow).
- No MILLM integration in this increment (contract designed for it — IDL-30).

## 6. Data / Types
- New table `cluster_profiles` (IDL-30 §1) + Alembic migration.
- Pydantic schemas + JSON (de)serializers for definition/bundle kinds.
- Frontend `ClusterProfile`, `ClusterDefinitionFile` types; `clusterProfilesStore` (list/save/load/import/
  export state).

## 7. Dependencies
- 012 (label tier 1 consumes profile name; cluster context identifies save source).
- 013 (strengths/budget/intensity being saved are its outputs; `formula_id`/constants echo).
- Existing: react-markdown (narrative), Alembic, settings/crypto NOT needed (no secrets).

## 8. Success Criteria
1. Profile survives grouping recompute + backend restart (integration-tested).
2. Round-trip export→import semantic equality (automated test on definition AND bundle).
3. Import blocks on n_features mismatch with actionable message; warns-and-binds on model mismatch.
4. Load-profile → Blended run titled by profile name; strengths/pins/intensity match saved values (E2E).
5. Exported JSON validates against the published schema (checked in as `docs/schemas/cluster-definition-v1.json`
   in-repo) and contains no local paths/secrets.
6. MCP agent can save + export a profile end-to-end.

## 9. Non-Goals
- MILLM import/runtime (separate BRD); marketplace mechanics (vision only); cross-SAE index remapping
  (definitions bind to compatible SAEs only); profile versioning/history (single mutable record v1);
  authored-profile sync to Neuronpedia.

## 10. Testing Requirements
- Backend unit: schema validation (kind/version/member bounds), compatibility matrix (block/warn/bind),
  serializer round-trip property test.
- Backend integration: recompute-survival; CRUD; import binds/blocks correctly.
- Frontend: store tests (save/load/import flows, stale-SAE guards); dialog validation.
- E2E: save → recompute index → load → steer → export → re-import → identical config; caps screenshot.

## 11. BRD Traceability

| BRD req | Covered by |
|---|---|
| BR-007 (name/narrative/persisted strengths) | §3.1 |
| BR-008 (portable standardized JSON export, single + multi) | §3.2 |
| BR-009 (import round-trip) | §3.3 items 9, 11, 12 |
| BR-010 (consumer-neutral standardized format) | §3.2 item 7, §3.3 item 10, IDL-30 |
