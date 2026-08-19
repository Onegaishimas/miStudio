# Technical Design Document: Cluster Authoring & Portable Definitions

**Document ID:** 014_FTDD|Cluster_Definitions
**Version:** 1.0
**Status:** Planned
**Related:** 014_FPRD · IDL-30 · consumes 012 (labels) + 013 (allocations)

---

## 1. Storage design

**`cluster_profiles`** (new table, Alembic migration; decoupled from recomputable grouping tables):

```sql
id               VARCHAR PK            -- clp_<hex12>
sae_id           VARCHAR NOT NULL      -- FK saes.id (RESTRICT delete: block SAE delete w/ profiles, or cascade-with-confirm)
model_id         VARCHAR NULL          -- miStudio model id at save time
extraction_id    VARCHAR NULL          -- soft context
source_group_id  VARCHAR NULL          -- SOFT reference (no FK) — groups are ephemeral
name             VARCHAR(120) NOT NULL
narrative        TEXT NULL             -- markdown
display_token    VARCHAR NULL
members          JSONB NOT NULL        -- [{feature_idx, label, similarity, activation_frequency,
                                       --   max_activation, strength, sign, pinned}]
budget           JSONB NULL            -- {B, B_dir, G, f_eff, formula_id, constants, intensity,
                                       --   intensity_range: [0,2]}
schema_version   VARCHAR NOT NULL      -- "1"
imported_from    JSONB NULL            -- provenance of an import {original kind/version, created_at, source note}
created_at / updated_at TIMESTAMPTZ
```

Members/budget are **snapshots by design** (IDL-30 tradeoff): the profile records what the user tuned
against; it is not a live join. Layer/hook live inside the SAE reference (single-layer per 013).

## 2. Interchange format

```jsonc
// mistudio.cluster-definition/v1
{
  "kind": "mistudio.cluster-definition",
  "schema_version": "1",
  "name": "fear",
  "narrative": "…markdown…",
  "display_token": "fear",
  "model": { "hf_id": "LiquidAI/LFM2.5-1.2B-Instruct", "mistudio_model_id": "m_88d55564" },
  "sae":   { "mistudio_sae_id": "sae_eb8374929894", "layer": 12, "hook_type": "residual",
             "n_features": 32768, "d_model": 2048, "source_hint": "hf:repo/path (optional, no local paths)" },
  "members": [ { "feature_idx": 10091, "label": "…", "similarity": 0.82,
                 "activation_frequency": 0.164, "max_activation": 4.01,
                 "strength": 0.4, "sign": 1, "pinned": false } ],
  "budget": { "B": 2.1, "B_dir": 2.4, "G": 0.87, "f_eff": 0.19,
              "formula_id": "freq-budget/sim-alloc@1",
              "constants": { "a": 2.9, "b": 2.6, "m": 1.0, "M": 3.0 },
              "intensity": 1.0, "intensity_range": [0, 2] },
  "provenance": { "created_at": "…", "mistudio_version": "0.5.0", "exported_at": "…" }
}
// mistudio.cluster-bundle/v1: { "kind": "mistudio.cluster-bundle", "schema_version": "1",
//                               "definitions": [ <definition>, … ] }
```

Design rules: consumer-neutral (no miStudio-internal ids required to *use* it — indices + SAE shape
suffice); formula id + constants travel with it (any consumer can rescale/recompute — IDL-30); **no
secrets, no absolute local paths**; JSON Schema published in-repo at
`docs/schemas/cluster-definition-v1.json` and validated on export AND import (dogfood the contract).

## 3. Compatibility matrix (import)

| Condition | Action |
|---|---|
| kind/schema_version unknown major | reject, explicit message |
| local SAE with same `mistudio_sae_id` | bind silently |
| id differs, `n_features` + layer match | warn, user picks binding SAE (default: matching model+layer) |
| `n_features` mismatch (any candidate) | **block** binding (indices meaningless); allow import as unbound profile |
| model hf_id mismatch | warn + allow |
| no local SAE at all | import unbound; steerable later |
| duplicate profile name | allow (names not unique); flag in UI |

## 4. Architecture

```
frontend clusterProfilesStore ── REST /cluster-profiles CRUD/export/import ── ClusterProfileService
   │ save: snapshot from steeringStore (members+strengths+pins+sign+intensity + clusterBudget + 012 ctx)
   │ load: hydrate steeringStore (selection, strengths, pins, intensity, label ctx tier-1 = profile name)
   └ export/import: file download / picker → server validates against JSON Schema
MCP tools (list/get/save/export) call ClusterProfileService directly.
```

- Save is a **frontend snapshot → backend persist** (the tuned truth lives in the store); load is the
  inverse. The backend validates member bounds against the bound SAE at save/load.
- Export runs server-side from the stored profile (single source), NOT from live UI state — save first,
  then export (UI enforces).
- Import creates profiles; binding per §3; response carries warnings array the UI toasts.

## 5. Type changes

- Backend: `models/cluster_profile.py`, Alembic migration, `schemas/cluster_profile.py` (Profile CRUD +
  Definition/Bundle models with strict validators), `services/cluster_profile_service.py`, endpoints file,
  MCP `tools/profiles.py` (4 tools).
- Frontend: `types/clusterProfile.ts`, `api/clusterProfiles.ts`, `stores/clusterProfilesStore.ts`,
  dialog + list components, SteeringPanel/Clusters panel actions. `blendedTitle` tier 1 (012) consumes the
  loaded profile name; steeringStore gains `activeProfile: {id, name} | null` (cleared like clusterContext).

## 6. Risks

| Risk | Mitigation |
|---|---|
| Profile drift vs re-tuned reality | snapshots are the point; `updated_at` + explicit re-save; no silent sync |
| SAE deletion orphans profiles | FK RESTRICT + UI: block delete while profiles exist (or cascade with explicit confirm listing profiles) |
| Schema evolves for MILLM needs | versioned kind + published JSON Schema; additive-minor / new-major discipline recorded in IDL-30 |
| Import of hostile JSON | strict Pydantic validation + size caps (≤1 MB, ≤50 definitions/bundle, ≤20 members) + no path/secret fields honored |
| Round-trip float drift | strengths stored at 0.1 grain; serializer test asserts exact equality |
