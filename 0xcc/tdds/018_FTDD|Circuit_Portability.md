# Technical Design Document: Circuit Review, Evidence Ladder & Portability

**Document ID:** 018_FTDD|Circuit_Portability
**Version:** 1.0
**Status:** Planned
**Related:** 018_FPRD · IDL-33, IDL-35 · CIRCUITS-002 A.1/A.9 normative · gates 016/017 (shared enum + contract land first)

---

## 1. Evidence ladder model (single source)

```python
# backend/src/schemas/evidence_ladder.py — THE one definition (IDL-35)
class EvidenceRung(IntEnum):
    MINED = 0
    ATTRIBUTION_SUPPORTED = 1
    CAUSALLY_VALIDATED = 2
    FAITHFULNESS_TESTED = 3

RUNG_LANGUAGE = {0: "associated", 1: "suggested (attribution-supported)",
                 2: "causally validated (edge)", 3: "faithfulness-tested (circuit)"}

class EdgeEvidence(BaseModel):
    rung: EvidenceRung              # highest PASSED
    tested_and_failed: list[EvidenceRung] = []   # failures recorded, never demote
```

- Circuit displayed rung = `min(edge.rung for edges)`; faithfulness status separate field.
- TS mirror `frontend/src/types/evidenceLadder.ts` generated/synced (test compares literal values).
- MCP responses embed `rung` + `rung_language` (server-rendered string — one language source).
- Copy audit (017's shared suite) asserts `RUNG_LANGUAGE` is the only causal-word source.

## 2. Edge-type classifier (A.9)

```
classify(edge, signals) -> {type, signals_disclosed}
  persistence iff ≥2 of:
    weight_prior ≥ P_hi (default 0.9)
    token_identity_overlap(top contexts) ≥ O_hi (default 0.8)
    label_embedding_sim ≥ S_hi (default 0.85)   # embeddings via existing labeling stack
  attention_mediated iff Tier-2.5 evidence present (mediating_heads non-null)  # future data
  else computed
```

- Runs: on discovery results (016 report echo counts) and at circuit assembly (edge records).
- Ranking composition (BR-020): default candidate ordering = f(PMI, support, attribution?) with the
  prior contributing ONLY via `distinctness = 1 − echo_confidence`; disclosed in the run report.
- Audit fixture: `backend/tests/fixtures/edge_type_audit.json` — hand-labeled echoes + computed edges
  (seeded from known L13→L14 persistence pairs + validated computed edges); regression-gated
  (≥90%/≤10%).

## 3. Circuit storage

```sql
circuits (
  id            VARCHAR PK        -- crc_<hex12>
  name          VARCHAR(120) NOT NULL
  narrative     TEXT NULL
  granularity   VARCHAR           -- feature | cluster
  members       JSONB             -- [{layer, sae_id, member_kind, feature_idx | cluster_ref,
                                  --   label, strength, sign, pinned, stats…}]  (≤20 PER LAYER)
  edges         JSONB             -- [{up, down, type, rung, tested_and_failed, coactivation,
                                  --   weight_prior, attribution, validation_manifest_ref, position{...}}]
  budget        JSONB             -- per-layer budgets + global intensity (IDL-31 formula id)
  faithfulness  JSONB NULL        -- {necessity, sufficiency?, k, metric_id, manifest_ref}
  discovery_run VARCHAR NULL      -- soft ref
  schema_version VARCHAR          -- "1"
  created_at / updated_at
)
```

A promoted circuit IS the loadable steering profile (015 hydrates from members) — no dual entity.
017 writes validation/faithfulness into these records; rungs recompute on write.

## 4. Contract & projection

- `mistudio.circuit-definition/v1` per IDL-33 §1 (fields mirror §3 + saes[] + provenance/discovery).
  Vendored schema `docs/schemas/circuit-definition-v1.json`, pydantic-generated, sync-tested (014's
  discipline verbatim); cross-repo review vs the anticipated miLLM runtime before freeze.
- Projection: `slice(circuit, layer) -> ClusterDefinitionV1` — that layer's members (feature_ref
  expansion for cluster members), that layer's budget, meta `{projection_of: crc_…, parent_rung,
  partial_rendering: true}` (display-only per the member-meta contract rev). Slices MUST validate
  against the SHIPPED v1 schema (test) — miLLM imports them with zero changes.
- Round-trip: definition → import → definition semantic equality property test (incl. rungs, types,
  attribution, manifest refs, position nulls).

## 5. Review & promotion flow

```
Discovery run (016) → candidates (+classifier types, +017 validation)
  → Review tab: assemble/edit circuit (per-layer caps enforced live) → name/narrate
  → POST /circuits (draft) → /promote (marks steerable; rung computed)
  → 015 loads (members → SelectedFeatures w/ sae_id/layer; budget → layerBudgets; title + rung chip)
  → export definition / slices
```

## 6. Architecture / types

```
evidence_ladder.py (shared enum + language)         — lands FIRST
CircuitEdgeTypeService (classifier + audit fixture) — consumed by 016 reports + circuit assembly
CircuitService (CRUD, promote, rung computation, per-layer caps)
CircuitContractService (to/from definition, slices, schema publish)
endpoints: circuits.py; MCP: 5 tools + rung/type fields on existing circuit-returning tools
frontend: Review tab (CircuitsPanel), RungChip.tsx, EdgeTable.tsx, CapsMeter.tsx; evidenceLadder.ts
tables: circuits (+ Alembic single-head check)
```

## 7. Risks

| Risk | Mitigation |
|---|---|
| Enum divergence across surfaces (the IDL-35 defect class) | single module + TS sync test + server-rendered rung_language |
| Classifier thresholds misfire | audit fixture regression gate; thresholds config; signals disclosed per edge |
| Contract churn at freeze | pre-freeze review vs miLLM runtime; nullable Tier-2.5 fields; 014 sync discipline |
| Projection mistaken for the whole | parent_rung + partial_rendering markers; manual page states it; slice meta test |
| Per-layer caps break existing single-layer flows | cap validator applies per layer — single-layer behavior identical to the 20-cap today (regression test) |
| Review UX overload | progressive disclosure: rung chip + type first; stats/manifests behind expanders |
