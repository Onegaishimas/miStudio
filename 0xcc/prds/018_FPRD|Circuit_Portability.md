# Feature PRD: Circuit Review, Evidence Ladder & Portability

**Document ID:** 018_FPRD|Circuit_Portability
**Version:** 1.0
**Status:** Planned
**Related:** BRD-MIS-CIRCUITS-001 (BR-010, BR-011, BR-012, BR-013, BR-014) as amended by BRD-MIS-CIRCUITS-002 (BR-020, BR-021, BR-025, BR-026) · 000_PPRD §3.19 · IDL-33, IDL-35 · normative: CIRCUITS-002 A.1/A.9 · SEQUENCED FIRST in the increment (the ladder enum + contract gate everything else)

---

## 1. Overview

### 1.1 Purpose
Turn discovered circuits into first-class, honestly-labeled, portable artifacts: one evidence-ladder
vocabulary across every surface, typed edges, an evidence-rich review/promotion flow, the
`mistudio.circuit-definition/v1` contract (amended pre-freeze), and the per-layer projection that keeps
today's single-SAE consumers working.

### 1.2 User Problem
As capability tiers accumulate (mined → attribution → intervention → faithfulness), claim inflation
becomes the product's biggest credibility risk: a mined correlation presented with causal language, an
echo edge indistinguishable from computed structure, a projection mistaken for a validated whole. And
without a durable circuit entity + contract, discovery output remains trapped in run records.

### 1.3 Solution
- **Evidence ladder (BR-026):** ONE shared rung model (mined / attribution_supported /
  causally_validated / faithfulness_tested) as a single backend enum consumed by UI, MCP, and contract;
  language rules enforced by a copy audit.
- **Edge typing (BR-020/021):** computed | persistence | attention_mediated; the weight prior becomes an
  echo-detector input; persistence de-ranked but never hidden.
- **Review & promotion (BR-011/012):** evidence-rich review; promotion to loadable multi-layer profiles;
  badge, not gate; per-layer caps (BR-025).
- **Contract (BR-013) + projection (BR-014):** circuit-definition/v1 with rung/type/position/attribution/
  manifest fields; per-layer cluster-definition/v1 slices with parent-rung markers.

## 2. User Stories
- **US-1:** Every edge and circuit I see — list, detail, steering panel, MCP response, export — shows its
  rung; a tooltip tells me exactly what would move it up one rung.
- **US-2:** I review a candidate circuit: members with labels/examples, PMI/support/null percentile,
  attribution score, validation status + manifest links, edge types with their classification signals. I
  edit members, name it, write a narrative, and promote it.
- **US-3:** A promoted-but-unvalidated circuit steers immediately (015) and displays rung 0/1 language
  ("associated") — never "causal".
- **US-4:** I filter the edge list to persistence edges when I want robustness steering; they're typed
  and queryable, not hidden.
- **US-5:** I export a circuit; the JSON carries rungs, edge types, attribution scores, manifest refs,
  and discovery provenance; re-import reproduces it losslessly.
- **US-6:** I export per-layer slices; miLLM imports them unchanged; each slice shows the parent circuit
  ref, the parent's rung, and a partial-rendering marker.
- **US-7 (agent):** MCP returns rungs and edge types on every circuit/edge; I can filter by type/rung and
  drive review → promotion → export end-to-end.

## 3. Functional Requirements

### 3.1 Evidence ladder (BR-026) — IDL-35
1. One shared enum (backend model → UI, MCP, contract): rungs 0–3; edge rung = highest PASSED; failed
   tests record `tested_and_failed` at that rung without demoting; circuit displayed rung = MIN over
   member edges; faithfulness status displayed separately.
2. Language mapping enforced: rung 0–1 "associated/suggested"; rung 2 "causally validated (edge)"; rung
   3 "faithfulness-tested (circuit)"; the word "causal" forbidden below rung 2 (shared copy-audit suite
   with 017).
3. Rungs render in every UI surface where a circuit/edge appears; every MCP tool returning circuits or
   edges includes rung fields; exports and projections carry them.

### 3.2 Edge typing (BR-020, BR-021) — A.9
4. Classifier: `persistence` when ≥2 of {weight prior ≥ high threshold, top-context token-identity
   overlap ≥ threshold, label-embedding similarity ≥ threshold}; `computed` otherwise;
   `attention_mediated` reserved for Tier-2.5 evidence. Classification signals disclosed per edge.
5. In RANKING the weight prior participates only combined with distinctness signals (BR-008 disclosure
   updated); low prior + high association is never down-ranked for the low prior.
6. Persistence edges: de-ranked from default views; queryable via filter; steerable; exportable; always
   visibly typed. The 016 candidate tables and run reports consume this classifier (echo-filter counts).
7. A hand-labeled audit fixture (known echoes + known computed edges) gates the classifier: ≥90%
   persistence recall, ≤10% computed misclassification.

### 3.3 Review & promotion (BR-011, BR-012, BR-025)
8. Review surface per candidate circuit: members (labels, examples, per-layer grouping), edges (type,
   rung, statistics, attribution, validation status, manifest links), run-report context.
9. Edit members (within per-layer caps), name, narrative (markdown); cluster-granularity circuits show
   member clusters with drill-down.
10. Promotion creates a durable circuit record that IS a loadable multi-layer steering profile (015
    loads it; per-member sae_id/layer set). **Badge, not gate** at every rung.
11. **Per-layer member caps (BR-025):** the 20-member cap applies PER LAYER; validators + UX enforce and
    display per layer.

### 3.4 Contract (BR-013) — IDL-33
12. `mistudio.circuit-definition/v1` (NEW kind; cluster-definition/v1 untouched): `saes[]`, `members[]`
    (+ `layer`, `member_kind: feature_ref|cluster_ref`), `edges[]` {type, rung, tested_and_failed[],
    coactivation stats incl. null percentile + replicated_heldout, weight_prior, attribution, 
    validation_manifest_ref, position fields (nullable)}, per-layer budgets + global λ, faithfulness
    block, provenance + discovery block (mode, granularity, corpus, split, thresholds, dates).
13. Vendored JSON schema published beside the v1 schema, pydantic-sync-tested; reviewed against the
    anticipated miLLM circuits runtime BEFORE freeze.
14. Export→import round-trip preserves rungs, edge types, attribution scores, manifest refs, provenance
    (semantic equality).

### 3.5 Projection (BR-014)
15. Per-layer slices export as valid cluster-definition/v1 files (that layer's members + budget slice)
    with display-only meta `{projection_of, parent_rung, partial_rendering: true}`; slices validate
    against the shipped v1 schema; miLLM imports unchanged.

## 4. User Interface
- **Circuits panel — Review tab:** candidate/circuit list (rung chips, type filters, granularity);
  detail view (members by layer with caps meter, edge table with evidence columns + manifest links,
  faithfulness scores); edit/name/narrate; Promote button; Export / Export slices.
- **Rung chips everywhere:** list rows, detail headers, steering panel (loaded circuit), export
  dialogs; tooltip = "what moves this up one rung".
- **Steering integration:** loading a promoted circuit into 015 shows the rung chip beside the circuit
  title; hazard banner shows rung-labeled evidence (015 consumes).

## 5. API / Integration
- `GET/POST /api/v1/circuits` (from discovery candidates or manual assembly) · `GET/PATCH/DELETE /{id}`
  · `POST /{id}/promote` · `GET /{id}/export` · `POST /{id}/export-slices` · edge-type/rung filters on
  list.
- Edge-type classifier runs on discovery results (service consumed by 016's report) and at circuit
  assembly.
- MCP: `list_circuits`, `get_circuit`, `promote_circuit`, `export_circuit_definition`,
  `export_circuit_slices` (+ rung/type fields on ALL existing circuit-returning tools).
- 015 consumes promoted circuits (multi-layer profile loading + hazard edges); 017 writes
  validation/faithfulness into circuit records.

## 6. Data / Types
- `circuits` table (`crc_` ids, JSONB members/edges/budget/faithfulness, schema_version, discovery-run
  soft ref) + Alembic (single-head check).
- Shared `EvidenceRung` enum module (backend single source; mirrored TS type generated/kept in sync by
  test).
- `docs/schemas/circuit-definition-v1.json` (vendored, sync-tested).

## 7. Dependencies
- SEQUENCED FIRST: the rung enum + contract land before 016/017 populate them (002 sequencing).
- 016 candidates/reports (typing feeds back), 017 validation/faithfulness (rung 2–3 sources), 015
  loading (consumes promoted circuits), 014 contract discipline (schema sync pattern), cluster profiles
  (cluster_ref members + audit fixture seeds).

## 8. Success Criteria
1. One rung enum, three consumers (UI/MCP/contract) — divergence test proves single-source.
2. Copy audit green: no causal language below rung 2 anywhere (shared suite).
3. Edge-type audit fixture: ≥90% persistence recall, ≤10% computed misclassification.
4. Promote → load in 015 → steer: circuit-titled, rung-chipped, per-layer caps enforced.
5. Export→import round-trip semantic equality (incl. rungs/types/attribution/manifest refs); slices
   validate against v1 schema and carry parent-rung markers; existing cluster-definition/v1 suite passes
   unchanged.
6. MCP loop (list → review data → promote → export) works agent-only.

## 9. Non-Goals
- Attention-mediated evidence production (Tier-2.5 fast-follow — fields ship nullable); circuit
  narration/auto-labeling; marketplace; miLLM runtime changes; v1 freeze itself (a release decision
  after the contract review).

## 10. Testing Requirements
- Unit: rung transitions (passed/failed history, min-over-edges, faithfulness separate), language
  mapper, classifier matrix + audit fixture, per-layer cap validators, contract validators
  (kind/version/members/edges/nullable position fields), projection slicing + marker, round-trip
  property tests.
- Integration: promote→load→steer path; classifier over a real discovery run; schema-sync (pydantic ↔
  vendored file); TS-enum sync test.
- Frontend: rung chips + tooltips, type filters, review detail rendering, caps meter, export dialogs.
- E2E: review a seeded candidate → promote → steer (015) → export definition + slices → re-import;
  screenshot `0xcc/caps/miStudio_Circuit_Review_<date>.png`.

## 11. BRD Traceability

| BRD req | Covered by |
|---|---|
| BR-010 (triviality policy, disclosed) | §3.2 items 4–7 |
| BR-011 (evidence-rich review) | §3.3 items 8–9 |
| BR-012 (promotion; badge-not-gate) | §3.3 item 10 |
| BR-013 (circuit-definition/v1, lossless round-trip) | §3.4 |
| BR-014 (per-layer v1 projection, marked partial) | §3.5 |
| BR-020 (002: weight prior = echo detector; ranking composition) | §3.2 items 4–5 |
| BR-021 (002: typed persistence edges, never hidden) | §3.2 items 4, 6 |
| BR-025 (002: per-layer member caps) | §3.3 item 11 |
| BR-026 (002: evidence ladder across all surfaces) | §3.1 |
