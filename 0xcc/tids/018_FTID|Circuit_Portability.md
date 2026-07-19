# Technical Implementation Document: Circuit Review, Evidence Ladder & Portability

**Document ID:** 018_FTID|Circuit_Portability
**Version:** 1.0
**Status:** Planned
**Related:** 018_FPRD · 018_FTDD · IDL-33, IDL-35 · CIRCUITS-002 A.1/A.9 normative

---

## 1. Implementation Order (gates the increment — land 1–3 before 016/017 code)

1. `evidence_ladder.py` (enum + language + TS mirror + sync tests + copy-audit hooks).
2. Contract models + vendored schema + round-trip/property tests (+ projection slicer).
3. `circuits` table + CircuitService (CRUD, rung computation, per-layer caps).
4. Edge-type classifier + audit fixture (016's reports consume).
5. Endpoints + MCP tools (+ rung/type fields threaded into existing circuit-returning tools).
6. Review tab UI (RungChip/EdgeTable/CapsMeter, promote/export flows).
7. 015 integration (circuit loading) + E2E + manual.

## 2. File-by-file

### 2.1 `backend/src/schemas/evidence_ladder.py` (NEW — single source)
- `EvidenceRung(IntEnum)`, `RUNG_LANGUAGE`, `EdgeEvidence` per FTDD §1. NO other rung/status enums may
  exist (grep-guard test). `frontend/src/types/evidenceLadder.ts` mirror + literal-value sync test.

### 2.2 `backend/src/schemas/circuit_definition.py` (NEW) + `docs/schemas/circuit-definition-v1.json`
- `CircuitDefinitionV1` per IDL-33 (strict validators; members ≤20 PER LAYER; nullable position/
  attention fields; edges carry rung/type/stats/attribution/manifest_ref). Schema generated via the 014
  `_generate()` pattern; sync test; NOTE for the future cross-repo pin (miLLM consumes post-runtime-BRD).
- Projection: `to_layer_slice(defn, layer) -> ClusterDefinitionV1` with meta markers; slice-validates
  against the SHIPPED v1 schema (import the existing validator — do not duplicate).

### 2.3 `backend/src/models/circuit.py` (NEW) + Alembic
- `circuits` per FTDD §3; single-head check; up+down tested. Soft `discovery_run` ref (no FK — runs
  prunable).

### 2.4 `backend/src/services/circuit_service.py` (NEW)
- CRUD; `promote(id)`; `recompute_rungs(circuit)` (min-over-edges; faithfulness separate); per-layer cap
  enforcement on every member write; `from_candidates(run_id, selection)` assembly.

### 2.5 `backend/src/services/circuit_edge_type_service.py` (NEW)
- Classifier per FTDD §2: weight_prior (015's resolvers), token-identity overlap (top contexts from
  Feature rows), label-embedding sim (existing labeling/embedding stack — check what's available; else
  substring/token-set fallback with the signal marked weaker). Signals disclosed per edge.
- `tests/fixtures/edge_type_audit.json` + regression test (≥90% persistence recall, ≤10% computed
  misclassification). Ranking composition helper `distinctness = 1 − echo_confidence` exported for
  016's ordering.

### 2.6 `backend/src/services/circuit_contract_service.py` (NEW)
- to/from definition (round-trip property test incl. all evidence fields), export slices (one v1 file
  per layer; Content-Disposition naming `<name>.L<k>.cluster.json`), import (kind-keyed; unknown major
  reject).

### 2.7 `backend/src/api/v1/endpoints/circuits.py` (NEW) + MCP
- Routes per FPRD §5 (list filters: rung ≥, type, granularity). MCP `tools/circuits.py` += 5 tools;
  ALL circuit-returning tools embed `rung` + `rung_language` + edge `type` (audit: grep MCP responses in
  smoke test).

### 2.8 Frontend
- `types/evidenceLadder.ts` (mirror), `types/circuits.ts` extensions, `api/circuits.ts`,
  `circuitStore.ts` extensions.
- `RungChip.tsx` (NEW — chip + what-moves-it-up tooltip; used in lists/details/steering title row),
  `EdgeTable.tsx` (NEW — type + rung + stats + attribution + manifest link columns; type filter;
  persistence rows de-emphasized but present), `CapsMeter.tsx` (NEW — per-layer member counts vs cap).
- CircuitsPanel **Review tab**: candidate list → detail (members by layer, EdgeTable, faithfulness
  block) → edit/name/narrate → Promote → Export / Export slices.
- SteeringPanel: rung chip beside circuit-titled results (015's title row).

### 2.9 Manual
- `circuits.md` += ladder (lead section — the rung table + language rules), review/promotion, contract
  + projection ("a slice is NOT the circuit"); mcp-server.md tool additions.

## 3. Pitfalls

- The ladder module MUST land before 016/017 write any status fields — enforce via implementation order
  (and the grep-guard against local enums).
- Projection slices must validate against the SHIPPED v1 schema — import the existing validator; any
  new field belongs in meta (display-only), never top-level.
- `min(edge.rung)` on an empty-edge circuit (hand-assembled) ⇒ rung 0 (mined) — define, don't crash.
- Cluster_ref member expansion at steering/projection time must use CURRENT profile membership —
  record the membership snapshot in the definition at export (portability) while live records resolve
  dynamically (document the difference).
- Per-layer caps: validator counts by layer — a 3-layer circuit may hold 60 members total; UI CapsMeter
  must show per-layer, not total.
- Label-embedding availability: if no embedding stack is reachable, the classifier runs on 2 signals
  with thresholds requiring 2/2 — record the degradation in the disclosed signals.

## 4. Testing

- Unit: rung transitions/min/empty-edges; language mapper; TS sync; classifier matrix + audit fixture;
  per-layer caps (incl. single-layer regression); contract validators; projection slice validity +
  markers; round-trip property.
- Integration: assemble-from-run; promote→015-load→steer; classifier over a real discovery run;
  schema-sync; MCP rung/type embedding smoke.
- Frontend: RungChip tooltip, EdgeTable filters, CapsMeter, review flows.
- Copy audit: shared suite (rung language only from RUNG_LANGUAGE).
- E2E: review seeded candidate → promote → steer → export definition + slices → re-import → equality;
  screenshot `0xcc/caps/miStudio_Circuit_Review_<date>.png`.
