# Task List: Circuit Review, Evidence Ladder & Portability

**Document ID:** 018_FTASKS|Circuit_Portability
**Version:** 1.1
**Status:** ✅ COMPLETE — 3 review rounds done (49+26+12 findings; R1: 21 fixed, R2: 15 fixed, R3: all 3 must-fix + 6 ride-alongs fixed). R3 verdict: **GO for 016** (record: `.claude/context/sessions/review_feature018_R3_2026-07-19.md`). Deferred items live in 016/017 FTASKS Phase 0 / Task 3.0. Phase 6 E2E/manual at close-out.
**Source:** 018_FPRD · 018_FTDD · 018_FTID · IDL-33, IDL-35 · CIRCUITS-002 A.1/A.9 normative
**Sequencing:** Phases 1–2 GATE the increment — land before 016/017 implementation begins (002 order: d → b → c → a-hazards).

**Progress (2026-07-19):** Phases 1–5 core SHIPPED (commits ccf6bf8, 9532bc4, f1be274 + R2 wave):
ladder + TS mirror + grep-guard; contract + vendored schema + projection; circuits table (2 migrations,
up/down/up verified) + service (contract-backed validation); classifier + audit fixture (O_HI calibrated,
recorded in FTDD); REST (incl. IMPORT — BR-013 round-trip real-stack-tested lossless incl. model ref/
granularity/created_at) + 9 MCP tools + Review UI (rung chips, markdown narrative, persistence toggle,
caps meter, edit name/narrative, import/export/slices buttons, reversible promotion). 121 circuit pins.

**Recorded deferrals (review R2-P1):**
- `from_candidates` assembly seam → Feature 016 FTASKS (its discovery runs are the caller).
- Member-level editing UI + list filter CONTROLS (API supports; UI renders none) → with 016's Review-tab expansion.
- SteeringPanel rung chip on circuit-titled results → Feature 015 (its titling work).
- Optimistic-concurrency precondition on PATCH/edge-writes → MUST land with 017's writer (R2-Q2: 3-writer world).
- SQL-side pagination/edge_type filtering → with 016 (mass-created circuits make it real; R2 B8).
- typed type_signals model at v1 freeze (R2-A2) + Phase 6 E2E/screenshot/manual → close-out session.

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Evidence ladder (gate) | 1 task | ⏳ |
| Phase 2: Contract + projection (gate) | 2 tasks | ⏳ |
| Phase 3: Circuit storage + service | 2 tasks | ⏳ |
| Phase 4: Edge-type classifier | 1 task | ⏳ |
| Phase 5: Endpoints + MCP + UI | 3 tasks | ⏳ |
| Phase 6: Verification + acceptance | 2 tasks | ⏳ |

---

## Phase 1: Evidence ladder (gate)

### Task 1.1: evidence_ladder.py
- [ ] `EvidenceRung` IntEnum + `RUNG_LANGUAGE` + `EdgeEvidence` (highest-passed; tested_and_failed history) · TS mirror + literal-value sync test · grep-guard: no other rung/status enums anywhere · copy-audit hook: causal words only from RUNG_LANGUAGE

## Phase 2: Contract + projection (gate)

### Task 2.1: CircuitDefinitionV1 + vendored schema
- [ ] Strict models per IDL-33 (members ≤20 PER LAYER; nullable position/attention; edges w/ rung/type/stats/attribution/manifest_ref; discovery block) · `docs/schemas/circuit-definition-v1.json` via the 014 `_generate()` pattern + sync test · round-trip property test (all evidence fields)

### Task 2.2: Projection slicer
- [ ] `to_layer_slice()` → valid SHIPPED-v1 cluster-definition (existing validator imported, not duplicated) · meta `{projection_of, parent_rung, partial_rendering}` · slice-validity + marker tests · membership snapshot at export vs dynamic live resolution documented

## Phase 3: Circuit storage + service

### Task 3.1: circuits table + migration
- [ ] `crc_` ids, JSONB members/edges/budget/faithfulness, soft discovery_run ref, schema_version · Alembic single-head check, up+down

### Task 3.2: CircuitService
- [ ] CRUD · `from_candidates(run, selection)` assembly · `promote()` · `recompute_rungs` (min-over-edges; empty-edges ⇒ rung 0; faithfulness separate) · per-layer cap enforcement on every member write (single-layer regression pin: today's 20-cap behavior identical)

## Phase 4: Edge-type classifier

### Task 4.1: circuit_edge_type_service.py
- [ ] persistence = ≥2 of {prior ≥ 0.9, token-identity overlap ≥ 0.8, label-embedding sim ≥ 0.85 (2/2 fallback when no embedding stack — degradation disclosed)} · computed default · attention_mediated reserved · signals disclosed per edge · `distinctness` export for 016 ranking · hand-labeled audit fixture + regression gate (≥90% / ≤10%)

## Phase 5: Endpoints + MCP + UI

### Task 5.1: circuits endpoints
- [ ] CRUD + promote + export + export-slices + rung/type/granularity list filters · kind-keyed import (unknown major reject) · hostile-input tests

### Task 5.2: MCP
- [ ] `list_circuits`, `get_circuit`, `promote_circuit`, `export_circuit_definition`, `export_circuit_slices` · rung + rung_language + edge type embedded in ALL circuit-returning tools (smoke-tested)

### Task 5.3: Review tab UI
- [ ] RungChip (tooltip: what moves it up) · EdgeTable (type/rung/stats/attribution/manifest columns; persistence de-emphasized not hidden; filters) · CapsMeter (per-layer) · review detail (members by layer, faithfulness block) · edit/name/narrate → Promote → Export/Export-slices · SteeringPanel rung chip on circuit-titled results

## Phase 6: Verification + acceptance

### Task 6.1: Integration + E2E
- [ ] Assemble-from-run → promote → 015 load → steer (rung chip shown) · classifier over a real discovery run · export definition + slices → re-import equality · slices validate against shipped v1 schema · E2E + cap `0xcc/caps/miStudio_Circuit_Review_<date>.png`

### Task 6.2: Acceptance (per instruct 007)
- [ ] FPRD §8 criteria 1–6 verified (single-source enum proof, copy audit, audit fixture, round-trips) · cluster-definition/v1 suite passes unchanged · suites green · manual ladder-led circuits page · CLAUDE.md + PPRD row 19 status update

---

## Relevant Files

| File | Purpose |
|------|---------|
| `backend/src/schemas/evidence_ladder.py` (NEW) + TS mirror | the ladder (single source) |
| `backend/src/schemas/circuit_definition.py` (NEW) · `docs/schemas/circuit-definition-v1.json` (NEW) | contract |
| `backend/src/models/circuit.py` (NEW) + migration · `services/circuit_service.py` (NEW) | storage/promotion |
| `backend/src/services/circuit_edge_type_service.py` (NEW) + audit fixture | typing |
| `backend/src/services/circuit_contract_service.py` (NEW) | export/import/slices |
| `backend/src/api/v1/endpoints/circuits.py` (NEW) · MCP tools/circuits.py | REST + MCP |
| `frontend/src/components/circuits/RungChip.tsx`, `EdgeTable.tsx`, `CapsMeter.tsx` (NEW) + Review tab | UI |
| `manual/docs/**` | docs |

## Coverage audit (instruct 007)
- Data ✅ (Ph3, migration both directions) · API ✅ (Ph5) · MCP ✅ (Ph5) · UI/State ✅ (Ph5) ·
  Tests ✅ (enum sync/grep-guard, round-trip property, audit fixture, slice validity, copy audit, E2E) ·
  Docs ✅ (Ph5-6) · Acceptance ✅ (Ph6). Security: import kind/size validation (house caps);
  no secrets/paths in exports (inherited v1 discipline + tests).
