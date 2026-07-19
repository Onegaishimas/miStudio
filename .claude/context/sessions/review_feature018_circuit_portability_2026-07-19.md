# Review Record — Feature 018: Circuit Review, Evidence Ladder & Portability

**Date:** 2026-07-19
**Scope:** commits `ccf6bf8..9532bc4` (i.e. commit 9532bc4 — "018 Ph3-5: circuit storage/service, edge-type classifier, REST+MCP+Review UI"). Base ccf6bf8 (Ph1-2 gates: evidence ladder + circuit-definition/v1 contract) read for context and coverage accounting but out of scope for findings unless a Ph3-5 consumer breaks it.
**Process:** house `/review` — four perspective reviews (Product, QA, Architect, Test), READ-ONLY.
**References:** `0xcc/prds/018_FPRD|Circuit_Portability.md`, `0xcc/tids/018_FTID|Circuit_Portability.md`, `0xcc/tasks/018_FTASKS|Circuit_Portability.md`, `0xcc/prds/BRD-MIS-CIRCUITS-002.md` (locked: ladder everywhere, badge-not-gate, per-layer caps, typed persistence edges).

**Files reviewed:** `backend/src/models/circuit.py`, `backend/alembic/versions/9a7da58fcd50_add_circuits_table.py`, `backend/src/services/circuit_service.py`, `backend/src/services/circuit_edge_type_service.py`, `backend/src/api/v1/endpoints/circuits.py`, `backend/src/mcp_server/tools/circuits.py`, `backend/src/mcp_server/config.py`, `backend/src/services/steering_service.py` (resolve_encoder_weight), `backend/tests/unit/test_circuit_service.py`, `test_circuits_api.py`, `test_circuit_edge_types.py`, `backend/tests/fixtures/edge_type_audit.json`, `frontend/src/components/panels/CircuitsPanel.tsx`, `frontend/src/components/circuits/RungChip.tsx`, `frontend/src/api/circuits.ts`, `frontend/src/types/circuits.ts` (+ Ph1-2 context: `evidence_ladder.py`, `circuit_definition.py`, `evidenceLadder.ts`, their tests).

---

## Verdict: **SHIP-WITH-NOTES**

The locked BRD decisions are genuinely implemented and pinned by tests: **badge-not-gate** (service + endpoint + tests assert promotion at rung 0), **per-layer caps** (contract validator counts cluster expansions per layer; 422 surfaced with the exact violation), **typed persistence edges never hidden** (classifier + de-emphasized-but-present rows in the edge table), and **ladder everywhere on backend surfaces** (every REST/MCP response carries `rung` + server-rendered `rung_language` + `rung_next_step`). The validation-via-contract pattern is clean and DRY. What keeps this at SHIP-WITH-NOTES rather than SHIP: two interface bugs that will bite the moment 016 wires the classifier into edges (ARCH-2 `type_signals` shape mismatch, QA-1 unvalidated scalar columns → 500s), the import half of the "lossless round-trip" story missing entirely (PE-2), and a classifier gate that was calibrated against its own 10-case fixture (ARCH-3/TE-3). None of these endanger current users — 016/017 have not landed and the panel is read-mostly — but ARCH-2, QA-1, PE-2, and PE-3 should be closed **before 016 implementation starts**, since 016 consumes exactly those seams.

Finding counts: **21 findings** — 6 P1, 10 P2, 4 P3, plus 2 positive/info notes.

---

## 1. PRODUCT ENGINEER — requirements alignment

Traceability check against FPRD §3 (15 functional requirements), §4 UI, §5 API, §8 success criteria, and BRD-002 locked decisions.

**Covered (verified in code/tests):** §3.1 items 1–3 backend-side (ladder single-source, language map, rung on every REST/MCP response — Ph1-2 + this commit); §3.2 item 4 (classifier 2-of-3 with disclosure — with deviations, see ARCH-3), item 6 partially (typed + de-ranked in UI via opacity; queryable server-side only via stored JSONB, no list filter); item 7 (audit fixture + gate test); §3.3 item 10 (promotion → durable record, badge-not-gate), item 11 (per-layer caps + single-layer regression semantics preserved); §3.4 items 12–13 (contract + vendored schema + sync test, Ph2); §3.5 item 15 (slices validate as real ClusterDefinitionV1; markers in name + provenance.source_note); §5 REST routes all present incl. filters (promoted/min_rung/granularity); §5 MCP: all 5 named tools + a bonus `create_circuit`.

### Findings

**PE-1 · P1 — Review-tab authoring absent: no member editing, no naming, no narrative authoring in the UI.**
FPRD §3.3 item 9 / US-2 / Task 5.3 require edit-members (within caps), name, and markdown-narrative authoring in the Review tab. `CircuitsPanel.tsx` is read-only + promote/delete/export; `PATCH /circuits/{id}` exists and validates, but no UI calls it (frontend `circuitsApi` has no `update`).
*Recommendation:* Add an edit drawer (name, narrative, member add/remove with per-layer counts) wired to PATCH; until then, mark item 9 as ❌ in FTASKS, not partially done.

**PE-2 · P1 — Import path missing entirely; "lossless round-trip" (§3.4 item 14, US-5, §8 criterion 5) cannot be exercised.**
There is no `POST /circuits/import` (or equivalent), no `circuit_contract_service.py` (FTID §2.6: kind-keyed import, unknown-major reject), and no MCP import tool. Round-trip equality is proven only at pydantic level (`test_circuit_definition.py::test_json_round_trip_semantic_equality`) — a user or agent who exports a circuit has no way to bring it back.
*Recommendation:* Add kind-keyed import (reject unknown kind/major with a clear 422) + an export→import→export equality test at the API level. This is also the natural home for the Task 5.1 hostile-input tests.

**PE-3 · P1 — `from_candidates(run_id, selection)` assembly not implemented.**
FTID §2.4 and FPRD §5 ("POST /circuits ... from discovery candidates") specify assembly from a discovery run. `CircuitService.create` accepts only fully-formed payloads. When 016 lands, its candidates have no assembly interface, and the classifier (§3.2 item 6: "016 candidate tables and run reports consume this classifier") has zero callers today.
*Recommendation:* Either implement the assembly seam now (even stubbed on run-record shape) or explicitly move it into 016's FTASKS with a cross-reference — right now it's silently dropped between the two features.

**PE-4 · P2 — Evidence-surface UI gaps: no EdgeTable component, no CapsMeter, no filters, no slice-export UI.**
Against FPRD §4 / FTID §2.8: (a) the inline `EdgeRow` shows type/rung/PMI/support/attr/prior but omits `null_percentile`, `replicated_heldout`, `tested_and_failed`, validation status, and **manifest links** (§3.3 item 8); (b) no edge-type filter UI (US-4: "filter the edge list to persistence edges") and no rung filter in the list; (c) no `CapsMeter.tsx` (per-layer member counts vs cap — FTID pitfall explicitly warns per-layer, not total); (d) `export-slices` is reachable via API/MCP but has no UI action (FPRD §4 "Export / Export slices"); (e) cluster members render name-only, no drill-down (§3.3 item 9).
*Recommendation:* Land EdgeTable + CapsMeter + slice-export button as the first 018-follow-up; these are the review surfaces 017's manifest refs will need anyway.

**PE-5 · P2 — SteeringPanel rung-chip integration not done.**
FPRD §4 ("rung chips everywhere ... steering panel (loaded circuit)") and Task 5.3. `RungChip` is used only inside the Circuits panel. Defensible while 015 (multi-SAE loading) is unbuilt, but the FPRD assigns the chip surface to 018 — and §8 criterion 4 (promote → load in 015 → steer, rung-chipped) is unmeetable this increment.
*Recommendation:* Record as deferred-to-015-landing in FTASKS with an explicit hand-off note so it isn't lost between features.

**PE-6 · P2 — Copy-audit does not cover frontend strings; panel hand-writes ladder-adjacent language.**
§3.1 item 2 requires the causal-language audit as a shared suite. The backend test checks only `RUNG_LANGUAGE` itself. `CircuitsPanel.tsx` hand-writes "discovery results are Tier-1 (same-token) associations until validated" — currently compliant, but nothing prevents a future edit from introducing "causal" below rung 2 in UI copy.
*Recommendation:* Extend the grep-style audit to `frontend/src` (forbid /causal/i outside `evidenceLadder.ts`-mirrored server strings and rung-2+ contexts), shared with 017 as the FPRD specifies.

**PE-7 · P3 — MCP loop is create/promote/export only; no update/delete; manual not updated.**
US-7 ("drive review → promotion → export end-to-end") — an agent cannot edit members or narrative via MCP. FTID §2.9 manual additions (`circuits.md` ladder-led page, mcp-server.md tool list) are absent from `manual/docs`.
*Recommendation:* Add `update_circuit`/`delete_circuit` MCP tools (respecting the approvals pattern if destructive) and the manual pages in Phase 6.

**PE-8 · Info (positive) — BR traceability of the locked decisions is real, not aspirational.**
Badge-not-gate is asserted in three layers (service test, API test, MCP docstring); per-layer caps tested incl. the 21-member rejection with exact message; persistence de-rank-not-hide is visible in the UI (opacity, never filtered out); `DEFAULT_CATEGORIES` gains `circuits` consistent with the profiles precedent.

---

## 2. QA ENGINEER — error handling, validation, security

### Findings

**QA-1 · P1 — Unvalidated scalar fields reach DB column limits → unhandled 500s.**
`CircuitCreate.granularity` is a free `str` (DB `String(16)`); `discovery_run_id` free str (`String(36)`); `model_id` free str (`String(255)`). A hostile or merely sloppy payload (`granularity: "x"*32`) passes pydantic and the contract round-trip (granularity/model_id/discovery_run_id are NOT part of `CircuitDefinitionV1`), then raises `StringDataRightTruncation` at commit → raw 500, transaction left mid-flight. Same class: `granularity` accepts any string, so list-filter semantics ("feature"|"cluster") silently degrade.
*Recommendation:* `granularity: Literal["feature", "cluster"]`; `max_length` on `discovery_run_id`/`model_id`; a regression test posting each overflow.

**QA-2 · P2 — Export filename shaping is Unicode-permissive → Content-Disposition header encode failure.**
`circuits.py::export_circuit`: `safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" ...)` — `str.isalnum()` is True for all Unicode letters (CJK, Cyrillic, etc.). Starlette encodes headers latin-1; a circuit named "回路A" yields a header value that raises `UnicodeEncodeError` → 500 on export. (Injection itself is handled — quotes/CR/LF are replaced — this is an availability bug, not an injection one.)
*Recommendation:* Restrict to `ch.isascii() and ch.isalnum()`, fall back to the circuit id when the result is empty/all-dashes, and add a test with a non-ASCII name. Consider RFC 5987 `filename*` if Unicode names should survive.

**QA-3 · P2 — `GET /circuits` is unbounded and returns full JSONB bodies per row.**
No `page`/`limit` (house API standard per CLAUDE.md: `?page=1&limit=50`), and `_out` inlines complete `members`/`edges` (up to 20/layer × 16 layers + 200 edges each) for every circuit in the list. Tens of circuits → multi-MB list responses; the panel refetches the whole list after every promote/delete.
*Recommendation:* Add pagination + a summary list shape (counts + rung + name), full body on GET-by-id only.

**QA-4 · P2 — `recompute_rung`/serializer trust stored JSONB; out-of-range rung raises unhandled.**
`EvidenceRung(e.get("rung", 0))` raises `ValueError` for any rung outside 0–3. `recompute_rung` is the documented 017 write-path hook ("017 writes validation results, then calls this") but performs **no contract validation** — unlike `update()`. If 017 (or a manual DB fix) writes a malformed edge, recompute 500s; and `_out`'s `EvidenceRung(circuit.rung)` has the same failure mode, poisoning every read of that circuit.
*Recommendation:* Clamp/validate inside `recompute_rung` (or route 017's edge writes through `update()` and document that recompute alone is not a safe write path); serializer falls back to rung 0 with a logged warning rather than 500.

**QA-5 · P2 — Task 5.1 hostile-input tests absent; several endpoint error paths uncovered.**
No tests for: oversized narrative via PATCH, wrong-type JSONB payloads (e.g. `edges: [{"rung": "high"}]` — surfaces as 422 via the contract, but unpinned), export of a circuit whose stored JSONB no longer validates (the intended 500 "Stored circuit invalid" path), granularity filter with junk values.
*Recommendation:* A small hostile-payload matrix test against POST/PATCH; one test seeding invalid JSONB and asserting the export 500 shape.

**QA-6 · P3 — Response envelope inconsistent with house `{data, meta}` standard.**
`{"circuits": [...], "total": n}`, bare object on GET, `{"deleted": id}` on DELETE. Consistent with some sibling endpoints (cluster_profiles), so this is drift the house already tolerates — noting for the eventual sweep.
*Recommendation:* Leave as-is for now; align if/when the API-envelope cleanup happens.

Security review otherwise: JSONB payloads are schema-validated before storage (pydantic contract) and re-validated on export; narrative renders through React text nodes (`whitespace-pre-wrap`, no `dangerouslySetInnerHTML`) so no XSS; no path construction from user input anywhere in the new code; MCP tools proxy through the authenticated client with no shell/file surface. No injection findings.

---

## 3. ARCHITECT — design consistency, coupling for 016/017

### Findings

**ARCH-1 · Info (positive) — Validation-via-contract is the right pattern and is executed well.**
Every service write round-trips `CircuitDefinitionV1`, making the contract validators (per-layer caps, edge-endpoint integrity, layer ascension, SAE-ref completeness) the single structural authority — no duplicated validation, and export-is-a-projection falls out of the JSONB-mirrors-contract storage decision. Static-method service style matches the house (cluster-profile/steering services). The migration is clean: soft `discovery_run_id` (documented rationale — runs prunable), `(promoted, rung)` index matches the list-filter access path, symmetric downgrade, single Alembic head (verified during review: `9a7da58fcd50` is the sole head across 86 revisions).

**ARCH-2 · P1 — `type_signals` contract type cannot hold the classifier's disclosure payload.**
`CircuitEdge.type_signals: Optional[Dict[str, float]]` (`circuit_definition.py:93`) vs `classify_edge()` returning `signals = {"weight_prior": float, ..., "label_method": "token_set" (str), "thresholds": {...} (dict), "votes": {...} (dict of bool), "echo_confidence"/"distinctness": float}`. The FIRST attempt (016, or anyone today) to store the disclosed signals on an edge — the very disclosure BR-021/§3.2 item 4 mandates — fails contract validation with a type error. The two halves of this feature don't compose.
*Recommendation:* Widen to a typed `EdgeTypeSignals` model (floats + `label_method: str` + `thresholds: Dict[str,float]` + `votes: Dict[str,bool]`) in the contract *now*, pre-freeze (amendments land inside v1 per the module header), regenerate the vendored schema, and add a test that `classify_edge()["signals"]` validates as `CircuitEdge.type_signals`. This is the cheapest moment it will ever be.

**ARCH-3 · P2 — Classifier deviates from spec, and the deviation is calibrated to its own fixture.**
(a) FTASKS 4.1 specifies token-overlap threshold **0.8**; code ships `O_HI_DEFAULT = 0.5` with a comment claiming calibration "on the audit fixture". Verified empirically during review: at the spec's 0.8, fixture case `echo_overlap_and_label_low_prior` (Jaccard exactly 0.5) classifies computed → persistence recall drops to 75% → gate fails; 0.5 is precisely the value that makes the fixture pass. (b) The FTID §3 degradation rule — no embedding stack ⇒ require 2/2 of the remaining strong signals — is not implemented; token_set label similarity counts as a full third vote, and the `classify_edge` docstring describes a special counting rule the code doesn't contain (it is plain 2-of-3). The disclosure (`label_method: "token_set"`) is honest; the vote-weighting is not what the FTID prescribed.
*Recommendation:* Either implement the degraded-mode 2-of-2 rule, or amend FTID/FTASKS to the shipped rule with rationale — but not silently. Revisit O_HI once the fixture grows (TE-3); a threshold chosen by one boundary case is not calibration.

**ARCH-4 · P2 — Distinctness lets the weight prior down-rank a computed edge on its own — tension with BR-020.**
For non-persistence edges `echo_confidence = max(prior/p_hi, overlap/o_hi, label_sim/s_hi)/2`: a computed edge with a lone high prior (fixture case `computed_one_strong_signal_only`) gets distinctness ≈ 0.47 (verified) — the prior alone halves the only ranking-facing output, while FPRD §3.2 item 5 says the prior participates in ranking "only combined with distinctness signals". The pinned test (`test_low_prior_high_association_is_computed_not_penalized`) covers only the low-prior direction.
*Recommendation:* For edges classified computed with exactly one firing signal, cap any single signal's echo_confidence contribution (or require ≥2 partial signals before echo_confidence exceeds ~0.25), and pin the high-prior-alone case.

**ARCH-5 · P2 — Panel state: local `useState` vs the house zustand pattern (FTID prescribed `circuitStore.ts` extensions).**
Acceptable for this read-mostly v1 panel taken alone — no cross-component sharing, no WS. But 016/017 add discovery-run progress (WS channel), candidate tables, and validation status *into this same panel*, and the steering panel must show rung chips for loaded circuits (PE-5) — cross-panel shared state. Every sibling panel that grew those needs runs on a store; retrofitting later means churning this component.
*Recommendation:* Extract `circuitsStore.ts` (list + selected + refresh + optimistic promote) before 016's tabs land; treat as a 016-precursor task.

**ARCH-6 · P3 — `resolve_encoder_weight` pre-landed with no callers and no tests.**
The steering-service addition is a sensible companion to `resolve_decoder_weight` and the right single-orientation source for the IDL-32 weight prior — but nothing exercises it, and the `[d_sae, d_model]` orientation claims across tied/JumpReLU/Linear families are exactly the kind of silent-transpose bug 016's cosine math would mask.
*Recommendation:* A unit pin per SAE family asserting shape/orientation (mirror the decoder resolver's pins) before 016 consumes it.

**ARCH-7 · Info — 016/017 interface readiness (requested by review charter).**
Ready: rung enum import path (`src.schemas.evidence_ladder` — the grep-guard test prevents forks); MCP category design (`circuits` in DEFAULT_CATEGORIES mirrors the `profiles` read-write precedent; millm_* stay opt-in); circuit read path. **Needs the notes above before 016/017 land:** classifier consumption (ARCH-2 signals shape; PE-3 no assembly seam; zero callers today), and the 017 edge-write path — 017 must write edges through `CircuitService.update()` to get contract validation; `recompute_rung` alone neither validates nor tolerates bad rungs (QA-4). Document the sanctioned write path in 017's FTID.

---

## 4. TEST ENGINEER — strategy effectiveness: what can silently break

### Findings

**TE-1 · P1 — API layer tested almost entirely against mocks; real router→service→DB wiring unexercised.**
`test_circuits_api.py` patches `CircuitService.get/promote` for every success path; only the create-422 test runs real code (and it fails *before* the DB). Never tested at the API level: `GET /circuits` (filter param parsing — `promoted` bool coercion, `min_rung` ge/le bounds), `PATCH` (exclude_unset merge semantics), `DELETE`, `GET /export` (JSONResponse + Content-Disposition shaping — where QA-2 lives, invisible to the suite), the real `POST /export-slices` path. The service tests do hit a real DB, so the gap is specifically the seam — the classic place a rename or param change breaks silently.
*Recommendation:* One happy-path integration test per route against the test DB (the `async_engine` fixture already exists in the service tests); explicitly test the export filename header with an awkward name.

**TE-2 · P2 — No migration up/down test and no single-head guard in CI.**
FTASKS 3.1 requires "single-head check, up+down". Manual verification for this review confirms a single head today, but this repo has a prior multi-head incident (merge `cd6c46abac48`), and nothing automated prevents recurrence; the downgrade path has never executed anywhere.
*Recommendation:* A pytest that runs `alembic heads` (assert exactly one) and `upgrade head` → `downgrade -1` → `upgrade head` against the test DB.

**TE-3 · P2 — Audit fixture too small for the gates it enforces, and the classifier was tuned on it.**
10 cases: 4 persistence, 5 computed, 1 attention. The ≥90% recall gate degenerates to "all 4"; the ≤10% misclassification gate to "0 of 5" — one flipped case is the difference between 100% and gate-failure, so the thresholds have no statistical meaning at this n. Combined with ARCH-3 (O_HI chosen to be exactly one case's Jaccard), the gate currently certifies the fixture, not the classifier. FPRD §7 anticipated seeding from real cluster profiles.
*Recommendation:* Grow to ≥20 per class (seed from actual cluster-profile members + real discovery output once 016 runs), keep the current 10 as a smoke subset, and only then treat the 90/10 numbers as meaningful.

**TE-4 · P3 — TS mirror sync pins enum literals only; `EdgeEvidence` shape and frontend `circuitRung` unpinned; zero frontend tests.**
`test_ts_mirror_in_sync` regexes `NAME = value` pairs — the `EdgeEvidence` interface and the reimplemented `circuitRung()` min-rule in `evidenceLadder.ts` can drift freely (backend changing empty-edge or min semantics would fail nothing frontend-side). No Vitest coverage for `RungChip` (contract: server strings verbatim, tooltip from `rung_next_step`) or `CircuitsPanel` (FPRD §10 lists frontend tests explicitly).
*Recommendation:* Extend the sync test to assert `EdgeEvidence` field names; add RungChip + panel render tests (empty state, chip text verbatim from props, persistence-row de-emphasis present-not-hidden).

**TE-5 · P2 — No E2E, no MCP tool invocation tests; §8 criterion 4 structurally open.**
FPRD §10/§8: E2E (review → promote → steer → export → re-import + screenshot cap) absent; promote→015-load→steer is blocked on 015 (fine) but §8 criterion 4 must be tracked as open, not assumed; Task 5.2's "rung + rung_language embedded in ALL circuit-returning tools (smoke-tested)" has no test — the six MCP tools in `tools/circuits.py` are never invoked by any test (only the category-set assertion in `test_mcp_server_foundation.py` was touched).
*Recommendation:* An MCP smoke test invoking each circuits tool against a mocked client asserting `rung`/`rung_language` passthrough; E2E + screenshot in Phase 6 as planned — keep FTASKS honest that Phase 6 is entirely open.

**TE-6 · P3 — List ordering and behavior-under-growth unpinned.**
`updated_at desc` ordering has no test; no pagination exists (QA-3) so nothing defines behavior at scale; `test_list_filters` asserts exact single-element lists that will become order-dependent once pagination lands.
*Recommendation:* Pin ordering explicitly when adding pagination tests.

---

## Summary table

| # | Perspective | Severity | Finding |
|---|---|---|---|
| PE-1 | Product | P1 | No review-tab authoring UI (edit members / name / narrative) — PATCH API exists, unreachable from UI |
| PE-2 | Product | P1 | Import path missing entirely (no endpoint, no contract service, no MCP tool) — round-trip criterion unmeetable |
| PE-3 | Product | P1 | `from_candidates` discovery-assembly seam not implemented; classifier has zero callers |
| PE-4 | Product | P2 | UI evidence gaps: no EdgeTable (manifest links, null-percentile, tested_and_failed), no type/rung filters, no CapsMeter, no slice-export UI, no cluster drill-down |
| PE-5 | Product | P2 | SteeringPanel rung-chip integration not done (blocked on 015 — needs explicit hand-off) |
| PE-6 | Product | P2 | Copy audit doesn't cover frontend strings; panel hand-writes ladder-adjacent copy |
| PE-7 | Product | P3 | No MCP update/delete; manual pages (circuits.md, mcp-server.md) not written |
| PE-8 | Product | Info+ | Locked decisions (badge-not-gate, per-layer caps, typed persistence, ladder-on-every-response) verifiably implemented |
| QA-1 | QA | P1 | granularity/discovery_run_id/model_id unvalidated → DB truncation 500s; granularity should be a Literal |
| QA-2 | QA | P2 | Unicode-permissive export filename → latin-1 header encode 500 (availability, not injection) |
| QA-3 | QA | P2 | List endpoint unbounded, full JSONB bodies per row — no pagination per house standard |
| QA-4 | QA | P2 | `recompute_rung`/serializer trust stored JSONB; out-of-range rung → unhandled 500; recompute path skips validation |
| QA-5 | QA | P2 | Task 5.1 hostile-input tests absent; export-of-invalid-stored-circuit 500 path untested |
| QA-6 | QA | P3 | Response envelope drifts from `{data, meta}` house format (consistent with siblings) |
| ARCH-1 | Architect | Info+ | Contract-round-trip validation pattern sound; service style, model, migration quality house-consistent; single Alembic head verified |
| ARCH-2 | Architect | P1 | `type_signals: Dict[str, float]` cannot hold `classify_edge()` disclosure (str/nested dicts) — 016's first write fails contract validation |
| ARCH-3 | Architect | P2 | Classifier deviates from spec: O_HI 0.8→0.5 tuned to its own fixture (verified); FTID degraded-mode 2/2 rule not implemented; docstring describes uncoded behavior |
| ARCH-4 | Architect | P2 | Lone high weight prior halves distinctness of a computed edge — tension with BR-020 "prior never a standalone ranking input" |
| ARCH-5 | Architect | P2 | Panel on local useState vs FTID-prescribed zustand store — acceptable now; extract before 016/017 tabs and steering-chip sharing |
| ARCH-6 | Architect | P3 | `resolve_encoder_weight` pre-landed with no callers, no orientation pin tests |
| ARCH-7 | Architect | Info | 016/017 readiness: enum path/MCP category ready; classifier consumption + 017 edge-write path need ARCH-2/QA-4/PE-3 closed; document the update()-only write rule |
| TE-1 | Test | P1 | API success paths fully mocked — list/patch/delete/export routes and param plumbing never executed by any test |
| TE-2 | Test | P2 | No migration up/down test, no single-head CI guard (house has a prior multi-head incident) |
| TE-3 | Test | P2 | 10-case audit fixture makes the 90%/10% gates degenerate (4/4, 0/5) and self-referential |
| TE-4 | Test | P3 | TS sync test pins enum literals only (EdgeEvidence shape, circuitRung drift-able); zero frontend tests for RungChip/panel |
| TE-5 | Test | P2 | No E2E, no MCP tool invocation tests; §8 criterion 4 open pending 015 — track, don't assume |
| TE-6 | Test | P3 | List ordering/pagination behavior unpinned |

## Recommended fix order (before 016 implementation starts)
1. **ARCH-2** — widen `type_signals` in the contract + regenerate vendored schema + `classify_edge` compatibility test (cheapest pre-freeze).
2. **QA-1** — Literal granularity + length caps + overflow tests.
3. **PE-2** — kind-keyed import endpoint + API-level round-trip test.
4. **PE-3 / ARCH-7** — decide the assembly seam's home (018 follow-up vs 016) and document 017's sanctioned edge-write path.
5. **TE-1 / TE-2** — real-wiring route tests + migration/single-head guard.
Then: QA-2/3/4, ARCH-3/4 (with fixture growth TE-3), UI batch (PE-1/PE-4, ARCH-5), and the Phase-6 items (E2E, manual, MCP smoke, screenshot cap).

---
*Review conducted 2026-07-19 · scope commit 9532bc4 · verdict SHIP-WITH-NOTES (6 P1 / 10 P2 / 4 P3 / 2 positive).*
