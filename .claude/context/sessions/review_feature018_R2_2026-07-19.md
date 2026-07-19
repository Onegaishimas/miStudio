# Review Record — Feature 018: Circuit Review, Evidence Ladder & Portability — ROUND 2

**Date:** 2026-07-19
**Scope:** post-fix state at HEAD `f1be274` ("018 review round 1 — 49 findings triaged, 21 fixed"), reviewed AGAINST THE CURRENT CODE, not the diff. Round 1 record: `review_feature018_circuit_portability_2026-07-19.md` (verdict SHIP-WITH-NOTES, 6 P1 / 10 P2 / 4 P3).
**Process:** house `/review` — four perspectives (Product, QA, Architect, Test), READ-ONLY.
**Live verification performed:** circuit test files executed against the real test DB (all green; `TestRealWiring::test_full_lifecycle_through_real_stack` runs and PASSES in 0.98s against `mistudio_test` — not skipped); model-ref/granularity round-trip loss reproduced with a scripted repro (see ARCH-1).

---

## Verdict: **SHIP** (with 4 fix-now items, all small)

The round-1 P1 wall is down. Verified in current code: **import is real** (`POST /circuits/import`, kind-keyed reject, MCP `import_circuit_definition`, exercised end-to-end by the real-wiring lifecycle test including re-import); **`type_signals` now holds the classifier's full disclosure** (`Dict[str, Any]`, schema regenerated); **scalar validation landed** (granularity `Literal`, length caps on name/narrative/discovery_run_id/model_id, granularity-typo 422 pinned); **the API seam is exercised for real** (create→list→patch→promote→slices→export→import→delete through router→service→DB, no mocks); **promotion is reversible** ({promoted:false} + Unpromote button); **list is slim + paginated** (limit/offset, summary rows pinned); **filename ASCII-safe** (pinned with a non-ASCII name); **recompute_rung clamps** malformed rungs; **inline name/narrative editing, persistence show/hide, per-layer caps meter (n/20), slices button** all present in the panel; **O_HI 0.8→0.5 deviation and the token_set positive-evidence rule are now recorded in the FTDD** (no longer a silent deviation).

What keeps four items at fix-now rather than defer: a **new** BR-013 losslessness gap the fix wave exposed (the `model` ref is dropped on import AND emitted empty on export — verified by repro; granularity of imported cluster circuits silently defaults to "feature"), the **FTASKS file that still says "Planned" with every box unchecked** (the deferrals the increment is relying on are recorded nowhere), the **empty-state copy that promises a UI import affordance that does not exist**, and the **unbounded import payload surface** (no app-level size cap; `type_signals` is now deliberately `Any`). Everything else is honestly deferrable to 016/017-landing — itemized per finding below.

Finding counts: **15 findings** — 0 P1, 6 P2, 7 P3, 2 positive/info.

---

## 1. PRODUCT — FPRD §3 (1–15), §4 UI, §8 criteria re-audit

Round-1 P1 status: PE-1 **partially fixed** (name/narrative inline edit ✅; member editing ❌), PE-2 **fixed** (import real, round-tripped in tests), PE-3 **still open by design** (deferral to 016 — but see R2-P1, it is recorded nowhere).

Requirement-by-requirement delta vs R1: items 1–3 ✅ (unchanged); 4 ✅ (deviations now documented in FTDD — resolves the "silently" half of R1 ARCH-3); 5 ✅ (computed edges now distinctness=1.0 by definition — the prior can no longer de-rank them at all, the safe reading of BR-020); 6 ◐ (typed + toggle in detail; **no list-level type/rung filter controls** — API has them, UI doesn't render them); 7 ✅ (fixture gate; still 10 cases — TE side); 8 ◐ (edge table now shows manifest refs + ES; null_percentile/tested_and_failed still absent); 9 ◐ (name/narrative ✅, member editing ❌ in UI — PATCH/MCP cover it); 10–11 ✅; 12–13 ✅; 14 ◐ (**round-trip loses the model ref — see ARCH-1**); 15 ✅ (+ single-sourced parent rung, pinned).

### Findings

**R2-P1 · P2 · fix now (docs-only, ~15 min) — FTASKS/FPRD do NOT record the deferrals this increment leans on.**
`018_FTASKS` is still **Status: Planned, every checkbox `[ ]`**, all six phases "⏳". Task 3.2 still lists `from_candidates(run, selection)` as an 018 deliverable with no hand-off note to 016; Task 5.3 still lists member-edit UI + EdgeTable filters + SteeringPanel chip; Phase 6 (E2E, screenshot, manual, acceptance) is open but indistinguishable from "not started because nothing is". Round 1 asked for exactly this honesty (PE-1/PE-3/PE-5 recommendations) and the fix wave shipped code but not the record. The deferral story currently lives only in commit messages and review records.
*Recommendation:* Mark done items `[x]`, and annotate: `from_candidates` → **deferred to 016 FTASKS** (it is 016's input shape; classifier + create/import are the 018-side seams and they exist); member-edit UI + list filter controls → deferred to 016-landing (API/MCP cover the workflows meanwhile); SteeringPanel rung chip → deferred to 015-landing; Phase 6 E2E/screenshot/manual → open, blocked items marked blocked-on-015/016. Update FTASKS status header.
*Deferral verdicts (explicit, as charged):* `from_candidates` to 016 — **acceptable** (it consumes 016's run-record shape; building it now would invent that shape twice). Member-edit UI — **acceptable to defer** (PATCH validates caps server-side; MCP `update_circuit` gives agents the loop). Rung/type filter controls — **acceptable to defer** to when 016 floods the list with candidates (today's list is hand-curated and short); the API being ready makes this pure UI work. E2E + screenshot (§8 crit 4, Task 6.1) — **acceptable only as a tracked-open item**: crit 4 is structurally blocked on 015, but the export→import→export equality E2E and the screenshot are NOT blocked and should land with 016's Playwright pass at the latest.

**R2-P2 · P3 · fix now (one line or one button) — Empty-state copy promises an import UI that doesn't exist.**
`CircuitsPanel.tsx` empty state: "…or by importing a `.circuit.json` file." There is no file input, no import button anywhere in the panel; `circuitsApi.importDefinition` has **zero callers** in the frontend. A user reads the sentence, looks for the button, and concludes the feature is broken.
*Recommendation:* Either add the file-picker import button (the API client method already exists — ~30 lines mirroring the cluster-profile import pattern) or reword the copy to name the MCP/API route only. Adding the button is the better spend; it also makes §8 criterion 5 user-demonstrable.

**R2-P3 · P3 · defer (note the drift) — Persistence edges are now hidden-by-default, not de-emphasized-by-default.**
R1 praised "de-emphasized (opacity), never filtered out." Current detail view defaults `showPersistence=false`: persistence rows are removed from the table with a "show N persistence edges" count-button. The count-button preserves *disclosure* (the user always sees they exist), so this is defensible under FPRD item 6 "de-ranked from default views; queryable" — but it is a silent strengthening beyond both the R1-reviewed behavior and the BRD's "de-emphasized not hidden" phrasing.
*Recommendation:* Either restore visible-but-dimmed as the default with the toggle inverting to "hide", or record the hidden-by-default choice as the intended reading of BR-020 in the FTDD. One sentence either way.

**R2-P4 · Info (positive) — The R1 UI batch genuinely landed and matches the FPRD's shapes.**
Inline name/narrative edit wired to PATCH with error surfacing; caps meter is per-layer `(n/20)` counting cluster expansions (the exact pitfall the FTID warned about); slices button downloads client-side with a sanitized filename; unpromote is one click with honest hover copy; markdown narrative renders through react-markdown (no `dangerouslySetInnerHTML`); the promoted badge and rung chip coexist on the row. §8 criterion 5's semantic core (export→import preserving rungs/edge types/discovery) is now machine-verified in `TestRealWiring`.

---

## 2. QA — hostile inputs, error propagation, concurrency

Round-1 status: QA-1 **fixed** (Literals + length caps + pinned 422s), QA-2 **fixed** (pinned), QA-3 **fixed at the serializer** (see ARCH-3 for the SQL half), QA-4 **fixed** (clamp + pin), QA-5 **partially open** (matrix below).

### Findings

**R2-Q1 · P2 · fix now (small middleware or endpoint guard) — No application-level size cap on import/create; `type_signals: Any` made the surface unbounded.**
Checked every layer: nginx allows `client_max_body_size 100M` (docker/k8s confs) and **`0` = unlimited in `docker/nginx.gcp.conf`**; FastAPI has no body-size middleware; `CircuitCreate`/import cap name (120), narrative (10k), edges (200), saes (16), per-layer members (20) — but each edge's `type_signals` is now `Dict[str, Any]` (the R1 #1 fix) with **no depth or size bound**, and `budget`/`discovery`/member extras ride the same JSON body. Consequences: a 100MB import parses, validates, and lands in one JSONB row (list stays slim, but every detail GET/export of that circuit ships it back); a pathologically deep `type_signals` raises `RecursionError` during parse/validation → raw 500. Non-dict payloads are fine (FastAPI 422s a JSON array/string body — verified by the parameter type).
The FTASKS coverage-audit line literally promised "Security: import kind/**size** validation (house caps)" — the size half never shipped.
*Recommendation:* (a) a Content-Length guard on `/circuits/import` and `POST /circuits` (reject > ~2MB with a clear 413/422 — a legitimate circuit is tens of KB); (b) bound `type_signals` serialized size per edge (e.g. reject > 16KB) inside the contract validator so the cap travels with the contract; (c) fix `nginx.gcp.conf`'s `client_max_body_size 0` independently — that's a global exposure beyond circuits.

**R2-Q2 · P2 · defer to pre-017 (re-flagged from R1, unfixed as predicted) — No optimistic concurrency; three writers are coming.**
Confirmed NOT fixed in the current code: no version column, no `If-Match`/`updated_at` precondition on PATCH; `CircuitService.update()` merges request-fields over the row read in the same request (classic lost-update between a GET and a PATCH); `recompute_rung` does read-edges→write-rung with no guard; the panel's `saveEdit` PATCHes a name+narrative snapshot taken when the edit opened (stale clobber of a concurrent MCP rename). Single-human-writer today, so nothing is bleeding — but 017 writes edges + calls `recompute_rung` while agents PATCH via MCP and the UI edits narratives: that is a genuine 3-writer world where an interleaved edge-write and rung-recompute can commit a rung that matches neither edge set.
*Recommendation:* Before 017's write path lands (hard precondition, put it in 017's FTASKS): add `updated_at` (or an integer `version`) as an optional precondition on PATCH → 409 on mismatch, and make 017's edge-write + rung-recompute a single `update()` transaction rather than two commits. **Severity escalates to P1 the day 017 starts writing.**

**R2-Q3 · P3 · defer (bundle with 016 hardening) — Import error branches thin; hostile matrix from R1 QA-5 still unwritten.**
Tested today: unknown-kind 422 ✅, re-import round-trip ✅. Untested: valid-kind + invalid definition (the `ValidationError` branch — whose detail collapses to **only the first error's `msg`, dropping the `loc`**, so "Invalid circuit definition: 1 error(s): Input should be a valid integer" arrives with no field path); the post-validation `CircuitValidationError` branch; `schema_version: "2"` (rejected via the Literal — works, but the message is a generic literal-mismatch, not the FTID's "unknown major version" wording); oversized payload; deep nesting. fetchAPI propagation itself is sound (server `detail` string or FastAPI array → joined `loc: msg` → APIError.message → panel banners show it verbatim) — the weakness is what the server puts in `detail` on the import path, not how the client shows it.
*Recommendation:* Include `loc` in the import 422 detail; add the 5-case hostile matrix when R2-Q1's caps land (they share test scaffolding).

---

## 3. ARCHITECT — fix-architecture cleanliness, API design, 016/017 readiness

Round-1 status: ARCH-2 **fixed** (by widening, see R2-A2), ARCH-4 **fixed** (computed ⇒ distinctness 1.0, persistence grades by vote count — clean resolution of the BR-020 tension), ARCH-5 **open by agreement** (store extraction remains a 016-precursor), ARCH-6 **NOT fixed** (see R2-A4).

### Findings

**R2-A1 · P2 · fix now (~20 lines + one test line) — Round-trip loses the `model` ref both directions; import mislabels cluster-granularity circuits. NEW — exposed by the import fix wave.**
Verified by scripted repro against current code: (a) `CircuitService._validate`/`to_definition` never pass `model`, so **every export emits an empty `DefinitionModelRef`** (`{hf_id: null, mistudio_model_id: null}`) even when the row's `model_id` column is set; (b) `POST /circuits/import` drops the incoming `defn.model` on the floor (never threads it to `model_id` or anywhere recoverable); (c) import never sets `granularity`, so a definition whose members are `cluster_ref`s lands with `granularity="feature"` — the list filter and the UI badge then lie about it. BR-013 says the round-trip preserves provenance; the model ref is the single most operationally important provenance field (015 must refuse to load a circuit onto the wrong model) — and it is exactly the field that doesn't survive. The R1 real-wiring test can't see this because its payload never sets a model.
*Recommendation:* Thread `model` through `_validate`→storage (either a `model` JSONB column or map to/from `model_id` + `hf_id`), derive import `granularity` from members (`any cluster_ref ⇒ "cluster"`), and extend `TestRealWiring` to assert `exported["model"]` equality and granularity fidelity. Do this **before 015 consumes promoted circuits** — 015's model-mismatch hazard check needs the field to exist.

**R2-A2 · P3 · defer (revisit at v1 freeze) — `type_signals: Dict[str, Any]` is an escape hatch, not the typed disclosure R1 asked for.**
R1 ARCH-2 recommended a typed `EdgeTypeSignals` model (floats + `label_method: str` + `thresholds: Dict[str,float]` + `votes: Dict[str,bool]`); the fix chose maximal widening to `Any`. It works — the classifier's payload now stores and round-trips — and pre-freeze it keeps 016's options open. But the contract now guarantees nothing about the disclosure's shape (the vendored JSON schema says "object"), there is **no compatibility pin** asserting `classify_edge()["signals"]` conforms to any expectation, and R2-Q1's size concern exists precisely because this field is unbounded. Keep-or-collapse verdict on the *promotion* fix architecture, same question asked of this one: widening was the right pre-freeze call, **but it must not survive the v1 freeze as `Any`**.
*Recommendation:* At freeze time (018's declared release decision), replace with the typed model + `extra="allow"` for forward-compat, and add the classifier-output-validates pin now (one test, works with either typing).

**R2-A3 · P3 · fix now (5 min) + defer (SQL) — Collapse the `promote()` alias; note the list endpoint's Python-side pagination.**
(a) `set_promoted(db, circuit, promoted)` is the real API; `promote()` survives solely for **two call sites in `test_circuit_service.py`** — no production caller (the endpoint and MCP go through `set_promoted`). A back-compat alias for your own two tests is drift bait: collapse it, update the two tests. (b) The summary/detail split is the right design and correctly mirrored in the TS client (`CircuitSummary` vs `Circuit`, detail-on-expand) — but `CircuitService.list` still SELECTs full rows (all JSONB columns) for **all** matching circuits, then the endpoint applies `edge_type` filtering, `total`, and `offset/limit` in Python. Slim on the wire, fat in memory. Fine at today's scale; wrong the week 016 starts minting candidate circuits in bulk.
*Recommendation:* Alias collapse now. Push limit/offset/count into SQL and use `load_only`/deferred JSONB columns as a 016-precursor task (pairs naturally with the ARCH-5 store extraction already parked there). The `edge_type` filter can move to a JSONB containment query (`edges @> '[{"type": …}]'`) at the same time.

**R2-A4 · P3 · defer to pre-016 (re-flagged, unfixed) — `resolve_encoder_weight` still has zero tests and zero callers.**
The R1 fix added only a defensive-branch warning comment. The orientation claims (`[d_sae, d_model]` across tied/JumpReLU/Linear families) remain unexercised; 016's cosine prior will silently produce garbage on a transposed family. Unchanged recommendation: one shape/orientation pin per SAE family (mirror the decoder resolver's pins) as a named 016-precursor task — it should block 016's IDL-32 prior work, not 018.

**R2-A5 · Info — 016/017 interface readiness, concretely (the charter's list).**
**READY:** (1) *Rung enum import path* — `src.schemas.evidence_ladder`, single-source enforced by the grep-guard test (`test_no_parallel_rung_enums`) and the TS mirror pin; safe to import from 016/017 today. (2) *Classifier import path* — `from src.services.circuit_edge_type_service import classify_edge`; output now storable in `type_signals` (verified shape-compatible since it's `Any`); distinctness is the only ranking-facing output as BR-020 demands. Still zero callers — first caller is 016 by design. (3) *Circuit create from discovery runs* — `CircuitService.create` + `POST /circuits` + MCP `create_circuit` accept fully-formed payloads including `discovery` provenance + `discovery_run_id` (threaded end-to-end since R1); the **`from_candidates(run_id, selection)` convenience seam is 016's to build** on top (must be written into 016's FTASKS — see R2-P1). (4) *Contract import path* — kind-keyed, usable by 016's "adopt candidate from another instance" flows.
**NOT READY / conditions:** (5) *Edge write path for 017 validation results* — the sanctioned path is `CircuitService.update(edges=…)` (contract-validated, rung recomputed); `recompute_rung` alone now clamps but still skips contract validation, so it is a *repair* hook, not a write path. **This rule is documented nowhere** — it lives in R1's ARCH-7 paragraph and a code comment. Write it into 017's FTID before implementation. (6) *Concurrency precondition* (R2-Q2) gates 017's writer. (7) *Encoder resolver pins* (R2-A4) gate 016's prior.

---

## 4. TEST — what the suite actually proves now

Round-1 status: TE-1 **fixed and verified live** (see R2-T1), TE-2 **open**, TE-3 **open** (deviation now documented, fixture unchanged), TE-4 **partially mitigated** (client rung computation deleted — the riskiest drift vector is gone), TE-5 **open** (Phase 6).

### Findings

**R2-T1 · Info (positive, verified by execution) — The real-wiring test runs against the real DB inside the suite, not skipped.**
Executed during this review: `TestRealWiring::test_full_lifecycle_through_real_stack` → **1 passed in 0.98s** against `mistudio_test` via the shared `async_engine` fixture (which creates real tables + enums per test). It exercises create→list→patch→promote→export-slices→export→**import**→delete through the genuine router→service→DB stack and asserts discovery-provenance losslessness and re-imported edge rungs. There is no skip guard — if the DB is down the test errors rather than skips, which matches house convention (memory: unit tests auto-create `mistudio_test`). All six circuit test files pass green.

**R2-T2 · P2 · fix now (one test file, ~30 lines) — Migration pair still manual-only; no single-head guard. (TE-2, re-flagged, risk now growing.)**
`3e9c439d9085` (discovery column + server-default timestamps) has a clean symmetric downgrade **on paper**; "up/down/up verified" exists only in the commit message. No test runs `alembic heads` (single-head assert) or `upgrade head → downgrade -1 → upgrade head`. The repo has a prior multi-head incident (`cd6c46abac48`), there are now **two** circuits migrations, and 016/017 will each add more within weeks. This is the cheapest insurance in the whole backlog and it keeps not being bought.
*Recommendation:* `tests/unit/test_alembic_sanity.py`: assert exactly one head; run up/down-1/up against the test DB (sync engine, `DATABASE_URL_SYNC`). Land it before 016's first migration.

**R2-T3 · P3 · fix now (2 assertions) — The R1 distinctness fix is unpinned; it can silently regress.**
The formula rewrite ("computed edges are NEVER de-ranked — distinctness 1.0 by definition") is the direct fix for R1 #3/ARCH-4, but no test asserts it: `test_low_prior_high_association…` asserts only `distinctness > 0.5` on the *low*-prior case; the fixture gate checks `type` only. The specific regression R1 verified empirically (lone high prior → distinctness ≈ 0.47) would pass today's suite if reintroduced.
*Recommendation:* Pin `classify_edge(weight_prior=0.95, disjoint tokens, unrelated labels)` → `type == "computed"` **and** `distinctness == 1.0`; pin persistence grading (`2-of-3 → 1/3`, `3-of-3 → 0.0`).

**R2-T4 · P3 · defer to 016-landing — Frontend remains at zero tests for circuits; TS mirror pins values only.**
Unchanged from TE-4, with one honest improvement: deleting the client-side `circuitRung()` removed the worst drift vector, and `evidenceLadder.ts` now documents the pin. Still true: `EdgeEvidence`'s field names are unpinned; `RungChip`, `CircuitsPanel`, and every R1-added behavior (edit save/cancel, persistence toggle, caps-meter math including cluster-expansion counting, slices download) have no Vitest coverage. The panel churn expected from 016's tabs argues for writing these against the post-store-extraction shape rather than twice.
*Recommendation:* Bundle with the ARCH-5 store extraction as 016-precursor work: RungChip (verbatim server strings), caps-meter counting (cluster expansion weight), persistence-toggle disclosure (count-button always present when hidden).

**R2-T5 · P2 · defer with named owner — Phase 6 remains entirely open; fixture still certifies itself.**
(a) No E2E, no screenshot cap (`0xcc/caps/` has no circuit capture), no manual pages (`manual/docs` has no circuits page — the only "circuit" hits are incidental words in steering docs), no MCP tool-invocation smoke (all 9 MCP circuit tools still never called by any test — the import/update/delete additions widened this gap). (b) TE-3 unchanged: 10-case fixture, gates degenerate to 4/4 and 0/5, O_HI equals one boundary case's Jaccard — now *disclosed* in the FTDD (real improvement: the deviation is a recorded decision, not a silent one) but statistically still self-certification. Both were explicitly planned as post-016 (fixture seeds from real discovery output; E2E needs candidates to review).
*Recommendation:* Acceptable to defer **only if R2-P1's FTASKS update names them as open with owners**: MCP smoke + manual pages → 018 Phase 6 (doable now, no 016 dependency — consider pulling the MCP smoke forward since import/update/delete are fresh); E2E + screenshot + fixture growth (≥20/class from real runs) → 016 acceptance gates.

---

## Summary table

| # | Perspective | Severity | Fix-now / Defer | Finding |
|---|---|---|---|---|
| R2-P1 | Product | P2 | **Fix now** (docs) | FTASKS still "Planned", all boxes unchecked; from_candidates/member-UI/filters/Phase-6 deferrals recorded nowhere |
| R2-P2 | Product | P3 | **Fix now** | Empty-state copy promises a `.circuit.json` import UI that doesn't exist (`importDefinition` has zero frontend callers) |
| R2-P3 | Product | P3 | Defer (note drift) | Persistence edges hidden-by-default behind a count-button vs the reviewed "de-emphasized, never hidden" |
| R2-P4 | Product | Info+ | — | R1 UI batch verifiably landed; §8 crit 5's semantic core machine-verified |
| R2-Q1 | QA | P2 | **Fix now** | No app-level size cap on import/create; `type_signals: Any` unbounded; `nginx.gcp.conf` allows unlimited bodies; deep nesting → 500 |
| R2-Q2 | QA | P2→P1 at 017 | Defer to pre-017 (hard gate) | No optimistic concurrency (re-flagged, unfixed): PATCH merge, recompute_rung, and UI snapshot-save all last-write-wins |
| R2-Q3 | QA | P3 | Defer (bundle w/ Q1) | Import 422 drops field `loc`; invalid-definition/oversize/deep-nest branches untested |
| R2-A1 | Architect | P2 | **Fix now** | Round-trip loses `model` ref both directions (verified repro); imported cluster circuits mislabeled `granularity="feature"` — blocks 015's model-mismatch check |
| R2-A2 | Architect | P3 | Defer to v1 freeze | `type_signals` widened to `Any` instead of typed disclosure — right pre-freeze, must not survive freeze; no classifier-output pin |
| R2-A3 | Architect | P3 | Fix alias now; SQL at 016 | Collapse `promote()` alias (2 test callers only); list pagination/edge_type filter still Python-side over full-row SELECTs |
| R2-A4 | Architect | P3 | Defer to pre-016 (gate) | `resolve_encoder_weight` still zero tests/callers (re-flagged); orientation pins must precede 016's cosine prior |
| R2-A5 | Architect | Info | — | 016/017 readiness: enum/classifier/create/import READY; 017 edge-write rule (update()-only) documented nowhere; concurrency + encoder pins gate the consumers |
| R2-T1 | Test | Info+ | — | Real-wiring lifecycle test verified running+passing against the real DB in-suite (not skipped); all circuit suites green |
| R2-T2 | Test | P2 | **Fix now** | Migration up/down + single-head still manual-only; prior multi-head incident; 016/017 migrations imminent |
| R2-T3 | Test | P3 | **Fix now** (2 asserts) | Distinctness rewrite (the R1 #3 fix) unpinned — the exact verified regression would pass today's suite |
| R2-T4 | Test | P3 | Defer to 016-landing | Zero frontend tests for circuits; TS mirror pins enum values only (client rung computation removal did shrink the risk) |
| R2-T5 | Test | P2 | Defer w/ named owner | Phase 6 wholly open (E2E, screenshot, manual, MCP smoke — 9 tools never invoked); 10-case fixture still self-certifying (now disclosed) |

## Fix order
**Fix-now batch (one short session):** R2-P1 (FTASKS honesty — everything else's deferrals depend on it) → R2-A1 (model/granularity round-trip + real-wiring assert) → R2-Q1 (size caps; include the gcp nginx line) → R2-T2 (alembic sanity test) → R2-T3, R2-A3(alias), R2-P2 (trivial).
**Pre-016 gates (named in 016 FTASKS):** encoder-weight pins (R2-A4), store extraction + frontend tests (R2-T4), SQL-side list pagination (R2-A3b), fixture growth + E2E/screenshot (R2-T5), from_candidates seam, list filter controls, member-edit UI.
**Pre-017 gates (named in 017 FTID/FTASKS):** optimistic-concurrency precondition + single-transaction edge-write (R2-Q2), documented update()-only write rule (R2-A5).
**At v1 freeze:** typed `EdgeTypeSignals` (R2-A2).

---
*Round 2 conducted 2026-07-19 · scope HEAD f1be274 (current code) · verdict SHIP (0 P1 / 6 P2 / 7 P3 / 2 positive) · live-verified: circuit suites green incl. real-DB wiring test; model-ref round-trip loss reproduced.*
