# Review Record — Feature 018: Circuit Review, Evidence Ladder & Portability — ROUND 3 (FINAL / CLOSING)

**Date:** 2026-07-19
**Scope:** post-R2-fix state at HEAD `69dbf97` ("018 review round 2 — 26 findings, 15 fixed"). Closing round after R1 (49/21) and R2 (26/15). READ-ONLY.
**Live verification performed:** all six circuit test files executed against the real test DB (`mistudio_test`) — **47 passed, exit 0**, including `TestRealWiring::test_full_lifecycle_through_real_stack` with the new lossless assertions; frontend `tsc --noEmit` clean; ESLint clean on all five 018 frontend files; tz-aware `created_at` import failure **reproduced live** against asyncpg (see R3-B1).

---

## Verdict: **GO for Feature 016** — with 3 must-fix items (2 small code, 1 docs), none of which block starting 016's Phase 1

Every R2 fix-now item verifiably landed and holds at runtime. The fresh sweep found **one new P2 regression introduced by an R2 fix** (tz-aware import 500), **one P2 residue of the R2-A1 losslessness fix** (hf_id still dropped), and a tail of P3/P4 pins-and-polish. The causal-language copy audit across the full 018 surface (backend, frontend, MCP docstrings) is **clean**.

---

## PART A — R2 fix verification table

| # | R2 fix | Verified how | Verdict |
|---|--------|-------------|---------|
| 1 | Lossless round-trip: export carries `model.mistudio_model_id`; import preserves `model_id`, granularity, `created_at` | Code read (`circuits.py:169-181`, `circuit_service.py:88-91,192`) + real-wiring test executed green against real DB (asserts `exported["model"]["mistudio_model_id"]`, reimported `model_id`/`granularity`/`created_at`) | **HOLDS** — with 2 caveats: (a) the cluster_ref→"cluster" derivation branch is unpinned (the test asserts `granularity == "feature"`, which is the default and would pass even without the fix); (b) the `created_at` half introduced a new bug for foreign files (R3-B1) |
| 2 | `buildQueryString` '?' composition | Code read: `client.ts:125-135` returns WITHOUT leading '?'; `circuits.ts` list now guards `qs ? \`?${qs}\` : ''` → `/circuits?promoted=true` | **HOLDS** — but the identical bare-concat bug is still live in `clusterProfiles.ts:16` (R3-B7, 014 surface) |
| 3 | Import size cap 413; error detail carries field `loc` | Code read: `MAX_IMPORT_BYTES = 1_048_576`, Content-Length guard → 413; ValidationError detail now `"first at '<loc>': <msg>"` | **HOLDS in code, UNTESTED** — no 413 or loc test exists (R3-B3); chunked-transfer bypass + malformed-header 500 (R3-B4) |
| 4 | Panel: save-error draft preservation, Import button, markdown components mapping | Full file read: separate `saveError` state (load `error` no longer unmounts the editor); hidden file input + Import button wired to `circuitsApi.importDefinition`; house dark-theme `components` mapping on ReactMarkdown (no prose classes) | **HOLDS** — ESLint + tsc clean; the second `import { useRef } from 'react'` line (panel line 16) does NOT trip lint (rule not enabled) — cosmetic only (R3-B8) |
| 5 | Distinctness pins + single-head guard | `TestDistinctnessPins` (lone strong prior → `computed` + `distinctness == 1.0`; persistence 2-vs-3-vote `echo_confidence` ordering) and `test_alembic_single_head.py::test_exactly_one_head` both exist and **executed green** | **HOLDS** |
| 6 | FTASKS deferral record | `018_FTASKS` v1.1: status header updated, progress block, **7 recorded deferrals each with an owner** (from_candidates→016, member-edit UI/filters→016, rung chip→015, concurrency→017, SQL pagination→016, typed type_signals→freeze, Phase 6→close-out) | **HOLDS on the 018 side** — but the pointed-at 016/017 FTASKS were not updated (R3-B5) |

All six circuit test files green in one run (47 passed, exit 0). Frontend type-check clean.

---

## PART B — Fresh sweep findings (ranked)

**R3-B1 · P2 · fix before 016 (~20 min) — Import 500s on any tz-aware `provenance.created_at`. NEW — introduced by the R2 created_at fix. LIVE-REPRODUCED.**
`DefinitionProvenance.created_at: Optional[datetime]` parses `"2026-07-19T10:00:00Z"` into a tz-aware datetime; `CircuitService.create` assigns it straight into the `DateTime` (timestamp-without-tz) column; asyncpg raises `DataError: can't subtract offset-naive and offset-aware datetimes` → raw 500 on `POST /circuits/import`. Reproduced against the real `mistudio_test` DB. miStudio's own exports emit naive `utcnow()` so the pinned round-trip passes — but ISO-8601-with-Z is the normal form for any foreign or hand-written export, which is exactly the cross-instance flow BR-013 exists for.
*Fix:* normalize in `create()` — `dt.astimezone(timezone.utc).replace(tzinfo=None)` when `tzinfo` is set — and pin with a Z-suffixed import test.

**R3-B2 · P2 · fix before 015 consumes circuits (residue of R2-A1) — `model.hf_id` is still dropped both directions.**
Import threads only `defn.model.mistudio_model_id` (`circuits.py:180`); the `Circuit` row has no hf_id storage; export emits `DefinitionModelRef(mistudio_model_id=…)` with `hf_id: null` (`circuit_service.py:192`). The R2-A1 fix made the round-trip lossless **same-instance only**: `mistudio_model_id` is instance-local, `hf_id` is the cross-instance-stable identifier — an imported foreign circuit whose `model.hf_id` is set (and `mistudio_model_id` null, as it will be) loses its model provenance entirely, which is the field 015's model-mismatch hazard check needs. *Fix:* store hf_id (column or thread through the members/saes JSONB pattern — a `model` JSONB column is cleanest), emit both on export, pin with an hf_id-only import.

**R3-B3 · P3 · fix now (~30 min, one test class) — The R2 fix wave repeated the R2-T3 mistake: four of its own fixes are unpinned.**
No test covers: the 413 cap; the `loc`-bearing 422 detail; the cluster_ref→"cluster" granularity derivation (the only granularity assertion checks the default value); PATCH-granularity persistence (the B6 fix — `test_granularity_param_validated` is a list-param test, not a PATCH test). Each is 3–6 lines in `test_circuits_api.py`.

**R3-B4 · P3 · bundle with B3 — The import cap is header-trusting and single-endpoint.**
(a) A chunked request (no Content-Length) skips the guard entirely and FastAPI parses the full body; (b) a malformed `Content-Length: abc` → `int()` ValueError → 500; (c) `POST /circuits` and PATCH have **no cap at all** — R2-Q1(a) asked for both create and import; (d) per-edge `type_signals` size bound (R2-Q1 b) not done; (e) `docker/nginx.gcp.conf:29` still `client_max_body_size 0` (unlimited) — R2-Q1(c) unfixed, and it is a global exposure beyond circuits.

**R3-B5 · P3 · fix before 016 kickoff (docs, ~15 min) — The deferral hand-off is half-recorded: 018 points at 016/017 FTASKS, which were never updated.**
`016_FTASKS` contains none of the promised precursor tasks: no encoder-weight orientation pins (R2-A4 — gates IDL-32 prior work), no `from_candidates` seam, no store-extraction/frontend-test bundle, no SQL-side pagination task; worse, its file table still lists `CircuitsPanel.tsx (NEW)` — stale, the panel exists. `017_FTASKS`/`017_FTID` carry neither the optimistic-concurrency precondition (R2-Q2 — escalates to P1 when 017's writer lands) nor the documented `update()`-only edge-write rule (R2-A5). Since 016 executes from its FTASKS via 007_process-task-list, the deferrals must live where the executor will read them. (Noted positively: 017 FTASKS already plans `test_causal_language_audit.py` and imports 018's shared enum.)

**R3-B6 · P3 · defer to 016 (note in FTASKS) — MCP surface asymmetries + still zero MCP smoke coverage.**
`promote_circuit` cannot unpromote (no `promoted` param; UI can); `update_circuit` doesn't expose `granularity` (the B6 PATCH fix is unreachable from MCP); `list_circuits` lacks `edge_type`/`limit`/`offset`. All 9 MCP circuit tools remain invoked by zero tests (re-flag of R2-T5; recorded owner: close-out session).

**R3-B7 · P3 · ride-along (014 surface, same defect class as R2 B2) — `clusterProfilesApi.list` still bare-concats `buildQueryString`.**
`clusterProfiles.ts:16`: `` `/cluster-profiles${buildQueryString(params ?? {})}` `` → filtering by `sae_id` or `search` (both passed by `clusterProfilesStore.fetchProfiles`) produces `/cluster-profilessae_id=…` → 404. The R2 B2 fix was applied only to circuits.ts. One-line fix + it is the last bare-concat caller in the codebase.

**R3-B8 · P4 · cosmetic — Duplicate react import in CircuitsPanel.**
Line 9 `import { useCallback, useEffect, useState } from 'react'` + line 16 `import { useRef } from 'react'`. Verified NOT a lint break (ESLint clean — `import/no-duplicates` not enforced) and tsc-clean. Merge on next touch.

**R3-B9 · P4 · note — Slices download drops the parent-rung wrapper from the saved file.**
The panel writes only `r.slices` to `.slices.json`; `parent_rung`/`parent_rung_language` from the response are discarded. Not a losslessness violation — each slice's `provenance.source_note` carries the `projection_of=…; parent_rung=N; partial_rendering=true` marker — but the human-readable rung language doesn't travel in the file.

**R3-B10 · P4 · assessed honestly (BR-026 "rungs included in exports") — sufficient, with one soft spot.**
The export document carries per-edge `rung` ints + `tested_and_failed`; the circuit-level rung is derivable (min-over-edges, and the ladder itself is contract-vendored) — machine-lossless, so BR-013/BR-026 hold. The soft spot: MCP `export_circuit_definition` returns the raw definition — the ONE circuits response with no `rung_language` string. Acceptable because the same agent gets language from `get_circuit`/`list_circuits`, and injecting display fields into the contract document would corrupt saved files; worth one docstring sentence pointing agents at `get_circuit` for language.

**R3-B11 · P4 · demo polish — minor UI roughness in CircuitsPanel.**
Import has no busy state and no success feedback (silent list refresh; errors do surface in the banner); delete uses `window.confirm` (house panels mostly use styled dialogs); no list-level filter controls (recorded deferral); R2-P3's persistence-hidden-by-default FTDD sentence was never added (FTDD untouched since f1be274 — carry as an open one-liner).

**R3-B12 · Info (positive) — Copy audit CLEAN across the full 018 surface.**
Grepped backend schemas/services/endpoints, MCP tool docstrings, and all frontend circuits files for `causal|causally`: every hit is either the rule's own meta-text (`evidence_ladder.py` docstring, IDL-35 references), the rung-2 entries in `RUNG_LANGUAGE`/`RUNG_NEXT_STEP` (allowed), the rung-2 description in the MCP `list_circuits` docstring (allowed — it labels rung 2), or the enum member name. No below-rung-2 causal phrasing anywhere. RungChip renders server strings verbatim; ladder remains single-sourced with the grep-guard + TS mirror pin.

---

## PART C — Gate check for Feature 016

### Verdict: **GO**

The 016-facing seams are ready and verified: rung enum importable (`src.schemas.evidence_ladder`, grep-guarded), classifier importable (`classify_edge` — keyword-only, `{type, signals}` disclosure, distinctness the only ranking-facing output, now regression-pinned), `CircuitService.create` + REST + MCP accept fully-formed discovery payloads with provenance threading (real-stack tested), contract import is kind-keyed and capped, migrations are single-head-guarded. Nothing found in R3 undermines 016's Phase 1 (store IO).

### Must fix BEFORE (or in the first session of) 016 — all small, ~2h total
1. **R3-B5 (docs, gating the process):** write the deferral tasks into `016_FTASKS` (encoder-weight pins as a named gate on the IDL-32 prior, `from_candidates` seam, store extraction + frontend tests, SQL pagination; fix the stale "CircuitsPanel NEW" row) and into `017_FTASKS`/FTID (concurrency precondition, `update()`-only edge-write rule). 016 executes from its FTASKS — undone here means silently dropped there.
2. **R3-B1 (code, ~20 min):** tz-normalize imported `created_at` — foreign-file import is currently a 500, live-reproduced.
3. **R3-B2 (code, ~45 min):** carry `hf_id` through storage/round-trip — hard precondition for 015's model-mismatch check and the real cross-instance BR-013 story; cheaper now than after 016 starts minting circuits.

### Can ride along with 016
R3-B3/B4 (pins + cap hardening — bundle into 016's first hardening pass), R3-B6 (MCP asymmetries + smoke), R3-B7 (clusterProfiles '?' one-liner), R3-B8–B11 (cosmetic/polish). Carried-open R2 gates unchanged and correctly sequenced: encoder pins before 016's prior work; optimistic concurrency + edge-write rule before 017's writer (escalates to P1 there); typed `type_signals` at v1 freeze; Phase 6 (E2E, screenshot, manual, MCP smoke, fixture growth) at the named close-out/016-acceptance points.

### Finding counts
R3: **12 findings** — 0 P1, 2 P2, 5 P3, 4 P4, 1 positive/info. All six R2 fix-now items verified holding at runtime.

---
*Round 3 (final) conducted 2026-07-19 · scope HEAD 69dbf97 · verdict GO for 016 (3 must-fix: 1 docs + 2 small code) · live-verified: 47 circuit tests green vs real DB; tz-import failure reproduced; tsc + ESLint clean.*
