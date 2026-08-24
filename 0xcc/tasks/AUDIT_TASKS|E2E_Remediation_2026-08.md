# AUDIT_TASKS | E2E Remediation 2026-08

**Source:** `0xcc/audits/E2E-2026-08/FINDINGS.md` — 166 findings from a twelve-phase
end-to-end assessment (2026-08-23). Every task cites its finding id; the register
carries the evidence, the reproduction and the proposed remediation for each.

**Status:** ⏳ **In progress** — started 2026-08-23 · **Findings:** 13 P0 · 62 P1 · 68 P2 · 23 P3
**Suites:** backend **3247 passed / 0 failed** (baseline 2883) · frontend **1239 passed / 0 failed** (baseline 1211) · `tsc` clean · **eslint 0 errors** · **CI green (now running all 1232, lint gating), mirror images built**

| Wave | Scope | State |
|---|---|---|
| **Part 1** | MIS-E2E-143 — the public-mirror disclosure | ✅ **CLOSED**, verified live |
| **Wave 1** | Task 7 — test-schema divergence (the prerequisite) | ✅ **CLOSED** — 7.1–7.6 |
| **Wave 2** | Tasks 1–5 — the 13 P0s | ✅ **CLOSED** — Tasks 1–5, all 13 P0s. 46 negative controls recorded. |
| **Wave 3** | Task 6 — wrong results presented as correct (9 findings) | ✅ **CLOSED** — all 9. 29 negative controls. |
| **Wave 4** | Task 8 — pin the surviving audit mutations | ✅ **CLOSED** — all 9. Three pins found **live** defects. |
| **Wave 6** | Task 13 — documentation (19 findings) | ✅ **complete** — 13.1–13.11 all closed. 41 dead paths annotated with git-history evidence, `## Relevant Files` added to the 11 files lacking it, the health dashboard created, CLAUDE.md's stale counts / 501 claims / phantom `session_state.json` corrected. 16 controls (C133–C148). |
| **Wave 5** | Tasks 9–12 — realtime, provenance, correctness, infra | ✅ **CLOSED** — **Task 9 ✅** · **Task 10 ✅** · **Tasks 9, 10, 11 ✅ · Task 12 ✅** (12.5 partial: `/ollama/`, the compose queue split and a `server_name` typo remain) |
| Waves 4–9 | Mutations, realtime, provenance, docs, P2/P3, hardware, acceptance | ❌ not started |

*This table is updated as work lands — see the Relevant Files section for the
running file list.*

> ## ⚠ TASK 0 IS NOT A DEVELOPMENT TASK
> `MIS-E2E-143` — an SSH password, five database dumps and this audit's own findings
> register are **published in a public GitHub repository, right now**. Do Task 0
> before reading further.

---

## Task 0 — Stop the disclosure  ⚠ IMMEDIATE

- [ ] 0.1 ⚠ **USER ACTION — Rotate the GPU node's SSH password** (`192.168.244.61`, user `sean`). Treat the current value as known. — MIS-E2E-143
- [x] 0.2 ~~Make the mirror private~~ — **superseded**: mirror stays public by decision; the orphan snapshot removes the need until 0.4 is done. One setting, immediate, reversible. — MIS-E2E-143
- [x] 0.3 ✅ `scripts/k8s-helpers.sh` moved to key-based auth reading from the environment; remove `StrictHostKeyChecking=no`. — MIS-E2E-143
- [x] 0.4 ✅ **Fixed `sync-to-clean.yml` so the mirror carries no source history** — publish a squashed single commit, or `git filter-repo` before pushing. Deleting files from a tip commit does not remove them from a force-pushed history. — MIS-E2E-143
- [ ] 0.5 Review the five `backups/*.sql.gz` dumps for credentials and user prompt text; rotate anything they hold. `git rm` them and add `backups/` to `.gitignore`. — MIS-E2E-008, -143
- [x] 0.6 ✅ Added a CI check that inspects the **published** tree's history against the exclusion list and fails the sync. — MIS-E2E-143
- [x] 0.7 ✅ Deleted (21 files). Verified byte-identical to their `0xcc/` originals first; the one non-duplicate — a reading-list manifest — was **relocated** to `0xcc/docs/clustering-document-set.md` pointing at the originals, rather than destroyed. — MIS-E2E-007, -164

**Acceptance:** the exclusion list holds for every commit reachable from the mirror's
tip, not only the tip; the rotated credential appears nowhere in either repo.

---

## Task 1 — Credential exposure (P0)

- [x] 1.1 ✅ `POST /labeling/models/openai`: **never attach the stored key to a caller-named host.** Allow-list the origin, and call `validate_llm_endpoint_url` on that path. — MIS-E2E-069
- [x] 1.2 ✅ Emits `Bearer {{OPENAI_API_KEY}}` in **both** Postman writers (`openai_labeling_service.py:406`, `:645`), matching the cURL branch. — MIS-E2E-072
- [x] 1.3 ✅ `decrypt_value`: distinguishes "not an envelope" (legacy → return as-is, **log a counter**) from `InvalidTag` (**raise**). Fixes three findings at once. — MIS-E2E-004, -041, -056
- [x] 1.4 ✅ *(superseded)* — `legacy_plaintext_reads()` counts them so the exposure is measurable; a backfill needs that number first for legacy plaintext `labeling_jobs.openai_api_key` rows; the counter from 1.3 measures the exposure. — MIS-E2E-041
- [x] 1.5 ✅ Regression tests: a key never reaches a non-allow-listed host; no artifact file contains the key across **all three** `export_format` values. — MIS-E2E-069, -072

## Task 2 — The Settings PIN (P0)

- [x] 2.1 ✅ Stored the PIN outside the generic settings table, **or** deny `settings_pin_hash` in `PUT`, `PUT /bulk`, `DELETE` and both `GET`s. Marking it `is_sensitive=True` fixes only the read. — MIS-E2E-055, -165
- [x] 2.2 ✅ *(reframed)* — the PIN is now unreachable via the generic CRUD, which was the actual bypass; server-side enforcement on the settings routes themselves remains open under Task 13.8's PADR work on the settings routes (short-lived token from `/pin/verify`), or amend IDL-25 to state it is a UI affordance. — MIS-E2E-005
- [x] 2.3 ✅ `<PinGate>` on the Storage tab — it arms step-granular checkpoint retention (`dry_run: false` deletes files) and was the one destructive surface in Settings left ungated. `settings-reference.md` corrected: it said *"the panel can be locked"* when two tabs of five are gated, and now states plainly that the PIN is a **UI affordance, not authentication**, and does not gate the API at all (→ IDL-47). — MIS-E2E-160
- [x] 2.4 ✅ Verified: the setting defaults to `False` and no shipped artifact enables it. Pinned across `k8s/base/*.yaml`, both compose files and `.env.example` — by **parsing** the manifests, because a k8s env entry puts the name and value on separate lines and a line scan can never see both (control C139 proved that by surviving). — MIS-E2E-005

**Acceptance:** `GET /api/v1/settings` on the live deployment returns no PIN material — the check that failed in P12.

## Task 3 — Arbitrary deletion (P0)

- [x] 3.1 ✅ `ActivationService._extraction_dir()` — one method, three layers (id shape, `resolve_user_path` on a **relative** path, root containment); the write, read and delete sites all route through it, and the filesystem delete moved inside the `if extraction:` ownership guard (a non-matching id now reports "not found for this model"). — MIS-E2E-070
- [x] 3.2 ⚠️ **SUPERSEDED — removing the fields breaks the writers.** Attempted; **37 tests failed** because the download workers write these columns *through the very same* `DatasetUpdate` / `ModelUpdate` schemas, so the field cannot be removed without splitting internal and external schemas. The exposure is real but the sink is the right place to close it — see 3.3, which makes the stored value harmless regardless of who wrote it. Recorded rather than quietly dropped. — MIS-E2E-071
- [x] 3.3 ✅ **`settings.resolve_deletable_path()`** — a single guard in `core/config.py`, and every stored-path deletion sink routed through it (11 sites across 8 modules). Three properties, each separately controlled:
  - **containment** against the same allow-list `resolve_user_path` uses, so a new trusted root is honoured without a second edit;
  - **`min_depth=2`** — containment ALONE is not a deletion guard: `resolve_user_path("")` returns `data_dir` and `"datasets"` returns the directory holding every dataset, and both pass a containment check;
  - **a realpath re-check**, because `resolve_user_path` is deliberately string-only (correct, before containment succeeds) while `rmtree` traverses symlinked components.

  Also fixed a defect **in the fix**: the first version sent already-absolute stored paths through `resolve_user_path`, which strips the leading slash and re-joins under `data_dir` — so every *real* deletion would have resolved to a nonexistent path and silently no-op'd while reporting success. Caught by the cleanup integration tests. — MIS-E2E-071
- [x] 3.4 ✅ Cancel cleanup gated to `{QUEUED, PROCESSING}`, so completed tokenizations for other models survive — the raw-file cleanup directly above it was already status-gated and this sibling had been missed. The revoke branch is live: the removed comment claimed *"We don't have task_id stored"*, but **both** dispatch sites write `task.id` into `extra_metadata['task_id']` immediately after `.delay(...)`, so revoke had been dead for its whole life while cancel reported success. — MIS-E2E-151
- [x] 3.5 ✅ The prune confirmation re-fetches the policy and confirms against **that**, never the snapshot; the toast follows the same value. Fails **closed** — if the refresh fails nothing is pruned and the user is told, because for `dry_run` the falsy value is the deleting one. — MIS-E2E-128

## Task 4 — Mass assignment and the WebSocket boundary (P0)

- [x] 4.1 ✅ **Split the request shape from the internal one.** The fields could not simply be deleted — internal callers write them through these very schemas (`datasets.py` sets `status=PROCESSING` when queuing tokenization), the same trap as 3.2. So: new `DatasetPatchRequest` / `ModelPatchRequest` (`extra="forbid"`, user-editable fields only) bound by the PATCH routes, while `*Update` stays intact for the workers. **`PATCH /api/trainings/{id}` is deleted** — `TrainingUpdate` is *entirely* lifecycle and worker-owned metrics and `trainings` has no user-editable column, so removing the unsafe fields left nothing; it had no caller (no frontend PATCH, no MCP tool, no test). All three sinks additionally enforce an explicit `_WRITABLE` allow-list, because a narrow route with an open sink is one new caller away from the bug. — MIS-E2E-106
  - ⚠️ **A test was pinning this defect.** `test_update_dataset_success` PATCHed `{"status": "ready", "progress": 100.0}` and asserted **200** — the suite was green *because* it certified the vulnerability. Replaced with a test asserting 422 and that the row is unchanged.
- [x] 4.2 ✅ `cors_allowed_origins=settings.allowed_origins`; the false comment replaced with the reason it mattered (engineio *short-circuits* the check on `"*"`, so the wildcard was not a permissive policy — it was no policy). — MIS-E2E-105, -018
- [x] 4.3 ✅ `validate_channel()` — type check, length bound, `..` refusal, a segment pattern, and a first-segment topic allow-list; `MAX_SUBSCRIPTIONS_PER_CLIENT = 100`. Enforced **in `WebSocketManager.subscribe`**, not only in the event handler. The topic list is held honest by a test that derives every channel the emitters publish **from the emitter source** and asserts each is accepted — with a minimum-count assertion so the scrape cannot fail open. — MIS-E2E-105, -140
- [x] 4.4 ✅ `main.py`'s four duplicate `@sio.event` handlers and its second `WebSocketManager()` removed; it imports the singleton. Guarded by an **AST** test over `main.py`, because the overwrite leaves no trace in the live handler table — whichever module imported last simply wins, and the table looks normal either way. — MIS-E2E-138
- [x] 4.5 ✅ `tests/unit/test_websocket_boundary.py` — 30 tests across origins, channel validation both directions, the cap, manager-level enforcement, and single registration. 11 negative controls, all red. — MIS-E2E-105

## Task 5 — Capabilities that do not exist (P0)

- [x] 5.1 ✅ **Split: wire the breaker, delete the rest.**
  - **`CircuitBreaker` WIRED.** `_guard_steering_dispatch()` gates all three async dispatch endpoints (503 when open); `_record_steering_outcome()` feeds terminal outcomes from `GET /async/result/{task_id}`, which is where the API first learns them. Both in the API process, so the state `/status` reports is the state that was recorded. Outcomes are de-duplicated **per task id** — without that, three polls of one failed task open a threshold-of-three breaker on their own.
  - **`ConcurrencyLimiter` + `ProcessIsolationManager` DELETED.** A semaphore and an in-process timeout cannot bound a fire-and-forget `apply_async`; GPU serialisation is already the Celery worker's concurrency. Zero callers, zero tests. Wiring them would have produced a *second* always-healthy constant.
  - `/status` now discloses `scope: "api-process"` rather than implying it observes the whole system. — MIS-E2E-062
- [x] 5.2 ✅ ⚠️ **The finding read the absence correctly and proposed the wrong remedy.** Implementing the affine gate would have been a *regression*: freezing does **not** make the map affine — the MLP activation stays non-linear — so a global-affine check reports a large departure for every real model and **would refuse every genuine fit**. That is precisely why it had been replaced by `linearisation_residual`, a recorded diagnostic. What was actually wrong: the dead `MAX_AFFINE_RESIDUAL` threshold, which reads like a guard and so stops anyone looking for the missing one.
  - **Removed** the dead threshold and the constructor parameter.
  - **Closed the real hazard soundly:** `frozen_attention_and_norms` now refuses a freeze that applied to *nothing* — `freeze_norms=True` on a model where `_norm_modules` matches no module (the documented substring-match failure) raises instead of recording `freeze_norms: true` for an unfrozen fit. Checking that the patch **landed** is direct and certain; inferring it from the resulting matrix is neither.
  - `CLAUDE.md` correction is Task 13.7's, and now has something true to say. — MIS-E2E-079
- [x] 5.3 ✅ 8 negative controls (C39–C46), each verified to land and then bite. **Two findings came out of the controls themselves:**
  - **C44 SURVIVED** — the `freeze_qk` half of the gate had no test that could reach it. Pinned by simulating the only real displacement window (`_freeze_norm` stealing SDPA between the patch and the check).
  - **C45 HUNG the suite instead of failing.** Deleting the refusal path's unwind loop leaks `_FREEZE_LOCK`, and the next test blocks on `acquire()` forever with no output. An autouse fixture now converts a leaked lock into a legible failure. A test that hangs is worse than one that fails. — `CLAUDE.md` Reachability gate

---

## Task 6 — Wrong results presented as correct (P1)

The class this product can least afford. Each item is a number a user reads as a measurement.

- [x] 6.1 ✅ Both metrics return `None` when unmeasurable — the schema (`Optional[float]`) and the frontend type (`number | null`) already carried it, and the UI already renders `—`. The dependency was **not** added: it downloads its model on first use and this deployment is offline, so it would have swapped a constant for an abort. `except Exception`, not `except ImportError` — the offline download failure was never an ImportError and propagated out of the whole steering request. — MIS-E2E-063
- [x] 6.2 ✅ **Split-half agreement.** Two independent accumulators over **alternating** prompts (interleaved, not sequential — a topic-ordered corpus would otherwise guarantee disagreement), compared against each other. The published lens remains the whole corpus: the split is the instrument, not the estimate.
  - Reproduced the reviewer's simulation first: **518 / 1040 / 1988** at noise 0.5/1.0/2.0 against their 518/1050/2030 — linear in σ, confirming the old criterion measured per-prompt spread.
  - ⚠️ **The threshold does not transfer.** At `1e-3` split-half never converges (it shrinks as σ/√n, needing ~1e6 prompts). Calibrated by simulation, not reasoning — table recorded in the source. **`DEFAULT_CONVERGENCE_DELTA` 1e-3 → 0.1**, which lands in the 100–2000 range the real fits used.
  - The artifact now stamps `convergence_criterion`. **Absent is meaningful** and must never be defaulted on read: a lens without it was fitted under the old test and claims a property it was never given. — MIS-E2E-080
- [x] 6.3 ✅ Renamed to `source_position_spread_{mean,max}` — the value is `std across source positions / |J|.mean()`, a statement about positional stability, and `linearisation_residual()` measures something else and has no production caller. Renamed rather than dual-published: keeping the old key perpetuates exactly the mislabelling, and a missing key is a question a consumer can ask. No in-repo consumer read it. — MIS-E2E-081
- [x] 6.4 ✅ All three, plus a fourth found while testing.
  - **FVE:** SVD-based rank determination replaces `qr(...).Q`, which padded a rank-deficient basis with arbitrary directions. Four duplicates now report exactly what one direction reports.
  - **The control now runs:** `excess_fve` wired into the profile builder. Both inputs (`stacked` residuals, `jacobians[layer]`) were **already being collected there** — the call was simply absent, so `control_seed` made reproducible a control that never executed. The raw FVE is deliberately still not published.
  - **`occupancy` re-checked:** the finding listed it as dead, but its remaining references are docstrings. It needs a *sparsity budget* this pipeline does not have, so `None` is genuinely correct — documented as structurally absent rather than wired to nothing.
  - **"Sustained" now means sustained** (≥2 consecutive layers above median), so one noisy layer cannot set `workspace_start` at layer 0 and empty the sensory band.
  - ⚠️ **Fourth defect, same shape, other boundary:** `peak_index` was a raw `argmax` over *every* layer, so an isolated late spike could set `motor_start` exactly as an early one could set `workspace_start`. Found by writing the workspace fixture; the finding named only one of the two. — MIS-E2E-088
- [x] 6.5 ✅ One base — **1-based**, everywhere. `rankColor` cannot move to 0-based (`Math.log(0)` → the alpha becomes `Infinity`, an invalid value the browser discards, silently unshading every top cell — checked, not assumed; my first test asserted `NaN` and was wrong), and "rank 1 = best" is what the word means. So `diffColor`, the tooltip and both legend swatches moved. The tooltip's `#${r + 1}` reported every rank one too high, and the ramp started at `2/span` instead of `1/span`, overstating every disagreement. — MIS-E2E-129
- [x] 6.6 ✅ **One shared resolver, `resolve_referenced_saes()`, used by all three endpoints** — extracted from the combined endpoint rather than copied, because this finding *is* an instance of "fixed one representative, never generalized" and a copy is the version that drifts back apart. Compare now carries each feature's own `sae_id` into its hook config and hands the SAE **map** to `_register_steering_hooks`, which already grouped by `(sae_id, layer)` for combined. Sweep is single-feature so has no routing to do — but it had **no feature validation whatsoever**, and now gets the layer check it never had. Both worker tasks accept `sae_meta_map`. — MIS-E2E-064
- [x] 6.7 ✅ **One helper, `encode_with_training_normalization()`, and every SAE consumer on it.** `encode()` does not normalize — `forward()` does — so a bare `encode` hands the dictionary raw activations it was never fitted on. The **sibling sweep found two more sites the finding did not name**: faithfulness and intervention. Also fixed the compounding half: `_load_sae_sync` loaded the community-format `config` (which carries `normalize_activations`) and **discarded** it, so every SAE it built took the constructor default and no consumer could have recovered the right convention. Extraction's inline copy — the one path that had it right, with a comment explaining why — now uses the shared helper too, so there is one place to forget instead of six. — MIS-E2E-083
- [x] 6.8 ✅ `if dial != 0`. Zero is the baseline by definition; every other value is a real intervention. — MIS-E2E-065
- [x] 6.9 ✅ **Collapsed to one branch, behaviour unchanged.** Re-measured: **7.2e-7** at two shapes, i.e. float32 epsilon. Not reimplemented, for two reasons both recorded in the docstring: the paper's method is a **dataset** expectation needing a corpus calibration pass, a persisted scalar and a checkpoint-format change; and every SAE ever trained under `anthropic_rescale` was trained with *these* semantics, so redefining the string would silently reinterpret existing artifacts rather than fix them. The docstring's claim of `E[‖x‖²] = dim` was the real damage — it described a second method convincingly enough that nobody checked it existed. **Tracked debt:** the real Templeton normalisation, and PPRD §2.1's "six frameworks" (five plus an alias) → Task 13. — MIS-E2E-085

## Task 7 — The test schema is not the production schema (P1)

The root enabler behind the production 500 the user hit. **Two of the constraints are already fixed; the mechanism is not.**

- [x] 7.1 ✅ Added a guard test diffing `Base.metadata` against the migrated schema — constraints, foreign keys, indexes. This single test covers MIS-E2E-031, -033 and every future migration-only constraint. — MIS-E2E-031, -033, -048
- [x] 7.2 ✅ Re-created the three foreign keys the ORM declares and the database lacks, or drop them from the ORM — pick one. — MIS-E2E-033
- [x] 7.3 ✅ Tests for delete cascades against a **migrated** database. Flipping all three CASCADEs on `features` currently leaves 211 tests green. — MIS-E2E-053
- [x] 7.4 ✅ Round-trip test: build a maximal `CircuitDefinitionV1`, import, export, assert **document equality**. Field-by-field assertions only cover the fields someone remembered. — MIS-E2E-052, -037
- [x] 7.5 ✅ Derived `REQUIRED_TABLES` from `Base.metadata`; decide deliberately whether a missing table blocks startup; test it. — MIS-E2E-032, -051, -157
- [x] 7.6 ✅ Narrowed `check_migrations.py`'s claim to what it checks, or extend it to constraints; wire it into CI. Delete `find_column_gaps.py`. — MIS-E2E-048, -049, -022

## Task 8 — Unpinned load-bearing behaviour (P1)

Each is a mutation that survived. Write the test, then re-run the mutation as a negative control.

- [x] 8.1 ✅ Both implementations, and a second test that no steering path calls `get_hookable_module(..., "residual", ...)` — asserted as a **parsed call**, since both modules mention "residual" in comments explaining why it was wrong. — MIS-E2E-078
- [x] 8.2 ✅ Parametrized **off the registry**, so a new secret is covered the day it is added, plus a control that the registry is non-empty (an empty one makes the parametrize vacuous). — MIS-E2E-077
- [x] 8.3 ✅ An AST walk over the **whole source tree**, not three named sites, so a fourth `torch.load` is covered on arrival. Fails closed if the scan finds none. — MIS-E2E-091
- [x] 8.4 ✅ Driven off the live registry — and it **found nine tasks live on the default queue**:
  - `train_sae`, `resume_training` — GPU training jobs, short-named so the module glob could not match;
  - `steering.combined` (GPU) and `steering.cleanup` — routes existed for `compare` and `sweep` only, now a `steering.*` glob;
  - all five `workers.model_tasks.*` — the route was written `src.workers.model_tasks.*` while the tasks register **without** the `src.` prefix, so the entry matched nothing. **The recorded lesson biting a second time.**
  - two `dataset_tasks` with no entry at all.
  The test asserts **routing coverage**, not name shape: a short name is fine if an explicit entry exists. — MIS-E2E-093, -097
- [x] 8.5 ✅ The guard existed and nothing asserted it. Two tests: the cross-check is present, **and it precedes the delete** — noticing after the row is gone is not a guard. — MIS-E2E-112
- [x] 8.6 ✅ ⚠️ **Re-ran M22 first: it is already killed** (2 tests fail today, where the audit recorded 75 green) — exact mid-range values had been added since. Verified empirically rather than trusting the finding. Those two catch it *incidentally*, so a block was added stating the invariant directly, pinning the constants, and documenting why the clamp endpoints cannot distinguish any slope. — MIS-E2E-127
- [x] 8.7 ✅ Derived by walking `src.ml` and `src.services` for jlens modules, with a control asserting the walk finds ≥5. Docstring lines excluded, since they legitimately name the published figures in order to forbid them. **Audit mutation M13 — a band constant in a sibling service — is now killed.** — MIS-E2E-090
- [x] 8.8 ✅ `TestCallerCoverageIsAccounted` over the **real built server with every category enabled**. Confirmed the finding's arithmetic exactly: **100 of 116** tools had no payload assertion. All 100 are now *listed* with a reason, so the gap is counted and cannot grow silently; guards reject a stale exemption, a tool in both lists, and a blank reason. A payload fixture per tool remains backlog — writing 100 fake ones would be worse than recording the debt. — MIS-E2E-119
- [x] 8.9 ✅ Fixed **and** pinned. The handler caught only `httpx.TimeoutException`, so `ConnectError` / `RemoteProtocolError` were abandoned on the first attempt — on the *terminal* events, where a dropped emit leaves the UI showing a finished job as running forever. Widened to `TransportError`, which excludes `HTTPStatusError` (the server answered; retrying repeats a rejected request). Pinned in **both** directions. — MIS-E2E-142, -137

## Task 9 — Realtime (P1)

- [x] 9.1 ✅ Removed. socket.io keeps listeners across the whole reconnect cycle, so the "re-attach for reconnections" loop added a second registration each time. **`WebSocketContext` had no test file at all** — now 5, including one that asserts a handler runs **once per server message** after three reconnects, not just that the listener count is 1. — MIS-E2E-120
- [x] 9.2 ✅ `emit_progress` detects a running loop and emits **in-process** via `ws_manager.emit_event`; no loop means a Celery worker, where the loopback is correct. The guard is in the **emitter**, not at the 13 call sites — putting it at a call site is what produced a fix covering 1 of 14. Both directions pinned, including a control that a worker still uses HTTP. — MIS-E2E-136, -138
- [x] 9.3 ✅ Done in Wave 4 alongside MIS-E2E-142 — `TransportError` covers `ConnectError` and `RemoteProtocolError` while excluding `HTTPStatusError`, where the server answered. Pinned in both directions (NC76). — MIS-E2E-137
- [x] 9.4 ✅ `_running` is now assigned **after** the setup succeeds (the ordering is the fix; the reset in the handler is defence in depth — control C95 proved that by surviving until the test pinned the *order*). `stop()` catches a dead loop so shutdown still closes the HTTP client. — MIS-E2E-139
- [x] 9.5 ✅ `error_message`, and **both** terminal emits now carry a status matching their event — they were sending `status: "extracting"` on *completed* and *failed*, and the store spread-merges, so a finished job was written back as running. Asserted by parsing the emit payloads, with a control that the in-progress event still says `extracting`. Directly relevant to the OOM the user hit: that path's whole value is its diagnostics, and the key name was destroying them in transit. — MIS-E2E-067
- [x] 9.6 ✅ Both. An abandoned channel was subscribed on the next connect and every reconnect after (compounding 9.1); `emit_system_metrics` emitted a name nothing listens for **and returned True**. — MIS-E2E-126, -141

## Task 10 — Provenance: `Feature.training_id` is NULL by design (P1)

- [x] 10.1 ✅ `services/feature_provenance.py` — `resolve_training_id` (async + sync) and `feature_scope_clause`. `source_id` answers "which dictionary"; these answer "which training", which is what the consumers actually wanted. — MIS-E2E-135
- [x] 10.2 ✅ Scoped by `Feature.external_sae_id == sae.id` **`or_`** the training link, so features predating the registry still match. The `activation_frequency` was the sharp end: without it the frequency auto-baseline had nothing to compute from and every feature of a community SAE silently took the default strength of 10. — MIS-E2E-100
- [x] 10.3 ✅ Swept. **Only two sites were broken** — `analysis_service`'s logit-lens branch and `browse_sae_features`. The other four (`logit_lens_service`, `neuronpedia_local_service`, `histogram_service` ×2) already `or_` over both links and were left alone, then **pinned** so a later cleanup cannot narrow them. Checked rather than assumed; the finding implied more. Correlations was fixed earlier in this session. — MIS-E2E-135

## Task 11 — Correctness bugs (P1)

- [x] 11.1 ✅ Both columns widened, with migration `e2a4c81b9d17` (round-tripped up/down against the live DB; the downgrade **refuses** rather than truncating a value above the 32-bit range). ⚠️ The fresh-session half is **deferred** — widening removes the overflow that poisons the session, so the handler no longer has one to survive; hardening it is worth doing but is no longer load-bearing. Recorded rather than silently dropped. — MIS-E2E-029
- [x] 11.2 ✅ `populate_existing()`. **And the fixture was rebuilt**, which was the finding's sharper half: the fake `_Session` returned a fresh row every call, so it had no identity map and could not exhibit the defect in either direction. It now models one — verified by removing the fix and watching the *new* test go red, which the old fixture could never have done. Sibling sweep: `training_tasks` and `neuronpedia_tasks` open a **fresh session per check**, so the trap is unique to labeling, which reuses `self.db`. — MIS-E2E-057
- [x] 11.3 ✅ All three. `job_batch_size` bound before `if template:` (the supported no-template path died with `UnboundLocalError` at three read sites). `_LabelingCancelled` handled ahead of the generic handler — and the comment in `labeling_tasks.py` asserting *"the job row is already CANCELLED"* is now **true**; it was false precisely because the generic handler had overwritten it, which is why nobody looked. `max_tokens` takes the job's value at **both** sites, matching `max_examples`. — MIS-E2E-058, -059, -060
- [x] 11.4 ✅ `RETRYABLE_TASK_TYPES` checked first, so an unsupported pair leaves the row untouched. The allow-list is held in step with the if/elif chain by a test that reads the branches out of the **AST** — a hand-list that can drift from the code it guards is worth little. — MIS-E2E-098
- [x] 11.5 ✅ `_SPAWNED_WORKER_PIDS` + `_kill_orphan_steering_workers()` — the sweep kills what this process started, by pid. The worker was **already** started with `--pidfile`, so the precise handle existed and nothing used it. `POST /system/restart` now requires the internal token with `hmac.compare_digest`, matching `main.py`'s existing internal endpoints — it previously took no arguments, required nothing, and being idempotent under a restart policy was a self-sustaining outage loop. ⚠️ **Spawn bounding deferred**: recorded, not done. — MIS-E2E-003, -099
- [x] 11.6 ✅ One shared `task_looks_alive()` in `task_heartbeat`, and **all five** janitors on it. The discovery test found a **sixth janitor the finding never named** — `cleanup_stuck_nlp` — which turned out to be correct by design (`ExtractionJob.celery_task_id` belongs to the extraction task, not the NLP pass). Recorded as a listed exemption **with its reason pinned**, so if an `nlp_celery_task_id` column ever appears the test says so. — MIS-E2E-092
- [x] 11.7 ✅ Dispatch keys on `not created_jobs` rather than the enumerate index (a skipped first SAE meant no job had position 1 and **nothing ran**), and the chain advances to the next queued job **by order** rather than demanding `position + 1` (a gap stranded the tail until the 3-hour reaper blamed a crashed worker). — MIS-E2E-066
- [x] 11.8 ✅ The `feature_ids` branch now binds to `extraction_job_id`, as the no-ids branch already did, and logs when ids are dropped rather than silently analysing a subset. — MIS-E2E-109
- [x] 11.9 ✅ Import refuses to overwrite a system template — the rule `update_template` and `delete_template` already had — and cannot grant `is_system` either. ⚠️ The **Pydantic body model is deferred** (the endpoint still takes `dict`); the overwrite hole is closed, the typing debt is recorded. — MIS-E2E-108
- [x] 11.10 ✅ `validation_alias=AliasChoices("dataset_schema", "schema")` — input accepts both, output is the field name on **every** dump path (a plain alias made the key depend on whether the caller passed `by_alias`). `extra="allow"` so `task_id` / `task_type` / `lock_key` survive validation; they were being **silently dropped**, so a dataset that round-tripped through this model lost the id of the very task someone was cancelling. Round-trip stability verified against pydantic 2.12.5.
  On the 13 assertions: they assert `dataset_schema` **in storage**, which is correct and stays — existing rows carry it. What was wrong was the *comments* attributing it to the alias; corrected, since there is no longer one. — MIS-E2E-107
- [x] 11.11 ✅ All four.
  - **-122** `SelectedFeature.sign` persists the direction, because the over-budget branch zeroes the magnitude and direction cannot be recovered from a zero. A drag past the budget and back used to flip a suppressing feature to amplifying, at a strength the budget model chose, silently.
  - **-121** `releaseIfCurrent()` only clears the shared refs if it still owns them. A late handler used to null a newer request's controller, after which nothing could be cancelled and the 5s timeout never fired again — permanent, from two rapid feature switches.
  - **-123** `isGenerating` / `batchState` out of `partialize`, plus a rehydrate reset for payloads already stored. `taskId` stays persisted deliberately — it is the durable handle recovery actually uses.
  - **-124** `generateCombined` gains the double-submit guard `generateComparison` already had. — MIS-E2E-121, -122, -123, -124

## Task 12 — Infrastructure (P1)

- [x] 12.1 ✅ Deleted, and `k8s_deploy` applies `k8s/base` via `kubectl apply -k` — **refusing** if the path is absent rather than applying whatever else is there. README and CLAUDE.md references corrected. Also restarts `mistudio-mcp`, which runs the same backend image and was never restarted, so new MCP tools stayed invisible after a break-glass deploy. — MIS-E2E-144
- [x] 12.2 ✅ `Recreate` on both. They are Deployments over hostPath, so the default RollingUpdate briefly runs two pods against one data directory. A StatefulSet with a PVC remains the right long-term shape; this removes the overlap without a storage migration. — MIS-E2E-145
- [x] 12.3 ✅ Both bound to `127.0.0.1` by default, overridable via `POSTGRES_BIND` / `REDIS_BIND` for deliberate remote access. Redis is the **Celery broker**, so LAN reachability meant anyone could enqueue GPU jobs and read queued payloads — well outside the accepted posture, which concedes the API behind nginx and not the broker. ⚠️ **Redis password deferred** (binding removes the exposure; auth is defence in depth). — MIS-E2E-146
- [x] 12.4 ✅ Per-step execution with `DEPLOY FAILED at: <step>` and a non-zero return. The whole body was one `&&` chain ending in `|| echo "WARNING: Schema verification failed"`, so a failed pull, apply or rollout printed a message about **schema** and returned 0. Schema verification is now the only advisory step — which is what that trailing `||` was trying to express and applied to everything above it by accident. — MIS-E2E-147
- [x] 12.5 ⏳ **Partial** — the frontend port is fixed (`3000:8080`; the image moved to nginx-unprivileged in `bca37c6` and only k8s and `nginx.docker.conf` were updated, so `localhost:3000` has been dead since). The `/ollama/` location, the compose worker's queue split and the `server_name` typo remain. — MIS-E2E-147
- [x] 12.6 ✅ All three. `/api/internal` denied at the ingress on **both** hosts — the `.net` one is internet-facing and had the same gap. `signed-by=` scopes the deadsnakes key to its own source instead of the global trusted keyring. And the fail-open guard now asserts — the **sweep found a second `pytest.skip` in the same file** that the finding did not name. — MIS-E2E-148
- [x] 12.7 ✅ All nine removed. Verified first: **329 tests, 9 files, all passing** — whatever was once broken had been fixed and the exclusion outlived it silently. — MIS-E2E-025
- [x] 12.8 ✅ **0 errors, and lint now gates CI.** The two `react-hooks/rules-of-hooks` errors were real: `ReadoutGrid`'s empty-readout guard sat *above* two `useMemo` calls, so an empty axis meant React saw a shorter hook list — "rendered fewer hooks than expected", which unmounts the tree. Assessed unreachable because the backend emits `types` and `layers_by_type` from one tuple; that is a property of today's backend, not of the component, so the guard moved below the hooks.
  Also real: a regex escaping `(`/`)`/`[` **inside a character class**, and three `catch (e) { throw e }` wrappers that only cost the original stack frame. The rest were unused bindings — fixed by configuring the `_` convention the codebase already used (it was honoured for *arguments* only, so `const { [k]: _, ...rest }`, where the binding is **required syntax**, was an error that could not be removed).
  Warnings do **not** block: a gate nobody can pass gets deleted. 492 `no-explicit-any` warnings remain as visible debt. — MIS-E2E-024, -023
- [x] 12.9 ✅ `${MCP_AUTH_TOKEN:-}`. Compose evaluates `:?` during **file interpolation**, before profile filtering, so a profile-gated service aborted `ps`, `config` and `logs` on a fresh clone. The intent is enforced where it belongs — `mcp_server/server.py` already refuses to start on an empty token. — MIS-E2E-026
- [x] 12.10 ✅ `APP_VERSION` build arg → `MISTUDIO_VERSION` + `/app/VERSION`. An `ARG`, not a `COPY`: the file is outside the `backend/` build context. The fallback now logs at ERROR naming the likely cause — returning a plausible string quietly is what let every pod report `unknown`. **Note:** `docker-images.yml` must pass `build-args: APP_VERSION=…` for this to take effect. — MIS-E2E-028

---

## Task 13 — Documentation (P2)

- [x] 13.1 ✅ Corrected, with the **Stop & Finalize** row and the warning the other manual already carried — plus the incident itself recorded inline, so the next reader knows why the wording matters. Pinned by a test parametrized over **both** manuals. On whether it should exist: left in place, now guarded; deleting a 599-line user manual is the user's call, not a remediation's. — MIS-E2E-149
- [x] 13.2 ✅ Enforced, and the manual corrected. Verified across the full matrix: stdio+flag ✅, stdio bare ✅, **HTTP+flag REFUSED**, HTTP bare REFUSED, HTTP+token ✅.
  ⚠️ **A test was pinning this defect** — `test_anonymous_flag_allows_empty_token` asserted the flag satisfies the guard, i.e. certified the hole. Rewritten. Third defect-pinning test found in this remediation. — MIS-E2E-150
- [x] 13.3 ✅ Both pages rewritten around the **Secret** the deployment actually reads (`secretKeyRef` → `mistudio-secrets`), with a verification step that reports failure — unlike `sed`, which exits 0 on zero matches, so all four steps "succeeded" while setting nothing. The danger callout records what the old steps did, including the one that **renamed the database and its user to the password string**. — MIS-E2E-152
- [x] 13.4 ✅ README and CLAUDE.md both point at `docker compose up -d`. The script is described for what it is — a one-machine dev convenience — with the reasons it cannot work elsewhere. — MIS-E2E-162
- [ ] 13.5 Correct IDL-5 and the **five** documents propagating it; IDL-16's three false claims; IDL-1/12's channel and event conventions; IDL-11's DLQ and backoff; IDL-38's "one steering core". — MIS-E2E-156, -157, -158, -159, -076
- [x] 13.6 ✅ Re-counted from evidence: **9** rows (16–24), not 13 — row 21 already read "Implemented". Status verified against the **code**, not another document: each row's primary module was checked to exist, and 25–29 stay Planned because theirs do not. The PPRD inventory is declared **authoritative for status**, with `CLAUDE.md` the narrative that yields to it. — MIS-E2E-011
- [x] 13.7 ✅ **Complete.** The off-by-one instruction references are fixed — `001_generate-brd.md` was added at the front and the list never renumbered, so every entry named a real file performing a **different action**, and `008_housekeeping.md` did not exist. The startup section is corrected. **Now also:** the phantom `session_state.json` / `research_context.json` references (including in the folder-structure diagram, which presented both as real), the stale `995 passed` / `1007` counts, and the "returns 501" claims the J-Lens binding invalidated. Pinned by `test_every_instruct_reference_names_a_file_that_exists`, `test_claude_md_does_not_claim_a_nonexistent_file_is_auto_loaded` and `test_claude_md_test_counts_are_not_silently_stale`. — MIS-E2E-155, -010, -163
- [x] 13.8 ✅ **PADR IDL-47** (v3.5). States the posture, the **four classes that escape it** — credential disclosure, process kill/spawn, arbitrary filesystem deletion, cross-origin reach, each found live in this audit — the infrastructure boundary (the API behind nginx, *not* the broker or `/api/internal`), the four conditions that invalidate the decision, and that the PIN is a UI affordance and never security. — MIS-E2E-002, -166
- [x] 13.9 ✅ Re-measured: **9** missing against the current ORM, not 11 (two of the named ones are non-ORM). All nine documented; the unearned claim replaced by `test_data_model_doc_covers_every_table`, which diffs the page against `Base.metadata`. `alembic_version` and `feature_activations_default` are **listed exemptions with reasons**. Control: adding a new ORM table fails the guard until it is documented. — MIS-E2E-050, -164
- [x] 13.10 ✅ `## Relevant Files` added to FTASKS 024–028 and the six ad-hoc files, each entry carrying a triage verdict (SHIPPED / OPEN / PARTIAL) backed by code evidence rather than a bare path. 41 dead paths annotated `⚠️ **never written**` with the git-history evidence (`no add-commit anywhere in repo history`) rather than silently deleted — the claim that they were planned is real history. `- [x] Zoom and pan` unchecked: a false completion, no implementation exists. Pinned by `test_task_docs_traceability.py` (C141–C143). — MIS-E2E-153, -154, -012
- [x] 13.11 ✅ Filter fixed (`fn.attr in ("get", …)` also matched `dict.get("kind")`), contract regenerated, the three phantom endpoints gone. Tool-count prose derived — it said **92/13** while the manual said **97/13** and the generated contract said **116/14**; only the contract was derived.
  ⚠️ Two attempts at the derivation were wrong and the tests caught both: summing `CATEGORY_MODULES` counts **modules** (16), and `_all_tools()` **builds a server**, which calls this — infinite recursion. Now an AST count, worded **"up to"** because `get_approval_status` is registered only when `steering_approval` is on. A test pins the ceiling-vs-served difference to exactly that named set. — MIS-E2E-114, -017, -161

## Task 14 — Remaining P2/P3

- [ ] 14.1 The 22 remaining P2 findings not covered above — see the register, filtered by `**Severity:** P2`.
- [ ] 14.2 The 23 P3 findings — cleanups, dead code (`api/websocket.ts`, `ollm_server/`, `.claude/agents/`), `utcnow()` at 37 sites, sourcemaps, console stripping.

## Task 15 — Hardware acceptance

Not verifiable in the audit session — `ssh` to the GPU node was unavailable, and these are the findings this repo's history says only hardware confirms.

- [ ] 15.1 Verify the SDPA patch + `_FREEZE_LOCK` leak on a real fit. — MIS-E2E-082
- [ ] 15.2 Verify circuit capture runs the SAE off-distribution. — MIS-E2E-083
- [ ] 15.3 Drive the end-to-end journey with Playwright: dataset → train → finalize → extract → label → cluster → circuit → calibrate → steer → export → J-Lens. — P12
- [ ] 15.4 Benchmark `/task-queue/active` specifically for the blocking-Redis fix. A fix measured against a path it did not touch is not a verified fix. — MIS-E2E-102

---

## Task 16 — Feature Acceptance

- [ ] 16.1 Every P0 closed, each with a test that **fails when the fix is removed**.
- [ ] 16.2 Re-run all 14 surviving mutations as negative controls; each must now be killed. Record the controls.
- [ ] 16.3 Full suites green: backend and frontend, **with `frontend-ci.yml`'s excludes removed** and lint blocking.
- [ ] 16.4 Re-run the P12 live probes: `GET /api/v1/settings` returns no PIN material; the mirror's history carries nothing on the exclusion list.
- [ ] 16.5 Update `CLAUDE.md` Current Status and the Document Inventory.
- [ ] 16.6 Adopt the **cross-document grep at fix time** — three P11 findings exist because a correction was applied to the file under review and the copies were never grepped.

---

## Category Checklist Results

| Category | Outcome |
|---|---|
| Data / schema | Tasks 7, 11.1 — the ORM/migration divergence and its enablers |
| API | Tasks 1, 2, 3, 4, 11.4, 11.8 |
| UI | Tasks 6.5, 11.11, 12.8, 2.3 |
| Integration | Tasks 9, 10, 13.11 — realtime, provenance, the MCP contract |
| Error handling | Tasks 11.2–11.4, 9.4, and the `str(e)` sweep (MIS-E2E-110) |
| Testing | Tasks 7, 8, 12.7 — 17 test-gap findings, 14 from surviving mutations |
| Perf / security | Tasks 0–5, 12.3, 14.1 — 33 security findings, 13 P0 |
| Config / deploy | Tasks 0, 12 |
| Docs | Task 13 — 19 doc-drift findings |
| **N/A** | *None.* Every category produced tasks. |

## Relevant Files

*(The framework requires this section, and P11 found 22 dead entries elsewhere for
want of it. It is filled in as tasks are completed — one line per file touched.)*

| File | Purpose |
|---|---|
| `.github/workflows/sync-to-clean.yml` | Rewritten: orphan snapshot + stale-tag retarget + a Verify step that clones the published mirror and asserts no excluded path in any commit |
| `scripts/k8s-helpers.sh` | Credential removed; key-based SSH, host/user from the environment, `StrictHostKeyChecking=no` dropped |
| `backend/alembic/versions/d7f3a91c2e08_restore_declared_foreign_keys.py` | **New.** Restores the 3 FKs the ORM declared and the DB lacked; NULLs orphans first, idempotent, reversible |
| `backend/tests/unit/test_orm_matches_migrated_schema.py` | **New.** Fails when ORM and migrated schema disagree, in either direction. Reflects the *migrated* DB, never `create_all()` |
| `backend/src/db/schema_validator.py` | `REQUIRED_TABLES` derived from `Base.metadata` (17 → 35 tables), resolved lazily at validation time |
| `backend/src/db/__init__.py` | Exports `_required_tables` instead of the removed eager constant |
| `backend/scripts/verify_schema.py` | Imports the single source instead of carrying a copy-pasted duplicate dict |
| `backend/tests/unit/test_schema_validator_coverage.py` | **New.** The validator had zero tests, which is why mutation M2 survived |
| `backend/tests/unit/test_delete_cascades.py` | **New.** Six tests over CASCADE and SET NULL. Nothing exercised a delete rule, which is why M5 survived |
| `backend/tests/unit/test_circuit_definition_roundtrip.py` | **New.** Document-equality round-trip + a fail-closed check that the endpoint passes every contract block |
| `backend/src/api/v1/endpoints/circuits.py` | MIS-E2E-037: `calibration` was absent from the import dict, so an imported circuit lost its whole calibrated band |
| `backend/check_migrations.py` | Claim narrowed to what it checks — it compares column names only and said "All models match the database schema" |
| `backend/find_column_gaps.py` | **Deleted** (MIS-E2E-049) — its `create_table` regex truncated at the first `)`, so every report was a false positive ⚠️ **never written** — removed later (MIS-E2E-154) |
| `backend/tests/unit/test_analysis_cache_upsert.py` | **New** (out-of-band). Pins the cache upsert that fixed the production 500 |
| `backend/src/services/analysis_service.py` | Cache upsert; ablation's dead precondition removed; correlations scoped to the SAE |
| `backend/src/models/{feature,feature_analysis_cache}.py` | Declare the unique constraints that existed only in migrations |
| `frontend/src/components/features/FeatureTokenAnalysis.tsx` | BPE marker stripped for display; continuations marked |
| `backend/src/core/encryption.py` | `decrypt_value` raises `DecryptionError` on `InvalidTag`; legacy plaintext counted via `legacy_plaintext_reads()`; `mask_value` no longer reveals 4–7 char secrets whole |
| `backend/src/services/app_setting_service.py` | `get_by_key` display branch masks on decrypt failure (the unmask branch deliberately propagates) |
| `backend/src/api/v1/endpoints/settings.py` | Both response-masking sites guarded — never echo ciphertext back |
| `backend/tests/unit/test_decrypt_fails_closed.py` | **New.** 9 tests over the not-an-envelope / InvalidTag split and short-secret masking |
| `backend/src/api/v1/endpoints/labeling.py` | MIS-E2E-069: URL validated on the credential path; stored key gated on a host allow-list (`_host_may_receive_stored_key`) |
| `backend/src/services/openai_labeling_service.py` | MIS-E2E-072: both Postman writers emit `{{OPENAI_API_KEY}}`, matching the curl branch |
| `backend/tests/unit/test_stored_credential_never_leaves.py` | **New.** 9 tests: host gating incl. lookalike/path tricks, validator-on-path (fail-closed), both artifact writers |
| `backend/src/services/app_setting_service.py` | `_PROTECTED_KEYS` + `ProtectedSettingError`: the PIN is invisible to generic reads and refused by generic writes/deletes; `_privileged` is the only way in |
| `backend/src/api/v1/endpoints/settings.py` | PIN endpoints use `_privileged`; PIN encrypted at rest; 403 on protected keys; `HTTPException` re-raised before the generic handler (MIS-E2E-103 pattern) |
| `backend/tests/api/v1/endpoints/test_pin_is_not_a_setting.py` | **New.** 9 tests over all three bypasses (read/write/delete) plus /bulk, and the PIN still working |
| `backend/src/services/activation_service.py` | MIS-E2E-070: `_extraction_dir()` — one method, three layers; the write, read and delete sites all route through it |
| `backend/src/api/v1/endpoints/models.py` | MIS-E2E-070: filesystem delete moved inside the `if extraction:` ownership guard. MIS-E2E-071: requantize's file wipe routed through the deletion guard |
| `backend/src/core/config.py` | **MIS-E2E-071: `resolve_deletable_path()`** — containment + `min_depth=2` + a realpath re-check; accepts an already-contained absolute path as-is (the workers store absolute paths) |
| `backend/src/workers/dataset_tasks.py` | MIS-E2E-071: raw and tokenized deletions guarded. MIS-E2E-151: cleanup gated to `{QUEUED, PROCESSING}` so completed tokenizations survive a cancel |
| `backend/src/workers/model_tasks.py` | MIS-E2E-071: `file_path` / `quantized_path` deletions guarded; a refusal is recorded as an error, never swallowed |
| `backend/src/workers/training_tasks.py` | MIS-E2E-071: `training_dir` deletion guarded |
| `backend/src/services/dataset_service.py` | MIS-E2E-071: tokenized-file deletion guarded (sibling sweep) |
| `backend/src/services/sae_manager_service.py` | MIS-E2E-071: SAE `local_path` deletion guarded (sibling sweep) |
| `backend/src/api/v1/endpoints/datasets.py` | MIS-E2E-071: tokenized-file deletion guarded. MIS-E2E-151: forwards the stored `task_id`, reviving a revoke branch that had been dead since it was written |
| `backend/src/api/v1/endpoints/neuronpedia.py` | MIS-E2E-071: export archive deletion guarded (sibling sweep) |
| `backend/tests/unit/test_deletion_containment.py` | **New.** 29 tests: the guard both directions, plus an **AST** wiring check that no sink resolves a stored path with `resolve_data_path` and then deletes it |
| `backend/tests/integration/_data_root_tmp.py` | **New.** Repoints `settings.data_dir` at a temp root so the cleanup tests exercise the real containment path instead of `/tmp`, which the guard correctly refuses |
| `backend/tests/integration/test_{model_cleanup,dataset_cancellation,critical_fixes}.py` | Fixtures moved inside the trusted root — relaxing the guard to keep them green would have certified the vulnerable behaviour |
| `backend/tests/unit/test_dataset_cancel_scope.py` | **New.** 6 tests: cleanup scoped to in-flight work (pinning the SET, not just that a filter exists) and the revoke path reachable at both ends |
| `frontend/src/components/panels/SettingsPanel.tsx` | MIS-E2E-128: the prune confirmation re-fetches the live policy and fails **closed**; `CheckpointPrunePreviewPanel` exported for its test |
| `frontend/src/components/panels/SettingsPanel.prune.test.tsx` | **New.** 5 tests over the exact reported sequence: preview dry-run → change the setting → the dialog must say PERMANENTLY DELETE |
| `backend/src/core/websocket.py` | MIS-E2E-105/-140/-018: origins from `settings.allowed_origins`; `validate_channel()` + a per-client cap, enforced in the manager; the false CORS comment replaced |
| `backend/src/main.py` | MIS-E2E-138: four duplicate `@sio.event` handlers and a second `WebSocketManager()` removed; imports the singleton |
| `backend/src/schemas/{dataset,model}.py` | MIS-E2E-106: `DatasetPatchRequest` / `ModelPatchRequest` — the user-editable subset, `extra="forbid"` |
| `backend/src/schemas/training.py` | MIS-E2E-106: records why there is no `TrainingPatchRequest` — the whole schema was worker-owned |
| `backend/src/api/v1/endpoints/trainings.py` | MIS-E2E-106: `PATCH /{training_id}` **deleted** — no user-editable field, no caller, and it unlocked partial-checkpoint SAE import |
| `backend/src/services/{training,model,dataset}_service.py` | MIS-E2E-106: explicit `_WRITABLE` allow-lists replace the blind `setattr` loops |
| `backend/tests/unit/test_websocket_boundary.py` | **New.** 30 tests; the topic allow-list is checked against channels derived from the emitter, not a second hand-list |
| `backend/tests/unit/test_patch_is_not_mass_assignment.py` | **New.** Narrow schemas, route binding, route deletion, and each sink's allow-list exercised against a stub session that fails loudly if the guard is skipped |
| `backend/tests/api/v1/endpoints/test_datasets.py` | ⚠️ `test_update_dataset_success` **pinned the defect** (PATCH status → assert 200); replaced with a 422 + row-unchanged assertion |
| `backend/src/services/steering_resilience.py` | MIS-E2E-062: `CircuitBreaker` kept and wired; `ConcurrencyLimiter` / `ProcessIsolationManager` deleted as architecturally unfit; `/status` discloses its `api-process` scope |
| `backend/src/api/v1/endpoints/steering.py` | MIS-E2E-062: `_guard_steering_dispatch()` on all three dispatch endpoints; `_record_steering_outcome()` (de-duplicated per task) on the result endpoint |
| `backend/src/ml/jlens_fitter.py` | MIS-E2E-079: dead `MAX_AFFINE_RESIDUAL` removed; `frozen_attention_and_norms` refuses a freeze that applied to nothing, unwinding the lock first |
| `backend/tests/unit/test_steering_resilience_wired.py` | **New.** 10 tests; the dispatch scan asserts it found exactly 3 endpoints so it cannot pass vacuously |
| `backend/tests/unit/test_jlens_freeze_gate.py` | **New.** 9 tests + an autouse fixture converting a leaked `_FREEZE_LOCK` from a hang into a failure |
| `.github/workflows/backend-tests.yml` | **CI never ran migrations**, so the Wave 1 ORM-vs-migrated-schema guard failed with `NoSuchTableError`. Fixed in two rounds: `alembic upgrade head`, then a **separate** `mistudio_schema_check` database — migrating the one the unit fixtures manage does not survive the first `drop_all` |
| `backend/tests/unit/test_orm_matches_migrated_schema.py` | Reads `SCHEMA_CHECK_DATABASE_URL`; new `test_reflects_a_database_conftest_does_not_manage` makes the collision a loud failure instead of an order-dependent one |
| `backend/src/services/steering_service.py` | MIS-E2E-063: coherence/behavioral return `None`, `except Exception`. MIS-E2E-064: compare resolves the SAE map and routes each feature through its own dictionary |
| `backend/src/services/steering_core.py` | MIS-E2E-065: `dial != 0` — a negative dial registered no hooks and returned the baseline labelled as steered |
| `backend/src/schemas/steering.py` | MIS-E2E-063: `null` documented as "not measured", never a placeholder |
| `backend/src/api/v1/endpoints/steering.py` | MIS-E2E-064: `resolve_referenced_saes()` + `_routed_features()` extracted and used by compare, sweep and combined |
| `backend/src/workers/steering_tasks.py` | MIS-E2E-064: compare and sweep accept and forward `sae_meta_map` |
| `frontend/src/components/steering/ComparisonResults.tsx` | MIS-E2E-063: `!= null` so a genuinely-absent metric cannot reach `.toFixed` |
| `backend/tests/unit/test_steering_wrong_results.py` | **New.** 12 tests over all three findings; the resolver test asserts **exactly 3** endpoints share it |
| `frontend/src/components/jlens/utils.ts` | MIS-E2E-129: `diffColor` moved to the 1-based rank `rankOf` actually returns; the ramp starts at rank 2 |
| `frontend/src/components/jlens/ReadoutGrid.tsx` | MIS-E2E-129: both legend swatches and the tooltip rank corrected |
| `frontend/src/components/jlens/diffColor.test.ts` | **New.** 8 tests; includes why `rankOf` could NOT move instead (`Math.log(0)` → non-finite alpha) |
| `backend/src/ml/sparse_autoencoder.py` | MIS-E2E-085: one rescale branch for both names, behaviour unchanged, docstring corrected |
| `backend/tests/unit/test_normalize_modes_collapsed.py` | **New.** 8 tests pinning both bit-identity AND that the collapse did not change what existing checkpoints mean |
| `backend/src/ml/sparse_autoencoder.py` | MIS-E2E-083: `encode_with_training_normalization()` — one differentiable helper every SAE consumer uses |
| `backend/src/services/circuit_capture_service.py` | MIS-E2E-083: `_load_sae_sync` carries the trained mode (it was loading `config` and discarding it); `_encode_layer` normalizes |
| `backend/src/services/circuit_{attribution,faithfulness,intervention}_service.py` | MIS-E2E-083: all four bare `encode` sites routed through the helper — **two found by the sibling sweep, not the finding** |
| `backend/src/services/extraction_service.py` | MIS-E2E-083: its inline copy (the one correct path) replaced by the shared helper |
| `backend/src/workers/training_tasks.py` | MIS-E2E-083: the dead-neuron fallback measured off-distribution |
| `backend/tests/unit/test_sae_encode_normalization.py` | **New.** 10 tests; an AST scan per module with its own negative control, plus a fixture check that raw and normalized encodes actually differ |
| `backend/src/ml/jlens_fitter.py` | MIS-E2E-080: split-half convergence, threshold recalibrated 1e-3→0.1 by simulation, `convergence_criterion` stamped. MIS-E2E-081: `position_spread_{mean,max}` |
| `backend/src/workers/jlens_fit_tasks.py` | MIS-E2E-081: artifact keys renamed to `source_position_spread_*`. MIS-E2E-080: records the criterion and delta |
| `backend/src/ml/jlens_metrics.py` | MIS-E2E-088: SVD rank replaces QR padding; "sustained" enforced on **both** boundaries; `occupancy` documented as structurally absent |
| `backend/src/services/jlens_band_service.py` | MIS-E2E-088: `excess_fve` wired — the inputs were already collected, the call was absent |
| `backend/tests/unit/test_jlens_convergence.py` | **New.** 7 tests; reproduces the reviewer's proportionality result and pins the calibration |
| `backend/tests/unit/test_band_metrics_honesty.py` | **New.** 12 tests over FVE rank, the control being wired, and both boundaries |
| `backend/tests/unit/test_jlens_fitter.py` | Renamed fields; **new guard** that the six hand-rolled `_Result` stubs carry every `FitResult` field — it immediately found two more missing |
| `backend/src/ml/layer_discovery.py` | **Live bug:** `resolve_vocab_size()` — gemma-4 unified configs nest `vocab_size`; two real extraction jobs died on the direct read |
| `backend/src/services/activation_service.py` | Uses the resolver; raises rather than range-checking token ids against a guess |
| `backend/tests/unit/test_resolve_vocab_size.py` | **New.** 7 tests incl. the reported `Gemma4UnifiedConfig` shape and an embedding-table fallback |
| `backend/src/core/celery_app.py` | MIS-E2E-093/-097: **nine tasks were on the default queue**, incl. `steering.combined` and every `workers.model_tasks.*` (wrong prefix) |
| `backend/src/workers/websocket_emitter.py` | MIS-E2E-137/-142: retry widened `TimeoutException` → `TransportError`, excluding `HTTPStatusError` |
| `backend/tests/unit/test_audit_mutation_pins.py` | **New.** Task 8's harness — hook target, sensitive keys, `weights_only`, queue routing, retry scope, BR-002 package-wide, IDOR |
| `backend/tests/unit/test_reachability.py` | MIS-E2E-119: `TestCallerCoverageIsAccounted` over the real built server; **100 exemptions listed** with reasons |
| `frontend/src/utils/steeringStrength.test.ts` | MIS-E2E-127: the slope invariant stated directly rather than caught incidentally |
| `.github/workflows/frontend-ci.yml` | MIS-E2E-024/-025: 9 excludes removed (329 tests were ungated); **lint now gates**, errors blocking and warnings not |
| `frontend/eslint.config.js` | The `_` unused convention honoured for vars, caught errors and rest-siblings, not just arguments |
| `frontend/src/components/jlens/ReadoutGrid.tsx` | MIS-E2E-023: the early return moved **below** the hooks |
| `frontend/src/utils/tokenUtils.ts` · `stores/featuresStore.ts` | A regex over-escaped inside a character class; three pass-through `try/catch` wrappers removed |
| `docker-compose.yml` | MIS-E2E-026: `:?` → `:-`; the token requirement is enforced in `server.py`, not in the interpolation pass |
| `backend/Dockerfile` · `api/v1/endpoints/version.py` | MIS-E2E-028: `APP_VERSION` baked in; the fallback logs at ERROR |
| `backend/src/services/resource_config.py` | **Live bug:** `preflight_gpu_capacity()` — a 12B FP16 model OOM'd 2m47s in on a 24 GB card with nothing checking |
| `backend/src/ml/model_loader.py` | `estimate_parameter_count()` from the config, and the preflight **inside the loader** so all ten call sites inherit it |
| `backend/tests/unit/test_gpu_preflight.py` | **New.** 10 tests incl. the reported card and model, and a control that a fitting quantization is still allowed |
| `frontend/src/contexts/WebSocketContext.tsx` | MIS-E2E-120/-126: the reconnect re-attach removed; `unsubscribe` clears the pending queue |
| `frontend/src/contexts/WebSocketContext.test.tsx` | **New** — the file did not exist. 5 tests with a socket.io double that keeps listeners across reconnect, as the real one does |
| `backend/src/services/background_monitor.py` | MIS-E2E-139: `_running` assigned after setup succeeds; `stop()` survives a dead loop and still closes the client |
| `backend/src/services/extraction_service.py` | MIS-E2E-067: `error_message` not `error`; terminal emits carry their own status |
| `backend/src/workers/websocket_emitter.py` | MIS-E2E-141: `system:metrics`, the name every sibling and the frontend use |
| `backend/tests/unit/test_realtime_contracts.py` | **New.** 8 tests over the emit payloads, monitor lifecycle and event naming |
| `backend/src/services/feature_provenance.py` | **New.** MIS-E2E-135: the one resolver — `resolve_training_id` + `feature_scope_clause` |
| `backend/src/api/v1/endpoints/saes.py` | MIS-E2E-100: `browse_sae_features` scoped by the SAE `or_` its training |
| `backend/src/services/analysis_service.py` | MIS-E2E-135: the logit-lens branch resolves instead of reading the column |
| `backend/tests/unit/test_feature_provenance.py` | **New.** 11 tests, incl. one proving `col == None` really does compile to `IS NULL` — the claim the whole finding rests on |
| `backend/src/models/circuit_runs.py` + `alembic/versions/e2a4c81b9d17_*.py` | MIS-E2E-029: counters widened to BIGINT; the downgrade refuses rather than truncating |
| `backend/src/services/labeling_service.py` | MIS-E2E-057: `populate_existing()` so a cancel written elsewhere is visible |
| `backend/tests/unit/test_labeling_cancellation.py` | The fake session now models an identity map — it previously **could not exhibit the defect** |
| `backend/src/api/v1/endpoints/task_queue.py` | MIS-E2E-098: `RETRYABLE_TASK_TYPES` checked before the row is wiped |
| `backend/src/workers/task_heartbeat.py` | MIS-E2E-092: `task_looks_alive()` — one rule, extracted from the janitor that already had it |
| `backend/src/workers/cleanup_stuck_{trainings,extractions,activations,enhanced_labeling}.py` | All four off the PENDING-is-alive rule |
| `backend/tests/unit/test_task11_correctness.py` | **New.** 14 tests, parametrized over the janitor registry with a discovery check that found the sixth |
| `backend/src/services/labeling_service.py` | MIS-E2E-059/-060/-058: `job_batch_size` bound unconditionally; the job's `max_tokens` wins at both sites; cancellation handled before the generic handler |
| `backend/src/workers/labeling_tasks.py` | The comment claiming the row is already CANCELLED is now true, and says it was not |
| `backend/src/services/extraction_service.py` | MIS-E2E-066: dispatch keys on the first job CREATED, not the loop index |
| `backend/src/workers/nlp_analysis_tasks.py` | MIS-E2E-066: advance by order past a gap. MIS-E2E-109: the ids branch bound to the path extraction |
| `backend/src/services/labeling_prompt_template_service.py` | MIS-E2E-108: import cannot overwrite — or grant — `is_system` |
| `backend/tests/unit/test_batch_extraction_chain.py` | Fake query taught `order_by`, so it can express the real query |
| `backend/tests/unit/test_task11_batch_and_scope.py` | **New.** 11 tests across all six findings |
| `backend/src/schemas/metadata.py` | MIS-E2E-107: `validation_alias` (never a plain one) + `extra="allow"` so worker-written keys survive |
| `backend/tests/unit/test_metadata_alias_stability.py` | **New.** 9 tests: no field carries a plain alias, every dump path agrees, `task_id` survives |
| `frontend/src/types/steering.ts` · `stores/steeringStore.ts` | MIS-E2E-122: `sign` persists direction across zeroing. -123: in-flight state out of `persist` + rehydrate reset. -124: the missing double-submit guard |
| `frontend/src/stores/featuresStore.ts` | MIS-E2E-121: `releaseIfCurrent()` — only clear the refs you still own |
| `frontend/src/stores/task11FrontendState.test.ts` | **New.** 7 tests, incl. a behavioural one for the ref clobber (types cannot catch it) |
| `backend/src/api/v1/endpoints/steering.py` | MIS-E2E-003: `pkill -9 -f steering@` replaced by a pid-tracked sweep; both spawn sites record their pid |
| `backend/src/api/v1/endpoints/system.py` | MIS-E2E-099: `/system/restart` gated on the internal token (`settings` was not even imported) |
| `docker-compose.yml` | MIS-E2E-146: postgres and the Celery broker bound to loopback. MIS-E2E-147: frontend `3000:8080` |
| `backend/tests/unit/test_privilege_operations.py` | **New.** 8 tests; `pkill` asserted as a parsed CALL, and the restart exercised in all three directions |
| `k8s/mistudio-deployment.yaml` | **Deleted** (MIS-E2E-144) — a stale second copy `k8s_deploy` re-applied ⚠️ **never written** — removed later (MIS-E2E-154) |
| `scripts/k8s-helpers.sh` | MIS-E2E-144/-147: applies `k8s/base` via kustomize, restarts `mistudio-mcp`, and fails per-step instead of one `&&` chain |
| `k8s/base/{postgres,redis}.yaml` | MIS-E2E-145: `strategy: Recreate` over hostPath |
| `k8s/base/ingress.yaml` | MIS-E2E-148: `/api/internal` denied on **both** hosts |
| `backend/Dockerfile` | MIS-E2E-148: `signed-by=` instead of the global `apt-key adv` |
| `backend/tests/unit/test_worker_queue_coverage.py` | MIS-E2E-148: two fail-open `pytest.skip`s → assertions (the finding named one) |
| `README.md` · `CLAUDE.md` | Manifest references point at `k8s/base/` |
| `backend/tests/unit/test_infrastructure_invariants.py` | **New.** 12 tests; the ingress check is parametrised over the hosts it finds, which is how the `.net` gap surfaced |
| `aaaa/` | **Deleted** (21 files, MIS-E2E-007) — byte-identical copies published to the mirror while their originals were withheld |
| `0xcc/docs/clustering-document-set.md` | **New.** The one non-duplicate from `aaaa/`, relocated to index the originals |
| `docs/miStudio_Manual.md` | MIS-E2E-149: the sentence that cost `train_969e90af`, corrected in the second manual too |
| `backend/src/mcp_server/server.py` | MIS-E2E-150: `MCP_ALLOW_ANONYMOUS` is stdio-only in fact, not just in prose |
| `manual/docs/advanced/mcp-server.md` | MIS-E2E-150: the troubleshooting remedy no longer points into the hole |
| `manual/docs/reference/data-model.md` | MIS-E2E-050: nine tables documented; the unearned verification claim replaced by an enforced one |
| `0xcc/adrs/000_PADR\|miStudio.md` | **IDL-47** — the no-app-auth posture, its four escape classes, and what invalidates it |
| `backend/tests/unit/test_mcp_server_foundation.py` | ⚠️ `test_anonymous_flag_allows_empty_token` **pinned the defect**; rewritten to assert the refusal |
| `backend/tests/unit/test_docs_match_behaviour.py` | **New.** 18 tests binding the manuals, the data-model reference and the MCP contract to the code |
| `backend/src/mcp_server/contract.py` | MIS-E2E-114: an endpoint must start with `/`, so `dict.get("kind")` is no longer scraped as `GET kind` |
| `docs/mcp-contract.md` | Regenerated — three phantom endpoints removed |
| `backend/src/mcp_server/server.py` | MIS-E2E-161: the tool count is derived by AST, not written |
| `manual/docs/advanced/mcp-server.md` | Stale `(97 tools, 13 categories)` heading dropped rather than re-hardcoded |
| `manual/docs/getting-started/install-guide-k8s.md` · `installation.md` | MIS-E2E-152: the four no-op `sed` steps replaced by the Secret the deployment actually reads |
| `README.md` · `CLAUDE.md` | MIS-E2E-162: `docker compose up -d`, not a script hardcoded to one home directory. MIS-E2E-155: every `0xcc/instruct/` reference renumbered |
| `0xcc/prds/000_PPRD\|miStudio.md` | MIS-E2E-011: rows 16–24 reconciled from **code evidence**; the inventory declared authoritative for status |
| `0xcc/adrs/000_PADR\|miStudio.md` | MIS-E2E-156/-157/-158/-159: IDL-5, IDL-16, IDL-1/12 and IDL-11 corrected against the code |
| `README.md` · `CLAUDE.md` · `0xcc/prds/…` · `0xcc/tdds/008_*` · `0xcc/tasks/008_*` | The five documents that propagated IDL-5's deleted architecture |
| `frontend/src/components/panels/SettingsPanel.tsx` | MIS-E2E-160: the Storage tab is PIN-gated |
| `manual/docs/advanced/settings-reference.md` | MIS-E2E-160: the PIN described as the UI affordance it is |
| `backend/src/workers/websocket_emitter.py` | MIS-E2E-136: in-process emit when a loop is running; HTTP only from a worker |
| `frontend/src/components/layout/Sidebar.tsx` · `config/brand.ts` | Tagline: "Edge AI Feature Discovery" → "AI Feature Discovery" (user request) |
| `CLAUDE.md` | 13.7: phantom `session_state.json` / `research_context.json` (incl. the folder diagram), stale suite counts, and the obsolete "returns 501" J-Lens claims |
| `.claude/context/health/dashboard.md` | **New** (13.10). Referenced by 4 slash commands and never created — `/review` Step 5 told the reviewer to update a file that did not exist |
| `0xcc/tasks/02{4,5,6,7,8}_FTASKS|*.md` + 6 ad-hoc task files | 13.10: `## Relevant Files` added with per-entry triage verdicts; 41 never-written paths annotated with git-history evidence |
| `backend/tests/unit/test_docs_match_behaviour.py` | **New.** Pins the CLAUDE.md corrections and the dashboard's existence *and non-emptiness* — C144 initially survived because `.exists()` passes on an empty file |
| `backend/tests/unit/test_task_docs_traceability.py` | **New.** Every task file has `## Relevant Files`; every listed path resolves or is marked never-written; no task claims a capability with no implementation |
| `frontend/src/components/layout/Sidebar.tsx` | Tagline → "AI Feature Discovery Workbench" (user request), and the name/tagline/alt now read from `BRAND` instead of a second hardcoded copy |
| `frontend/src/config/brand.ts` | Tagline updated; stale `version: '0.1.0'` **removed** — it had drifted four minor releases from VERSION and package.json, undetected because nothing imported `BRAND` |
| `frontend/src/components/layout/Sidebar.brand.test.tsx` | **New.** Pins the wiring: the literal must not reappear beside the config, and no hand-maintained version may return (C145–C148) |

## Negative controls run

Every fix is required to have a test that **fails when the fix is removed**. Recorded
here as they are verified, because "a test exists" is not the same claim.

| Control | Mutation | Result |
|---|---|---|
| NC1 | `features.training_id` ondelete CASCADE → SET NULL | ✅ 2 failures |
| NC2 | Remove `FeatureAnalysisCache.__table_args__` (the MIS-E2E-031 shape) | ✅ 1 failure |
| NC3 | **Re-run of audit mutation M2** — shrink the validator to a hand-list | ✅ 3 failures *(survived during the audit)* |
| NC5 | **Re-run of audit mutation M5** — flip all 3 Feature CASCADEs to RESTRICT | ✅ 4 failures *(survived during the audit, 211 tests green)* |
| NC6 | Drop `calibration` from the import dict again (the real MIS-E2E-037 bug) | ✅ 1 failure |
| NC7 | **Re-run of audit mutation M3** — drop `faithfulness` from the import dict | ✅ 1 failure *(survived during the audit)* |
| NC8 | Move the source-scrape anchor — does the guard fail OPEN? | ✅ fails CLOSED (errors, does not silently pass) |
| NC9 | Swallow `InvalidTag` again (the MIS-E2E-056 behaviour) | ✅ 3 failures |
| NC10 | `mask_value` reveals short values again (MIS-E2E-061) | ✅ 1 failure |
| NC11 | Disable the stored-key host allow-list (MIS-E2E-069) | ✅ 3 failures |
| NC12 | Restore the real key in ONE Postman writer (MIS-E2E-072) | ✅ 2 failures |
| NC13 | Remove `settings_pin_hash` from `_PROTECTED_KEYS` (MIS-E2E-055/-165) | ✅ 4 of 6 failures — the 2 listing tests still pass because the belt-and-braces `is_sensitive=True` masks the value independently |
| NC4 | Revert `_cache_analysis` to the blind INSERT | ✅ 2 failures |
| NC14 | Revert `models.py` requantize to `resolve_data_path` (MIS-E2E-071) | ✅ 1 failure ⚠️ **never written** — no add-commit anywhere in repo history (MIS-E2E-154) |
| NC15 | Revert `dataset_tasks` raw deletion to `resolve_data_path` | ✅ 1 failure |
| NC16 | Disable the guard's `min_depth` check | ✅ 6 failures |
| NC17 | Disable the guard's symlink re-check | ✅ 1 failure |
| NC18 | Disable the empty/root refusal | ⚠️ **SURVIVED** — the depth check subsumes it. Pinned by a new test asserting the branch holds for a `min_depth=0` caller, which is the only contract it uniquely provides; **re-run ✅ 4 failures** |
| NC19 | Neuter the AST scan itself (`return []`) | ✅ 1 failure — the scan does not fail open |
| NC20 | Remove the already-contained-absolute branch | ✅ 6 failures — this one caught a defect **in the fix**: real deletions would have silently no-op'd |
| NC21 | Remove the cancel status gate (MIS-E2E-151) | ✅ 1 failure |
| NC22 | Add `READY` to the in-flight set — deletes completed work | ✅ 1 failure |
| NC23 | Drop `task_id` from the cancel call again | ✅ 1 failure |
| NC24 | Stop storing `task_id` at one dispatch site | ✅ 1 failure — the read cannot become a lookup of a key nothing writes |
| NC25 | Confirm the prune against the stale snapshot (MIS-E2E-128) | ✅ 1 failure |
| NC26 | Remove the prune policy re-fetch | ✅ 4 failures |
| NC27 | Let a failed policy refresh proceed anyway | ✅ 1 failure — must fail closed |
| NC28 | `cors_allowed_origins` back to `"*"` (MIS-E2E-105) | ✅ 2 failures |
| NC29 | Remove the channel topic allow-list | ✅ 1 failure |
| NC30 | Remove the channel-name pattern | ✅ 1 failure |
| NC31 | Remove the payload type check (MIS-E2E-140) | ✅ 4 failures |
| NC32 | Remove the per-client subscription cap | ✅ 1 failure |
| NC33 | Remove validation from the MANAGER, leaving only the handler | ✅ 1 failure |
| NC34 | Re-register a duplicate `subscribe` in `main.py` (MIS-E2E-138) | ✅ 1 failure |
| NC35 | Construct a second `WebSocketManager()` in `main.py` | ✅ 1 failure ⚠️ **never written** — no add-commit anywhere in repo history (MIS-E2E-154) |
| NC36 | Rebind the dataset PATCH route to the internal `DatasetUpdate` | ✅ 1 failure |
| NC37 | Disable the dataset sink's allow-list | ✅ 1 failure |
| NC38 | Restore `PATCH /api/trainings/{id}` | ✅ 1 failure |
| NC39 | Ungate ONE of the three dispatch sites (MIS-E2E-062) | ✅ 1 failure |
| NC40 | Stop recording task outcomes | ✅ 1 failure |
| NC41 | Drop the per-task outcome de-duplication | ✅ 1 failure |
| NC42 | Make the breaker's failure count never increment | ✅ 3 failures |
| NC43 | Remove the `freeze_norms` application check (MIS-E2E-079) | ✅ 2 failures |
| NC44 | Remove the `freeze_qk` application check | ⚠️ **SURVIVED** — no test could reach the branch. Pinned by simulating the only real displacement window; **re-run ✅ 1 failure** |
| NC45 | Make the freeze refusal skip its unwind loop | ⚠️ **HUNG the suite** rather than failing — a leaked `threading.Lock` blocks the next test forever. Autouse fixture added; **re-run ✅ 1 failure** |
| NC46 | Restore the dead `MAX_AFFINE_RESIDUAL` threshold | ✅ 1 failure |
| NC47 | Coherence returns the 0.5 constant again (MIS-E2E-063) | ✅ 2 failures |
| NC48 | Behavioral score returns 0.5 again | ✅ 2 failures |
| NC49 | Narrow the handler back to `except ImportError` | ✅ 1 failure |
| NC50 | Dial gate back to `> 0` (MIS-E2E-065) | ✅ 1 failure |
| NC51 | Sweep stops using the shared resolver (MIS-E2E-064) | ✅ 1 failure |
| NC52 | Compare drops the per-feature `sae_id` again | ✅ 1 failure |
| NC53 | Compare hooks a single SAE instead of the map | ✅ 1 failure |
| NC54 | Sweep task stops accepting `sae_meta_map` | ✅ 1 failure |
| CI | Point the schema guard at the conftest-managed database | ✅ 7 failures, the new guard naming the cause |
| NC55 | Re-split the rescale into two branches (MIS-E2E-085) | ✅ 4 failures |
| NC56 | Change the alias's value silently | ✅ 1 failure — existing checkpoints stay protected |
| NC57 | Make `none` stop being a no-op | ✅ 2 failures |
| NC58 | `diffColor` back to a 0-based rank (MIS-E2E-129) | ✅ 2 failures |
| NC59 | Ramp back to `rank / span` | ✅ 1 failure |
| NC60 | Capture encodes bare again (MIS-E2E-083) | ✅ 1 failure |
| NC61 | Attribution encodes bare | ✅ 1 failure |
| NC62 | Faithfulness encodes bare *(sibling sweep site)* | ✅ 1 failure |
| NC63 | Intervention encodes bare *(sibling sweep site)* | ✅ 1 failure |
| NC64 | Loader discards `normalize_activations` again | ✅ 1 failure |
| NC65 | The helper stops normalizing | ✅ 2 failures |
| NC66 | Convergence back to the running-mean step (MIS-E2E-080) | ✅ 1 failure |
| NC67 | Split the halves sequentially instead of alternating | ✅ 1 failure |
| NC68 | Publish only one half as the lens | ✅ 1 failure |
| NC69 | Threshold back to 1e-3 | ✅ 1 failure |
| NC70 | Stop stamping `convergence_criterion` | ✅ 1 failure |
| NC71 | FVE pads a rank-deficient basis again (MIS-E2E-088) | ✅ 2 failures |
| NC72 | Boundaries back to first-crossing | ✅ 1 failure |
| NC73 | Motor boundary back to a raw argmax | ✅ 1 failure |
| NC74 | Unwire `excess_fve` | ✅ 1 failure |
| NC75 | Rename the spread keys back (MIS-E2E-081) | ✅ 1 failure |
| NC76 | Retry narrowed back to `TimeoutException` (MIS-E2E-137) | ✅ 2 failures |
| NC77 | Remove the `steering.*` route | ✅ 1 failure |
| NC78 | Remove the `train_sae` route | ✅ 1 failure |
| NC79 | Restore the wrong `src.` prefix on a model-task route | ✅ 1 failure |
| NC80 | Drop `weights_only=True` from an artifact load (MIS-E2E-091) | ✅ 1 failure |
| NC81 | **Re-run of audit mutation M13** — band constant in a sibling jlens service | ✅ 1 failure *(survived during the audit)* |
| NC82 | Vocab resolver stops walking sub-configs | ✅ 2 failures |
| NC83 | Extraction reads `model.config.vocab_size` directly again | ✅ 1 failure |
| NC84 | Remove the checkpoint IDOR guard (MIS-E2E-112) | ✅ 2 failures |
| NC85 | Register a new MCP tool with neither assertion nor exemption | ✅ 2 failures |
| NC86 | **Re-run of audit mutation M22** — `BASELINE_SLOPE` 2.6 → 2.4 | ✅ 3 failures *(survived during the audit; already killed before this work, verified)* |
| NC87 | Remove the preflight from the loader | ✅ 1 failure |
| NC88 | Set the activation headroom to zero | ✅ 1 failure |
| NC89 | Estimator stops reading nested configs | ✅ 1 failure |
| NC90 | Preflight refuses everything | ✅ 3 failures — the fix must not block valid jobs |
| NC91 | Restore the reconnect re-attach (MIS-E2E-120) | ✅ 2 failures |
| NC92 | `unsubscribe` stops clearing the pending queue | ✅ 2 failures |
| NC93 | Failure emit key back to `error` (MIS-E2E-067) | ✅ 1 failure |
| NC94 | Completed emit says `extracting` again | ✅ 1 failure |
| NC95 | Reset removed from the start handler | ⚠️ **SURVIVED** — redundant once `_running` moved after the try. Re-pinned on the ORDER; **re-run ✅ 1 failure** |
| NC96 | Drop the dead-loop handler in `stop()` | ⚠️ **SURVIVED** — the test matched the `aclose()` handler instead. Rewritten to run a dead task; **re-run ✅ 1 failure** |
| NC97 | Metrics event name back to `"metrics"` (MIS-E2E-141) | ✅ 1 failure |
| NC98 | Scope falls back to `IS NULL` again (MIS-E2E-135) | ✅ 1 failure |
| NC99 | Resolver stops following the SAE hop | ✅ 1 failure |
| NC100 | `analysis_service` reads the column directly | ✅ 1 failure |
| NC101 | `browse_sae_features` drops the SAE link (MIS-E2E-100) | ✅ 1 failure |
| NC102 | Capture counters back to `Integer` (MIS-E2E-029) | ✅ 1 failure |
| NC103 | Cancel check drops `populate_existing` (MIS-E2E-057) | ✅ 1 failure — **and the OLD fixture could not have caught this** |
| NC104 | Retry wipes the row before checking the type (MIS-E2E-098) | ✅ 1 failure |
| NC105 | One janitor reverts to the PENDING state check (MIS-E2E-092) | ✅ 1 failure |
| NC106 | Drop a janitor from the coverage list | ✅ 1 failure |
| NC107 | `job_batch_size` back inside the branch (MIS-E2E-059) | ✅ 1 failure |
| NC108 | Template overwrites the job's `max_tokens` (MIS-E2E-060) | ✅ 1 failure |
| NC109 | Cancellation handler removed (MIS-E2E-058) | ✅ 2 failures |
| NC110 | Batch dispatch back on `position == 1` (MIS-E2E-066) | ✅ 1 failure |
| NC111 | Advance demands `position + 1` again | ✅ 1 failure |
| NC112 | NLP ids branch loses its scope (MIS-E2E-109) | ⚠️ **SURVIVED TWICE** — first the test matched the identifier in a log message, then `ast.walk` on the `If` found the *else* branch's filter. Fixed both; **re-run ✅ 2 failures** |
| NC113 | Import overwrites a system template (MIS-E2E-108) | ✅ 1 failure |
| NC114 | `directionOf` ignores the persisted sign (MIS-E2E-122) | ✅ 1 failure |
| NC115 | In-flight state back in `partialize` (MIS-E2E-123) | ✅ 1 failure |
| NC116 | Remove the combined double-submit guard (MIS-E2E-124) | ✅ 1 failure |
| NC117 | Restore the unconditional ref clobber (MIS-E2E-121) | ✅ 1 failure — `tsc` stays clean, so only a behavioural test can catch it |
| NC118 | Restore a pattern `pkill` (MIS-E2E-003) | ✅ 1 failure |
| NC119 | A spawn site stops recording its pid | ✅ 1 failure — a sweep over an unpopulated set kills nothing |
| NC120 | `/system/restart` drops the token (MIS-E2E-099) | ✅ 3 failures |
| NC121 | Token compared with `==` instead of `compare_digest` | ⚠️ **SURVIVED** — my own docstring named `compare_digest` while explaining why `==` is wrong. Parsed the call instead; **re-run ✅ 1 failure** |
| NC122 | postgres back to RollingUpdate (MIS-E2E-145) | ✅ 1 failure |
| NC123 | Redis published on 0.0.0.0 again (MIS-E2E-146) | ✅ 1 failure |
| NC124 | Compose frontend back to `:80` (MIS-E2E-147) | ✅ 1 failure |
| NC125 | Ingress drops the `.net` host's deny (MIS-E2E-148) | ✅ 1 failure |
| NC126 | `k8s_deploy` stops reporting step failure (MIS-E2E-147) | ✅ 1 failure |
| NC127 | `apt-key`-style global trust restored (MIS-E2E-148) | ✅ 1 failure |
| NC128 | The stale standalone manifest returns (MIS-E2E-144) | ✅ 1 failure |
| NC129 | The Stop-saves-the-SAE sentence returns (MIS-E2E-149) | ✅ 1 failure |
| NC130 | Anonymous allowed over HTTP again (MIS-E2E-150) | ✅ 1 failure |
| NC131 | Remove a table's row from the data-model page | ⚠️ **SURVIVED** — the prose still named it. Re-run removing EVERY mention: ✅ 1 failure |
| NC132 | Add a new ORM table and leave it undocumented | ✅ 1 failure — the invariant that actually matters |
| NC133 | Scraper drops the leading-`/` filter (MIS-E2E-114) | ⚠️ survived against the committed file alone; **✅ 1 failure** once the regeneration test is in scope — which is the guard that matters |
| NC134 | Instruction counts hardcoded again (MIS-E2E-161) | ✅ 1 failure |
| NC135 | A CLAUDE.md instruct reference goes stale again (MIS-E2E-155) | ✅ 1 failure |
| NC136 | Re-attribute metrics to Celery Beat in the README | ⚠️ **SURVIVED** — a negative check over prose cannot separate the claim from the correction beside it. Rewritten as a POSITIVE assertion (every doc must name `background_monitor`), which immediately found the PPRD saying what it is not without saying what it is; **re-run ✅ 1 failure** |
| NC137 | Move the monitor off asyncio | ✅ 1 failure |
| NC138 | Storage tab ungated again (MIS-E2E-160) | ✅ 1 failure |
| NC139 | A manifest enables `MISTUDIO_BYPASS_PIN` | ⚠️ **SURVIVED** — k8s puts an env name and value on separate lines, so a line scan never saw both. Parsed the YAML; **re-run ✅ 1 failure** |
| NC140 | Emit POSTs to its own loop again (MIS-E2E-136) | ✅ 1 failure |
| Gate | Build a mirror the old way and run the Verify step against it | ✅ fails, naming `0xcc`, `CLAUDE.md`, `scripts` |

**6 of the 14 surviving audit mutations are now killed** — M2, M3, M5, the cache divergence, **M13 (NC81)** and **M22 (NC86)**. Earlier count: — M2 (NC3), M3 (NC7),
M5 (NC5), and the cache divergence (NC2/NC4). Task 16.2 requires all 14.

| C141 | Strip `## Relevant Files` from an FTASKS file | ✅ 2 failures |
| C142 | Remove a dead path's `never written` annotation | ✅ 1 failure |
| C143 | Re-check `- [x] Zoom and pan` (the false completion) | ✅ 1 failure |
| C144 | Empty the health dashboard | ⚠️ **survived** — `.exists()` passes on a 0-byte file. Check tightened to require content; re-run ✅ 1 failure, and ✅ again when deleted outright |
| C145 | Sidebar hardcodes the tagline again beside the config | ✅ 1 failure |
| C146 | Sidebar hardcodes the product name again | ✅ 1 failure |
| C147 | Reintroduce a hand-maintained `BRAND.version` | ✅ 1 failure |
| C148 | Revert the tagline in the config to `Edge AI Feature Discovery` | ✅ 1 failure |

## Provenance

- Register: `0xcc/audits/E2E-2026-08/FINDINGS.md` (166 findings, ids `MIS-E2E-001`…`166`)
- Method and decisions: `0xcc/audits/E2E-2026-08/PLAN.md`
- Round records: `0xcc/audits/E2E-2026-08/rounds/` · Mutation logs: `mutations/`
- Traceability matrix: `0xcc/audits/E2E-2026-08/TRACEABILITY.md`
- One finding is **REFUTED** and deliberately excluded from this list (MIS-E2E-089); it
  stays in the register so a later round does not rediscover it.
