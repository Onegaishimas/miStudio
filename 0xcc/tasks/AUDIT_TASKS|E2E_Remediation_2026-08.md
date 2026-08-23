# AUDIT_TASKS | E2E Remediation 2026-08

**Source:** `0xcc/audits/E2E-2026-08/FINDINGS.md` — 166 findings from a twelve-phase
end-to-end assessment (2026-08-23). Every task cites its finding id; the register
carries the evidence, the reproduction and the proposed remediation for each.

**Status:** ⏳ **In progress** — started 2026-08-23 · **Findings:** 13 P0 · 62 P1 · 68 P2 · 23 P3
**Suites:** backend **3079 passed / 0 failed** (baseline 2883) · frontend **1224 passed / 0 failed** (baseline 1211) · `tsc --noEmit` clean · **CI Backend Tests green**

| Wave | Scope | State |
|---|---|---|
| **Part 1** | MIS-E2E-143 — the public-mirror disclosure | ✅ **CLOSED**, verified live |
| **Wave 1** | Task 7 — test-schema divergence (the prerequisite) | ✅ **CLOSED** — 7.1–7.6 |
| **Wave 2** | Tasks 1–5 — the 13 P0s | ✅ **CLOSED** — Tasks 1–5, all 13 P0s. 46 negative controls recorded. |
| **Wave 3** | Task 6 — wrong results presented as correct (9 findings) | ✅ **CLOSED** — all 9. 29 negative controls. |
| **Wave 4** | Task 8 — pin the surviving audit mutations | ⏳ **next** |
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
- [ ] 0.7 Delete `aaaa/` — 20 byte-identical copies of authoritative `0xcc/` documents, not in the exclusion list. — MIS-E2E-007, -164

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
- [ ] 2.3 Gate the **Storage** tab, which arms irreversible checkpoint deletion, not only `api_keys`; correct `settings-reference.md:56`. — MIS-E2E-160
- [ ] 2.4 Verify `MISTUDIO_BYPASS_PIN` is false in every shipped manifest. — MIS-E2E-005

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

- [ ] 8.1 Assert the steering hook target **is** `structure.layers_module[L]` on **both** implementations. Failure mode: `steered == unsteered at every dial`; cost a hardware round to find. — MIS-E2E-078
- [ ] 8.2 Parametrized test that every `_SENSITIVE_KEYS` member stores ciphertext even when the client sends `is_sensitive: false`. — MIS-E2E-077
- [ ] 8.3 Assert `weights_only=True` at all three `torch.load` sites — the only guard against artifact RCE. — MIS-E2E-091
- [ ] 8.4 Assert every **registered** task resolves to its intended queue, driven off the registry. Catches `train_sae` landing on `datasets`. — MIS-E2E-093, -097
- [ ] 8.5 IDOR test on `DELETE /trainings/{tid}/checkpoints/{cid}`. — MIS-E2E-112
- [ ] 8.6 Assert an exact **mid-range** value for the auto-baseline — the existing file samples only where the slope vanishes or clamps. — MIS-E2E-127
- [ ] 8.7 Derive BR-002's band-constant scan from the jlens **package**, not a two-module list; extend it to the frontend. — MIS-E2E-090
- [ ] 8.8 Parametrize the MCP payload assertion off the registry; make absence from `EXPECTED_CALLS` an explicit, listed exemption rather than silence. — MIS-E2E-119
- [ ] 8.9 Two tests for emit retry: a `ReadTimeout` retries; a `RemoteProtocolError` retries. — MIS-E2E-142, -137

## Task 9 — Realtime (P1)

- [ ] 9.1 Drop the handler re-attach in `WebSocketContext` — socket.io does not detach on disconnect, so every event fires N+1 times after N reconnects. — MIS-E2E-120
- [ ] 9.2 Call `ws_manager.emit_event()` directly from async contexts instead of the HTTP loopback; `asyncio.to_thread` is the stopgap the one fixed site uses. Requires 4.4 first. — MIS-E2E-136, -138
- [ ] 9.3 Retry on `httpx.TransportError`, not `TimeoutException`. — MIS-E2E-137
- [ ] 9.4 Reset `_running` in a `finally` and log the setup failure, so the monitor cannot die silently. — MIS-E2E-139
- [ ] 9.5 Fix the emit key `"error"` → `"error_message"`, and type the emit payloads so a rename cannot pass silently. — MIS-E2E-067
- [ ] 9.6 Clear `pendingSubscriptionsRef` on unsubscribe; rename the `"metrics"` event to `"system:metrics"`. — MIS-E2E-126, -141

## Task 10 — Provenance: `Feature.training_id` is NULL by design (P1)

- [ ] 10.1 One resolver — "the training/model behind this feature" — that walks the `ExternalSAE` row. `Feature.source_id` already exists and nothing uses it. — MIS-E2E-135
- [ ] 10.2 Fix `browse_sae_features` so external-SAE features keep labels, stats and `activation_frequency` — the third consumer with this assumption, and the one still open. — MIS-E2E-100
- [ ] 10.3 Sweep every direct `Feature.training_id` read. — MIS-E2E-135

## Task 11 — Correctness bugs (P1)

- [ ] 11.1 `BigInteger` for `circuit_runs.bytes_total`/`events_total`; give the error handler a fresh session so a poisoned one cannot swallow the failure. — MIS-E2E-029
- [ ] 11.2 `refresh`/`populate_existing` before the cancel check — the identity map means a cancel can never be observed. The fake session in its test has no identity map. — MIS-E2E-057
- [ ] 11.3 Bind `job_batch_size` before the branch; catch `_LabelingCancelled` before the generic handler; give `max_tokens` the precedence `max_examples` has. — MIS-E2E-058, -059, -060
- [ ] 11.4 Resolve the dispatch branch **before** `increment_retry_count` — retry currently erases the failure evidence, commits, then 400s, stranding the row forever. — MIS-E2E-098
- [ ] 11.5 Track spawned worker PIDs instead of `pkill -9 -f steering@`; bound worker spawn. HMAC-gate `POST /system/restart`. — MIS-E2E-003, -099
- [ ] 11.6 Route all four janitors through `looks_abandoned`, and parametrize the regression test over the janitor **registry**. — MIS-E2E-092
- [ ] 11.7 Fix the batch-extraction dispatch to track created job ids. — MIS-E2E-066
- [ ] 11.8 Scope NLP analysis writes to the path extraction. — MIS-E2E-109
- [ ] 11.9 Guard the template-import overwrite on `is_system`; validate the body with a schema. Same rule missed in two migrations. — MIS-E2E-108, -045
- [ ] 11.10 Fix the `metadata.py` plain `alias`, give `DatasetMetadata` `extra="allow"`, and **rewrite the 13 assertions that pin the defect**. — MIS-E2E-107
- [ ] 11.11 Frontend state: sign-before-zeroing in rebalance; exclude in-flight state from `persist`; generation tokens on the feature fetches and the cleanup abort. — MIS-E2E-121, -122, -123, -124

## Task 12 — Infrastructure (P1)

- [ ] 12.1 Delete the root `k8s/mistudio-deployment.yaml`; have `k8s_deploy` apply `k8s/base` via kustomize. It currently reverts the queue-split and SQL-echo fixes. — MIS-E2E-144
- [ ] 12.2 `strategy: Recreate` on postgres and redis, or StatefulSets with PVCs. — MIS-E2E-145
- [ ] 12.3 Bind compose postgres/redis to `127.0.0.1`; set a Redis password. The broker is the LAN-writable one. — MIS-E2E-146
- [ ] 12.4 Fix `k8s_deploy`'s `&&`-chain so a failed pull/apply/rollout is not reported as success. — MIS-E2E-147
- [ ] 12.5 Fix the compose frontend port (`3000:80` → 8080), add the `/ollama/` location, split the compose worker's queues, fix the `server_name` typo. — MIS-E2E-147
- [ ] 12.6 Scope `MCP_TOOL_CATEGORIES` off the ingress `/api` prefix or deny `/api/internal/*` there; use `signed-by=` instead of `apt-key adv`. — MIS-E2E-148
- [ ] 12.7 **Remove the 9 `--exclude` flags from `frontend-ci.yml`** — all 329 tests pass; 27% of the suite is ungated. — MIS-E2E-025
- [ ] 12.8 **Run lint in CI** and fix the 34 errors. Nothing gates on it today, which is how a Rules-of-Hooks violation shipped. — MIS-E2E-024, -023
- [ ] 12.9 Fix `${MCP_AUTH_TOKEN:?}` so it does not break every `docker compose` command. — MIS-E2E-026
- [ ] 12.10 Ship `VERSION` in the backend image and make the fallback loud. — MIS-E2E-028

---

## Task 13 — Documentation (P2)

- [ ] 13.1 **Fix `docs/miStudio_Manual.md:349`** — the exact sentence that cost a real SAE. Then decide whether that manual should exist. — MIS-E2E-149
- [ ] 13.2 Make `MCP_ALLOW_ANONYMOUS` genuinely stdio-only, which is what both the manual and the guard's own error message already promise. — MIS-E2E-150
- [ ] 13.3 Rewrite the K8s install guide's four `sed` steps — none matches, and one renames the database to the password. — MIS-E2E-152
- [ ] 13.4 Point README at the real Compose quickstart, not `start-mistudio.sh` (hardcoded `/home/x-sean`, needs a venv it never creates, different domain). — MIS-E2E-162
- [ ] 13.5 Correct IDL-5 and the **five** documents propagating it; IDL-16's three false claims; IDL-1/12's channel and event conventions; IDL-11's DLQ and backoff; IDL-38's "one steering core". — MIS-E2E-156, -157, -158, -159, -076
- [ ] 13.6 Reconcile PPRD §2.1 — **13 rows** mark "Planned" work that shipped — and decide which document is authoritative for status. — MIS-E2E-011
- [ ] 13.7 Fix CLAUDE.md: the off-by-one instruction references (which point at the **wrong action**), the phantom paths, the self-contradictions, the stale test counts, the "returns 501" claims. — MIS-E2E-155, -010, -163
- [ ] 13.8 Add a PADR IDL stating the no-app-auth posture, its boundary, and what invalidates it. Currently undocumented and indistinguishable from an oversight. — MIS-E2E-002, -166
- [ ] 13.9 Document the 11 missing tables in `data-model.md` and remove its unearned "verified against the ORM models"; add a doc test diffing it against `Base.metadata`. — MIS-E2E-050, -164
- [ ] 13.10 Add `## Relevant Files` to FTASKS 024–028 and the six ad-hoc files; triage the 348 unchecked boxes; fix the 22 dead paths. — MIS-E2E-153, -154, -012
- [ ] 13.11 Regenerate `docs/mcp-contract.md` after fixing the `startswith("/")` filter — it lists three endpoints that do not exist and a test pins them. Derive the tool-count prose from the registry. — MIS-E2E-114, -017, -161

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
| `backend/find_column_gaps.py` | **Deleted** (MIS-E2E-049) — its `create_table` regex truncated at the first `)`, so every report was a false positive |
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
| NC14 | Revert `models.py` requantize to `resolve_data_path` (MIS-E2E-071) | ✅ 1 failure |
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
| NC35 | Construct a second `WebSocketManager()` in `main.py` | ✅ 1 failure |
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
| Gate | Build a mirror the old way and run the Verify step against it | ✅ fails, naming `0xcc`, `CLAUDE.md`, `scripts` |

**4 of the 14 surviving audit mutations are now killed** — M2 (NC3), M3 (NC7),
M5 (NC5), and the cache divergence (NC2/NC4). Task 16.2 requires all 14.

## Provenance

- Register: `0xcc/audits/E2E-2026-08/FINDINGS.md` (166 findings, ids `MIS-E2E-001`…`166`)
- Method and decisions: `0xcc/audits/E2E-2026-08/PLAN.md`
- Round records: `0xcc/audits/E2E-2026-08/rounds/` · Mutation logs: `mutations/`
- Traceability matrix: `0xcc/audits/E2E-2026-08/TRACEABILITY.md`
- One finding is **REFUTED** and deliberately excluded from this list (MIS-E2E-089); it
  stays in the register so a later round does not rediscover it.
