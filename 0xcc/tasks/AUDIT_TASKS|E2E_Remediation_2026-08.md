# AUDIT_TASKS | E2E Remediation 2026-08

**Source:** `0xcc/audits/E2E-2026-08/FINDINGS.md` — 166 findings from a twelve-phase
end-to-end assessment (2026-08-23). Every task cites its finding id; the register
carries the evidence, the reproduction and the proposed remediation for each.

**Status:** ⏳ **In progress** — started 2026-08-23 · **Findings:** 13 P0 · 62 P1 · 68 P2 · 23 P3

| Wave | Scope | State |
|---|---|---|
| **Part 1** | MIS-E2E-143 — the public-mirror disclosure | ✅ **CLOSED**, verified live |
| **Wave 1** | Task 7 — test-schema divergence (the prerequisite) | ⏳ 7.1 ✅ · 7.2 ✅ · 7.3 ✅ · 7.5 ✅ · 7.4, 7.6 open |
| Waves 2–9 | P0s, then correctness, then docs | ❌ not started |

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

- [ ] 1.1 `POST /labeling/models/openai`: **never attach the stored key to a caller-named host.** Allow-list the origin, and call `validate_llm_endpoint_url` on that path. — MIS-E2E-069
- [ ] 1.2 Emit `Bearer {{OPENAI_API_KEY}}` in **both** Postman writers (`openai_labeling_service.py:406`, `:645`), matching the cURL branch. — MIS-E2E-072
- [ ] 1.3 `decrypt_value`: distinguish "not an envelope" (legacy → return as-is, **log a counter**) from `InvalidTag` (**raise**). Fixes three findings at once. — MIS-E2E-004, -041, -056
- [ ] 1.4 Add an encrypting backfill for legacy plaintext `labeling_jobs.openai_api_key` rows; the counter from 1.3 measures the exposure. — MIS-E2E-041
- [ ] 1.5 Regression tests: a key never reaches a non-allow-listed host; no artifact file contains the key across **all three** `export_format` values. — MIS-E2E-069, -072

## Task 2 — The Settings PIN (P0)

- [ ] 2.1 Store the PIN outside the generic settings table, **or** deny `settings_pin_hash` in `PUT`, `PUT /bulk`, `DELETE` and both `GET`s. Marking it `is_sensitive=True` fixes only the read. — MIS-E2E-055, -165
- [ ] 2.2 Enforce the PIN **server-side** on the settings routes (short-lived token from `/pin/verify`), or amend IDL-25 to state it is a UI affordance. — MIS-E2E-005
- [ ] 2.3 Gate the **Storage** tab, which arms irreversible checkpoint deletion, not only `api_keys`; correct `settings-reference.md:56`. — MIS-E2E-160
- [ ] 2.4 Verify `MISTUDIO_BYPASS_PIN` is false in every shipped manifest. — MIS-E2E-005

**Acceptance:** `GET /api/v1/settings` on the live deployment returns no PIN material — the check that failed in P12.

## Task 3 — Arbitrary deletion (P0)

- [ ] 3.1 `activation_service.delete_extraction`: route the id through `resolve_user_path`, reject ids not matching `^[A-Za-z0-9_-]+$` at the schema, **and** move the filesystem delete inside the `if extraction:` guard. Three layers, because this one deletes. — MIS-E2E-070
- [ ] 3.2 Remove `raw_path` / `file_path` / `quantized_path` from the create and update schemas — the workers write them. — MIS-E2E-071
- [ ] 3.3 Make every deletion sink re-assert containment with `resolve_user_path` immediately before `rmtree`. **The database is not a trust boundary while any API can write to it.** — MIS-E2E-071
- [ ] 3.4 Fix the dataset-cancel worker to delete only the tokenization being cancelled, and store the `task_id` so the revoke branch is not dead. — MIS-E2E-151
- [ ] 3.5 Re-fetch the prune preview immediately before the confirmation dialog, or confirm against the policy the backend will apply. — MIS-E2E-128

## Task 4 — Mass assignment and the WebSocket boundary (P0)

- [ ] 4.1 Remove `status` and the derived progress/metric fields from `TrainingUpdate`, `ModelUpdate`, `DatasetUpdate`; replace the blind `setattr` loops with explicit allow-lists (`cluster_profile_service.py:272` is the reference). — MIS-E2E-106
- [ ] 4.2 Set Socket.IO `cors_allowed_origins` to `settings.allowed_origins` and delete the false comment. — MIS-E2E-105, -018
- [ ] 4.3 Validate `subscribe` against a channel-pattern allow-list; cap subscriptions per connection; type-check the payload. — MIS-E2E-105, -140
- [ ] 4.4 Register the Socket.IO handlers **once** — the duplicates silently overwrite, so the acks never fire and `ws_manager` is always empty. — MIS-E2E-138
- [ ] 4.5 Regression test: a handshake from a foreign `Origin` is refused. Negative control: flip the setting back and require a red. — MIS-E2E-105

## Task 5 — Capabilities that do not exist (P0)

- [ ] 5.1 **Wire or delete** `steering_resilience.py`. Five state-mutating functions have zero callers, so `/steering/status` can only ever return `"healthy"` and `/steering/reset` is a no-op that reports success. — MIS-E2E-062
- [ ] 5.2 **Implement the `affine_residual` freeze-leak gate**, or correct `CLAUDE.md`. The threshold is stored and never read; per the project's own description this is the only point an incomplete freeze is detectable. — MIS-E2E-079
- [ ] 5.3 For both: delete the wiring line and require a red. A grep for a caller is the weaker form and is not sufficient. — `CLAUDE.md` Reachability gate

---

## Task 6 — Wrong results presented as correct (P1)

The class this product can least afford. Each item is a number a user reads as a measurement.

- [ ] 6.1 Add `sentence-transformers`, or return `None` and render "not measured". Every coherence/behavioral score ever shown is the constant `0.5`. Broaden the `except` either way. — MIS-E2E-063
- [ ] 6.2 Measure J-lens convergence against **held-out** prompts or split-half agreement. The current criterion is the shrinkage of a running mean — it stops at `n ≈ σ/δ`, proportional to variance. Until then, do not call it convergence in the artifact or the docs. — MIS-E2E-080
- [ ] 6.3 Rename `linearisation_residual_*` in the published artifact to what it measures, or compute the residual. It travels to HuggingFace. — MIS-E2E-081
- [ ] 6.4 Rank-check before QR in the band metrics and refuse or report a degenerate basis (FVE overstated 4.5×); wire the random-direction controls or stop documenting `control_seed`; implement the "sustained rise" the docstring describes. — MIS-E2E-088
- [ ] 6.5 Make `rankOf` and `diffColor` agree on one index base, and assert the "same top token" legend swatch is reachable. — MIS-E2E-129
- [ ] 6.6 Route each steered feature through its **own** `sae_id` in compare and sweep, and validate `layer == sae.layer`. — MIS-E2E-064
- [ ] 6.7 Carry `normalize_activations` on the SAE record so capture and attribution stop running the SAE off-distribution. — MIS-E2E-083
- [ ] 6.8 Gate the steering core on `dial != 0` — a negative dial currently returns the baseline labelled as steered, and negative strength is canonical. — MIS-E2E-065
- [ ] 6.9 Either implement `anthropic_rescale` per its paper or collapse it — it is arithmetically identical to `constant_norm_rescale` (2.4e-7). — MIS-E2E-085

## Task 7 — The test schema is not the production schema (P1)

The root enabler behind the production 500 the user hit. **Two of the constraints are already fixed; the mechanism is not.**

- [x] 7.1 ✅ Added a guard test diffing `Base.metadata` against the migrated schema — constraints, foreign keys, indexes. This single test covers MIS-E2E-031, -033 and every future migration-only constraint. — MIS-E2E-031, -033, -048
- [x] 7.2 ✅ Re-created the three foreign keys the ORM declares and the database lacks, or drop them from the ORM — pick one. — MIS-E2E-033
- [x] 7.3 ✅ Tests for delete cascades against a **migrated** database. Flipping all three CASCADEs on `features` currently leaves 211 tests green. — MIS-E2E-053
- [ ] 7.4 One round-trip test: build a maximal `CircuitDefinitionV1`, import, export, assert **document equality**. Field-by-field assertions only cover the fields someone remembered. — MIS-E2E-052, -037
- [x] 7.5 ✅ Derived `REQUIRED_TABLES` from `Base.metadata`; decide deliberately whether a missing table blocks startup; test it. — MIS-E2E-032, -051, -157
- [ ] 7.6 Narrow `check_migrations.py`'s claim to what it checks, or extend it to constraints; wire it into CI. Delete `find_column_gaps.py`. — MIS-E2E-048, -049, -022

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
| `backend/tests/unit/test_analysis_cache_upsert.py` | **New** (out-of-band). Pins the cache upsert that fixed the production 500 |
| `backend/src/services/analysis_service.py` | Cache upsert; ablation's dead precondition removed; correlations scoped to the SAE |
| `backend/src/models/{feature,feature_analysis_cache}.py` | Declare the unique constraints that existed only in migrations |
| `frontend/src/components/features/FeatureTokenAnalysis.tsx` | BPE marker stripped for display; continuations marked |

## Negative controls run

Every fix is required to have a test that **fails when the fix is removed**. Recorded
here as they are verified, because "a test exists" is not the same claim.

| Control | Mutation | Result |
|---|---|---|
| NC1 | `features.training_id` ondelete CASCADE → SET NULL | ✅ 2 failures |
| NC2 | Remove `FeatureAnalysisCache.__table_args__` (the MIS-E2E-031 shape) | ✅ 1 failure |
| NC3 | **Re-run of audit mutation M2** — shrink the validator to a hand-list | ✅ 3 failures *(survived during the audit)* |
| NC5 | **Re-run of audit mutation M5** — flip all 3 Feature CASCADEs to RESTRICT | ✅ 4 failures *(survived during the audit, 211 tests green)* |
| NC4 | Revert `_cache_analysis` to the blind INSERT | ✅ 2 failures |
| Gate | Build a mirror the old way and run the Verify step against it | ✅ fails, naming `0xcc`, `CLAUDE.md`, `scripts` |

**3 of the 14 surviving audit mutations are now killed** — M2 (NC3), M5 (NC5), and
the cache divergence (NC2/NC4). Task 16.2 requires all 14.

## Provenance

- Register: `0xcc/audits/E2E-2026-08/FINDINGS.md` (166 findings, ids `MIS-E2E-001`…`166`)
- Method and decisions: `0xcc/audits/E2E-2026-08/PLAN.md`
- Round records: `0xcc/audits/E2E-2026-08/rounds/` · Mutation logs: `mutations/`
- Traceability matrix: `0xcc/audits/E2E-2026-08/TRACEABILITY.md`
- One finding is **REFUTED** and deliberately excluded from this list (MIS-E2E-089); it
  stays in the register so a later round does not rediscover it.
