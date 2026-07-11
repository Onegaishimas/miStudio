# Feature Review: 006 — Model Steering

**Reviewed:** 2026-07-11
**Reviewer:** deep code review (doc↔code accuracy + bug/correctness + quality/tech-debt)
**Feature docs:** [FPRD v2.0](../prds/006_FPRD|Model_Steering.md) (2026-03-21), FTDD, FTID
**Verdict:** The Celery-async steering migration (this session's work) is well-architected — GPU isolation, resilience primitives, experiment persistence. `forward_hooks.py` and the dynamic-discovery integration are clean. No P0. Main issues: a **frontend concurrency bug** in the module-level resolver singletons, **zero test coverage** for the whole feature, and the FPRD §5/§6 predating the async migration. The steering-worker config issues (F10 from the Celery review) carry over.

---

## 1. Scope

**Backend read:** `ml/forward_hooks.py` (253, full), `api/v1/endpoints/steering.py` (1198, full route inventory + async paths), `schemas/steering.py` (517, validation), `models/steering_experiment.py` (101), `models/prompt_template.py` (48). `services/steering_service.py` (2639) and `steering_resilience.py` (508) surveyed; `workers/steering_tasks.py` (754) reviewed in depth in this session's Celery review (F10).
**Frontend read:** `stores/steeringStore.ts` (resolver-coordination logic, lines 40-200).
**Tests present:** **NONE.** No `test_steer*` anywhere in the backend suite. (The QA agent context flagged "no unit tests for steeringStore core functions" in Jan 2026 — still true, and it extends to the entire backend steering path.)

---

## 2. Doc ↔ Code Accuracy

FPRD v2.0's Architecture Notes (§ end) accurately describe the Celery migration, but §5 (endpoints) and §6 (data model) and §8 (params) predate it:

### 2.1 API endpoints — §5 mostly superseded by async routes

| PRD (§5) | Reality |
|---|---|
| `POST /steering/generate` | **Does not exist.** Superseded by `/async/compare`, `/async/sweep`, `/async/combined`. |
| `POST /steering/calibrate` | **Does not exist** as a route. |
| `POST /steering/compare`, `/sweep`, `/combined` | Sync versions exist (`/compare`, `/sweep`) **plus** the real async ones `/async/compare`, `/async/sweep`, `/async/combined` (the ones the frontend uses). |
| (not in PRD) | `GET /async/result/{task_id}`, `DELETE /async/task/{task_id}`, `GET /status`, `POST /reset`, `POST /cleanup`, `GET /mode`, `POST /enter-mode`, `POST /exit-mode`, and a full `/experiments` CRUD (`GET/POST /experiments`, `GET/DELETE /experiments/{id}`, `POST /experiments/delete`). |

### 2.2 Data model — PromptTemplate wrong, SteeringExperiment undocumented

| PRD (§6) | Reality |
|---|---|
| `prompt_templates`: `content TEXT`, `variables JSONB`, `template_type`, `usage_count` | Real: `prompts JSONB` (an **array** of prompts — it's a multi-prompt template), `tags JSONB`, `description`, `is_favorite`. **No** `content`/`variables`/`template_type`/`usage_count`. `id` is UUID here (unlike most other tables). |
| `SteeringResult` (transient dataclass only) | Real: `SteeringExperiment` is a **persisted DB table** (`steering_experiments`) backing the `/experiments` CRUD and reproducibility. The PRD's §6.2 "transient" framing is obsolete — experiments are durable. |

### 2.3 Parameters — ranges wrong

| PRD | Reality |
|---|---|
| strength `-10 to +10` (FR-2.1) | `-300.0 to +300.0` ([steering.py schema:33](../../backend/src/schemas/steering.py#L33)) — Neuronpedia-compatible raw coefficients. |
| `max_new_tokens` `10-500` (§8) | `1-2048`. `top_k` default 50 — schema uses different bounds. |

### 2.4 Steering-hook mechanics — §3.1 illustrative, not literal

The PRD's `steering_hook` pseudocode (encode→modify→decode) is conceptual; the real implementation in `steering_service.py` applies steering vectors directly to the residual stream via `forward_hooks`-style injection (the §3.3.2 combined-mode pseudocode is closer to reality). Fine as illustration, but note it's not the actual code path.

**Recommendation:** rewrite FPRD 006 §5/§6/§8 to match the async architecture; the Architecture Notes section is the accurate part — promote its content into the main body.

---

## 3. Findings (severity-ranked)

### P1 — Frontend concurrency bug

**F1. Module-level resolver singletons hang the first operation when a second starts.**
[steeringStore.ts:53,108,161](../../frontend/src/stores/steeringStore.ts#L53) — `pendingBatchResolver` / `pendingSweepResolver` / `pendingCombinedResolver` are module-level singletons (one global each). `createBatchResolver` calls `cleanupBatchResolver()` first, which `clearTimeout`s and **nulls the previous resolver without resolving or rejecting its promise**. So if a user starts a second compare while the first is still running, the first `await generateBatchComparison(...)` **hangs forever** (its promise is orphaned — neither resolved, rejected, nor timed out, since its timeout was cleared). The `taskId` guard prevents cross-talk on the *completion* side, but the create-side cleanup drops the earlier promise. This is the residual of the Jan-2026 "pendingBatchResolver leak" QA finding — the timeout/cleanup was added, but the orphan-on-replace case remains.
**Fix:** on cleanup, reject the outgoing resolver (`reject(new Error('superseded by new request'))`) before nulling, or key resolvers by taskId in a Map instead of a singleton.

### P2 — Correctness (carried from Celery review, steering-specific)

**F2. Steering-worker config contradicts its own comments (Celery review F10).**
[celery.sh:405-431](../../backend/celery.sh#L405) — `--pool=solo --max-tasks-per-child=1` with a comment claiming per-task worker recycling, but **solo pool ignores `max-tasks-per-child`** (this was the root cause of the Jan-2026 steering hang; the `finally`-block state reset in `steering_tasks.py` is the actual compensating guard). The `time_limit=180` "SIGKILL guaranteed" docstring is also questionable under solo (no child to kill — the in-service `GenerationWatchdog` is what actually fires). **Fix:** correct the misleading comments; document that the watchdog + finally-reset is the real safety mechanism, not worker recycling.

**F3. `acks_late=True` + global `task_reject_on_worker_lost=True` → steering redelivery risk.** If a steering task OOM-kills its worker, the message is redelivered and re-executed on restart → potential crash loop on a pathological input. Consider `acks_late=False` for steering (accept-on-start) given it's non-idempotent GPU work.

### P3 — Quality / tech-debt

**F4. Zero test coverage for the entire steering feature.** No backend `test_steer*` and no `steeringStore` tests. For a feature with a 2639-line service, GPU isolation, resilience/circuit-breaker primitives, and a known concurrency bug (F1), this is the **biggest coverage gap in the app**. At minimum: a steeringStore resolver-lifecycle test (would catch F1), a `/async/compare` happy-path integration test, and an experiments-CRUD test.

**F5. `steering_service.py` is 2639 lines** — the largest single service file. It holds hook injection, generation, calibration, watchdog, emergency GPU cleanup, and experiment orchestration. Cohesive but a candidate for extracting the watchdog/resilience concerns (some already in `steering_resilience.py`).

**F6. Two module-level singletons pattern (F1) also means the UI silently supports only one in-flight steering op per type** — arguably a product limitation worth making explicit in the UI (disable Generate while running) rather than relying on the buggy replace path.

---

## 4. Test Coverage Notes

**The worst-covered P0 feature in the review.** Zero tests. The async migration, GPU isolation, resilience primitives, experiment persistence, and the resolver coordination are all untested. Given F1 (a real hang) and F2/F3 (worker semantics), this feature most needs a test harness. Priority tests: (1) steeringStore resolver lifecycle incl. concurrent-start (catches F1), (2) `/async/compare` + `/async/result` round-trip, (3) experiments CRUD, (4) strength validation bounds.

---

## 5. Summary — if you fix three things

1. **F1** — the resolver-singleton orphan-on-replace hang (real user-facing bug under concurrent steering). Reject the superseded promise or key by taskId.
2. **F4** — add a steering test harness (currently zero). This feature has the most complex runtime behavior and the least verification.
3. **F2** — correct the steering-worker config comments and document the real safety mechanism (watchdog + finally-reset, not worker recycling).

**The good:** `forward_hooks.py` (clean, GPU-memory-explicit, fail-loud), the dynamic-discovery integration (any transformer steers without a whitelist), the Neuronpedia-compatible strength calibration, and the Celery async migration with experiment persistence are all solid. The strength/param validation in the schema is thorough. The gaps are concurrency handling and verification, not core mechanics.
