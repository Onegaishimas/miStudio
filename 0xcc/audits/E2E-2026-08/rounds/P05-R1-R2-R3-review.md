# P05 — REST API surface & schemas: all three rounds

**Phase:** P05 · **Date:** 2026-08-23
**Scope:** `backend/src/api/` (25 modules, 270 routes, 16,071 lines),
`backend/src/schemas/` (25 files, 6,340 lines), `backend/src/main.py`

Mutation log: `mutations/P05-mutations.md` (2 run, 1 survived).

## R1 — findings (15)

| Id | Sev | Claim |
|---|---|---|
| **MIS-E2E-105** | **P0** | Socket.IO accepts **any origin**, `connect` authenticates nothing, `subscribe` joins **any string** |
| **MIS-E2E-106** | **P0** | `PATCH` lets a caller set lifecycle `status` and falsify every training metric |
| MIS-E2E-098 | P1 | Retry **erases the failure evidence and commits**, then 400s — the row is stranded forever |
| MIS-E2E-099 | P1 | `POST /system/restart` is unauthenticated, unrated and idempotent — a self-sustaining restart loop |
| MIS-E2E-100 | P1 | Features from **downloaded** SAEs lose labels, stats and `activation_frequency` in the Steering browser |
| MIS-E2E-107 | P1 | A plain `alias` renames on output: metadata keys destroyed, `task_id` stranded, and a **test pins it** |
| MIS-E2E-108 | P1 | Template **import** overwrites protected system templates — the guard exists on update and delete only |
| MIS-E2E-112 | P1 | *(R2)* The API's only IDOR guard has no test |
| MIS-E2E-101 | P2 | A failed activation extraction appears in neither `/active` nor `/failed` |
| MIS-E2E-102 | P2 | Synchronous Redis reads inside the async Monitor handler block the event loop |
| MIS-E2E-103 | P2 | The documented 409 is unreachable on the large-delete path; a 400 is re-raised as a 500 |
| MIS-E2E-109 | P2 | NLP analysis writes **across extraction boundaries** — the worker drops the scope when ids are supplied |
| MIS-E2E-110 | P2 | IDL-22's error-hardening never covered the modules written after it |
| MIS-E2E-104 | P2 | A training is dispatched before its `celery_task_id` is persisted, so it cannot be revoked |
| MIS-E2E-111 | P3 | An internal LLM endpoint URL is published in a response its sibling deliberately withholds |

## The two P0s both escape the accepted posture

MIS-E2E-002 concedes that anyone who can reach the host can read the API. Neither of
these is covered by that concession.

**MIS-E2E-105** changes *who* can reach the host. `cors_allowed_origins="*"` under a
comment claiming *"CORS is handled by FastAPI's CORSMiddleware in main.py"* — which
`main.py:85-86` explicitly contradicts, installing none. engineio short-circuits its
origin check on `"*"` (`base_server.py:301`), and its own source comment notes this
matters more for WebSocket precisely because browsers do not apply CORS to it. So any
page an operator visits can open a socket and `subscribe` to any channel — and those
channels carry verbatim corpus text and generated model output.

**This makes MIS-E2E-018 load-bearing.** That finding was filed at P3 as "two comments
disagree about who handles CORS". It is the same comment, and it is the stated reason
the wildcard was considered safe. A doc-drift finding turned out to be the cause.

**MIS-E2E-106** is a privilege operation, not a read: `PATCH {"status":"completed"}`
on a running training unlocks SAE import from a partial checkpoint **with no
`finalized_from_step` marker** — defeating the exact honesty mechanism Feature 21 was
built to provide — makes the job uncancellable, and lets `progress`, `current_loss`
and `current_dead_neurons` be written in the same request.

## R2 — mutation controls

**M18 SURVIVED.** The checkpoint-delete parent guard — the API's only IDOR check, and
the one the reviewer flagged as *"the whole route rests on that one line"* — has no
test (MIS-E2E-112). 277 tests green with it removed.

**M19 KILLED.** Reintroducing the plain-alias trap inside a contract module failed
both schema-sync tests. The guard is real and bites, which is exactly what makes
MIS-E2E-107 a **scope** finding: `schemas/metadata.py` sits outside the swept set, so
a trap documented in two modules and guarded in one is live in a third.

## R3 — verification

| Verdict | Ids |
|---|---|
| **CONFIRMED** | 098, 099, 105, 106, 107, 112 |
| **PLAUSIBLE** | 100, 101, 102, 103, 104, 108, 109, 110, 111 |
| **REFUTED** | none |

Six confirmations were re-read at source rather than taken from the review: the retry
commit-then-400 ordering (including `increment_retry_count`'s field resets and its
`db.commit()`), the whole `/system/restart` handler, the Socket.IO wildcard and the
unvalidated `subscribe`, and `TrainingUpdate`'s field list against its `setattr` sink.
MIS-E2E-107 was reproduced by the reviewer against pydantic 2.12.5.

## Notable "checked and clean" — R2 of a later phase should attack these

The security pass was unusually precise about what it verified versus assumed, and
those distinctions are worth preserving:

- **The two HMAC-gated internal routes are genuinely correct.** `compare_digest`,
  a derived 64-hex secret, `secret_key` with `min_length=32` and **no default**, a
  uniform 403 for both a missing and a wrong token, and nginx `deny all` on top.
  One nit not filed: a non-ASCII header value makes `compare_digest` raise
  `TypeError` → 500 rather than 403, which reveals that the header is compared and
  nothing about the secret. **Assumed, not verified:** that `secret_key` is actually
  distinct per deployment — the k8s secret was not inspected. **P10 should.**
- **IDOR on nested paths is otherwise clean** — all 11 enumerated, 10 bind in-query.
- **`extra="forbid"` never co-occurs with an alias** anywhere in `schemas/`.
- **No route returns a bare ORM row**; all 109 handlers without a `response_model`
  project fields by hand, and the projections withhold `store_path`, the `*_task_id`s
  and `error_traceback`.
- Two weak spots noted rather than filed: `TaskQueueData`'s redaction is a top-level
  denylist that does not walk nested dicts, and `AppSettingResponse` masking lives at
  three call sites rather than in the schema.
- Two items the reviewer noted and deliberately did not file: `CircuitService.create`
  honours a caller-supplied `created_at` (timestamp forgery on an evidence artifact,
  reachable only through the explicit bring-your-own-document import path), and
  `retry`'s unconstrained `param_overrides` lets a caller replace `repo_id` on a
  retried download.

## Phase closed

**15 findings** (MIS-E2E-098 … 112), **2 P0**. Mutations: 2 run, 1 survived.
Tree verified clean.

**The one sentence for the synthesis:** this phase's two P0s are both cases where a
control that exists elsewhere in the same file was not applied here — the `is_system`
guard present on update and delete but not import, and an origin check disabled by a
comment describing a middleware that was deliberately never installed.
