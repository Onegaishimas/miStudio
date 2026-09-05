# Debt remediation tracker — cancellation, estimators, miLLM lifecycle

Plan: `my-goal-is-to-linear-ember.md` in the operator's plan store (outside this repo, so it is deliberately not in Relevant Files). One property unites every
defect here: **it reports success while doing nothing.** A cancel endpoint
returns 200 and emits a "cancelled" event while the GPU runs for hours; an
estimator says 8.8 h for a 68 h job; a preflight says 3.9 GB for a 15.4 GB load.
None fails loudly, so none was caught by a test or a review.

**The governing constraint.** Every Celery worker here is `--pool=solo -c 1`.
`revoke(terminate=True)` signals a pool child and solo has none; a busy solo
worker is not reading the control queue at all, so the revoke is never
delivered. It returns cleanly and does nothing. Cancellation must be
cooperative, and SIGKILL is not a fallback — it crashed the pool once and
stranded an `acks_late` message for the full 12-hour visibility timeout.

| Phase | Status | Commit |
|---|---|---|
| 0 — repair the hand-edited quantization rows | ✅ done | (pre-`efc576a3`) |
| 1.0 — the module, no callers | ✅ done | `efc576a3` |
| 1 — the guard everywhere, still no checkers | ✅ done | this record |
| 2 — `extract_activations`, the GPU proof | ✅ code; hardware acceptance pending | this record |
| 3 — the remaining compute tasks | ▢ | |
| 4 — the downloads | ▢ | |
| 5 — shim the five healthy conventions | ▢ | |
| 6 — the missing routes | ▢ | |
| 7 — cleanup | ▢ | |
| R1 / R2 / R3 review rounds | ▢ | |

---

## Relevant Files

| File | Purpose |
|---|---|
| `backend/src/core/cancellation.py` | The one cooperative-cancellation mechanism: `OperatorCancelled`, `CancelScope`, the `SCOPES` registry, `cancel_checker`, `request_cancel`, `record_progress`, `guard_allows`, `@cooperative_cancel` |
| `backend/tests/unit/test_cancellation_core.py` | The rule in isolation — throttle, identity-map defeat, `missing_row` policies, the guard matrix |
| `backend/tests/unit/test_progress_guard.py` | Shape B against the REAL writers: a cancellation is not overwritten by the next progress write |
| `backend/tests/unit/test_extraction_cancellation.py` | Phase 2 Shapes A and C: the extraction actually stops, and the endpoint's write is what the worker reads |
| `backend/src/workers/model_tasks.py` | `extract_activations` + `build_extraction_progress_callback` — the checkpoint |
| `backend/src/api/v1/endpoints/models.py` | `cancel_extraction` — now asks, rather than pretending to terminate |
| `backend/src/services/extraction_db_service.py` | `update_progress` for activation extraction — routed through `record_progress` |
| `backend/src/services/extraction_service.py` | `update_extraction_status{,_sync}` for the SAE lifecycle — guarded via `guard_allows` |
| `backend/src/workers/neuronpedia_tasks.py` | `update_export_progress` — the progress-only writer |
| `backend/src/workers/training_tasks.py` | `update_training_progress` — the tracker, now shimmed onto core |
| `backend/src/workers/jlens_progress.py` | `update_row` — now a shim; the guard rule originated here |
| `backend/tests/unit/test_extraction_aborts_without_a_row.py` | The phantom-extraction guard; its progress-warning test is now driven rather than scraped |
| `backend/tests/unit/test_jlens_cancel.py`, `test_task_heartbeat.py`, `test_training_tasks.py` | Existing suites whose fakes had to learn the fresh-read chain |

---

## Phase 0 — the quantization rows (done)

`model_service.py` derives the cache directory as `{repo}--{QUANT}`, and
**deletion recomputes that path** rather than reading the stored `cache_path`.
Two rows had been edited by SQL while diagnosing, so a delete would have
targeted the wrong directory — stranding 17–23 GB or removing only a symlink.

miLLM `granite-4.2-8b` reconciled:

```
before:  quant=FP16  cache_path=…--Q8    est_mem=10070   + a hand-made --FP16 symlink
after:   quant=FP16  cache_path=…--FP16  est_mem=20141   symlink gone
```

Verified by a real load, not by reading the row: **35.00 tok/s warm**, so the
6.6 → 34.6 tok/s Q8→FP16 speedup survived the repair.

## Phase 1.0 — `src/core/cancellation.py` (done, `efc576a3`)

635 lines, 34 tests, **zero callers** — nothing could regress. Surface:
`OperatorCancelled`, `CancelScope`, `SCOPES`, `cancel_checker`,
`request_cancel`, `record_progress`, `@cooperative_cancel`.

Three decisions worth restating, because the rest of the plan rests on them:

- **`OperatorCancelled` derives from `BaseException`.** Three sites already in
  the tree would otherwise swallow it: `activation_service`'s
  `except Exception: logger.warning("Progress callback failed")` around the
  progress callback, labeling's outer handler (MIS-E2E-058, which wrote FAILED),
  and `neuronpedia_tasks`' bare handler that calls `mark_export_failed`. All
  three would turn "the operator stopped it" into "it crashed".
- **Throttle on time, not call count.** The repo proves count throttles wrong in
  both directions: circuit's `% 5` over attribution batches is up to twenty
  minutes; training's `% 25` over steps can be milliseconds.
- **A lazy `model` callable, not a class**, so `src.core` never imports
  `src.models` — the cycle `jlens_progress.py` already avoids the same way.

**Mutation control C3 SURVIVED on the first pass, and it was a code finding.**
Two independent mechanisms both guaranteed "the first call always polls" — an
explicit `n == 0` branch and a `0.0` sentinel for the last-poll timestamp. With
the redundancy, neither line was individually load-bearing, so no mutation of
either could turn the suite red. Removing the duplicate (rather than writing a
cleverer test) made the control turn **8 tests** red.

---

## Phase 1 — the guard everywhere, still no checkers (done)

**Why this had to land before any checker.** Cancellation is asynchronous by
construction: the endpoint writes the terminal status and the task notices at
its next checkpoint. In between, the task is still narrating. Every progress
writer it narrates through was assigning status unconditionally — so adding a
checker to `extract_activations` first would have meant the endpoint's
`CANCELLED` write was overwritten with `EXTRACTING` within ~10 samples and the
checker would usually **never see the flag**. The guard is the load-bearing
half; the checker is the visible half.

This phase is invisible to every previously-green test and changes no
user-facing behaviour on the success path.

### `guard_allows` — one statement of the rule

Extracted from `record_progress` so the two writers that *cannot* borrow its
session mechanics still share its semantics rather than growing a fourth
near-identical guard: `ExtractionService.update_extraction_status` is async and
holds an `AsyncSession`, and `NeuronpediaTask.update_export_progress` already
has the row open in a context manager it also emits from.

> **A terminal row accepts only an error message, or a deliberate
> terminal → terminal transition.**

| row | write | outcome |
|---|---|---|
| live | anything | allowed |
| terminal | non-terminal status | **refused entirely** — the case that loses a cancellation |
| terminal | terminal status | allowed, so `cleanup_orphaned_*` can still fail an abandoned row |
| terminal | no status, but progress | **refused** — a cancelled export must not go on announcing "packaging, 60%" |
| terminal | error_message only | allowed, so a stopping task records *where* it stopped |

The progress-only row is new this phase and is not decoration: the export writer
sets **no status at all**, so it cannot lose a cancellation by overwriting one —
it loses it by contradicting it, and that is the only shape the guard can catch.

### The writers now behind it

| writer | scope | note |
|---|---|---|
| `ExtractionDatabaseService.update_progress` | `activation_extraction` | routed through `record_progress`; fires every ~10 samples — the worst offender |
| `ExtractionService.update_extraction_status_sync` | `sae_extraction` | `guard_allows`; refusal also suppresses the WebSocket emission |
| `ExtractionService.update_extraction_status` | `sae_extraction` | async; `guard_allows` |
| `NeuronpediaTask.update_export_progress` | `neuronpedia_export` | new scope; progress-only guard |
| `TrainingTask.update_training_progress` | `training` | now shimmed onto `record_progress` |
| `jlens_progress.update_row` | `jlens_task` | now a shim; the rule originated here |

`jlens_progress.update_row` and `training_tasks` had each **independently
rediscovered the same guard**. That duplication is what the registry replaces —
`SCOPES` is now the one place the project's several terminal vocabularies are
written down.

### Two defects found while wiring it

**1. The guard could not have seen the cancel.** Every one of these writers runs
on a long-lived Celery task session. SQLAlchemy's identity map hands back the
row as it looked when the task started, so a guard reading `row.status` compares
against a stale non-terminal value and waves the write through — present,
readable, reviewed, and inert. This is MIS-E2E-057's shape. All four sync reads
now use `.populate_existing()`; the async one sets
`.execution_options(populate_existing=True)`, which is the same option by the
only route a `Select` offers. **Four mutation controls (C9–C12) exist purely to
stop a future edit dropping it**, because a test fake satisfies
`populate_existing` silently.

**2. The status string was not writable on half the columns.**
`activation_extractions.status` is `SQLEnum(ExtractionStatus)` declared
**without** `values_callable`, so SQLAlchemy persists the member *name*. Scopes
speak lowercase strings, and a bare `"cancelled"` assigned there is not a key
SQLAlchemy can look up — it fails at flush, inside a write nobody is watching,
**on the cancellation path itself**. `_coerce_status` translates at the boundary;
unknown values pass through so the column still rejects them loudly.

### One behaviour deliberately made stricter

The training tracker previously refused only the *status* write on a terminal
row and went on writing metrics. It now refuses the whole write. The loop checks
for PAUSED every `status_check_interval` (≤25) steps, so a paused run was
accruing steps and losses for up to 25 steps after the operator paused it — and
the row it resumed *from* then disagreed with the checkpoint it resumed *at*.

### What the full suite found — 18 regressions, two root causes

The unit tests written alongside the change were green while **eighteen existing
tests were red**. Both causes are worth recording because neither is a bug in
the guard.

**1. Every existing fake was a `MagicMock` chain that did not model the fresh
read** (`test_jlens_cancel`, `test_task_heartbeat`, `test_training_tasks`). With
`.populate_existing()` in the chain, `db.query(...).filter(...).populate_existing()`
returns a fresh `MagicMock` and `.first()` a different one — so the writer
operated on a mock instead of the row and every assertion about the row failed.
The fakes were taught the chain; the production behaviour did not change. This
is the ordinary cost of the identity-map fix and is the reason C9–C12 exist.

**2. A source-scrape guard caught a real observability regression.**
`test_extraction_aborts_without_a_row.py` asserted the string
`"not found for progress update"` was present in `extraction_db_service.py`.
Routing through `record_progress` moved the warning out — and `record_progress`
did not log at all for a missing row, so a phantom extraction (the 2026-08-24
incident: **3 h 24 m at 100% GPU against a row that never existed**, ~300
ignored warnings) would now run in total silence. Fixed in the code, not the
test: `record_progress` warns when the row is gone.

The test itself was then rewritten from a scrape to a **driven** assertion — it
calls the writer against a missing row and requires the warning to reach the
log. A scrape cannot distinguish "moved" from "deleted", which is this repo's
recorded failure mode (*a source-scrape guard fails OPEN, twice observed*). It
happened to fire correctly this time; that is luck, not design.

Baseline confirmed by stashing the change and running the same subset: the only
remaining failures, 8 in `tests/integration/test_extraction_templates_api.py`,
fail identically on a clean tree and are unrelated.

### Mutation controls — 13 run, 13 verified biting

| id | mutation | result |
|---|---|---|
| P1-C1 | activation-extraction writer: guard removed (literally the pre-Phase-1 code) | KILLED (4 red) |
| P1-C2 | sae-extraction sync writer: guard disabled | KILLED (4 red) |
| P1-C3 | sae-extraction async writer: guard disabled | KILLED (1 red) |
| P1-C4 | export writer: guard disabled | KILLED (3 red) |
| P1-C5 | training tracker: guard removed (pre-Phase-1 code) | KILLED (4 red) |
| P1-C6 | `guard_allows`: a terminal row accepts a progress move again | KILLED (3 red) |
| P1-C7 | `guard_allows`: the terminal set is never consulted | KILLED (20 red) |
| P1-C8 | `_coerce_status`: the column's enum is not consulted | **survived, then KILLED** — see below |
| P1-C9 | `record_progress`: the identity map is not defeated | KILLED (1 red) |
| P1-C10 | sae-extraction sync writer: identity map not defeated | KILLED (1 red) |
| P1-C11 | sae-extraction async writer: identity map not defeated | KILLED (1 red) |
| P1-C12 | export writer: identity map not defeated | KILLED (1 red) |
| P1-C13 | `record_progress`: a missing row is silent again | KILLED (1 red) |

C9's anchor moved when the missing-row warning was inserted, and the harness
reported **DID NOT LAND** rather than SURVIVED — which is the whole point of
checking that a mutation actually applied before drawing a conclusion from it.

**C8 survived first, and it was a code finding — the same shape as Phase 0's
C3.** `_coerce_status` matched the incoming string against both `member.value`
and `member.name`. Every status enum in this project spells its value as the
lowercase of its name, so the two branches always agree: disabling one passed
silently on the other, and neither was individually load-bearing. Removing the
name branch (values are the vocabulary the scopes are written in; matching by
name was guessing) made the control bite. **Twice now the surviving mutation has
been redundancy, not a missing test.**

### Tests

`backend/tests/unit/test_progress_guard.py`, 27 tests. They drive the **real
writers**, not `record_progress` — `test_cancellation_core.py` already covers
the rule, and what would otherwise be unproven is that each writer is actually
standing behind it.

### Known limits of this phase, recorded honestly

- **No task can be cancelled yet.** This phase only stops a cancellation being
  *lost*. Nothing polls the flag until Phase 2.
- The scope registry covers six lifecycles. The nine remaining (datasets,
  models, faithfulness, calibration, record, enhanced labeling, feature
  grouping, dataset tokenization, circuit discovery) arrive in Phases 3–6, and
  the registry-completeness test (Shape D) cannot be switched on until Phase 6.
- `extraction_jobs.progress` is a **fraction** (0.0–1.0) while every other
  progress column is 0–100. `record_progress` clamps to [0, 100], which is a
  no-op on a fraction, so the clamp is inert for that one scope. Not fixed here:
  changing the column's units is a migration plus a frontend change, and it is
  unrelated to cancellation.

### Process note

The full suite was started **concurrently with the mutation script**, which is
the exact hazard CLAUDE.md records ("never run a mutating agent concurrently
with a reading one" — a reviewer once read an in-flight mutation out of the tree
and reported it as a committed defect). That run was killed and re-run against a
clean tree rather than trusted.

### Carried into Phase 2

`@cooperative_cancel(kind, target)`'s `target` parameter is **dead** — the
wrapper stores it on `__cooperative_cancel_target__` and nothing, including the
Shape-D registry test it was written for, ever reads it. It is removed before
the first caller passes it, on the same reasoning that killed C3 and C8: an
argument nothing depends on cannot be mutation-tested, and entrenching it across
a dozen task decorators makes it permanent.


---

## Phase 2 — `extract_activations`, the proof on a real GPU task

Chosen first for four reasons, all of which held up: the checkpoint already
existed as `on_extraction_progress`; it was **the worst lie in the system**; its
service swallows callback `Exception`s, so it tests the `BaseException` decision
empirically rather than by argument; and it runs for hours, so the hardware
acceptance is unambiguous.

### What the endpoint used to do

`POST /models/{id}/extractions/{id}/cancel` called
`revoke(terminate=True, signal='SIGTERM')`, wrote CANCELLED, emitted a
"cancelled" WebSocket event and returned **"Extraction cancelled
successfully"**. The extraction worker is `--pool=solo -c 1`, `terminate`
signals a pool child, and solo has none — and a busy solo worker is not reading
the control queue, so the revoke was never delivered. Every layer of the
response was true except the one the operator cared about: the GPU ran on for
hours. It now calls `request_cancel`, and the response and WebSocket message
both carry `outcome.detail` — *"A running job stops at its next checkpoint,
which is bounded by one indivisible unit of work"* — instead of a success
claim.

### Three checkpoints, not one

| where | why |
|---|---|
| the progress callback (~every 10 samples) | the finest boundary at which a partially written batch can be abandoned |
| `poll_now()` immediately before `extract_activations` | everything past it is one indivisible call whose first checkpoint is not reached until the model is on the GPU — minutes for a large model |
| `@cooperative_cancel` at the task boundary | turns the raise into the canonical `{"status": "cancelled", …}` **return**, which is what ACKS the `acks_late` message; raising would redeliver it against a 12-hour visibility timeout |

### A decision that had to be made, not defaulted

An operator who cancels during the **SAVING** phase gets no further checkpoints:
the task runs to the end and reaches `mark_completed`. `record_progress` would
allow that write — terminal → terminal is deliberately permitted so the janitors
can fail an abandoned row — and the row would read COMPLETED, telling the
operator their cancel did nothing.

Both facts are true, so both are recorded: **the row stays CANCELLED**, and the
artifact that does exist keeps its `statistics` and `saved_files` and is named
in `error_message`, so it is not silently orphaned. Reporting "failed" would
have been a lie about a complete artifact; reporting "completed" a lie about the
operator's request.

### The reachability rule bit, and it was worth it

The first Shape-A test drove a hand-written **reconstruction** of the callback,
because the real one was a closure inside the task and unreachable. Mutation
control P2-C1 — delete the `raise_if_cancelled` from production — then turned
exactly **one** test red, and it was a source scrape. By the rule (*a capability
is not shipped until a test FAILS when its wiring is removed*), the checkpoint
was not shipped: the only thing standing behind it was a guard of the kind this
repo has twice watched fail open.

The callback was hoisted to `build_extraction_progress_callback` at module
level so the test drives the thing production drives. P2-C1 now turns **three**
behavioural tests red, and a new P2-C9 (the task builds its own callback,
bypassing the factory) covers the wiring.

### Mutation controls — 9 run, 9 verified biting

| id | mutation | result |
|---|---|---|
| P2-C1 | the callback no longer polls — the only checkpoint removed | KILLED (3 red) |
| P2-C2 | the task loses `@cooperative_cancel` | KILLED |
| P2-C3 | nothing polls before the model load | KILLED |
| P2-C4 | the endpoint goes back to `revoke(terminate=True)` | KILLED |
| P2-C5 | `mark_completed` overwrites a cancellation again | KILLED |
| P2-C6 | `is_cancelled` never recognises a cancelled row | KILLED |
| P2-C7 | `OperatorCancelled` becomes an ordinary `Exception` | KILLED (5 red) |
| P2-C8 | the checker's first call is throttled away | KILLED (9 red) |
| P2-C9 | the task builds its own callback, bypassing the checkpoint | KILLED |

### Two of my own test bugs, recorded because both are the standard traps

* The Shape-A test asserted against a list it never populated — the value was
  returned from a function that raises, so it was always `[]`. It reported
  "0 units executed" as a *failure of the code* when it was a failure of the
  test.
* The "endpoint no longer pretends terminate works" assertion matched the
  **comment explaining why terminate is wrong**. It failed against correct code,
  and would equally have passed against a re-added call under a comment that did
  not mention it. Comments are now stripped before the check.

### Two prerequisites cleared first

* `@cooperative_cancel(kind, target)`'s `target` was dead — stored on the
  wrapper and never read, including by the Shape-D test it was written for.
  Removed before the first caller entrenched it.
* `_task_source()` in `test_extraction_aborts_without_a_row.py` used a single
  `__wrapped__` hop. With the decorator in place that lands on the
  **cancellation wrapper**, so six assertions about the task body would have
  been read against `core/cancellation.py` — passing or failing for unrelated
  reasons. Switched to `inspect.unwrap`, verified against the live Celery
  `PromiseProxy`.

### Outstanding

**Hardware acceptance on k8s is not done.** The unit suite cannot show the
properties that matter: work stopping at the next 10-sample boundary, **VRAM
actually freed**, the row still CANCELLED after the last in-flight progress
write, and the worker picking up the next queued job. Both `finally:` blocks
that release the GPU (`model_tasks.py` and `activation_service.py`) do run on a
propagating `BaseException`, which is necessary but not sufficient — the repo's
record is explicit that GPU bugs are found only on GPUs.
