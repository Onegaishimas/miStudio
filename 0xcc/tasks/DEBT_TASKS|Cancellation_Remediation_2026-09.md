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
| 2 — `extract_activations`, the GPU proof | ✅ **including hardware acceptance** | this record |
| 3 — the remaining compute tasks | ✅ except dataset tokenize | `fea78286` + this record |
| 4 — the downloads + tokenization | ✅ | this record |
| 5 — shim the five healthy conventions | ✅ | this record |
| 6 — the missing routes | ✅ | this record |
| 7 — cleanup | ✅ | this record |
| R1 / R2 / R3 review rounds | ✅ | `3da8f578`, `0f477da7`, + this |

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
| `backend/src/api/v1/endpoints/saes.py` | `cancel_sae_extraction` — wrote FAILED for a cancel; now CANCELLED |
| `backend/src/services/neuronpedia_export_service.py` | `_cancel_point`, the export's only real checkpoint; also the `utc_now` NameError |
| `backend/tests/unit/test_faithfulness_cancellation.py` | the resurrected faithfulness path |
| `backend/tests/unit/test_sae_extraction_cancellation.py` | SAE extraction Shapes A and C |
| `backend/tests/unit/test_export_cancellation.py` | export checkpoint + the failure writer |
| `backend/tests/unit/test_no_undefined_names.py` | the ratcheted F821 gate |
| `backend/alembic/versions/f3c8a92b1e07_add_cancel_requested_at.py` | the timestamp column for the three native-enum lifecycles |
| `backend/src/workers/tqdm_websocket_bridge.py` | the only owner-process checkpoint inside a forked `Dataset.map` |
| `backend/src/workers/dataset_tasks.py` | tokenize + download tasks, `remove_partial_download` |
| `backend/tests/unit/test_download_tokenize_cancellation.py` | Phase 4 shapes |
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


---

## Phase 3 — the remaining compute tasks

Three of four done. **Dataset tokenization is deferred to Phase 4**, because it
shares the `Dataset.map(num_proc=N)` fork-pool problem with the downloads and
its only owner-process checkpoint is the same tqdm bridge.

### The one-liner, and a correction to the plan

`circuit_validation_tasks.py` passed `cancel_check=None`, so the poll, the
`_FaithfulnessCancelled` raise and the handler that turns it into a cancelled
status were all unreachable. Passing a real checker was the whole fix.

**The plan said that path was "written, tested and dead". It was written and
NEVER TESTED** — nothing in the suite had ever constructed
`_FaithfulnessCancelled`, so deleting it outright would have gone unnoticed. It
has eight tests now, one driving the real `_behavior` loop to prove a cancelled
run costs **zero forward passes** rather than one more prompt.

Not yet operator-reachable: no cancel ROUTE exists for faithfulness. Phase 6
adds it. Until then the capability is real but only a janitor or a direct write
can trigger it, and by the reachability rule it is not shipped.

### Three latent NameErrors, found by chasing the class

`execute_export` called `utc_now()` as its first statement while the file's only
`from ..core.clock import utc_now` sat **inside a README template's f-string** —
a python code block in text the export writes for its users. **Every export died
with NameError** before reaching its own `try:`, so even the FAILED write never
ran and the row stranded in COMPUTING. Verified by AST: zero clock imports in
that module.

Running the linter over the whole tree found two more:

* `steering.py` called `signal.SIGKILL` in the orphan-worker reaper without
  importing `signal`. That reaper is the **sanctioned** replacement for
  `pkill -f steering@`, which is forbidden here — so the supported path raised
  NameError and the forbidden one was the only thing that worked.
* `jlens_fit_tasks.py` had `except ArtifactQualityRegression` with the name
  unimported in that function. Evaluating an except clause resolves the name,
  so **any** exception from the publish call became a NameError masking it.

Three further hits are false positives, verified by reading and listed with
reasons behind a ratchet that fails if one stops being reported.

**The gate had the bug it hunts.** Its first version passed an
`--output-format` ruff 0.1.15 rejects; the command errored, the parse found
nothing, and it reported a clean tree — **a lint guard failing OPEN**. It now
refuses any exit code that is not "clean" or "findings", with a negative control.

### `extract_features_from_sae`

The service had **no injection point at all** — no `cancel_check`, no callback.
It gained one, plus three checkpoints: before the multi-minute base-model load;
at the top of the phase-1 sampling batch loop (which commits nothing, so
abandoning leaves the database untouched); and per-feature in the phase-2 write
loop (which does commit, which is why the row must end CANCELLED — a partial
feature set claiming success would read as a finished extraction).

**Three guard bypasses fixed.** The task wrote `status = FAILED` straight onto
the ORM row in its failure handler and in two validation branches, so an
exception arriving after an operator cancelled relabelled the row FAILED and
lost the cancellation at the last possible moment.

**The endpoint wrote the wrong word.** `saes.py` set `FAILED` + "Cancelled by
user" while the sibling lifecycle wrote `CANCELLED` for the identical action;
`ExtractionStatus.CANCELLED` existed the whole time. The UI showed a deliberate
stop as a crash, and a checker looking for "cancelled" could never see a cancel
spelled "failed". That is what Shape C is for.

### The export path — two assumptions in the plan were wrong

* **The Celery task is dead code.** Nothing dispatches
  `neuronpedia.execute_export`; the live path is a FastAPI `BackgroundTasks`
  call running `execute_export` **inside the API process**. A checkpoint added
  to the Celery task would have fired on nothing. It went into the service,
  which both paths share.
* **A checkpoint reading `job.status` would have been inert.** `execute_export`
  loads the row once with `db.get(...)`, which returns the identity-mapped
  instance without emitting SQL, and both session factories are
  `expire_on_commit=False` — so the in-memory status is frozen at whatever it
  was when the export began. `_cancel_point` does a fresh `populate_existing`
  read.

Also closed here: a **deleted** row now stops the run (`DELETE /export/{id}`
removes it outright, and the default `missing_row="continue"` would have run
the export to completion against nothing); `mark_export_failed` no longer
relabels a cancellation as a crash; and re-cancelling an already-cancelled job
no longer reports a fresh cancellation.

### Mutation controls — 13 run, 13 verified biting

P3-C1…C13 cover: the one-liner reverted, the faithfulness scope pointed at the
wrong column, the service loop's poll deleted, the SAE checker removed, the
sampling poll deleted, the endpoint writing FAILED again, the task's failure
path bypassing the guard, the export checkpoint trusting the identity map, a
deleted row no longer stopping, `_update_stage` no longer a checkpoint, the
completion write unguarded, `mark_export_failed` relabelling again, and the
F821 gate narrowed.

**P3-C11 survived first, and it was a fail-open assertion of exactly the kind
this arc keeps finding.** The test located the checkpoint with `str.rfind`,
which returns **-1** when it is absent — and `-1 < complete` is true, so the
assertion passed precisely in the case it existed to catch. Fixed to require
presence before ordering; the control then bit.

That is the third fail-open guard in this arc (the source-scrape that could not
tell "moved" from "deleted", the lint gate that reported a clean tree from a
failed command, and this). The pattern is worth naming: **an assertion built on
a sentinel return value passes when the thing is missing.**


---

## Phase 4 — the fork-pool lifecycles: downloads and tokenization

Tokenization was moved here from Phase 3: it shares `Dataset.map(num_proc=N)`
with the downloads and needs the same mechanism, so building it once is the
point.

### Where a check can even run

`Dataset.map(num_proc=N)` forks a worker pool and the mapper executes in the
children. A child cannot coordinate a stop with its siblings, and one that dies
becomes *"One of the subprocesses has abruptly died"* — a cancellation
indistinguishable from a crash. But `datasets` funnels every batch's progress
back through a manager queue and ticks the progress bar in the **parent**. That
tick is the only owner-process checkpoint that exists. For a HuggingFace
download, tqdm is the only in-process callback of any kind.

So `TqdmWebSocketCallback.update()` polls, before the parent bookkeeping and
before any emission. The raise crosses that module's own `except Exception`
handlers and `datasets`' internal ones — which is the BaseException decision
earning its keep for the third time.

### A timestamp instead of an enum member (migration `f3c8a92b1e07`)

`datasets`, `models` and `dataset_tokenizations` are native Postgres enums with
no CANCELLED member, and `ALTER TYPE … ADD VALUE` is non-transactional. They
carry a nullable `cancel_requested_at` instead. It is also the better model: it
separates *the operator asked* from *the job stopped*.

Their status still ends at `error` — the enum offers nothing better, and
extending it is the thing being avoided. **That is not a regression; it is what
they already did.** What changes is that a row with `status = error` AND a
non-null request is now distinguishable from a crash. Before this column they
were the same row and no reader could tell them apart.

Their `terminal_values` are their own vocabularies (`ready`/`error`), not the
default `{completed, failed, cancelled}` — with the default the guard would
never consider any of these rows terminal and a straggling write could revive a
finished download.

### Three hazards closed

* **The `BaseException` catch-all.** `tokenize_dataset_task` catches it
  deliberately, for `SystemExit` from its signal handler — so it also caught
  `OperatorCancelled` and would have recorded a cancel as
  `Tokenization failed: …`, or, if the data happened to be saved already, as a
  **success**. An explicit handler now precedes it. The signal handler stays:
  it is not dead code, it is what lets `Dataset.map`'s pool children die to
  their own default handler instead of raising `SystemExit` into the owner's
  inherited one.
* **`rmtree` on a live directory.** The cancel path deleted the directory the
  download was actively writing into (`revoke(terminate=)` is inert on a solo
  pool, so the task was very much alive), and the task then recreated parts of
  it. Deletion moved into the cancelled task's own handler — writer and deleter
  are now the same process — with the endpoint cleaning up only a job that had
  **not** started, the one case where nobody else ever will.
* **SIGKILL in the tokenization endpoint.** Explicitly not a fallback here:
  killing a solo worker mid-task crashed the pool and stranded an `acks_late`
  message for the full 12-hour visibility timeout.

### A latent bug the tests surfaced

`self.desc` is **unset** when tqdm is constructed with `disable=True` — its
`__init__` returns early — so the bridge's emission path raised `AttributeError`,
which the surrounding `except Exception` logged as a dropped progress tick.
HuggingFace disables bars when its env flag is set, and a Celery worker has no
tty. Same failure shape as the seven-month `ImportError` already recorded in
that file: the row silently freezes while the job runs to completion.

### Mutation controls — 12 run, 12 verified biting (four needed a second pass)

**P4-C2** survived because the ordering assertion matched `raise_if_cancelled`
inside the **comment explaining the ordering**, which stays above
`super().update()` however far the actual call moves. Comments are stripped now.
That is the second time this exact trap has appeared in this arc.

**P4-C3 and P4-C4** survived because nothing asserted that either task *asks*
for a cancellable bar. The bridge was tested in isolation; a `CancellableTqdm`
nobody passes a scope to is an ordinary progress bar. Wiring assertions added.

**P4-C11** survived because the cleanup was inline in the cancellation handler
and therefore unreachable from a test — the only guard was a source scrape,
which cannot see `if False:`. Hoisted to `remove_partial_download`, now driven
against a real temp directory, including the refusal path for a `raw_path`
outside the deletable roots.

**P4-C5 was a bad mutation, not a test gap** — it renamed the bound variable,
so the clause still caught `OperatorCancelled` and behaviour was unchanged.
Replaced with one that actually disables the handler.

### Recorded debt

Moving deletion out of the endpoint leaves one case uncovered: a download whose
worker died without running its handler keeps its partial directory, and the
endpoint will no longer remove it because the row says the job had started.
This is a deliberate trade — deleting a directory that is being written is worse
than briefly leaking one — but it is a real gap and belongs to a janitor, not to
the cancel path.


---

## Phase 5 — one implementation

Fifteen registered scopes; the five healthy conventions now delegate.

| convention | shim |
|---|---|
| `jlens_progress.cancel_checker` / `request_cancel` | delegate to core |
| `jlens_progress.TaskCancelled` | **alias** of `OperatorCancelled` |
| `circuit_capture_tasks._cancel_checker` | keeps its signature; a `_SCOPE_FOR` map resolves (model, column) to a registered scope |
| `LabelingService._raise_if_cancelled` / `_LabelingCancelled` | delegate; alias kept |

The aliases are aliases, not subclasses, so every existing `except`-by-name and
`pytest.raises` keeps working — and both silently gain `BaseException`, which
generalises the MIS-E2E-058 fix to the last two paths whose outer
`except Exception` could still turn an operator's stop into a crash report.

`_cancel_checker` also loses its `% 5` count throttle. Five is a number chosen
against one loop's unit cost and true of no other: over attribution batches on a
large model it is up to twenty minutes of latency.

### Two things a "low risk" refactor got wrong

**I introduced a throttle that broke a tested contract.** Caching a
`CancelCheck` per labeling job so the 2-second budget would bite meant a fast
batch loop ran its whole length inside one window and never re-polled;
`test_a_batch_loop_stops_once_the_job_is_cancelled` went red. The old code
polled every batch, and a per-batch caller is already at the right granularity.
A fresh checker per call always polls, because the first call always does.

**An existing test was pinning the defect.** `test_missing_row_does_not_raise`
asserted a vanished row must NOT stop the job — but `delete_labeling_job`
revokes (inert) and then DELETES THE ROW, so deletion *is* how that path stops a
job, and returning quietly meant labelling every remaining feature against a row
that no longer existed. The premise was checked before inverting it: the
endpoint commits the row before `.delay()`, so a missing row can only mean
deleted, never not-yet-created.

### The fifth source-scrape guard of the arc

`test_the_cancel_check_bypasses_the_identity_map` asserted `.populate_existing()`
appeared in the *text* of `_raise_if_cancelled`. The shim moved the call one
level down with behaviour unchanged, and the test failed. Now driven against a
fake that models the identity map.

That is five in this arc, all the same defect shape — **an assertion about
source text or a sentinel return value, standing in for an assertion about
behaviour**:

| guard | failure |
|---|---|
| the progress warning string | text moved to core; hid a real observability loss |
| `terminate=True` absent | matched the comment explaining why it is wrong |
| `raise_if_cancelled` ordering | matched the comment explaining the ordering |
| `rfind(...) < complete` | `-1` satisfies `<`, so it passed when absent |
| `.populate_existing()` in one method | failed against unchanged behaviour |

### Mutation controls — 9 run, 9 verified biting

Two needed a second attempt for reasons worth recording. **P5-C2 was inert**:
removing `min_interval_s=0.0` changed nothing, because a fresh checker always
polls on its first call — so that argument was untestable redundancy and was
deleted rather than pinned (the same call the C3 and C8 survivors got).
**P5-C3 was a syntax error** — a duplicate keyword argument, which kills the
whole suite and proves nothing about behaviour; rewritten to flip the existing
value.

### Pre-existing, recorded not fixed

`circuit_capture_tasks` and `circuit_validation_tasks` import each other;
importing the former first raises ImportError. Reproduces on a clean tree —
Celery's autodiscovery happens to use the working order.


---

## Phase 6 — the missing routes, and Shape D turns on

Five lifecycles were **startable and not stoppable**: faithfulness,
calibration, the steering recorder, enhanced labeling and feature grouping each
had a launch route, a status column, and no way for an operator to reach a
running job short of restarting the pod — which on a `--pool=solo` worker also
strands the in-flight `acks_late` message for the full 12-hour visibility
timeout. Nineteen scopes are now registered.

`EnhancedLabelingStatus` and `GroupingRunStatus` gained a `CANCELLED` member.
Both columns are plain `String`, not native Postgres enums, so this is an
addition with no migration — and without it an operator's stop had to be
written as FAILED, the conflation this whole arc exists to remove.

### Shape D found four real gaps the phase would otherwise have shipped

The registry-completeness harness asserts four independent properties per
scope, because each is invisible from the others: every declared column exists
on its table; the cancelled values are a subset of the terminal ones; an
operator route reaches it; and something actually polls it.

| gap | what it meant |
|---|---|
| `dataset_download` had no `request_cancel` | the worker wrote `status = ERROR` directly, so the tqdm poll added in Phase 4 had **no flag to read** |
| `neuronpedia_export` had no `request_cancel` | the service wrote CANCELLED inline; correct only by coincidence of spelling |
| `model_download` registered, polled by nothing | Phase 4 wired the dataset half and left the model half a scope with no callers |
| `labeling` wrote its own status | worked only because both sides happened to spell it the same way — which is exactly what `saes.py` did **not** |

### `app.routes` is not the assembled surface

The route half of Shape D first read `app.routes`, which in this FastAPI
version holds lazy `_IncludedRouter` objects that carry no `.path` until the app
is built — ten framework paths and none of the real surface. It failed closed
here, but the mechanism was wrong: `app.openapi()` forces the resolution and is
what the harness uses.

### The model download's honest limit

`snapshot_download` offers no abort hook, and the progress monitor runs on a
separate thread where a raise would die. So the monitor **observes** the request
and stops narrating, and the task acts on it at the boundaries it owns: before
the download begins, and after it completes but before quantization and
profiling. Mid-transfer interruption is **not implemented** — recorded here
rather than implied by the presence of a scope.

### Mutation controls — 14 run, 14 verified biting

Three survived first, all for the same reason: the "something writes the
request" check was a **substring match**, so the scope name appearing anywhere
— its own registration comment, a `cancel_scope=` on a tqdm bar, a
`guard_allows` call — satisfied it while the actual `request_cancel(...)` had
been deleted. It now reads the AST for real calls in three shapes: a direct
call, `run_in_threadpool(request_cancel, "scope", …)`, and the one shared
dispatch helper the three circuit routes use.

That is the sixth substring-or-sentinel guard in this arc to fail, and the
lesson is now unambiguous: **if an assertion can be satisfied by text that is
not the thing, it will be.**


---

## Phase 7 — cleanup

* **`every=` is gone.** It existed only so the circuit and J-lens shims could
  keep their count semantics through the migration. Nothing passes it, and the
  first-call rule is now the sole mechanism guaranteeing that a job cancelled
  before it starts is noticed — which is what makes it mutable.
* **`test_dataset_cancel_scope.py`'s revoke test is rewritten.** It asserted
  the worker's source still contained the word `revoke`, which pinned the
  *illusion*: `revoke(terminate=)` is inert on a solo pool, so that test would
  have stayed green through the entire period when cancelling a dataset
  download did nothing at all. It now asserts, via AST, that the worker writes
  the flag the running download polls — a substring check is satisfied by the
  tqdm bar's `cancel_scope=` in the same file.
* **One home for the solo-pool explanation.** It was duplicated with drift
  across seven files: every copy correct, none authoritative, each an
  invitation to rediscover the same finding a ninth time.
* **The PID-kill escape hatch is documented as an OPERATOR PROCEDURE**, in that
  same docstring, and explicitly not as a code path — including the
  `/proc/<pid>/cmdline` rule (a `pgrep -f` pattern that appears in its own
  command line matches the caller's shell, and a wait loop written that way
  never exits — twice-observed here), SIGTERM before SIGKILL, and the 12-hour
  `acks_late` cost of getting it wrong.

---

## The arc's durable finding

**Six guards in this remediation failed because they asserted about text or a
sentinel rather than about behaviour.** Each either failed against correct code
or passed against broken code:

| guard | how it failed |
|---|---|
| the progress-warning string | text moved to core; hid a real observability loss |
| `terminate=True` absent | matched the comment explaining why it is wrong |
| `raise_if_cancelled` ordering | matched the comment explaining the ordering |
| `rfind(...) < complete` | `-1` satisfies `<`, so it passed when absent |
| `.populate_existing()` in one method | failed against unchanged behaviour |
| the scope name in worker source | satisfied by a `cancel_scope=` while the real call was deleted |

Plus one of the same family in the tooling: the F821 lint gate reported a clean
tree from a command that had errored.

**If an assertion can be satisfied by text that is not the thing, it will be.**
Every one of these is now driven, or reads the AST, or reads the live registry.


---

## The three review rounds

Records: `.claude/context/sessions/review_debt_R{1,2,3}_2026-09-05.md`.

| round | findings | headline |
|---|---|---|
| R1 | 17 test + 9 correctness | Shape D's own guarantee was a **tautology**; the Phase-5 alias had **broken the J-lens fit cancel** |
| R2 | 7 | two of R1's fixes were **worse than the bugs**; one destroyed the run's result on the SUCCESS path |
| R3 | 9 | two of R2's fixes were worse again; one was **inert under the default config** |

**A fix was worse than the bug it replaced in all three rounds.** That is four
consecutive arcs in this repo with the same result, and it is the entire
justification for the third round existing.

### The three that mattered most

**R1c-01 — the alias broke the fit cancel.** Phase 5 rebound `TaskCancelled` to
`OperatorCancelled`, whose signature is `(scope, target_id, …)`.
`jlens_fit_tasks` raised it with one bare message, so a cancel became a
`TypeError` — not caught by `except TaskCancelled`, therefore caught by
`owns_its_failure`'s `except BaseException`, which called `fail_row`. **The
operator's cancellation was recorded as a crash**, on the one path the module
docstring cites as hardware-verified.

**R2-01 — `db.refresh` destroyed the result on every successful run.**
`Session.refresh()` expires the instance BEFORE autoflush, so the calibration
band, the intensity clamp, the version bump and the faithfulness scores were
all erased and only the status was committed. Clamping the served dial to the
measured band — the whole of Feature 20 — silently stopped happening. **And the
same commit added the fake that hid it**: a no-op `refresh` on `_FakeSyncDB`,
written to make the test pass.

**R3-03 — retry after cancel was permanently broken.** Nothing ever cleared
`cancel_requested_at`, which is what the tqdm poll reads. Cancel a dataset
download once and it could never be downloaded again. Neither reviewer reached
this; it fell out of chasing R3-02.

### The arc's durable finding

**Seven guards failed because they asserted about text or a sentinel rather
than behaviour** — a scrape matching the comment that explained it; `rfind`
returning `-1` and satisfying `<`; a lint gate reporting clean from a command
that errored; a glob matching the asserting file; `"cancel_check="` satisfied by
`cancel_check=None`; `index()` finding an assignment; and — inside the
regression test written for the sixth — an assertion matching the comment
explaining the fix.

*If an assertion can be satisfied by text that is not the thing, it will be.*

### Process failures, recorded

Three times this session I ran the mutation harness concurrently with a reading
agent — the exact hazard CLAUDE.md names. The R3 reviewer read three transient
mutations out of the tree and had to pin every finding to `git show HEAD:`; one
of my own edits landed on top of an injected mutation and had to be reverted.
Also: a suite run once reported `exit 0` having never executed, because
`--timeout` is not installed and pytest rejected the argument. **Green is a
count, not an exit code.**


---

## Phase 2 hardware acceptance — PASS (2026-09-05, RTX 3090)

LFM2.5-1.2B-Instruct at Q4, Bloomberg_Financial_News, layer 8, 8000 samples,
cancelled mid-run through `POST /models/{id}/extractions/{id}/cancel`.

| criterion | result |
|---|---|
| stops at the next boundary | **PASS** — 3356 → 3388, **+32 samples** |
| VRAM freed | **PASS** — 1.03 → 2.62 peak → 1.03 GB |
| row stays CANCELLED | **PASS** — 7 observations over 43 s, samples frozen at 3388 |
| worker takes the next job | **PASS** — the queued job ran and completed |

Criterion 3 is the one no unit test can reach: it is a race between the
endpoint's write and the worker's next `update_progress`, on a real solo-pool
worker. The Phase 1 guard held — every in-flight progress write after the
cancel was refused, and the row never left `cancelled`.

The endpoint's response is now honest too: *"A running job stops at its next
checkpoint, which is bounded by one indivisible unit of work"*, in place of
the "Extraction cancelled successfully" it used to return while the GPU ran on.

### What the hardware round found that four static rounds did not

**The returned `extraction_id` was a guess.** The endpoint generated
`ext_{model}_{now}` and never passed it to the task, which generated its own
from a second `datetime.now()`. They agree only inside one second — observed
straddling one, `..._185140` returned against a row created as `..._185141`.
Since **cancel keys on that id**, an operator cancelling with the id they were
handed got a 404 and the extraction ran on. Fixed by passing it; the task
already accepted one.

Recorded, not fixed: the id is second-granular, so two extractions started for
one model in the same second still collide. A test pins the note.

### Three harness defects, each of which produced a verdict about nothing

The first three runs reported FAIL against a working feature:

1. **The pod name was captured once.** ArgoCD replaced the pod mid-run, every
   later `kubectl exec` addressed a dead name, and empty responses read as
   `None` statuses and a `-1` GPU. *A test that cannot tell "the thing is
   broken" from "I lost my connection to it" is not a test.*
2. **Criterion 1 measured from a 5-second-old poll** while the job advanced
   ~450 samples per tick, attributing a tick of ordinary progress to the
   cancellation (+132 → +32 once read immediately before the POST).
3. **Criterion 4 looked for the queued job's row too early.** Rows are created
   by the *task* when the worker starts it, not by the endpoint — so a queued
   job has no row at all until the one ahead finishes.

### And one self-inflicted outage

I ran a manual `kubectl rollout restart` while ArgoCD's Image Updater was about
to roll the same deployment for the new digest. It rolled twice; the second
roll killed a running extraction and orphaned its row at 4404/8000. CLAUDE.md
says deploys here are GitOps and `k8s_deploy` is break-glass only — this is
what that rule is for.
