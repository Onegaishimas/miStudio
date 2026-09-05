# Review round 1 — cancellation remediation

Scope: `git diff efc576a3~1..HEAD`, 54 files. Two reviewers: production correctness,
and test quality. Findings numbered `R1-NN`.

**The round's verdict: the implementation is in better shape than the tests that
guard it.** 76 mutation controls were run during construction and all bit — but
they bit against the tests as they were, and this round found that several of
those tests are satisfied by something other than the behaviour. A control that
"bites" a weak test proves the test is coupled to the code, not that the code is
correct.

---

## Severity 1 — guarantees that are void

### R1-01 Shape D's "a Shape-A test exists" is a tautology
`tests/unit/test_cancel_registry_completeness.py:281-293`.
`_shape_a_test_source()` globs `tests/unit/test_*cancel*.py`, **which matches
`test_cancel_registry_completeness.py` itself** — and a sibling test requires
that file to contain every scope name as a quoted dict key (`OPERATOR_ROUTES`,
`NO_ROUTE_BY_DESIGN`). So the assertion is satisfied by the file doing the
asserting.

Verified: deleting every Shape-A test file in the repo leaves this green for all
19 scopes. **The headline claim of the whole harness — that nobody ships a
cancellable task without proving the work stops — is unenforced.**

### R1-02 Shape D's "something actually polls the scope" is a substring match
`test_cancel_registry_completeness.py:226-235` — the exact pattern the same file
condemns twenty lines earlier. For 6 of 17 non-skipped scopes the only match is
unrelated text: `source_type = "training"`, `"status": "labeling"` dict
literals, and the `_SCOPE_FOR` name-mapping table for the three circuit scopes.
Deleting every `_cancel_checker(...)` call site leaves it green.

### R1-03 `assert "cancel_check=" in src` is satisfied by `cancel_check=None`
`test_sae_extraction_cancellation.py:184`. This is **the exact Phase-3
faithfulness defect** — `cancel_check=None`, the thing the whole arc exists to
fix — and the test cannot see it. Highest-value single surviving mutation:
every SAE-extraction checkpoint becomes inert, suite green.

### R1-04 A literal tautology
`test_faithfulness_cancellation.py:129`:
`assert CircuitFaithfulnessService._behavior(**kwargs) is not None or True`.
`X or True` is `True`. Self-inflicted; the only real content is "does not raise".

---

## Severity 2 — tests that cannot detect their own defect

### R1-05 `test_phase6_cancellation.py` does not do what its docstring claims
Line 9 says *"drive the REAL loop, flip the flag, assert the work stops."* No
test in the file drives a loop. Every "it stops" test is `inspect.getsource`
plus a substring or index comparison. Concretely survivable mutations:
`cancel_check_for=None` with the scope name in a log line; keeping a message
string in a comment while deleting the `raise_if_cancelled` beside it;
`cancel_checker("model_download", "")`, where the message and the ordering both
survive and the poll never fires.

### R1-06 Identity-map fakes agree by construction, in five files
`test_progress_guard.py:274-313`, `test_cancellation_core.py:120-126`,
`test_jlens_cancel.py:35-37`, `test_task_heartbeat.py:550`,
`test_training_tasks.py` (several).

On a real SQLAlchemy `Query`, `populate_existing()` returns a **new** Query.
Writing `q.populate_existing()` and discarding the result reads the stale
identity-mapped row — MIS-E2E-057 reintroduced. Every one of these fakes wires
both `filter().first()` and `filter().populate_existing().first()` to the same
row, or merely logs that the call happened, so none can exhibit it.
`test_task11_correctness.py:69-93` is the only fake that models it correctly.

### R1-07 `except` ordering asserted by text position, not structure
`test_phase6_cancellation.py:87,121,153,191` and others compare
`src.index("except OperatorCancelled") < src.index("except Exception as")` over
the whole function, with no proof the two handlers hang off the same `try`.
Moving the cancellation handler into an inner `try` preserves the order and
changes the semantics.

### R1-08 `src[handler:]` runs past the handler
`test_download_tokenize_cancellation.py:283-294`. The slice is the rest of the
whole function. Moving `remove_partial_download(...)` from the cancel handler
into the generic one leaves it green — and leaks the partial directory, which is
the specific thing that class was added to prevent.

### R1-09 `job_had_started` ordering finds the assignment
`test_download_tokenize_cancellation.py:313-326`,
`test_phase6_cancellation.py:285-292`. `src.index("job_had_started")` matches the
assignment, which is trivially before the `rmtree`. Deleting the guard condition
survives both. **The model-download case has no behavioural test at all.**

### R1-10 A wrong `status_field` on a circuit scope is undetectable
`test_phase6_cancellation.py:197-212`. `circuit_faithfulness` and
`circuit_calibration` are both on the `circuits` table, and both writer and
reader read `scope.status_field` — so they agree whatever it points at.
Pointing calibration's `status_field` at `faithfulness_status` passes the
column-exists check (real column) and this test (self-consistent), while
cancelling a calibration silently cancels the faithfulness run.

---

## Severity 3 — fail-open tooling and thin assertions

### R1-11 The F821 gate still fails open on its most likely failure
`test_no_undefined_names.py:57-67`. A **missing ruff also exits 1** with empty
stdout, so `_f821_findings()` returns an empty set and the gate reports a clean
tree it never inspected — the exact fail-open its docstring says was closed.
(The sibling ratchet test notices, so the suite catches it; the gate itself does
not.)

### R1-12 `readme_block[:800]` is an arbitrary window
Same file. The bogus import re-added 900 characters into the README template
survives.

### R1-13 No integration test asserts `cancel_requested_at`
`tests/integration/test_dataset_cancellation.py`. Four tests assert
`status == ERROR` and `error_message == "Cancelled by user"` — the outcome
conflation that `cancel_requested_at` was added to replace. Deleting the
`request_cancel("dataset_download", …)` call leaves every integration test
green; only the AST check in `test_dataset_cancel_scope.py` catches it.

### R1-14 Fixed-size source windows, and comment-satisfiable substrings
`test_dataset_cancel_scope.py:42-54` uses the `src[start:start+900]` window that
another file in the same diff explicitly calls out as a bug pattern; its
`"tokenization.status" in loop_body` is satisfiable by the surrounding comment
prose. `:84-104`'s `assert "task_id=" in target` is satisfied by
`cancel_task(…, task_id=None)`.

### R1-15 `test_labeling_service.py:294-313` asserts call shape only
`request_cancel` is patched to `Mock(requested=True)`, so `assert result is True`
cannot distinguish the service from `return True`. `requested=False → returns
False` is untested, and `celery_task_id` is `None` in the fixture so nothing
asserts it is forwarded — the one case a plain `revoke()` genuinely handles.

### R1-16 The Shape D skip list is a one-way ratchet
8 of 19 scopes skip the route check, 4 skip the Shape-A check.
`test_the_by_design_list_only_names_real_scopes` only catches *deleted* scopes;
nothing catches a scope that has GAINED a route and should leave the list.
`model_download` and `training` both have routes today and are still excused.

### R1-17 Decoy patch target
`test_progress_guard.py:134` patches
`src.services.extraction_service.emit_progress` with `create=True`, but that name
is a function-local import — the patch binds a name nothing reads. Harmless to
the assertions, but it reads as coverage that does not exist.

---

## Self-inflicted, and worth naming

Findings R1-01 through R1-05 are all mine, all written *in this arc*, and all of
the same shape the arc's own durable finding names: **an assertion satisfied by
text that is not the thing.** I recorded that lesson six times while producing
six more instances of it. Writing the lesson down is not the same as applying
it; the only reliable application is to drive the code, or read the AST, or read
the live registry.

---

## Correctness findings (second reviewer)

Nine defects, three of which left the feature **worse than before** on their
paths. Findings numbered `R1c-NN`.

### R1c-01 (S1) — the Phase-5 alias BROKE the J-lens fit cancel
`jlens_fit_tasks.py:210` raised `TaskCancelled(msg)` with one argument.
Rebinding `TaskCancelled` to `OperatorCancelled` — whose signature is
`(scope, target_id, reason, detail)` — made that a `TypeError`, which
`except TaskCancelled` does **not** catch. It reached `owns_its_failure`, which
catches `BaseException` and calls `fail_row`, so **the operator's cancellation
was written as a crash**, on the one path the module docstring cites as
hardware-verified. Confirmed by direct construction before fixing.

Also closed here: the plan required teaching `owns_its_failure` to re-raise
`OperatorCancelled` rather than mark the row failed. That was never done.

### R1c-02 (S1) — `download_and_load_model`: no handler, and a disk leak
The task got a checkpoint but neither `@cooperative_cancel` nor an
`except OperatorCancelled`. The raise escaped the task: `except Exception`
cannot catch a `BaseException`, so celery never acked the `acks_late` message —
the 12-hour strand this design exists to avoid, reintroduced by adding a
checkpoint without a handler to receive it. And `cancel_download` had stopped
deleting the cache directory *on the promise that the task would*; nothing did,
so a cancelled 40 GB download was orphaned with `delete_model_files` unable to
find it (it resolves `model.file_path`, written only after the checkpoint).

### R1c-03 (S1) — `execute_neuronpedia_export`: same missing handler
Plus the half-built export tree was never removed.

### R1c-04 (S1) — cancelling a tokenization stranded the parent dataset
`tokenize_dataset_task` sets the `Dataset` row to PROCESSING; the
`except BaseException` handler my new branch pre-empts was what restored it to
READY. **Strictly worse than before the handler existed** — the cancel used to
be inert, so the task ran to completion and the dataset ended up READY.

### R1c-05 (S2) — the moved cleanup targeted a path that does not exist
`remove_partial_download(raw_path)` — but the tqdm checkpoint fires during
`load_dataset`, and `raw_path` is not created until `save_to_disk` **after** it
returns. So the cleanup deleted nothing while HuggingFace's real cache leaked.
**Replacing the endpoint's cleanup with a no-op is worse than the
live-directory delete it was meant to fix.**

### R1c-06 (S2) — the enhanced-labeling stop still paid the whole bill
`with ThreadPoolExecutor(...)` calls `shutdown(wait=True)` without
`cancel_futures`, so every already-submitted example still ran. The route's
stated benefit was exactly what did not happen.

### R1c-07 (S2) — a cancelled SAE extraction was still relabelled COMPLETED
The two sibling paths got an `is_cancelled` gate; this one was missed. The
window is always open, because the 2-second throttle means the final iterations
of the `latent_dim` loop usually do not poll at all.

### R1c-08 (S2/S3) — four circuit-family completion writes bypassed the guard
`circuit_record_tasks._complete` (also reading the long-lived session without
`populate_existing`), faithfulness, calibration, and grouping finalize.

### R1c-09 (S3) — task-start "running" clobbered a cancel-while-queued
A solo worker busy on another job is not reading the control queue, so the
revoke lands late; the task then stamps `running` over `cancelled` and every
later poll reads running. The endpoint had already told the operator
*"it will not run."*

### R1c-10 / R1c-11 (S3/S4)
`_cancel_point`'s `populate_existing` discarded the pending `PACKAGING` write,
so the export never reported that stage. And two async routes called the
synchronous `request_cancel` inline, blocking the event loop through a query, a
commit and a `revoke()` broadcast.

### Clean categories, verified
Stale guard reads: none — every guard re-reads or uses a fresh session. Scope
declarations: clean; every field resolves to a real column and every vocabulary
matches its live enum. Handler ordering: every ordering that exists is correct;
the defects were *missing* handlers.

### Noted, not fixed
The repo has **8 alembic heads**, so `alembic upgrade head` refuses — pre-existing,
but it now gates a load-bearing column (`cancel_requested_at`).
