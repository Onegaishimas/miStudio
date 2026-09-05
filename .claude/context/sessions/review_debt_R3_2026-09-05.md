# Review round 3 — re-review of R1's and R2's fixes

Scope: commits `3da8f578` (R1) and `0f477da7` (R2). Findings numbered `R3-NN`.

**A fix was worse than the bug it replaced in all three rounds.** That is now
four consecutive arcs in this repo with the same result, and the reason the
third round is not optional.

---

## R3-01 (High) — R2's "opt-in" guard was inert on every stock install

`dataset_tasks.py`. R2 gated the shared-cache deletion on
`settings.auto_cleanup_after_download` and wrote *"only when the operator has
already opted into"* — but that setting **defaults to `True`**
(`core/config.py`). On a stock deployment the cancel handler still `rmtree`d
`data/datasets/downloads`, which is the transfer cache **every** dataset
shares, throwing away another job's resumable chunks. The code read an opt-out;
the comment described an opt-in.

R2's own regression test only asserted that the *string*
`auto_cleanup_after_download` appeared after the list, never the default — so
it was green against the inert guard.

**Fixed by removing the deletion entirely.** A cancelled download cannot know
whether the chunks in a shared directory are its own; the success path clears
them under the setting, as it always did.

## R3-02 (High) — R2's tokenization guard checked the wrong row, and stranded on the other branch

R2 gated the parent-dataset restore on `dataset_obj.cancel_requested_at is
None`. A **tokenization** cancel writes that column on the
`dataset_tokenizations` row, never on `datasets` — so the guard was always true
and purely decorative.

And when it *was* set, it was permanent: **nothing in the codebase ever cleared
`cancel_requested_at`**. `cancel_dataset_download` accepts a dataset in
PROCESSING, so once that had happened, every later tokenization cancel took the
else branch and left the dataset in **PROCESSING forever** — the exact defect
the restore exists to fix.

Now keyed on `status == PROCESSING`, which answers the real question ("did this
task put the row here?") without depending on a flag nobody resets.

## R3-03 (High, found while fixing R3-02) — retry after cancel was permanently broken

Chasing the never-cleared column produced a defect neither reviewer reached.
`cancel_requested_at` is what the tqdm poll reads. Nothing cleared it. So a
**re-download of a previously cancelled dataset abandoned on its first tick** —
cancel a dataset download once and it could never be downloaded again.

Verified before fixing: a checker over a row carrying a leftover timestamp
returns `True` on call one. `clear_cancel_request` is the missing half of the
`request_field` mechanism, called at task START — not at cancel time, because
the flag must survive until the running task has seen it.

## R3-04 (Medium-High) — R2 fixed the success path and inverted the cancel path

`circuit_calibration_service` / `circuit_faithfulness_service`. The result
fields are assigned **before** the guard and `db.commit()` runs
unconditionally, so a cancel during the tail left the row `cancelled` **with
the new band applied and the version bumped** — which is what
`export_circuit_definition` → `millm_import_circuit` then ships. R1's
`refresh()` had accidentally prevented this by wiping everything.

Also: a circuit deleted mid-run made `_fresh` `None`, so `is_cancelled(None)`
was False, `"completed"` was written, and the flush matched zero rows —
`StaleDataError` out of a clean finish.

## R3-05 (Medium) — R2's own fixture agreed by construction

`_FakeSyncDB.execute` returned the in-memory attribute of the same fake circuit
— the entire point of the scalar select being that it reads the *committed*
row. A mutation replacing the select with `circuit.calibration_status` passes
it unchanged. The faithfulness twin has no behavioural test at all.

## R3-06 / R3-07 / R3-08 (Medium) — three more half-fixes

* R2's post-completion re-reads **duplicated a read `_complete` already
  performed**, unprotected, inside a `try` whose handler writes FAILED — so an
  error on the bookkeeping read would relabel a committed success as a failure.
  `_complete` returns the answer now.
* R2's export cleanup became **the one deletion sink bypassing
  `resolve_deletable_path`** — the hole R2 had just closed in `model_tasks`.
  `exports_dir / ""` is `exports_dir` itself.
* R2 replaced a bare `assert` with `ValueError` and claimed that removed the
  crash-report outcome. It did not: `ValueError` is an ordinary `Exception`, so
  the handler still marks the run FAILED. It has to be a `BaseException` to
  travel the path the cancellation itself does.

## R3-09 (Low) — durable-right, visible-wrong, again

R2 closed the model download's `task_queue` row with
`mark_task_queue_entries_completed`, which writes `status="completed",
progress=100.0`. A cancelled download showed in Active Operations as *finished
successfully* — the same shape R2 had fixed two hunks earlier in its own
commit. `TaskQueue` documents a `cancelled` value; it is used now.

---

## Verified sound this round

The scalar-read mechanism (`autoflush=False`, READ COMMITTED, `Circuit` in
scope in both functions); removing the PACKAGING flush is not a contract break
(two readers, no janitor sweeps export status, both frontend paths already
accept `computing`); `_exports_dir` is stable and its `mkdir` idempotent;
`owns_its_failure` returning a dict breaks no caller (no celery
chains/chords/links anywhere); the `jlens_fit_tasks` arity fix and its AST
guard, watched firing; `_refuse_if_cancelled`'s semantics; and
`enhanced_labeling`'s `shutdown(cancel_futures=True)`.

## Process failure, recorded

**I ran the mutation harness concurrently with the reviewing agent** — the
third time in this session, and the exact hazard CLAUDE.md records. The
reviewer read three different transient mutations out of the working tree and
had to pin every finding to `git show HEAD:` to be sure of anything. One of my
own edits landed on top of an injected mutation and had to be found and
reverted (`_NeverRaised` in `model_tasks.py`). Writing the lesson down is
demonstrably not the same as applying it.
