# Review round 2 — re-review of R1's fixes

Scope: commit `3da8f578` (R1's twelve fixes). Findings numbered `R2-NN`.

**Two of the twelve fixes were worse than the bug they replaced, and one was
the exact no-op R1's own commit message lectured against.** This is the third
consecutive arc in this repo where that has been true, and the reason round 3
is not optional.

---

## R2-01 (S1) — `db.refresh(circuit)` silently destroyed the run's result

`circuit_calibration_service.py` and `circuit_faithfulness_service.py`.

R1 added `db.refresh(circuit)` before the completion status check.
`Session.refresh()` **expires the instance BEFORE autoflush**, so every
un-flushed attribute set immediately above it — the band, the
`intensity_range` clamp, the version bump, the faithfulness scores — was
erased, and the commit persisted only the status.

**On every successful run** the feature produced nothing while reporting
`completed`. Calibration's entire purpose — clamping the served dial to the
measured band — silently stopped happening.

Verified against this repo's SQLAlchemy before fixing:

```
row.payload = "NEW RESULT"; s.refresh(row); row.status = "completed"; s.commit()
  ->  payload: old      status: completed
```

**And R1 added the fake that hid it.** `_FakeSyncDB.refresh` was given a no-op
body in the same commit so the completion write would stop raising — which made
the destructive call harmless in the test and left it asserting the pre-fix
behaviour. That is "fixtures agree by construction" in its purest form, created
deliberately, to make a test pass.

Replaced with a scalar column select, which reads the committed status without
touching the identity map or the pending writes. The sibling half of the same
R1 fix — `circuit_record_tasks._complete` — had used a fresh query with
`populate_existing()` and was correct.

## R2-02 (S2) — `await db.flush()` blinded the export's last checkpoint

R1 added a flush so `populate_existing` would stop discarding the pending
`PACKAGING` write. But `_cancel_point` re-reads **on the same transaction**, so
after the flush it read back its own uncommitted `packaging` and could never
see the API process's committed `cancelled` — and `_update_stage` then committed
the clobber. The export finished `COMPLETED` over the operator's stop.

R1 traded a cosmetic reporting gap for the destruction of the cancellation
itself. Reverted; the stage is display-only and now rides on `current_stage`,
written after the checkpoint has passed.

## R2-03 (S3) — the export cleanup deleted a directory that does not exist

R1 invented `data_dir/"neuronpedia_exports"`. The service writes to
`data_dir/"exports"/"neuronpedia"`; the invented path appears nowhere else in
the repo, so `exists()` was always False and the tree leaked exactly as before.
R1's own commit message, about the dataset case: *"a no-op cleanup is worse than
the live-directory delete it replaced."* Now taken from the service itself so
the two cannot drift.

## R2-04 (S4) — the cleanup rmtree'd a SHARED cache

`data_dir` is `settings.datasets_dir`, the `cache_dir` passed to `load_dataset`
for **every** dataset — so `downloads/` holds other jobs' resumable chunks, and
the success path only clears it when the operator has set
`auto_cleanup_after_download`. R1 deleted it on any cancel. Now only the
per-repo arrow tree and this job's own output, with the shared cache touched
only under the operator's existing setting.

## R2-05 (S5) — the durable state was right, the state the operator SEES was wrong

Both `extraction_tasks` and `circuit_record_tasks` emitted a `completed`
WebSocket event immediately after the guarded write had refused. The frontend
spread-merges those payloads, so the UI showed a completed job over a cancelled
row — a half-fix, and the visible half was the wrong one.

## R2-06 (S6) — the model handler's loose ends

`cache_dir` computed inside the `try`, so a failure on that line left it unbound
for the logger in the `except` — an error inside an error handler, caught by
nothing, reproducing the unacked-`acks_late` strand the handler exists to
prevent. It was also the one deletion sink in the change bypassing
`resolve_deletable_path` (MIS-E2E-071), and it never closed the `task_queue`
row, so a cancelled download showed as running in Active Operations.

## R2-07 (S7) — five smaller ones

* R1's rationale for `owns_its_failure` was **inverted**: a task's own
  `except TaskCancelled` sits inside the decorated function and runs first, so
  reaching the decorator means there is no local handler — and re-raising a
  `BaseException` there escapes celery unacked. It now returns the canonical
  cancelled result.
* The tokenization restore was the one status write in the change that bypassed
  the guard, and would resurrect a dataset whose *download* had been cancelled.
  It also left a READY row carrying an error message.
* `expire_all()` was redundant with `populate_existing()` and made the next
  attribute access raise `ObjectDeletedError` on a row deleted mid-run.
* `_refuse_if_cancelled`'s `except Exception: return True` was silently inert.
* A bare `assert` on the cancellation path — stripped under `-O`, and
  `AssertionError` is an ordinary `Exception`, so the handler would have turned
  a programming error on the cancel path into a FAILED run: the arc's founding
  bug, reintroduced by its own guard.

## Verified sound by the reviewer

The `jlens_fit_tasks` arity fix; the `circuit_record_tasks._complete` half of
the completion guard; `_refuse_if_cancelled`'s logic and contract;
`enhanced_labeling`'s `shutdown(cancel_futures=True)`; and `run_in_threadpool`'s
kwarg forwarding. Reachability of the dataset cleanup names was also verified
sound — `data_dir`, `raw_path` and `repo_id` are all bound before the only
checkpoint.

---

## The seventh instance, inside the fix for the sixth

Writing R2-01's regression test, the first version asserted
`"db.refresh(circuit)" not in source` — and matched **the comment explaining
why it was removed**. The trap appeared inside the test written to catch the
trap. Comments are stripped now.
