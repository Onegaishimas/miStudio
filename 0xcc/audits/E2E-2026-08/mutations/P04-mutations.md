# P04 — mutation control log

**Phase:** P04 Workers, Celery, task lifecycle · **Round:** 2 · **Date:** 2026-08-23

| # | Target | Mutation | Landed | Result |
|---|---|---|---|---|
| M16 | `workers/cleanup_stuck_circuit_runs.py:84` | Revert `looks_abandoned(...)` to the pre-fix `state not in ("PENDING","STARTED","RETRY")` rule | ✅ | **KILLED** — 3 failures in `test_cleanup_stuck_circuit_runs_pending.py` |
| M17 | `workers/steering_tasks.py:115` | Remove `queue="steering"` from the GPU steering task | ✅ | **SURVIVED** → MIS-E2E-097 |

**1 of 2 survived.**

## M16 — the fix is pinned, for exactly one janitor

Reverting the PENDING rule failed three tests, including
`test_a_RUNNING_row_stuck_at_PENDING_is_reclaimed` and
`test_a_task_celery_calls_live_is_not_reclaimed[RECEIVED]`. A genuinely good
regression test, with a fixture that makes the two behaviours differ.

But note the file name: **`test_cleanup_stuck_circuit_runs_pending.py`**. The test is
named for one janitor and covers one janitor. That is the structural reason
MIS-E2E-092 exists — the fix and its test were both written for a single instance of
a trap the code itself documents as general, and the four siblings inherited neither.
A test parametrized over the janitor registry would have failed for all four the day
it was written.

## M17 — nothing asserts a task's resolved queue

Removing `queue="steering"` from `steering.compare` — sending the GPU steering task to
the default `datasets` queue, to compete with dataset downloads on workers with no GPU
reservation — left **184 tests green**, including the whole of
`test_worker_queue_coverage.py`.

That file's three assertions are: every queue named in `task_routes` has a consumer;
every worker declares its queues; `low_priority` is not on the GPU worker. All three
read the routing **table**. None asks whether a registered task **resolves** to the
queue it is supposed to — so a queue can be declared, consumed, and permanently empty
and still pass. That is the live state of the `training` queue (MIS-E2E-093), and M17
shows the same blind spot would swallow a regression on any of the correctly-routed
tasks too.

## Equivalent mutants

None.
