# P04 — Workers, Celery, task lifecycle: all three rounds

**Phase:** P04 · **Date:** 2026-08-23
**Scope:** `backend/src/workers/` (41 files, 15,087 lines), `backend/src/core/celery_app.py`

Mutation log: `mutations/P04-mutations.md` (2 run, 1 survived).

## R1 — findings (6)

| Id | Sev | Claim |
|---|---|---|
| MIS-E2E-092 | P1 | **Four of five janitors treat PENDING as alive**; the fix exists in one and is documented there as general |
| MIS-E2E-093 | P1 | `train_sae` / `resume_training` never reach the `training` queue — short names miss the `src.workers.training_tasks.*` glob |
| MIS-E2E-094 | P2 | The solo pool discards `worker_max_tasks_per_child`; the comment promises a VRAM reclaim that never happens |
| MIS-E2E-095 | P2 | No beat entry sets `expires`; ~26 stale reconciles pile up behind a 13-minute NLP pass on one solo worker |
| MIS-E2E-096 | P2 | `update_state` without `beat()` wipes the liveness stamp mid-fit; a janitor message loses its only diagnostic |
| MIS-E2E-097 | P1 | *(R2)* No test asserts any task's **resolved** queue |

## Both headline findings verified against the live system, not read

**MIS-E2E-093** — resolved through `celery_app.amqp.router.route()` on the real app:
```
train_sae                                        -> datasets   (default)
resume_training                                  -> datasets   (default)
src.workers.training_tasks.delete_training_files -> training
```
The registry confirms why: `train_sae` and `resume_training` are registered under
**short names** with no decorator `queue=`, so the module glob misses them. The
`training` queue is declared, provisioned and consumed, and the only task that
reaches it is file deletion.

**MIS-E2E-092** — `looks_abandoned` occurs once in `cleanup_stuck_circuit_runs.py`
and **zero times** in each of the four siblings, whose PENDING handling is inline.

## A false positive of my own, recorded

An initial probe suggested `steering_compare_task` also routes to `datasets`. It does
not — I probed with the **function** name; the task is registered as
`name="steering.compare"` with `queue="steering"` in the decorator. The reviewer had
explicitly cleared `steering.*` and `model_tasks.*` on exactly that basis and was
right. Recorded because "probe the layer that owns the thing" is a standing lesson
here and I briefly did not.

## R2 — mutation controls

**M16 KILLED.** Reverting the one fixed janitor to the pre-fix rule failed three
tests. The fix is genuinely pinned — but the test file is named
`test_cleanup_stuck_circuit_runs_pending.py`, for one janitor. That is the structural
cause of MIS-E2E-092: a general trap fixed and tested at a single instance.

**M17 SURVIVED.** Stripping `queue="steering"` from the GPU steering task left 184
tests green, including all of `test_worker_queue_coverage.py` (MIS-E2E-097). That
file guards the routing table's coverage and nothing about where tasks actually go.

## R3 — verification & closure

| Verdict | Ids |
|---|---|
| **CONFIRMED** | 092, 093, 097 |
| **PLAUSIBLE** | 094, 095, 096 |
| **REFUTED** | the `steering.*` mis-routing hypothesis (mine), recorded above |

094–096 are read-only reads of unambiguous code (a Celery source constant, absent
`expires` keys, an assignment ordering). Confirming 095's pile-up would need a live
13-minute NLP pass with beat running, which is a P12 exercise, not a P04 one.

**Phase closed. 6 findings** (MIS-E2E-092 … 097). Mutations: 2 run, 1 survived.
Tree verified clean.

**The one sentence for the synthesis:** two of this phase's findings are guards that
check the shape of a declaration rather than the behaviour it is supposed to
produce — a janitor test named for one janitor, and a queue test that reads the
routing table instead of routing a task.
