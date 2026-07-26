# Feature PRD: Training Finalization & Checkpoint Lifecycle

**Document ID:** 020_FPRD|Checkpoint_Lifecycle
**Version:** 1.0
**Status:** Implemented (3 review rounds — 104 findings; 24 mutation controls)
**Related:** 000_PPRD §3.21 (row 21) · PADR IDL-39 · extends Feature 3 (SAE Training) · produces the `community_format/` artifact Features 4/16/18 consume

---

## 1. Overview

### 1.1 Purpose
Make a stopped training's SAE usable, and stop training checkpoints growing
without bound.

### 1.2 User Problem
Cancelling a training forfeited its SAE. `stop_training` set `status=cancelled`
and returned, skipping the training loop's finalize block that writes
`community_format/` — the only artifact downstream consumers read
(`sae_manager_service` scans it; circuit capture, Neuronpedia export and the
analysis service load through `load_sae_auto_detect`). The run's checkpoints
remained on disk, intact and often excellent, and nothing in the product could
consume them. The UI offered only **Retry**, which creates a brand-new training
from step 0 and orphans them.

Observed: `train_969e90af` (granite-4.1-8b, JumpReLU, layers 34/35/36) was
stopped at step 10,300 once its FVU flattened — **FVU 0.065, zero dead
neurons**, three 1.1 GB checkpoints on disk, and no way to import the SAE. The
manual actively misled, promising *"Stop: Gracefully end training (saves final
checkpoint)"*.

Separately, checkpoints were never reclaimed: **423 rows / 18 trainings / 78 GB**
under `/data/trainings`, on a volume at 81% capacity, with `file_size_bytes`
NULL on every row.

### 1.3 Solution
Two capabilities on one domain (the checkpoint lifecycle):
1. **Finalize** a stopped run from a checkpoint, producing the standard export.
2. **Retention** that prunes redundant checkpoint steps, safe by default.

---

## 2. User Stories

- As a researcher who stops a run because its FVU has flattened, I want the SAE
  I already paid for to be importable, so the compute is not wasted.
- As a researcher, I want to be told plainly that a salvaged run stopped early,
  so I never mistake it for a completed run when interpreting features.
- As a researcher whose run crashed, I want to build an SAE from the last good
  checkpoint rather than restart from zero.
- As an operator, I want to reclaim checkpoint disk without risking the weights
  a run's results depend on.
- As an operator, I want to see exactly what a prune would delete before any
  deletion happens.

---

## 3. Functional Requirements

### 3.1 Stop & Finalize
Stopping a run MAY also finalize it. The action stops the run, then rebuilds the
SAEs from the newest complete checkpoint and writes `community_format/`. If no
checkpoint exists the response says so rather than claiming a finalize.

### 3.2 Finalize an already-stopped run
A CANCELLED or FAILED run that has checkpoints MAY be finalized. FAILED requires
an explicit acknowledgement, because its checkpoints may predate the crash.

### 3.3 Honest completion state
Finalizing sets `status=COMPLETED` (which unlocks SAE import) but MUST leave
`progress` and `current_step` untouched, and MUST record `finalized_from_step`.
The UI MUST distinguish a finalized-early run from a completed one.

### 3.4 Export fidelity
The export MUST be produced by the same writer the success path uses, and its
`cfg.json` MUST agree with the weights beside it — dimensions and architecture
are read from the checkpoint tensors, not from stored hyperparameters.

### 3.5 Export integrity
The export MUST be atomic (a failure leaves the previous export intact) and MUST
only proceed from a checkpoint step verified complete. When no step is named,
finalize falls back to the newest complete step.

### 3.6 Retention policy
Configurable: enabled, dry-run, keep-last-N steps, keep-best, minimum age.
Selection operates on whole **steps**. Defaults: disabled, dry-run, keep best +
newest 2, 24h minimum age.

### 3.7 Retention safety
Pruning MUST never delete `is_best`, the newest step, any checkpoint of a
training in an active state, or a checkpoint younger than the minimum age. A
step whose row is promoted to best between planning and execution MUST be
skipped entirely.

### 3.8 Prune preview and manual prune
An operator MUST be able to see what would be deleted (read-only) and to prune a
single training on demand.

### 3.9 Checkpoint deletion
A single checkpoint MAY be deleted. The best checkpoint requires explicit
confirmation.

---

## 4. User Interface

- **Stop & Finalize** beside **Stop** on running/paused runs.
- **Finalize** on cancelled/failed runs that have checkpoints.
- An amber **"Finalized early @ N"** badge on salvaged runs.
- A **Storage** settings tab: the five retention settings, a read-only prune
  **Preview**, and a confirm-gated **Prune now**.
- Failures surface in the card rather than the browser console; 409s offer the
  escalation the API asks for.

---

## 5. API / Integration

| Method | Path | Purpose |
|---|---|---|
| POST | `/trainings/{id}/finalize` | write `community_format/` (`checkpoint_step`, `allow_failed`, `force`) |
| POST | `/trainings/{id}/control` | `action=stop_and_finalize` |
| GET | `/trainings/{id}/checkpoints/prune-preview` | read-only report |
| POST | `/trainings/{id}/checkpoints/prune` | prune one training now |
| DELETE | `/trainings/{id}/checkpoints/{ckpt_id}` | delete one checkpoint (`allow_best`) |

WebSocket: `training:completed` (carries `finalized_from_step` and the real
progress) and `training:finalize_failed`.

---

## 6. Data / Types

- `trainings.finalized_from_step` — nullable integer, migration `b4d19f0c73ae`.
- `app_settings` keys (category `general`): `checkpoint_prune_enabled`,
  `checkpoint_prune_dry_run`, `checkpoint_prune_keep_last`,
  `checkpoint_prune_keep_best`, `checkpoint_prune_min_age_hours`.

---

## 7. Dependencies
Existing `CheckpointService` writer, `create_sae`, the `app_settings` table, and
the Celery `low_priority` queue. No new external dependencies.

---

## 8. Success Criteria

1. A stopped run can be finalized and its SAE imported.
2. A finalized-early run is visibly distinguishable from a completed one.
3. `cfg.json` matches the exported weights for every layer.
4. A prune preview reports real numbers; nothing is deleted while dry-run is on.
5. No prune ever produces a partially-deleted step.
6. `train_969e90af` specifically becomes importable.

---

## 9. Non-Goals

- Resuming a stopped run (that is the existing `resume` path).
- Reclaiming orphaned checkpoint directories that have no database row.
- Automatic retention tuning; the operator sets the policy.
- Any change to the live serving path in miLLM.

---

## 10. Testing Requirements

Every guard must be pinned by a test that FAILS when the guard is removed.
Required negative controls: casing normalisation, keep-best, min-age,
active-training guard, step-atomic execution, fail-closed completeness check,
the `stop_and_finalize` handler branch, the emitter's real progress, the
Finalize button's visibility, and `db.commit()`.

---

## 11. Traceability

| Source | Covered by |
|---|---|
| PPRD §3.21 row 21 | §1–§9 |
| PADR IDL-39 decision 1 (reuse the writer) | §3.4 |
| PADR IDL-39 decision 2 (dims from checkpoint) | §3.4 |
| PADR IDL-39 decision 3 (honest completion) | §3.3, §4 |
| PADR IDL-39 decision 4 (atomic, complete step) | §3.5 |
| PADR IDL-39 decision 5 (fail closed) | §3.5, §10 |
| PADR IDL-39 decision 6 (step granularity) | §3.6, §3.7 |
| PADR IDL-39 decision 7 (unlink before commit) | §3.7, §3.9 |
| PADR IDL-39 decision 8 (ships inert) | §3.6 |
