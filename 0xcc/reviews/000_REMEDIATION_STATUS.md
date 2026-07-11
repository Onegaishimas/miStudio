# Remediation Status — miStudio Review Backlog

**Date:** 2026-07-11
**Context:** Implementation pass over the [cross-feature synthesis](000_SYNTHESIS.md) backlog. Tracks what was fixed, what was deferred, and why.

---

## ✅ Implemented (this pass)

| ID | Severity | Fix | File(s) |
|----|----------|-----|---------|
| **002-F1** | **P0** | Broken `....models.extraction` imports → `....models.activation_extraction` (2 dead endpoints restored) | `api/v1/endpoints/models.py` |
| **002-F2** | P1 | Detached-instance access in `cancel_extraction` WS emit — capture `progress` before session close | `api/v1/endpoints/models.py` |
| **002-F5** | P2 | `redownload_model` used raw `Path()` → `settings.resolve_data_path()` (native-mode stale-file bug) | `api/v1/endpoints/models.py` |
| **001-F1** | P1 | Tokenization-cancel `NameError` (`model_id` undefined → `tokenization.model_id`) | `api/v1/endpoints/datasets.py` |
| **001-F2** | P2 | `DatasetUpdate.status` regex accepted non-existent `ingesting` enum value → removed | `schemas/dataset.py` |
| **001-F5** | P3 | Removed token-length logging + `print()` (secret-adjacent) → `logger.debug` presence-only | `api/v1/endpoints/datasets.py`, `workers/dataset_tasks.py` |
| **003-F1** | P2 | `create_training` now validates model + extraction existence (400 at API time, not mid-run) + endpoint maps `ValueError`→400 | `services/training_service.py`, `api/v1/endpoints/trainings.py` |
| **003-F3** | P3 | `JumpReLUSAE` docstring corrected (was fraction-based/0.4; now count-based/1e-3, matches code) | `ml/sparse_autoencoder.py` |
| **003-F4** | P3 | Deduped 3× `normalize()`/`denormalize()` into module-level `normalize_activations`/`denormalize_activations` (uses `x[..., :1]`, safe for 2D+3D) | `ml/sparse_autoencoder.py` |
| **003-F7** | P3 | `control_training` error detail was a non-f-string (`"Failed to {action}"`) → f-string | `api/v1/endpoints/trainings.py` |
| **004-F1** | P1 | Enhanced-labeling no longer marks job FAILED / emits `failed` when a Celery autoretry will run (only on terminal failure) | `workers/enhanced_labeling_tasks.py` |
| **004-F2** | P2 | Duplicate-active-job guard made atomic via `SELECT ... FOR UPDATE` feature-row lock | `api/v1/endpoints/enhanced_labeling.py` |
| **004-F3** | P2 | `set_star_color` now `Literal["yellow","purple","aqua"]` (was unvalidated free-text) | `api/v1/endpoints/features.py` |
| **005-F1** | P2 | SAE delete consistency: hard delete (cascade to features) when `delete_files=True`; reversible soft delete only when files preserved | `services/sae_manager_service.py` |
| **BATCH-6 / 003-F6** | P2 | `delete_model` pre-checks referencing trainings → clean 409 instead of raw FK IntegrityError→500 | `services/model_service.py`, `api/v1/endpoints/models.py` |
| **006-F1** | **P1** | Steering resolver-singleton orphan-hang: cleanup now *rejects* the superseded promise before nulling (batch/sweep/combined) | `stores/steeringStore.ts` |

**Verification:** all modified backend files parse; frontend `tsc --noEmit` exit 0; backend suite run (see commit).

---

## ⏸ Deferred — blocked on the multi-head Alembic state

The migration chain currently has **two pre-existing heads** (`f3a7b1c2d4e5`, `t7u8v9w0x1y2`) — confirmed via `alembic heads`. Adding new migrations onto this without first a **merge migration** would create a third head and break `alembic upgrade head` in CI/deploy. The following remediations require schema changes and are therefore deferred to a dedicated *"merge alembic heads + schema"* task so they don't destabilize deployment:

| ID | Severity | What it needs | Note |
|----|----------|---------------|------|
| **002-F3** | P2 | `celery_task_id` column on `models` | The `cancel_download` task already accepts+revokes a `task_id`; only the storage column + endpoint wiring is missing. Model download-cancel still cleans files but doesn't revoke the running task until this lands. |
| **003-F2** | P2 | `UNIQUE(training_id, step, layer_idx)` on `training_metrics` | Prevents silent metric-row duplication under multi-hook/resume. Needs a de-dup data migration first (existing dupes would violate the new constraint). |
| **007-F1** | P2 | `NeuronpediaPushJob` ORM model for the raw-SQL `neuronpedia_pushes` table | Table exists (raw SQL); wrapping it in an ORM model is schema-tracked work best done with the merge. |

**Recommended follow-up task:** (1) `alembic merge -m "merge heads" f3a7b1c2d4e5 t7u8v9w0x1y2`; (2) then the three migrations above; (3) wire `celery_task_id` through model download + cancel.

---

## 📋 Not attempted this pass (larger scope / lower severity)

| ID | Why not now |
|----|-------------|
| **BATCH-1** (extraction rename) | Cross-cutting rename across 002/004/005 models+tasks+routes; high-value but a focused PR of its own (the P0 import bug it caused is already fixed). |
| **BATCH-4** (test seams) | New tests for steering resolver, logit-lens, BackgroundMonitor emit. Valuable; separate test-writing pass. |
| **006-F2/F3** (steering worker config) | Comment corrections + `acks_late` change in `celery.sh` — deploy-config, safer as its own reviewed change. |
| **006-F4, 007-F3, 008-F3** (zero test coverage) | Same as BATCH-4. |
| **God-file splits** (features.py, steering_service.py, community_format.py) | Pure refactors, no behavior change, high churn — defer. |
| **FPRD doc refresh** (§3 of synthesis) | The documentation pass — large, systematic; tracked separately. |
| **009-F1** (training GPU-selection doc overclaim) | Doc fix; part of the FPRD refresh. |

---

## Summary

**16 findings fixed** across P0→P3, spanning all the correctness bugs that don't require schema migrations. The single open P0 is closed. Remaining items are either (a) blocked on resolving the pre-existing multi-head Alembic state, or (b) larger refactors/test-writing/doc-refresh best done as focused follow-up tasks. No fix touched the research-critical core's *behavior* (the normalize dedup preserves numerics; verified by the test suite).
