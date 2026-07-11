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

## ✅ Implemented (follow-up pass — the previously-deferred schema items)

The multi-head Alembic state (`f3a7b1c2d4e5`, `t7u8v9w0x1y2`) was resolved with a
**merge migration** (`cd6c46abac48`), collapsing to a single head. On that base, the three
deferred items landed:

| ID | Severity | Fix | Migration / file |
|----|----------|-----|------------------|
| **007-F1** | P2 | `NeuronpediaPushJob` ORM model wraps the existing `neuronpedia_pushes` table; raw-SQL INSERT/UPDATE in the endpoint + push task converted to ORM | `models/neuronpedia_push.py` (new, no migration — table already existed) |
| **002-F3** | P2 | `models.celery_task_id` column added; download + redownload persist the task id at dispatch; `cancel_model_download` now passes it to `cancel_download(task_id=…)` which revokes the running task | migration `e1a2b3c4d5e6`; `models/model.py`, `api/v1/endpoints/models.py` |
| **003-F2** | P2 | `UNIQUE(training_id, step, layer_idx)` on `training_metrics` with a defensive de-dup (keep max(id) per group) in the migration | migration `f2b3c4d5e6f7`; `models/training_metric.py` |

Also: `scripts/k8s-helpers.sh` `k8s_migrate` switched to `alembic upgrade heads` (matching
`docker-entrypoint.sh`, tolerant of transient multi-head states).

**Verification:** single head after merge; upgrade + full downgrade + re-upgrade all clean;
ORM models map to live tables (no schema-validator warnings); focused tests
(`tests/unit/test_deferred_remediations.py`) cover the metric unique constraint (dup →
IntegrityError, NULL-layer rows distinct), the push-model round-trip, and the `celery_task_id`
column; full backend suite green.

**Note on the earlier "blocked" framing:** investigation showed production was never actually
broken — `docker-entrypoint.sh` already runs `alembic upgrade heads` (plural). The merge was
still the right move to give future migrations a clean linear base and fix the singular
`k8s_migrate` helper. 007-F1 needed no migration at all (the table pre-existed).

---

## ✅ Test-infrastructure fixes (pre-existing flakiness)

Two pre-existing test problems surfaced while verifying the above and were fixed:

| Issue | Fix | File |
|-------|-----|------|
| **conftest enum-isolation race** — `CREATE TYPE` hit `pg_type_typname_nsp_index` duplicate-key under interleaving, and `DROP TYPE ... CASCADE` in teardown dropped tables that `drop_all` then couldn't find ("enhanced_labeling_jobs does not exist"). Also `label_source_enum` was **missing `enhanced_llm`** (latent — any Feature with that label source would fail). | `async_engine` fixture now: drops leftovers (tables → enums) at setup, creates enums then tables, drops WITHOUT CASCADE (tables first, enums second) at teardown, and includes `enhanced_llm`. | `tests/conftest.py` |
| **flaky `test_failed_pass1_example_still_completes`** — used a positional `side_effect` list, but pass-1 runs examples in parallel (ThreadPoolExecutor), so consumption order was non-deterministic → intermittent `StopIteration`. | Rewrote the mock as a content-aware `side_effect` function that routes by prompt (which prime token / synthesis) instead of call order. | `tests/unit/test_enhanced_labeling_service.py` |

These were latent (order/timing-dependent) and are why single-file and some subset runs failed
before. Full-suite green now holds deterministically.

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
