# Technical Implementation Document: Training Finalization & Checkpoint Lifecycle

**Document ID:** 020_FTID|Checkpoint_Lifecycle
**Version:** 1.0
**Status:** Implemented
**Related:** 020_FTDD · 020_FPRD · PADR IDL-39

---

## 1. Implementation Order

1. **Migration + model** (`finalized_from_step`) — gates the honest-status work.
2. **`training_finalize_service`** — pure service, no Celery/HTTP. Testable first.
3. **`checkpoint_retention`** — session-free policy core.
4. **`checkpoint_service.delete_checkpoint_files`** — shared sync deleter both
   delete paths use; fix the missing `is_best` guard while here.
5. **Workers** + the three `celery_app.py` edits (route, beat, autodiscovery).
6. **Endpoints** — route ORDER matters (see Pitfalls).
7. **Frontend** — types → store → card → settings tab.
8. **Tests + mutation controls** throughout, not after.

## 2. File-by-file

| File | Change |
|---|---|
| `alembic/versions/b4d19f0c73ae_*.py` | (NEW) nullable `finalized_from_step`; `down_revision = 5cede2a1b3f7` |
| `src/models/training.py` | (extend) the column, with the comment explaining why status alone is insufficient |
| `src/schemas/training.py` | (extend) widen the control Literal; expose `finalized_from_step` |
| `src/services/training_finalize_service.py` | (NEW) discovery, rebuild, atomic export |
| `src/services/checkpoint_retention.py` | (NEW) policy core + `policy_from_values` + `load_policy` |
| `src/services/checkpoint_service.py` | (extend) `delete_checkpoint_files`; `allow_best` on `delete_checkpoint`; reorder to unlink-first |
| `src/workers/training_finalize_tasks.py` | (NEW) task, guarded status write, terminal WS events |
| `src/workers/prune_checkpoints.py` | (NEW) scheduled sweep + single-training prune |
| `src/core/celery_app.py` | (extend) task_routes, beat_schedule, autodiscovery — all three |
| `src/api/v1/endpoints/trainings.py` | (extend) finalize, prune-preview, prune, DELETE checkpoint, `stop_and_finalize` |
| `frontend/src/types/training.ts` | (extend) `finalized_from_step`, widened action, `CheckpointPrunePreview` |
| `frontend/src/stores/trainingsStore.ts` | (extend) 4 actions; `allow_best`/`allow_failed`/`force` query params |
| `frontend/src/components/training/TrainingCard.tsx` | (extend) buttons, badge, 409 escalation, error banner |
| `frontend/src/components/panels/SettingsPanel.tsx` | (extend) Storage tab + preview + Prune now |
| `frontend/src/hooks/useTrainingWebSocket.ts` | (extend) stop hard-coding `progress: 100` |
| `manual/docs/core-workflow/training-lifecycle.md` | (NEW) user-facing page |
| `manual/sidebars.ts` | (extend) register it — omitting this orphans the page |

## 3. Pitfalls

Each of these cost real time; none is theoretical.

1. **Celery routes match the TASK NAME, not the module path.** A short
   `name="prune_checkpoints"` never matches `"src.workers.prune_checkpoints.*"`,
   so the task silently uses the default queue. Fully qualify the name, and
   assert routing by asking `celery_app.amqp.router.route()` — reading
   `conf.task_routes` back just re-reads the literal you wrote.
2. **`hp['hidden_dim']` is corrected in memory and never persisted.** Read
   dimensions from checkpoint tensor shapes.
3. **`load_multilayer_checkpoint` is broken dead code** — legacy paths only; it
   raises against every current checkpoint.
4. **`migrate_mistudio_to_community` cannot handle JumpReLU** (`encoder.weight`
   vs `W_enc`).
5. **`cfg.json` is built from the hyperparameters you pass the writer** — pass
   the resolved values or the config contradicts the weights.
6. **Route order**: literal `/checkpoints/prune*` must precede
   `/checkpoints/{checkpoint_id}`.
7. **The control-action Literal gates the request** — a new action without it
   returns 422 before the handler runs.
8. **`AppSettingService` is async** and unusable from a Celery worker; query
   `AppSetting` directly with the sync session.
9. **`file_size_bytes` is NULL** for rows the training loop writes; size
   reporting must `stat()` and tolerate failure.
10. **A boolean setting must fail to its default, not to `False`** — for
    `dry_run`, `False` means delete.

## 4. Testing

- **Policy** — table-driven against plain objects (session-free core).
- **Execution** — `tmp_path`-backed: real files, real unlink, real rmdir.
- **Wiring** — behavioural, not existence: POST the control action and assert
  `.delay` was called **once with the exact arguments**; assert the task is in
  the live Celery registry and that the router sends it to `low_priority`.
- **Worker** — assert `session.commits == 1` alongside the field writes;
  otherwise replacing `db.commit()` with `pass` leaves the suite green.
- **Frontend** — override the default empty `fetchCheckpoints` mock, or
  `checkpoints.length > 0` is false in every test and the Finalize block can be
  deleted with the suite green.
- **Store-level** — component tests mock the store, so query-param building
  needs its own tests.
- **Mutation controls are mandatory** for every guard; re-run each after fixing.
