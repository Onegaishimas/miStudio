# Task List: Training Finalization & Checkpoint Lifecycle

**Document ID:** 020_FTASKS|Checkpoint_Lifecycle
**Version:** 1.0
**Status:** Implemented and merged-ready. Three review rounds (`/review` + `/code-review`) produced **104 findings** — R1 35 (18+17), R2 41 (18+23), R3 28 (14+14) — all triaged and fixed. **24 mutation controls verified biting**, each of which previously left the suite green. 141 backend feature tests, 105 frontend, full backend suite clean apart from 15 pre-existing CUDA-only failures in `test_activation_service.py` (this dev box has no GPU). Commits 8d44fa4 → 54197af on `feat/checkpoint-lifecycle`, PR #2.
**Source:** 020_FPRD · 020_FTDD · 020_FTID · PADR IDL-39

| Phase | Tasks | Status |
|---|---|---|
| 1. Schema + contract | 3 | [x] |
| 2. Finalize service | 5 | [x] |
| 3. Retention policy | 4 | [x] |
| 4. Workers + wiring | 4 | [x] |
| 5. API + UI | 6 | [x] |
| 6. Verification + acceptance | 5 | [~] |

---

## Phase 1: Schema + contract (gates everything)
- [x] Migration `b4d19f0c73ae` — nullable `finalized_from_step`, `down_revision=5cede2a1b3f7`
- [x] `Training.finalized_from_step` + `TrainingResponse` field
- [x] Widen `TrainingControlRequest` Literal with `stop_and_finalize`

## Phase 2: Finalize service
- [x] Checkpoint discovery (`list_checkpoint_steps`, `resolve_layer_dirs`, legacy `layer_{idx}/`)
- [x] Dimensions + architecture from checkpoint tensors, not hyperparameters
- [x] Rebuild via `create_sae` + strict `load_checkpoint` on CPU
- [x] Atomic staged export + swap; resolved values written into `export_hp`
- [x] Completeness check with newest-complete-step fallback, failing closed

## Phase 3: Retention policy
- [x] Session-free `plan_from_checkpoints` / `select_prunable_steps` (whole steps)
- [x] `policy_from_values` + `load_policy`; fail-safe parsers, both-ends clamping
- [x] Guards: is_best, newest, active training, min age, dry-run
- [x] `delete_checkpoint_files` (sync, shared) + `is_best` guard on `delete_checkpoint`

## Phase 4: Workers + wiring
- [x] `training_finalize_tasks` — guarded status write, terminal WS events, error clearing
- [x] `prune_checkpoints` — step-atomic execution, per-training isolation, failure counters
- [x] `celery_app.py` — routes, beat schedule, autodiscovery (all three)
- [x] Fully-qualified task names so route globs actually match

## Phase 5: API + UI
- [x] `POST /finalize` with `checkpoint_step` / `allow_failed` / `force` guards
- [x] `prune-preview`, `prune`, `DELETE /checkpoints/{id}` (fixes a pre-existing 404)
- [x] `stop_and_finalize` control action
- [x] Stop & Finalize / Finalize buttons; finalized-early badge; error banner
- [x] Storage settings tab + preview + confirm-gated Prune now
- [x] 409 escalation for `allow_failed` / `force` / `allow_best`

## Phase 6: Verification + acceptance
- [x] 141 backend feature tests; 105 frontend; `tsc` clean
- [x] 24 mutation controls verified biting
- [x] Migration applied and verified reversible
- [x] Full backend suite — no regressions
- [~] **Hardware acceptance outstanding**: finalize `train_969e90af` on k8s and
      import the resulting SAE. Blocked on deploy. This is the "use it for real"
      step that static review cannot substitute for.

---

## Relevant Files

| File | Purpose |
|---|---|
| `backend/src/services/training_finalize_service.py` | (NEW) rebuild + atomic export |
| `backend/src/services/checkpoint_retention.py` | (NEW) retention policy core |
| `backend/src/workers/training_finalize_tasks.py` | (NEW) finalize Celery task |
| `backend/src/workers/prune_checkpoints.py` | (NEW) scheduled + manual pruner |
| `backend/alembic/versions/b4d19f0c73ae_*.py` | (NEW) `finalized_from_step` |
| `backend/src/services/checkpoint_service.py` | (extend) shared sync deleter, `is_best` guard |
| `backend/src/api/v1/endpoints/trainings.py` | (extend) 4 routes + control action |
| `backend/src/core/celery_app.py` | (extend) routes, beat, autodiscovery |
| `backend/tests/unit/test_training_finalize.py` | (NEW) discovery, dims, reachability |
| `backend/tests/unit/test_checkpoint_retention.py` | (NEW) policy, table-driven |
| `backend/tests/unit/test_finalize_and_prune_execution.py` | (NEW) real files, real deletes |
| `backend/tests/unit/test_finalize_wiring.py` | (NEW) behavioural wiring + API guards |
| `frontend/src/components/training/TrainingCard.tsx` | (extend) buttons, badge, escalation |
| `frontend/src/components/panels/SettingsPanel.tsx` | (extend) Storage tab |
| `manual/docs/core-workflow/training-lifecycle.md` | (NEW) user documentation |

---

## Coverage audit (instruct 007)

- **Data** — migration + model + schema; applied and reversible. ✅
- **API** — 4 new routes + 1 action; guards return actionable 409s. ✅
- **MCP** — none added. Finalize is a UI/operator action; no agent surface was
  requested. *Recorded as a deliberate omission, not an oversight.*
- **UI + State** — buttons, badge, Storage tab, store actions, error surfacing. ✅
- **Tests** — 141 backend + 105 frontend; 24 mutation controls. ✅
- **Docs** — PPRD row 21/§3.21, PADR IDL-39, this chain, Docusaurus page +
  reference updates. ✅
- **Acceptance** — ⚠️ hardware run outstanding (see Phase 6).
- **Security** — the new routes are unauthenticated, consistent with the rest of
  the API, but they are the first that permanently delete multi-GB artifacts.
  The Storage tab is not PIN-gated while API Keys is. **Recorded as tracked
  debt** for a deliberate decision.

**Contract-crosses-to-miLLM:** no. The artifact produced is the existing
`community_format/` export that miLLM and Neuronpedia already consume; no
schema, no vendored contract, no mirror to keep in sync.

---

## Recorded follow-up debt

1. Retention is row-driven, so orphaned `checkpoint_*/` directories with no
   database row are never reclaimed. A disk-sweep reconciliation pass would
   close it.
2. Neither new task creates a `task_queue` row, so finalize/prune do not appear
   in Active Operations.
3. The prune result (`files_failed`, `bytes_freed`) is not polled by the UI.
4. Destructive routes are unauthenticated and the Storage tab is not PIN-gated.
