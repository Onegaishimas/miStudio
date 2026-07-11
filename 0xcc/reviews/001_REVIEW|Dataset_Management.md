# Feature Review: 001 — Dataset Management

**Reviewed:** 2026-07-11
**Reviewer:** deep code review (doc↔code accuracy + bug/correctness + quality/tech-debt)
**Feature docs:** [FPRD v1.0](../prds/001_FPRD|Dataset_Management.md) (2025-12-05), FTDD, FTID
**Verdict:** Solid, well-tested feature. **1 real bug (P1)**, **1 latent bug (P2)**, plus notable doc drift and quality cleanups. No changes made — findings only.

---

## 1. Scope

**Backend read in full:** `models/dataset.py` (220), `models/dataset_tokenization.py` (242), `services/dataset_service.py` (489), `services/tokenization_service.py` (1034), `workers/dataset_tasks.py` (1530), `api/v1/endpoints/datasets.py` (1421), `schemas/dataset.py` (286), plus dataset routing in `core/celery_app.py`, dataset funcs in `workers/websocket_emitter.py`, and dataset retry in `api/v1/endpoints/task_queue.py`.
**Frontend read:** `stores/datasetsStore.ts` (472), `hooks/useDatasetProgress.ts` (167), `hooks/useTokenizationWebSocket.ts` (108), `components/datasets/DatasetCard.tsx` (188), `components/panels/DatasetsPanel.tsx` (204).
**Tests present:** unit (`test_dataset_service`, `test_dataset_progress`, `test_tokenization_service`, `test_tokenization_metadata`), integration (`test_dataset_workflow`, `test_dataset_cancellation`, `test_tokenize_preview`), api (`test_datasets`).

---

## 2. Doc ↔ Code Accuracy

The FPRD is **v1.0 from 2025-12-05 and has not tracked the implementation since**. It predates the entire per-model multi-tokenization redesign. Divergences:

### 2.1 Data model — substantially wrong

| PRD claims (§6) | Reality |
|---|---|
| `datasets.status` = `pending, downloading, ready, failed` | Enum `DatasetStatus` = `downloading, processing, ready, error` (no pending/failed). |
| `datasets.file_path` | Column is `raw_path`. |
| `datasets.metadata` (JSONB) | Column is `metadata` in DB but mapped to attribute `extra_metadata`; `metadata` is reserved by SQLAlchemy. |
| — | PRD omits real columns: `source`, `hf_repo_id`, `num_samples`, `size_bytes`, `tokenization_filter_enabled/mode/junk_ratio_threshold`. |
| `dataset_tokenizations` keyed by `tokenizer_name, stride, num_tokens, num_sequences, vocab_size, token_file_path, statistics` | Real PK is a **string** `tok_{dataset}_{model}_{maxlen}`; keyed on `(dataset_id, model_id, max_length)` unique constraint. Columns are `model_id` (FK), `tokenizer_repo_id`, `tokenized_path`, `avg_seq_length`, `status`, `progress`, `celery_task_id`, `remove_all_punctuation`, `custom_filter_chars`. No `stride`/`num_sequences`/`statistics` columns. |

### 2.2 API endpoints — several wrong or missing

| PRD (§5) | Reality |
|---|---|
| `POST /datasets/{id}/download` | Actual route is `POST /datasets/download` (creates the record itself). |
| `POST /datasets/{id}/cancel` | Actual is `DELETE /datasets/{id}/cancel`. |
| `GET /datasets/{id}/statistics` | **Does not exist.** Stats live inside tokenization records / `GET .../tokenizations`. |
| `POST /datasets/{id}/tokenize` | Exists, but PRD omits `model_id` as the required body field. |
| (not in PRD) | Real extras: `GET /{id}/task-status`, `POST /tokenize-preview`, `GET/DELETE /{id}/tokenizations`, `GET/DELETE /{id}/tokenizations/{model_id\|tok_id}`, `POST /{id}/tokenizations/{tok_id}/cancel`, `DELETE /{id}/tokenization` (clear). |

### 2.3 WebSocket channels — wrong naming

| PRD (§7) | Reality |
|---|---|
| Channel `dataset/{id}`, events `download_progress`, `download_completed`, `tokenization_progress`… | Channel `datasets/{id}/progress`, events `dataset:progress` / `dataset:completed` / `dataset:error`. Tokenization uses `datasets/{id}/tokenization/{tok_id}` with `tokenization:progress` / `tokenization:status`. |

### 2.4 Key files — 2 of 6 frontend paths wrong

`TokenizationStatsModal.tsx` and `useDatasetWebSocket.ts` **do not exist**. Real equivalents: `DatasetDetailModal.tsx`, `TokenizationsList.tsx`, `TokenizationPreview.tsx`, `useDatasetProgress.ts` + `useTokenizationWebSocket.ts`. `DatasetTokenization` is its own model file (PRD lists it only as a SQL block).

**Recommendation:** FPRD 001 needs a v2.0 rewrite to reflect the per-model multi-tokenization architecture. It is the most doc-drifted feature seen so far — nearly every reference section (data model, endpoints, channels, files) is inaccurate.

---

## 3. Findings (severity-ranked)

### P1 — Real bug

**F1. `NameError` on successful tokenization-cancel response**
[datasets.py:1417](../../backend/src/api/v1/endpoints/datasets.py#L1417) — `cancel_dataset_tokenization` returns `"model_id": model_id`, but `model_id` is never defined in this function (the path param is `tokenization_id`; the value lives on `tokenization.model_id`). Every successful cancel raises `NameError: name 'model_id' is not defined` **after** the DB commit, WS emit, lock release, and file-cleanup have all run — so the cancel *succeeds server-side* but the client gets a 500. Confusing and makes the UI think cancel failed.
**Fix:** `"model_id": tokenization.model_id`.

### P2 — Latent bug

**F2. `DatasetUpdate.status` regex accepts a value the enum rejects**
[schemas/dataset.py:42](../../backend/src/schemas/dataset.py#L42) allows `status` in `^(downloading|ingesting|processing|ready|error)$`, but `DatasetStatus` has no `INGESTING`. In [dataset_service.py:232](../../backend/src/services/dataset_service.py#L232), `update_dataset` does `DatasetStatus[value.upper()]` — so a PATCH with `status="ingesting"` passes schema validation then raises `KeyError` (unhandled → 500). No current caller sends it, hence latent, but it's a validation/enum contract mismatch waiting to fire.
**Fix:** drop `ingesting` from the regex (and `pending`/`failed` are already absent — good), or better, validate against `DatasetStatus` enum members directly.

### P2 — Correctness / robustness

**F3. Redis lock only released on the tokenize *success* path**
The tokenize task acquires the lock in the endpoint ([datasets.py:482](../../backend/src/api/v1/endpoints/datasets.py#L482)) and releases it in the worker on success ([dataset_tasks.py:1140](../../backend/src/workers/dataset_tasks.py#L1140)). The `except BaseException` handler (line 1171+) has a recovery path but I did **not** see a `release_redis_lock()` call on every failure branch — if tokenization raises before the success block, the 1-hour lock TTL is the only thing that frees it, blocking re-tokenization of that dataset for up to an hour. Verify the failure handler calls `release_redis_lock(dataset_id)` on all exits (including the SystemExit/signal path). The lock has a TTL so it's self-healing, but a user retrying after a failure hits a spurious 409.

**F4. `cancel_dataset_download` task can't revoke the running task**
[datasets.py:714](../../backend/src/api/v1/endpoints/datasets.py#L714) calls the cancel *synchronously* and notes "We don't have task_id stored, so we can't revoke the specific task." For a download, `task_id` *is* stored in `dataset.extra_metadata['task_id']` (set at [datasets.py:417](../../backend/src/api/v1/endpoints/datasets.py#L417)), so the running download Celery task is **never actually revoked** — cancel just deletes files and flips status while the worker may still be downloading, racing the cleanup. (Tokenization cancel, by contrast, correctly revokes via `celery_task_id` — F-note: inconsistent between the two paths.)
**Fix:** pass `metadata['task_id']` into the cancel task and revoke it, mirroring the tokenization-cancel path.

### P3 — Quality / tech-debt

- **F5. Debug `print()` + token-length logging in the request path.** [datasets.py:380](../../backend/src/api/v1/endpoints/datasets.py#L380) prints `token_provided`/`token_length` for every download; `dataset_tasks.py` and `tokenization_service.py` have many `print("[TOKENIZER DEBUG] …")` / `print("[FILTER_CONFIG] …")` lines. These should be `logger.debug`. Logging even the *length* of an access token is an unnecessary secret-adjacent signal.
- **F6. Duplicated tokenization inner-loop logic.** `_TokenizationMapper.__call__` (multiprocess) and the `tokenize_function` closure (single-process) in `tokenization_service.py` re-implement the same clean→tokenize→filter sequence. Extract a shared helper to prevent the two paths drifting (they already differ: only the mapper applies `enable_filtering`; the single-process closure does not filter at all — a subtle behavioral gap if `progress_callback` ever forces single-process with filtering enabled).
- **F7. `download_dataset_task` holds no lock but `create_dataset` + queue + metadata-update is 3 sequential commits.** Minor; the `get_dataset_by_repo_id` uniqueness check (line 384) is not transactional with the insert, so two simultaneous downloads of the same repo can both pass the check and create duplicate rows. Low likelihood (UI-driven), but the tokenize path uses a Redis lock for exactly this reason while download does not.
- **F8. `get_dataset_samples` runs heavy `load_from_disk` / HF loads inside the request handler** (up to full-dataset load for `len()`), no size guard. For large datasets this blocks an event-loop worker. Consider offloading or caching. (Bytes-safe `sanitize_value` handling here is good — that was a prior fix.)
- **F9. Inconsistent cancel verb/semantics.** Dataset cancel is `DELETE /{id}/cancel` (returns 200 + body); tokenization cancel is `POST /{id}/tokenizations/{id}/cancel`. Mixed REST conventions across the same feature.

---

## 4. Test Coverage Notes

Good breadth: service unit tests, tokenization service + metadata, cancellation integration, workflow integration, preview, and API endpoint tests. Gaps worth noting:
- **No test exercises the tokenization-cancel success response** → F1's `NameError` slipped through (the integration cancel test likely asserts DB state, not the HTTP body).
- No test for the `status="ingesting"` validation/enum mismatch (F2).
- No test asserting lock release on tokenization *failure* (F3).
- No test that a download cancel actually revokes the Celery task (F4).

---

## 5. Summary — if you fix three things

1. **F1** — one-liner `NameError` on tokenization-cancel (real, user-visible 500).
2. **F4** — make download-cancel actually revoke the running task (currently races cleanup against an unstopped download).
3. **F2/F3** — tighten the status enum/regex contract and guarantee lock release on tokenize failure.

And schedule an **FPRD 001 v2.0 rewrite** — the doc's data model, endpoints, channels, and file list are all stale.
