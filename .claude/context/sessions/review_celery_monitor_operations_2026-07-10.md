# Multi-Agent Review: Celery Workers, Job Progress & Monitor Operations Sections

**Session ID:** review_20260710
**Date:** 2026-07-10
**Type:** review (deep, multi-agent)
**Scope:** Celery worker functionality (`backend/src/workers/`), job progress indicators (WebSocket emission + frontend tracking), Monitor page "Active Operations" / "Failed Operations" sections

---

## Executive Summary

The Celery task architecture is generally well-designed (queue isolation, GPU cleanup hooks, throttled emission in dataset tasks). However, the review found **one production-breaking bug** (system-metrics WebSocket emission has been silently 403-ing since the internal-token hardening), **two design-level gaps** that make the Monitor page's operations sections misleading (most job types never appear in Active Operations; Failed Operations misses failed labeling/push/training/extraction jobs), and **a retry-ghost lifecycle bug** in the task_queue table. Several smaller correctness and robustness issues follow.

---

## Findings by Severity

### P0 — Critical

**F1. System-metrics WebSocket emission silently broken (403 Forbidden)**
- `BackgroundMonitor._emit_to_channel` ([background_monitor.py:180-187](../../../backend/src/services/background_monitor.py)) POSTs to `/api/internal/ws/emit` **without the `X-Internal-Token` header**.
- The endpoint ([main.py:151](../../../backend/src/main.py)) rejects any request without a valid token → **every `system/*` metric emission fails with 403**. Celery workers' `emit_progress` sends the header correctly; only BackgroundMonitor was missed when the token check was added (security hardening, ~May 2026).
- Failure is silent: `_emit_to_channel` returns False, no error log fires (403 is not an exception).
- **Compounding frontend bug:** the polling fallback is keyed to *connection* state, not *data freshness*. `setIsWebSocketConnected(true)` stops polling as soon as the socket connects ([systemMonitorStore.ts:273-286](../../../frontend/src/stores/systemMonitorStore.ts)) — even though no `system:metrics` events will ever arrive. Result: Monitor page data freezes after the initial fetch; "Historical Trends" draws straight lines between stale endpoints (visible in production screenshots).
- **Fix:** (a) add the token header in BackgroundMonitor — or better, since it runs in the same process as FastAPI, call `ws_manager.emit_event()` directly and skip HTTP entirely; (b) add a data-staleness watchdog on the frontend (e.g., no metrics event for 10s while connected → resume polling).

**F2. "Active Operations" never shows most operation types**
- `task_queue` rows are **only created inside failure handlers** for 3 task types: model download ([model_tasks.py:424-467](../../../backend/src/workers/model_tasks.py)), dataset download, tokenization ([dataset_tasks.py:374,1237](../../../backend/src/workers/dataset_tasks.py)). `TaskQueueService.create_task_entry` is never called at dispatch time by any worker.
- `/api/v1/task-queue/active` supplements with raw-SQL unions for labeling jobs and Neuronpedia pushes only.
- **Net effect:** running trainings, extractions, SAE downloads/uploads, steering runs, NLP analysis, enhanced labeling, model/dataset downloads (first attempt), tokenizations **never appear in Active Operations**. The section shows "No active operations" while a training is saturating the GPU.
- **Fix options:** (a) create task_queue entries at dispatch and update on status transitions (single writer per lifecycle); or (b) make `/active` a read-only federation over the real job tables (trainings, extraction_jobs, etc.), which avoids dual-writing state.

**F3. Retry ghost entries in task_queue**
- `increment_retry_count` sets `status="queued"` and the retry endpoint dispatches a new Celery task — but **no success path ever updates the task_queue row**. On a successful retry the row stays `queued` forever → a permanent phantom "Queued" card in Active Operations (with stale progress/started_at, which are also not reset on retry).
- The failure handler's "existing queued entry" check handles retry-failure, but retry-success leaks.
- **Fix:** on task success, mark any queued/running task_queue row for the entity as `completed` (mirror of the failure-handler lookup).

### P1 — High

**F4. "Failed Operations" misses most failure types**
- `/failed` reads only `task_queue.status='failed'`. Failed labeling jobs, Neuronpedia pushes, trainings, and extractions live in their own tables and never surface here (labeling/push are unioned into `/active` but not `/failed` — asymmetric).
- The two frontend sections' label maps have already drifted: ActiveOperationsSection knows `labeling`/`neuronpedia_push`; FailedOperationsSection doesn't.

**F5. Training pause/cancel race — user action can be silently lost**
- `update_training_progress` ([training_tasks.py:52-85](../../../backend/src/workers/training_tasks.py)) unconditionally writes `status = RUNNING` every `log_interval` steps. If the API sets PAUSED/CANCELLED between the loop-top status check and the log-interval write, the pause/cancel is clobbered and training continues.
- **Fix:** stop writing status in the progress update, or use a guarded `UPDATE ... WHERE status='running'`.

**F6. Training loop queries the DB every step**
- The pause/cancel check ([training_tasks.py:1079-1088](../../../backend/src/workers/training_tasks.py)) opens a session and queries `Training` on **every** step — up to 100k round-trips per run, on the GPU hot path.
- **Fix:** check every N steps (e.g., 25) or on a wall-clock interval (e.g., every 2s).

**F7. task_queue rows are never cleaned up**
- `TaskQueueService.cleanup_completed_tasks` exists but is **never scheduled or called** (dead code). Beat schedule has cleanup for stuck extractions/trainings/activations/enhanced-labeling + GPU watchdog only. Combined with F3, the table only grows.

**F8. Shared `loading`/`error` state across both Monitor sections**
- ActiveOperationsSection (5s poll) and FailedOperationsSection (30s poll) share one `taskQueueStore` `loading`/`error` flag. The pollers race: each fetch flips the global spinner in *both* sections, and an error from one renders in both.
- **Fix:** per-fetch loading/error keys (e.g., `activeLoading`, `failedLoading`), or move to react-query-style per-request state.

### P2 — Medium

**F9. Extraction task's generic failure path may not persist FAILED**
- [extraction_tasks.py:136-152](../../../backend/src/workers/extraction_tasks.py) emits a `failed` WS event in the outer except, but only the two path-validation branches set `ExtractionJob.status = FAILED` in the DB. If the service raises after that, the DB row can stay EXTRACTING until `cleanup_stuck_extractions` (10-min cadence) catches it → UI refresh shows the job "extracting" again after the failure toast. Also: `extraction_job` may be unbound in the except handler (silently swallowed by the inner try). Verify the service persists FAILED on all raise paths; otherwise set it in the task's except.
- Also: the whole extraction runs inside one `with self.get_db()` — a DB connection held for the full GPU run (pool pressure).

**F10. Steering worker config contradicts its own comments; redelivery risk**
- [celery.sh:405-431](../../../backend/celery.sh) starts the steering worker with `--pool=solo --max-tasks-per-child=1` and comments "Recycle worker after EVERY task" — but solo pool **ignores** max-tasks-per-child (this was the documented root cause of the 2026-01 steering hang; the `finally`-block state reset was the compensating fix). The misleading comment should be corrected.
- `time_limit=180` "SIGKILL guaranteed" docstring in steering_tasks.py is questionable under solo pool (no child process to kill); the in-service watchdog is the real guard — verify and document which mechanism actually fires.
- `acks_late=True` + global `task_reject_on_worker_lost=True`: if a steering task kills the worker (OOM), the message is redelivered and re-executed on restart → potential crash loop on a pathological input.

**F11. Emitter connection churn + no retry for terminal events**
- `emit_progress` creates a **new `httpx.Client` (new TCP connection) per event** with a blocking 5s timeout, called from GPU hot paths (training every `log_interval`, dataset ops at 0.5s throttle, deletion batches every 500 rows).
- Default `retries=0` means a single 5s blip drops **terminal** events (`training:completed`, `training:failed`) — call sites only log a warning; the frontend then shows a running job until manual refresh. Terminal events deserve `retries=2`+; progress events are fine to drop.
- **Fix:** module-level pooled client; per-event-class retry policy.

**F12. Fragile payload-shape dispatch in system monitor hook**
- `useSystemMonitorWebSocket.handleMetrics` distinguishes CPU/memory/disk/network by sniffing field names (`data.percent !== undefined && data.count !== undefined`…). Adding a field to one payload can misroute another. Include an explicit `metric_type` (or the channel) in the payload.

**F13. Two competing polling controllers for the monitor fallback**
- Both `systemMonitorStore.setIsWebSocketConnected` and `SystemMonitor.tsx`'s `hasStartedPolling` effect ([SystemMonitor.tsx:92-118](../../../frontend/src/components/SystemMonitor/SystemMonitor.tsx)) start/stop polling on the same signal. Redundant and easy to desynchronize; consolidate in the store.

**F14. DownloadProgressMonitor thread can outlive `stop()`**
- `stop()` joins with a 2s timeout; `get_directory_size` on a 40GB model dir can exceed that mid-iteration, so the daemon thread can commit a stale progress value and emit "downloading" **after** the main thread commits READY/100%. Guard writes with `if not self.running: break` immediately before DB/WS calls, or use a threading.Event.

**F15. Task-queue endpoints: N+1 queries and no pagination**
- `/`, `/active`, `/failed` call `get_entity_info` per row (one query each) and return unbounded lists. Fine today; combine with F7 growth and it degrades.

### P3 — Low

- **F16.** ActiveOperations shows "Started: {created_at}" when `started_at` is null — mislabeled for queued rows.
- **F17.** FailedOperationsSection uses browser `confirm()`/`alert()` — inconsistent with the app's modal patterns (RetryConfirmDialog exists right next to it).
- **F18.** `/api/internal/ws/emit` `if not all([channel, event, data])` returns a `(dict, 400)` tuple (serialized as 200 array) and treats empty-dict `data` as missing.
- **F19.** `handleCompleted` stamps `completed_at` with the client clock instead of a server timestamp.
- **F20.** Monitor sections keep polling when the tab is hidden (5s/30s) — use `document.visibilityState`.
- **F21.** `system_monitor_tasks.py` Celery task is dead code (beat entry disabled in favor of BackgroundMonitor) — remove or clearly gate.
- **F22.** "Historical Trends (Last Hour)" buffer is `maxDataPoints: 3600` with comment "1h at 1s intervals", but metrics arrive every 2s (= 2h window); buffer also resets on page navigation, so "Last Hour" only holds if the page stayed open.
- **F23.** `TaskQueueService.update_task_status` `if error_message:` can't clear a previous error with an empty string.
- **F24.** `get_queue_lengths` creates queues as a side effect via `SimpleQueue` declaration.

---

## Agent Perspectives

**Product Engineer:** The Monitor page promises a "unified view of all background operations" but delivers ~3 of 10 job types (F2/F4). This directly contradicts the long-standing roadmap item "unified operations dashboard." Recommend making `/active` and `/failed` read-only federations over the real job tables — no dual-write, no lifecycle bugs, complete coverage.

**QA Engineer:** F1 is a regression introduced by the security-hardening change (token check added; one internal caller missed). No integration test covers the BackgroundMonitor → emit endpoint path, so a 403 regression was invisible. F5's guarded-update fix is one line. Shared store state (F8) is the same class of bug previously flagged in steeringStore.

**Architect:** The task_queue design has no single owner of lifecycle state: created-on-failure by workers, mutated by the retry endpoint, never touched on success (F2/F3/F7). Either commit to task_queue as a first-class job ledger (create at dispatch, update on every transition, scheduled cleanup) or demote it to a failure-retry ledger and build Active Operations from source-of-truth tables. The in-process BackgroundMonitor calling itself over HTTP (F1) should just call `ws_manager` directly.

**Test Engineer:** Highest-risk untested paths: (1) BackgroundMonitor emission (assert 200 with token wiring), (2) task_queue lifecycle across fail → retry → success (would have caught F3), (3) pause-during-log-interval race (F5 — simulate status write between loop check and progress write), (4) frontend staleness fallback (WS connected but silent → polling resumes). Suggest a `last_metric_received_at` gauge in the store plus a test that advances fake timers.

---

## Recommended Fix Order

1. **F1** — add token header / direct `ws_manager` call + frontend staleness watchdog (restores live Monitor page).
2. **F3 + F7** — mark retried rows completed on success; schedule `cleanup_completed_tasks` in beat.
3. **F5** — guarded status update in training progress write.
4. **F2/F4** — decide task_queue architecture (ledger vs. federation) and implement; align both sections' label maps.
5. **F8, F6, F11** — store state split; throttled cancel check; pooled emitter client + terminal-event retries.
6. Remainder as cleanup batch.

## Fixes Applied (same session, 2026-07-10)

All findings were addressed in the working tree immediately after the review:

- **F1/F12:** `X-Internal-Token` header added to `BackgroundMonitor._emit_to_channel` (+ non-200 logging); every metrics payload now carries `metric_type`, and the frontend handler dispatches on it (shape-sniffing kept as fallback).
- **F1b/F13:** `SystemMonitor.tsx` duplicate polling controller replaced with a data-staleness watchdog: WS connected + no data for 10s → polling resumes. Store remains the single connect/disconnect controller.
- **F2/F4:** `/api/v1/task-queue/active` and `/failed` now federate trainings, extraction jobs, labeling jobs, and Neuronpedia pushes (read-only, `can_retry=false`; schema + frontend type updated; Failed section hides Retry/Delete for federated rows). *Remaining follow-up: SAE downloads, NLP analysis, enhanced labeling, steering not yet federated.*
- **F3:** `mark_task_queue_entries_completed()` helper in base_task.py called from model-download, dataset-download, and tokenization success paths; retry endpoint also resets progress/started_at.
- **F5:** progress writer no longer overwrites PAUSED/CANCELLED/FAILED/COMPLETED status.
- **F6:** pause/cancel DB check throttled to every 25 steps.
- **F7:** new `cleanup_task_queue` beat task (hourly): completed >7d and stale queued/running >7d deleted.
- **F8:** taskQueueStore split into `activeLoading/activeError` and `failedLoading/failedError`.
- **F9:** extraction task persists FAILED in its outer exception handler.
- **F10:** celery.sh and steering_tasks.py comments corrected (solo pool ignores max-tasks-per-child; time limits not enforced by solo — watchdog is the real guard).
- **F11:** websocket_emitter uses a module-level pooled `httpx.Client`; `training:completed`/`training:failed` emit with `retries=2`; completed event carries server `completed_at` (consumed by F19 fix in useTrainingWebSocket).
- **F14:** DownloadProgressMonitor uses `threading.Event` (interruptible sleep, stop re-checked after directory walk; `running` property with setter kept for test compat).
- **F15:** list queries capped at 200 rows.
- **F16/F17/F20:** "Started:"/"Created:" label fixed; browser `confirm()`/`alert()` replaced with two-click inline confirm; both sections skip polling while the tab is hidden.
- **F18/F23:** internal emit endpoint returns proper HTTP 400 (empty-dict data allowed); `update_task_status` can clear error_message.
- **F21:** dead `system_monitor_tasks.py` module removed along with its route/autodiscover entries.
- **F22:** history buffer corrected to 1800 points (1h at the 2s cadence).
- **F24:** deliberately not changed — queue declaration via SimpleQueue is harmless with `task_create_missing_queues=True`, and a Redis-specific `llen` rewrite would couple the helper to one broker.

Tests: emitter unit + integration tests updated to the pooled-client pattern (24 passed); frontend type-check and build clean; frontend vitest failures (98) verified pre-existing on the unmodified tree.

## Files Reviewed

Backend: `core/celery_app.py`, `workers/base_task.py`, `workers/training_tasks.py` (progress/loop/failure paths), `workers/model_tasks.py` (progress monitor, failure handler), `workers/extraction_tasks.py`, `workers/steering_tasks.py` (structure), `workers/dataset_tasks.py` (emission/throttle/task_queue), `workers/system_monitor_tasks.py`, `workers/websocket_emitter.py`, `services/background_monitor.py`, `services/task_queue_service.py`, `models/task_queue.py`, `api/v1/endpoints/task_queue.py`, `main.py` (emit endpoint), `celery.sh`, `start-celery-worker.sh`.
Frontend: `components/SystemMonitor/{SystemMonitor,ActiveOperationsSection,FailedOperationsSection,RetryConfirmDialog}.tsx`, `stores/{taskQueueStore,systemMonitorStore}.ts`, `hooks/{useSystemMonitorWebSocket,useTrainingWebSocket}.ts`.
