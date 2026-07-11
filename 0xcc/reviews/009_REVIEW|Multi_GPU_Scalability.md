# Feature Review: 009 — Multi-GPU Scalability

**Reviewed:** 2026-07-11
**Reviewer:** deep code review (doc↔code accuracy + bug/correctness + quality/tech-debt)
**Feature docs:** [FPRD v1.1](../prds/009_FPRD|Multi_GPU_Scalability.md) (2026-05-09), FTDD, FTID
**Verdict:** The **most self-aware FPRD** (honestly marks Phase 3/DDP as planned) — but it still contains a **material overclaim (training GPU selection marked ✅ Complete when it doesn't exist)** and a **wholly fictional data model + config section**. Phase 1 (monitoring) is genuinely complete and solid; Phase 2 (GPU routing) is complete for *extraction only*, not training. No code bugs — this is a doc-accuracy review of a partially-built feature.

---

## 1. Scope

**Backend verified:** `services/gpu_monitor_service.py` (484, method inventory), `workers/model_tasks.py` (extraction `gpu_id` path), `workers/training_tasks.py` (GPU-selection search), `workers/gpu_watchdog_task.py` (328, scheduled), `schemas/model.py` + `schemas/training.py` (gpu_id presence), `models/training.py` (column check), `core/config.py` (settings check), `core/celery_app.py` (watchdog schedule), `celery.sh` (steering GPU pinning). This feature spans 002/003/008, so verification was cross-cutting rather than a single module read.
**Tests present:** **NONE** specific to multi-GPU (shares the monitoring zero-coverage from 008).

---

## 2. Doc ↔ Code Accuracy

FPRD v1.1 (May 2026) is commendably honest in the FR tables (Phase 3 = Planned, several Partials). But three problems:

### 2.1 Material overclaim — training GPU selection does NOT exist

| PRD claims | Reality |
|---|---|
| FR-1.3 "GPU selection for training jobs — ✅ Complete (Dec 2025)" | **False for training.** There is **zero `gpu_id` / `gpu_ids` / `CUDA_VISIBLE` / `device_id`** in `schemas/training.py`, `training_service.py`, `trainings.py`, or `training_tasks.py`. Training always runs on the default CUDA device. |
| FR-2.2 "`gpu_id` param wired through API → schema → worker" | **True for extraction only.** `schemas/model.py:137` has `gpu_id`, and `model_tasks.extract_activations(gpu_id=...)` uses it with per-GPU memory cleanup ([model_tasks.py:626,674](../../backend/src/workers/model_tasks.py#L626)). But this is the *extraction* path (Feature 002/004), not training. |
| §10 Phase 2 "GPU selection for extraction jobs" | Correctly scoped ✅ — this is the accurate line. FR-1.3's "training jobs" is the overclaim. |

**Net:** the honest status is *"GPU routing complete for extraction; training GPU selection not implemented."* FR-1.3 should be downgraded from ✅ to ❌/Planned. This matters because it's the difference between "pick a GPU for your training run" (doesn't exist) and "extraction runs on a chosen GPU" (works).

### 2.2 Data model §6 — entirely fictional

| PRD (§6) | Reality |
|---|---|
| `ALTER TABLE trainings ADD COLUMN gpu_ids INTEGER[]` / `distributed BOOLEAN` | **Neither column exists** in `models/training.py`. |
| `CREATE TABLE gpu_metrics (...)` with `job_id`, `UNIQUE(gpu_index, timestamp)` | **No `gpu_metrics` table or model exists.** GPU metrics are ephemeral — emitted over WebSocket every 2s, never persisted. There is no historical GPU-metrics storage anywhere. |

Both §6.1 and §6.2 describe schema that was never built (they're forward-looking DDL for the planned phases, presented as if current).

### 2.3 API endpoints §5 — wrong paths

| PRD (§5) | Reality |
|---|---|
| `GET /system/gpus`, `/system/gpus/{id}` | Real: `/system/gpu-list`, `/gpu-metrics`, `/gpu-metrics/all`, `/gpu-info`, `/gpu-processes`. No `/gpus` or `/gpus/{id}`. |
| `POST /trainings` "Extended with `gpu_ids` field" | **Not extended** — no `gpu_ids` (or `gpu_id`) on the training create schema (ties to 2.1). |
| `/system/metrics?view=aggregated\|per_gpu` | **No `?view=` param.** Aggregation is a frontend concern (the `viewMode: 'single'\|'compare'` toggle in `systemMonitorStore`); the backend has no view query. |

### 2.4 Config §8 — fictional

None of `multi_gpu_enabled`, `default_gpu_selection`, `monitor_view_default` exist in `core/config.py`.

### 2.5 What IS accurate

- **Phase 1 (monitoring) fully verified:** `GPUMonitorService.get_all_gpu_info()`, `get_all_gpu_metrics()`, `get_device_count()`, `is_available()`, `get_gpu_metrics(gpu_id)` all exist as claimed. Per-GPU WebSocket channels `system/gpu/{id}` work. The aggregated-vs-per-GPU toggle (`viewMode`) exists in the store.
- **`gpu_watchdog_task.py`** exists (328 lines) **and is scheduled** in Celery Beat ([celery_app.py:273](../../backend/src/core/celery_app.py#L273)) ✅ — FR-2.3's "GPU watchdog tracks usage" is real.
- **Extraction `gpu_id` routing** works with GPU validation and per-device memory cleanup ✅.

---

## 3. Findings (severity-ranked)

### P2 — Doc accuracy (this is primarily a doc-review feature)

**F1. FR-1.3 overclaims training GPU selection as Complete.** (§2.1) The single most misleading line — training has no GPU selection. Downgrade to Planned, or implement it (it would be a small change: add `gpu_id` to the training schema + `CUDA_VISIBLE_DEVICES`/`torch.cuda.device` in `train_sae_task`, mirroring the extraction path which already does it correctly).

**F2. §6 data model is fictional-as-current.** (§2.2) `gpu_ids`/`distributed` columns and the `gpu_metrics` table don't exist. Either mark §6 explicitly as "Planned schema (Phase 3)" or move it to a design-intent appendix so it isn't read as the current model.

### P3 — Quality / tech-debt

**F3. GPU metrics are never persisted — no historical GPU analysis possible.** The PRD's `gpu_metrics` table (§6.2) with `job_id` correlation is exactly what would enable "which training run used which GPU, and how hard." Today that's impossible (metrics are fire-and-forget WebSocket). This is a genuine capability gap the PRD *anticipated* but that was never built — worth tracking as a real backlog item, not just doc drift. (Relates to the PPRD roadmap "resource-job correlation" item.)

**F4. Multi-GPU functionality is scattered across three features' routes/files.** GPU listing/metrics under `/system` (008), extraction gpu_id under `/models` (002), watchdog as a standalone task, steering hard-pinned to GPU 0 in `celery.sh`. There's no single "multi-GPU" module — it's an aspect woven through others. Fine for a P2 partial feature, but means FR verification requires cross-feature grepping (as this review did). Document the cross-feature surface.

**F5. Steering is hard-pinned to `CUDA_VISIBLE_DEVICES=0`** ([celery.sh:425](../../backend/celery.sh#L425)) — so on a multi-GPU box, steering *cannot* use GPUs 1+, and can contend with a training run on GPU 0. Not wrong for a single-GPU deployment (the current K8s host has one RTX 3090), but it's an unstated single-GPU assumption that contradicts the multi-GPU framing.

---

## 4. Test Coverage Notes

No multi-GPU-specific tests. The extraction `gpu_id` path, GPU validation (index bounds), and the watchdog are untested. Given the current deployment is single-GPU, this is low-risk today, but the extraction `gpu_id` validation (rejecting an index ≥ device count) is worth one test since it's user-facing.

---

## 5. Summary — if you fix three things

1. **F1** — correct FR-1.3 (training GPU selection is *not* complete). The one materially misleading status in an otherwise honest doc. *Or* implement it (small, mirrors the working extraction path).
2. **F2** — mark §6 (data model) and §8 (config) as Planned/design-intent, not current — they describe schema that doesn't exist.
3. **F3** (backlog, not doc) — GPU-metrics persistence + job correlation is a real anticipated capability that was never built; track it explicitly.

**The good:** Phase 1 monitoring is genuinely complete and matches the doc method-for-method; the GPU watchdog is real and scheduled; extraction GPU routing works correctly with proper cleanup. The feature is honestly labeled "Partially Complete," and the honest 60% is solid. The gap is that the *specific* completeness claims (training GPU selection, the data model, the config) overstate what's built — which for a partial feature is exactly where doc precision matters most.

---

## Note on this being the final per-feature review

All nine feature reviews (001–009) are now complete. See **`0xcc/reviews/000_SYNTHESIS.md`** for the cross-feature analysis: batch-fix opportunities, interdependencies (esp. the extraction-architecture tangle spanning 002/004/005), the whole-app FPRD doc-refresh plan, and a consolidated severity-ranked backlog.
