# Feature Review: 002 — Model Management

**Reviewed:** 2026-07-11
**Reviewer:** deep code review (doc↔code accuracy + bug/correctness + quality/tech-debt)
**Feature docs:** [FPRD v1.1](../prds/002_FPRD|Model_Management.md) (2026-03-21), FTDD, FTID
**Verdict:** Core download/quantize/architecture flow is solid, and `layer_discovery.py` is genuinely well-built. But **2 API endpoints are completely broken (P0)** from a bad import, and the feature's endpoint scope has ballooned to include the entire extraction API (which belongs to Feature 004). No changes made — findings only.

---

## 1. Scope

**Backend read in full:** `models/model.py` (79), `services/model_service.py` (415), `api/v1/endpoints/models.py` (1278), `ml/layer_discovery.py` (444), `schemas/model.py` (399, key sections), plus model routing/naming in `core/celery_app.py`, model funcs in `workers/websocket_emitter.py`, model failure-handler in `workers/model_tasks.py` (reviewed earlier this session). `workers/model_tasks.py` (1149) skimmed for the download/progress/quantize path.
**Frontend read:** `stores/modelsStore.ts` (583, key sections), `components/models/ModelCard.tsx` (340, status handling), plus confirmed presence of `ModelsPanel`, `ModelDownloadForm`, `ModelPreviewModal`, `ModelArchitectureViewer`, `useModelProgress.ts` (419).
**Tests present:** `test_model.py`, `test_model_service.py`, `test_model_loader.py`, `test_model_download_progress.py`, integration `test_model_workflow.py`, `test_model_cleanup.py`. **No test covers extraction cancel/retry** (see F1).

---

## 2. Doc ↔ Code Accuracy

FPRD is v1.1 (2026-03-21) — newer than 001's, and the key-files list is accurate. But the reference sections still drift:

### 2.1 Data model — wrong PK type, status enum, quantization values

| PRD claims (§6) | Reality |
|---|---|
| `id UUID PRIMARY KEY` | `id String(255)` = `m_{uuid_hex[:8]}` (not UUID). |
| `status` = `pending, downloading, ready, failed` | Enum = `downloading, loading, quantizing, ready, error` (no pending/failed; adds loading/quantizing). |
| `quantization` = `none, bnb-4bit, bnb-8bit` | Enum `QuantizationFormat` = `FP32, FP16, Q8, Q4, Q2`. Completely different vocabulary. |
| `architecture JSONB` | Split: `architecture String(100)` (family name) + `architecture_config JSONB` (the dims). |
| `file_path`, `size_bytes` | Real: `file_path`, `quantized_path`, `memory_required_bytes`, `disk_size_bytes`, `params_count`. |

### 2.2 API endpoints — download path wrong, preview missing, extraction surface undocumented

| PRD (§5) | Reality |
|---|---|
| `POST /models/{id}/download` | Actual is `POST /models/download` (creates the record itself). |
| `POST /models/preview` | **Does not exist.** No preview endpoint in `models.py` at all — the "preview before download" is a frontend-only HF fetch. |
| (not in PRD) | Real extras belonging conceptually to *Model Management*: `POST /{id}/redownload`, `DELETE /{id}/cancel`, `GET /local-cache/list`, `GET /tasks/{task_id}`. |
| (not in PRD) | **Entire extraction API lives in models.py**: `POST /{id}/estimate-extraction`, `POST /{id}/extract-activations`, `GET /{id}/extractions[/active]`, `POST /{id}/extractions/{eid}/cancel`, `.../retry`, `DELETE /{id}/extractions`. This is Feature 004 (Feature Discovery) territory but physically routed under `/models`. |

### 2.3 WebSocket channels — wrong naming

| PRD (§7) | Reality |
|---|---|
| Channel `model/{id}`, events `download_progress`/`download_completed`/`download_failed` | Channel `models/{id}/progress`, events `model:progress`/`model:completed`/`model:error`. Extraction uses `models/{id}/extraction`. |

**Recommendation:** FPRD 002 needs (a) data-model/endpoint/channel corrections, and (b) an architectural decision: the extraction API is physically under `/models` but semantically Feature 004. Either document the coupling in both FPRDs or split the routes. This overlap will recur when we review 004.

---

## 3. Findings (severity-ranked)

### P0 — Broken endpoints

**F1. `cancel_extraction` and `retry_extraction` import a non-existent module → `ModuleNotFoundError`**
[models.py:968](../../backend/src/api/v1/endpoints/models.py#L968), [models.py:981](../../backend/src/api/v1/endpoints/models.py#L981), [models.py:1075](../../backend/src/api/v1/endpoints/models.py#L1075) do `from ....models.extraction import ActivationExtraction` / `ExtractionStatus`. **There is no `src/models/extraction.py`** — the class lives in `src/models/activation_extraction.py` (which the top of the same file imports correctly at line 17). Any call to `POST /{id}/extractions/{eid}/cancel` or `.../retry` raises `ModuleNotFoundError` before doing anything. Both endpoints are 100% dead. No test exercises them, which is why it shipped. Likely introduced during a refactor that renamed the module but missed these three function-local imports.
**Fix:** change the three imports to `....models.activation_extraction`. One-line each. **Add a test** hitting both endpoints.

### P1 — Correctness

**F2. Detached-instance access in `cancel_extraction` WS emit**
[models.py:1013-1022](../../backend/src/api/v1/endpoints/models.py#L1013) — the `extraction` object is loaded and mutated inside `with get_sync_db() as sync_db:` (closes at line 1011), then the WS emit at 1019 reads `extraction.progress` *after* the session closed. `progress` was loaded so it likely works via the identity map, but this is exactly the detached-SQLAlchemy pattern that breaks under expire-on-commit. (Moot until F1 is fixed, since the function can't currently run — but fix alongside.)

**F3. Model download-cancel never revokes the running task (same pattern as 001-F4)**
[models.py:681](../../backend/src/api/v1/endpoints/models.py#L681) — `cancel_model_download` comments "We don't have task_id stored, so we can't revoke the specific task" and calls `cancel_download(model_id)` synchronously. The download Celery task id is *not* stored on the model (unlike datasets, which stash it in metadata), so a cancel during an active download flips status + deletes files while the worker keeps downloading — racing the cleanup. Consider storing the task id at dispatch (`download_and_load_model.delay(...)` returns it) and revoking it, mirroring the extraction-cancel path which *does* revoke correctly (line 996).

### P2 — Correctness / robustness

**F4. `list_model_extractions` does N+1 sync DB work inside an async handler**
[models.py:767-932](../../backend/src/api/v1/endpoints/models.py#L767) opens multiple `with get_sync_db()` blocks (active check, list, per-fs-extraction dataset resolution loop at 858, trainings usage query) inside an async route, plus a filesystem scan (`activation_service.list_extractions()`). Blocking sync DB + FS I/O on the event loop; for a model with many extractions the dataset-resolution loop (line 863) runs a full `datasets` table scan per filesystem-only extraction. Offload or batch.

**F5. `redownload_model` uses raw `Path` instead of `settings.resolve_data_path`**
[models.py:154-164](../../backend/src/api/v1/endpoints/models.py#L154) does `Path(model.file_path)` / `Path(model.quantized_path)` directly for `shutil.rmtree`. Everywhere else in the codebase paths are resolved via `settings.resolve_data_path()` to handle Docker `/data/` vs native paths. If `file_path` is stored as a container path, the rmtree silently no-ops on native deploys (dir "doesn't exist"), leaving stale files, then re-downloads. Use the resolver.

### P3 — Quality / tech-debt

- **F6. Inconsistent extraction-model import style.** Even setting aside F1, the file imports `ActivationExtraction` from `activation_extraction` at module top (line 17) but re-imports it function-locally elsewhere (lines 968, 1075, 1167). Consolidate on the one top-level import.
- **F7. `estimate_extraction_resources` hardcodes FP16 with a good comment but silent coupling.** [models.py:529](../../backend/src/api/v1/endpoints/models.py#L529) — memory estimate always assumes FP16 because `activation_service.py` loads FP16 regardless of stored quantization. Correct today, but it's an invisible cross-file invariant: if activation loading ever honors quantization, this estimate silently diverges. Worth a shared constant or a note in `activation_service`.
- **F8. `generate_model_id()` uses `uuid4().hex[:8]`** (32 bits) as the PK. Collision probability is low but non-zero across many models, and there's no uniqueness retry — a collision would surface as a PK violation on insert. Minor.
- **F9. Debug logging.** `model_tasks.py` and `tokenization`-adjacent code carry `print("[TOKENIZER DEBUG]…")` style lines (noted in 001 too — same habit). Route to `logger.debug`.
- **F10. `download_model` swallows the real error.** [models.py:89-93](../../backend/src/api/v1/endpoints/models.py#L89) catches `Exception` and returns a generic 500 "Failed to initiate model download" — good for not leaking internals, but it also catches the 409-duplicate path's own logic is fine; just note the broad catch can mask DB/commit errors during debugging. (Consistent with the security-hardening pattern, so leave as-is unless debugging.)

---

## 4. Test Coverage Notes

Good coverage of download, loader, progress, cleanup, and workflow. Critical gaps:
- **No test for extraction cancel/retry** → F1 (two broken endpoints) shipped undetected. A single API test per endpoint would have caught the `ModuleNotFoundError`.
- No test for download-cancel actually revoking the task (F3).
- No test for `redownload_model` path resolution on native vs container paths (F5).

---

## 5. Summary — if you fix three things

1. **F1** — the three `....models.extraction` imports → `....models.activation_extraction`. Two endpoints are currently dead. Highest priority in the review so far (a P0, worse than 001's P1).
2. **F5** — `redownload_model` path resolution (silent stale-file / re-download bug on native deploys).
3. **F3** — make model download-cancel revoke the running task (races cleanup, same class as dataset-cancel).

And schedule **FPRD 002 corrections** + a decision on the extraction-API-under-/models coupling (revisit when reviewing 004).

**Cross-feature pattern emerging (001 + 002):** both have (a) stale data-model/endpoint/channel doc sections, (b) a download-cancel that doesn't revoke the running Celery task, and (c) debug `print()` logging. Worth fixing the cancel-revoke pattern and doc-refresh as batch operations across features rather than piecemeal.
