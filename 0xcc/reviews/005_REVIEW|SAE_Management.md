# Feature Review: 005 — SAE Management

**Reviewed:** 2026-07-11
**Reviewer:** deep code review (doc↔code accuracy + bug/correctness + quality/tech-debt)
**Feature docs:** [FPRD v1.2](../prds/005_FPRD|SAE_Management.md) (2026-03-21), FTDD, FTID
**Verdict:** Solid multi-source SAE management with clean HF/Gemma-Scope handling and SAELens interop. No P0/P1. One notable **soft-delete-vs-file-deletion inconsistency (P2)** that can orphan features, plus the usual doc drift. This feature is also the **third entry point into the 004 feature-extraction pipeline** — reinforcing the extraction-architecture cleanup theme.

---

## 1. Scope

**Backend read:** `models/external_sae.py` (149), `api/v1/endpoints/saes.py` (773, full route inventory + extract/delete), `services/sae_manager_service.py` (793, delete + list + import), `schemas/sae.py` (311, validation), `services/sae_converter.py` (located — PRD path wrong), `services/huggingface_sae_service.py` (469, surveyed), `ml/community_format.py` (1233, surveyed). Extraction coupling traced through `ExtractionService.start_extraction_for_sae`.
**Tests present:** `test_multi_sae_import.py`, `test_sae_converter.py`. **Thin** relative to the feature's size (see §4).

---

## 2. Doc ↔ Code Accuracy

### 2.1 Key files — 1 path wrong

`ml/sae_converter.py` (PRD §7) → actually `services/sae_converter.py`. `community_format.py` is correctly under `ml/`.

### 2.2 Data model — ExternalSAE substantially wrong

| PRD (§6) | Reality |
|---|---|
| `id UUID` | `String` = `sae_{uuid}`. |
| `model_name`, `layer`, `hook_name`, `d_in`, `d_sae`, `repo_id`, `metadata` | Real: `model_name` (yes) + `model_id` FK, `layer`, `hook_type` (not `hook_name`), `n_features` (not `d_sae`), `d_model` (not `d_in`), `hf_repo_id`/`hf_filepath`/`hf_revision` (not `repo_id`), `sae_metadata` (not `metadata`), plus `status`, `progress`, `description`, `file_size_bytes`, `downloaded_at`. |
| `source` = `huggingface, gemma_scope, upload` | Enum = `huggingface, local, trained` (no `gemma_scope`/`upload`; Gemma Scope is a `huggingface` source). |
| `format` = `community, mistudio` | Enum `SAEFormat` = `community_standard, ...`. |
| (no status field in PRD) | Real `SAEStatus` = `pending, downloading, converting, ready, error, **deleted**` (soft-delete sentinel — see F1). |

### 2.3 API endpoints — several wrong/missing

| PRD (§5) | Reality |
|---|---|
| `GET/POST /saes`, `/saes/{id}`, `DELETE /saes/{id}` | `GET ""`, `GET /{id}`, `DELETE /{id}` exist. **No bare `POST /saes`** create — creation is via `/download`, `/upload`, `/import/training`, `/import/file`. |
| `POST /saes/download-hf` | Actual: `POST /saes/download` (+ `POST /saes/hf/preview`). |
| `POST /saes/{id}/convert` | **Does not exist** as a route (conversion happens inline during download/import via `sae_converter`). |
| `GET /saes/{id}/config` | **Does not exist**; config is embedded in `GET /saes/{id}`. |
| (not in PRD) | `POST /upload`, `/import/training`, `/import/file`, `GET /training/{tid}/available`, `POST /delete` (batch), `GET /{id}/features`, `POST /{id}/extract-features`, `GET /{id}/extraction-status`, `POST /{id}/cancel-extraction`, `POST /batch-extract-features`. |

**Recommendation:** rewrite FPRD 005 §5/§6. The Gemma Scope §8 section is accurate and useful — keep it.

---

## 3. Findings (severity-ranked)

### P2 — Correctness

**F1. Soft-delete deletes files but keeps the row → orphaned features with missing files.**
[sae_manager_service.py:686-725](../../backend/src/services/sae_manager_service.py#L686) — `delete_sae` does BOTH: `shutil.rmtree(local_path)` (files gone) AND `sae.status = DELETED` (row survives, not `db.delete`). The endpoint docstring calls it "soft-delete" but it's a **hybrid**: soft in the DB, hard on disk. Consequences:
  - `Feature.external_sae_id` FK rows (extracted from this SAE) still point at a live, `DELETED`-status SAE row. Those features remain in the DB and are browsable via `/extractions/{id}/features`, but re-extraction or file-backed operations fail because the SAE files are gone.
  - The `external_saes` table accumulates `DELETED` tombstone rows indefinitely (list queries correctly filter `status != DELETED` at [line 98](../../backend/src/services/sae_manager_service.py#L98), so they're invisible, but never purged).
**Fix options:** (a) true hard delete with cascade to features/extractions (matches user intent "delete SAE"); or (b) genuine soft-delete that also *retains* files (so it's reversible); the current half-and-half is the worst of both. Also consider cascading a features cleanup, or blocking delete when features exist (like model-delete should — see 003-F6).

**F2. `delete_saes_batch` and `delete_sae` don't revoke a running extraction.**
An SAE mid-`extract-features` can be soft-deleted; the running extraction task keeps writing features referencing a now-DELETED SAE. The `cancel-extraction` endpoint ([saes.py:644](../../backend/src/api/v1/endpoints/saes.py#L644)) revokes correctly, but delete doesn't chain to it. (Consistent with the cross-feature "delete doesn't revoke" pattern.)

### P3 — Quality / tech-debt

**F3. Third feature-extraction entry point — extraction architecture needs consolidation.**
`/saes/{id}/extract-features` and `/saes/batch-extract-features` both funnel into `ExtractionService.start_extraction_for_sae` → the 004 `extract_features_from_sae` task. So the SAE→features pipeline (004) has entry points in **both** `features`-adjacent code and `saes.py`, while the model→activations pipeline (002) is separate under `/models`. Combined with 004-F5 (the `ActivationExtraction` vs `ExtractionJob` naming collision), the "extraction" surface spans 3 endpoints, 2 models, 2 tasks, and 2 ExtractionStatus enums. **This is the single biggest cross-feature interdependency in the app** — worth one consolidation pass documenting the two-stage pipeline (model→activations→SAE→features) and unifying the naming. Note: the inline error handler here ([saes.py:604](../../backend/src/api/v1/endpoints/saes.py#L604)) *does* set `ExtractionStatus.FAILED` correctly — better than the task's outer-except (004-F7).

**F4. Test coverage is thin** for the feature's size (793-line service, 773-line endpoint, 1233-line community_format) — only 2 unit tests (`multi_sae_import`, `sae_converter`). No test for delete semantics (F1), HF download flow, Gemma Scope parsing, or the extract-features entry point. Given community_format handles SAELens interop (research-critical), this under-coverage is a risk.

**F5. `community_format.py` is 1233 lines** handling both community-standard and miStudio formats plus detection/conversion. Large but cohesive; flagging for awareness, not action.

**F6. Debug-logging habit** present here too (batch cleanup candidate across 001–005).

---

## 4. Test Coverage Notes

Weakest coverage-to-size ratio in the review so far. `test_sae_converter` and `test_multi_sae_import` are valuable (converter correctness is research-critical), but there is no coverage for: soft-delete file/row semantics (F1), delete-during-extraction (F2), HF/Gemma-Scope download parsing, format auto-detection, or the batch-extract flow. Recommend adding a delete-semantics test and a Gemma Scope repo-parsing test at minimum.

---

## 5. Summary — if you fix three things

1. **F1** — resolve the soft-delete-vs-hard-file-delete inconsistency (currently orphans features and leaves tombstone rows). Decide: true hard delete w/ cascade, or true reversible soft delete.
2. **F3** (cross-feature) — consolidate/document the 3-entry-point extraction architecture (ties to 004-F5 and 002-F1). Highest holistic value.
3. **F2** — chain delete → cancel running extraction.

**The good:** multi-source handling (trained / HF / Gemma Scope), SAELens community-format interop, multi-select and multi-layer import, and dynamic-architecture support (no whitelist) are all implemented and the converter is tested. The soft-delete issue is the one real correctness gap.
