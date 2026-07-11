# Feature Review: 004 — Feature Discovery

**Reviewed:** 2026-07-11
**Reviewer:** deep code review (doc↔code accuracy + bug/correctness + quality/tech-debt)
**Feature docs:** [FPRD v1.4](../prds/004_FPRD|Feature_Discovery.md) (2026-04-26), FTDD, FTID
**Verdict:** The largest feature (~10k lines of service code across extraction, bulk labeling, enhanced labeling, NLP). The **enhanced-labeling subsystem (newest code) is clean and well-built**; the aqua-guard and star lifecycle are correctly implemented exactly as documented. No P0. The main issues are the **most-drifted endpoint doc in the review** and a couple of validation/atomicity gaps.

---

## 1. Scope

**Backend read:** `models/feature.py` (122), `models/feature_activation.py` (60), `models/enhanced_labeling_job.py` (89), `workers/enhanced_labeling_tasks.py` (229, full), `services/enhanced_labeling_service.py` (376, key sections), `api/v1/endpoints/enhanced_labeling.py` (187, full), `api/v1/endpoints/features.py` (1085, route inventory + star/patch/examples), `services/feature_service.py` (`set_star_color`), `services/labeling_service.py` (aqua-guard sites). `services/extraction_service.py` (2045) and the extraction task reviewed in this session's Celery review (F9). `openai_labeling_service.py` (1593), `labeling_service.py` (1660) surveyed, not line-by-line.
**Structure clarified:** two distinct "extraction" pipelines — `activation_extraction.py`/`model_tasks.extract_activations` (Feature 002: model→raw activations) vs `extraction_job.py`/`extraction_tasks.extract_features_from_sae` (Feature 004: SAE→features). Same word, different tables/tasks. This resolves the 002 coupling question: they are *not* the same thing routed twice — they're two stages, and the naming collision is the real problem (see F5).
**Tests (very strong, 17 files):** enhanced_labeling (api + unit), dual_labels, extraction (vectorized/progress/schemas/template), auto_labeling, labeling (service/endpoints/context ×3), feature_schemas.

---

## 2. Doc ↔ Code Accuracy

FPRD v1.4 is the **most detailed** FPRD (the §2.11–2.13 enhanced-labeling/star/context-aware sections are accurate and match the code precisely — clearly written alongside the implementation). But the **mechanical reference sections (§5 endpoints, §6 data model) are the most inaccurate in the whole review.**

### 2.1 API endpoints — §5 is almost entirely fictional

Not one of the PRD's listed feature routes matches. Actual routes in `features.py` (router has **no prefix**, mounted flat):

| PRD (§5) | Reality |
|---|---|
| `GET/PATCH /features` , `/features/{id}` | `PATCH /features/{feature_id}` exists; **no bare `GET /features` list** — features are listed via `GET /trainings/{tid}/features` and `GET /extractions/{eid}/features`. |
| `POST /features/extraction`, `GET /features/extraction/{id}` | Real: `GET /extractions`, `GET /extractions/{id}`, `DELETE /extractions/{id}`. Extraction is *started* via Feature 002's `POST /models/{id}/extract-activations` — no `/features/extraction` route exists. |
| `POST /features/labeling`, `GET /features/labeling/{id}` | Labeling is a **separate router** (`labeling.py`, mounted as `labeling.router`) — not under `/features`. |
| `POST /features/export` | **Does not exist** as shown; export is per-feature `GET /features/{id}/examples` + the Neuronpedia export feature (007). |
| (not in PRD) | Huge undocumented analysis surface: `/features/{id}/star`, `/favorite`, `/logit-lens`, `/correlations`, `/ablation`, `/token-analysis`, `/analyze-nlp`, `/nlp-analysis`, `/analysis/cleanup`, `/extractions/{id}/analyze-nlp`, `/cancel-nlp`, `/reset-nlp`. |
| `/features/{id}/label/enhanced[/latest]` | **Correct** ✅ (the one accurate pair — the newest routes). |

### 2.2 Data model — Feature and FeatureActivation both wrong

| PRD (§6) | Reality |
|---|---|
| `features.id UUID` | `String` = `feat_{training_id}_{neuron_index}`. |
| stats nested in `statistics JSONB` | Real: `activation_frequency`, `interpretability_score`, `max_activation`, `mean_activation` are **top-level indexed columns** (not a JSONB blob). PRD omits `extraction_job_id`/`labeling_job_id` FKs, `example_tokens_summary`, `nlp_processed_at`, `labeled_at`. |
| `feature_activations`: `activation_value`, `token`, `context_before`, `context_after`, `id UUID` | Real: `max_activation`, `tokens` (JSONB array), `activations` (JSONB array), `prefix_tokens`/`prime_token`/`suffix_tokens`, composite PK `(id BigInteger, feature_id)` for **range partitioning**. Completely different shape. |
| `extraction_jobs`: `sae_id`, `features_found` | (extraction_job.py not fully audited, but `Feature.extraction_job_id` FK confirms the table; field names likely drift too.) |
| `EnhancedLabelingJob` (§6.4) | **Accurate** ✅ (newest, written with the code). |

### 2.3 WebSocket channels — §7 mostly accurate

The enhanced-labeling channels (`enhanced_labeling/{job_id}` + namespaced events) and the extraction progress payload (§7.1) match the code well — these are recent. The older `extraction/{id}`/`labeling/{id}` entries use the real namespaced event names. Best channel doc in the review so far.

**Recommendation:** §2.11–2.13 and §7 can stay. §5 (endpoints) and §6.1/§6.2 (Feature/FeatureActivation) need a full rewrite — they describe a data model and API that never existed in this form.

---

## 3. Findings (severity-ranked)

### P2 — Correctness / robustness

**F1. Enhanced-labeling task marks job FAILED *then* lets Celery autoretry.**
[enhanced_labeling_tasks.py:216-229](../../backend/src/workers/enhanced_labeling_tasks.py#L216) — the task is decorated `autoretry_for=(ConnectionError, TimeoutError, OSError), max_retries=3`, but its `except Exception` handler sets `status=FAILED`, emits `enhanced_labeling:failed` to the client, **then `raise`s**. If the exception is a retryable one, Celery retries — but the feature already flipped to a failed state and the modal already showed failure, then it silently succeeds on retry (or flips failed→running→failed again). The user sees a failure that may not be final. **Fix:** distinguish retryable exceptions (don't mark FAILED / don't emit failed until retries exhausted) from terminal ones, or drop `autoretry_for` and handle retries explicitly.

**F2. Duplicate-active-job guard is not atomic.**
[enhanced_labeling.py:64-77](../../backend/src/api/v1/endpoints/enhanced_labeling.py#L64) — checks for an existing QUEUED/RUNNING job, then inserts if none. Two rapid sparkle-clicks (or a double-submit) can both pass the check and create two jobs for the same feature, both writing the same feature row. FR-11.8 ("Prevent duplicate active jobs") holds under normal use but not under a race. **Fix:** a partial unique index on `(feature_id) WHERE status IN ('queued','running')`, or a `SELECT ... FOR UPDATE` / advisory lock.

**F3. `set_star_color` endpoint accepts an unvalidated free-text color.**
[features.py:440-460](../../backend/src/api/v1/endpoints/features.py#L440) — `star_color: Optional[str] = Query(None)` with no Literal/enum constraint. A caller can `POST /features/{id}/star?star_color=chartreuse` and it's stored (the service only special-cases yellow/purple/aqua/None). The aqua-downgrade protection in the service is correct, but the value domain is unenforced. **Fix:** `Literal["yellow","purple","aqua"] | None`.

### P3 — Quality / tech-debt

**F4. Enhanced-labeling docstring says duplicate returns HTTP 200, decorator forces 201.**
[enhanced_labeling.py:55](../../backend/src/api/v1/endpoints/enhanced_labeling.py#L55) claims "returns it (HTTP 200)" for an existing job, but `status_code=status.HTTP_201_CREATED` applies to the whole route, so the dedup path also returns 201. Cosmetic contract inaccuracy.

**F5. The "extraction" naming collision is a real cognitive-load / bug-surface issue.** `ActivationExtraction` (model→activations, Feature 002) and `ExtractionJob` (SAE→features, Feature 004) coexist with near-identical names, both have `ExtractionStatus` enums, both have "extraction" tasks, and 002's `cancel_extraction`/`retry_extraction` already broke by importing the *wrong* extraction module (002-F1, P0). This collision directly *caused* a P0. **Recommendation:** rename one family (e.g., `ActivationExtraction` → `ActivationCapture`, or `ExtractionJob` → `FeatureExtractionJob`) and document the two-stage pipeline explicitly in both FPRD 002 and 004. This is the highest-value cross-feature cleanup.

**F6. `features.py` is a 1085-line, 21-route god-file** spanning features, extractions, NLP analysis, logit-lens, correlations, ablation, and analysis-cache cleanup. Several of these (logit-lens, correlations, ablation) are really *analysis tools* (a distinct concern, arguably Feature 10 in the PPRD roadmap). Consider splitting `analysis.py` out. Not urgent; flagging the accretion.

**F7. Extraction task FAILED-persistence** — carried over from the Celery review (F9): the outer `except` in `extract_features_from_sae_task` emits a failed WS event but relies on the service having set `ExtractionStatus.FAILED`; if the service raises before that, the row can linger `EXTRACTING` until the 10-min `cleanup_stuck_extractions` beat task. Already documented; re-noting for the batch-fix list.

**F8. Debug logging pattern** — same `print()`/verbose-log habit as 001–003 appears in the labeling/extraction services. Batch cleanup candidate.

---

## 4. Test Coverage Notes

The strongest coverage in the review (17 files) — enhanced labeling has both API and unit tests, labeling context has 3 dedicated tests, extraction is well-covered (vectorized, progress, schemas, template). Gaps:
- **No test for the enhanced-labeling retry-vs-FAILED interaction** (F1) — the autoretry + mark-failed path is untested.
- No test for the duplicate-job race (F2) — the happy-path dedup is likely tested, but not concurrent submission.
- No test asserting `set_star_color` rejects invalid colors (F3).

---

## 5. Summary — if you fix three things

1. **F1** — fix the enhanced-labeling autoretry/FAILED interaction (user sees non-final failures).
2. **F5** (cross-feature) — resolve the `ActivationExtraction` vs `ExtractionJob` naming collision that already caused 002's P0. Highest holistic value.
3. **F2/F3** — atomic duplicate-job guard + star_color Literal validation.

And a **full rewrite of FPRD 004 §5 (endpoints) and §6.1/§6.2 (data model)** — the most-drifted reference sections in the app. Keep §2.11–2.13 and §7 (accurate).

**The good news:** the enhanced-labeling subsystem — the headline feature of this PRD — is genuinely well-implemented. The two-pass service (official OpenAI SDK, reasoning-model token budgets, parallel Pass-1), the settings-driven method resolution, the star lifecycle (purple-on-start, aqua-on-complete, protected from downgrade), and the aqua-guard in all three bulk-labeling loops all match the documentation exactly.
