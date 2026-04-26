# Feature Tasks: Feature Discovery

**Document ID:** 004_FTASKS|Feature_Discovery
**Version:** 1.3
**Last Updated:** 2026-04-26
**Status:** Implemented
**Related PRD:** [004_FPRD|Feature_Discovery](../prds/004_FPRD|Feature_Discovery.md)

---

## Task Summary

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Database | 4 tasks | ✅ Complete |
| Phase 2: Extraction Service | 5 tasks | ✅ Complete |
| Phase 3: Token Analysis | 3 tasks | ✅ Complete |
| Phase 4: Labeling System | 6 tasks | ✅ Complete |
| Phase 5: API Endpoints | 4 tasks | ✅ Complete |
| Phase 6: Frontend Browser | 6 tasks | ✅ Complete |
| Phase 7: Labeling UI | 4 tasks | ✅ Complete |
| Phase 8: Batch Extraction & Live Metrics (Jan 2026) | 8 tasks | ✅ Complete |
| Phase 9: Enhanced Per-Feature Labeling (Mar–Apr 2026) | 13 tasks | ✅ Complete |
| Phase 10: Star Color System (Mar 2026) | 6 tasks | ✅ Complete |
| Phase 11: OpenAI API Integration & SDK Standardization (Apr 2026) | 5 tasks | ✅ Complete |
| Phase 12: Context-Aware Labeling Template (Apr 2026) | 3 tasks | ✅ Complete |
| Phase 13: Notes UX — Markdown Rendering (Apr 2026) | 3 tasks | ✅ Complete |

**Total: 80 tasks** (40 original + 40 post-March 2026)

---

## Phase 1: Database Setup

### Task 1.1: Create Feature Migration
- [x] Create features table
- [x] Create feature_activations table (partitioned)
- [x] Create extraction_jobs table
- [x] Add indexes for queries

**Files:**
- `backend/alembic/versions/xxx_create_feature_tables.py`

### Task 1.2: Create SQLAlchemy Models
- [x] Define Feature model
- [x] Define FeatureActivation model
- [x] Define ExtractionJob model
- [x] Configure relationships

**Files:**
- `backend/src/models/feature.py`
- `backend/src/models/extraction_job.py`

### Task 1.3: Create Pydantic Schemas
- [x] FeatureResponse schema
- [x] FeatureActivationResponse schema
- [x] ExtractionJobCreate schema
- [x] ExtractionConfig schema

**Files:**
- `backend/src/schemas/feature.py`
- `backend/src/schemas/extraction.py`

### Task 1.4: Run Migrations
- [x] Apply migrations
- [x] Verify partitioning works
- [x] Test cascade deletes

---

## Phase 2: Extraction Service

### Task 2.1: Create Extraction Service
- [x] Implement create_job() method
- [x] Implement extract_features() method
- [x] Implement save_features() method
- [x] Handle batch processing

**Files:**
- `backend/src/services/extraction_service.py`

### Task 2.2: Implement Feature Statistics
- [x] Calculate activation_frequency
- [x] Calculate max_activation
- [x] Calculate mean_activation
- [x] Calculate std_activation

### Task 2.3: Implement Top Token Aggregation
- [x] Get top activating tokens per feature
- [x] Aggregate by token
- [x] Store in denormalized JSONB

**Files:**
- `backend/src/services/token_aggregator_service.py`

### Task 2.4: Create Extraction Task
- [x] Define extract_features_task
- [x] Configure queue routing
- [x] Emit progress via WebSocket

**Files:**
- `backend/src/workers/extraction_tasks.py`

### Task 2.5: Vectorized Extraction
- [x] Implement batch processing
- [x] Optimize memory usage
- [x] Add chunked processing

**Files:**
- `backend/src/services/extraction_vectorized.py`

---

## Phase 3: Token Analysis

### Task 3.1: Token Statistics
- [x] Count token occurrences
- [x] Calculate mean activation per token
- [x] Calculate max activation per token
- [x] Sort by frequency

### Task 3.2: Context Extraction
- [x] Extract context before token
- [x] Extract context after token
- [x] Store with activation

### Task 3.3: Stop Words Filter
- [x] Define stop words list
- [x] Add filter option
- [x] Enable by default

---

## Phase 4: Labeling System

### Task 4.1: Create Labeling Models
- [x] Create labeling_jobs table
- [x] Create labeling_prompt_templates table
- [x] Define default template

**Files:**
- `backend/src/models/labeling_job.py`
- `backend/src/models/labeling_prompt_template.py`

### Task 4.2: Create OpenAI Labeling Service
- [x] Implement label_feature() method
- [x] Implement batch labeling
- [x] Add rate limiting
- [x] Handle JSON response format

**Files:**
- `backend/src/services/openai_labeling_service.py`

### Task 4.3: Create Context Formatter
- [x] Format top tokens for prompt
- [x] Format example contexts
- [x] Format statistics

**Files:**
- `backend/src/services/labeling_context_formatter.py`

### Task 4.4: Create Labeling Task
- [x] Define label_features_task
- [x] Process in batches
- [x] Update features with labels
- [x] Emit progress

**Files:**
- `backend/src/workers/labeling_tasks.py`

### Task 4.5: Labeling Template Management
- [x] CRUD for templates
- [x] Default template handling
- [x] Template variables

### Task 4.6: Dual Label System
- [x] Support semantic_label
- [x] Support category_label
- [x] Allow manual override

---

## Phase 5: API Endpoints

### Task 5.1: Feature Endpoints
- [x] GET /features - List with filters
- [x] GET /features/{id} - Get details
- [x] GET /features/{id}/activations - Get activations
- [x] PUT /features/{id}/label - Update label

**Files:**
- `backend/src/api/v1/endpoints/features.py`

### Task 5.2: Extraction Endpoints
- [x] POST /extractions - Start extraction
- [x] GET /extractions/{id} - Get status
- [x] POST /extractions/{id}/cancel - Cancel

### Task 5.3: Labeling Endpoints
- [x] POST /labeling - Start labeling job
- [x] GET /labeling/{id} - Get status
- [x] GET /labeling/templates - List templates
- [x] POST /labeling/templates - Create template

### Task 5.4: Search/Filter API
- [x] Search by label text
- [x] Filter by category
- [x] Sort options
- [x] Pagination

---

## Phase 6: Frontend Browser

### Task 6.1: Create Types
- [x] Define Feature interface
- [x] Define FeatureActivation interface
- [x] Define ExtractionJob interface

**Files:**
- `frontend/src/types/features.ts`

### Task 6.2: Create API Client
- [x] listFeatures() function
- [x] getFeature() function
- [x] getActivations() function
- [x] updateLabel() function

**Files:**
- `frontend/src/api/features.ts`

### Task 6.3: Create Features Store
- [x] Feature list state
- [x] Selected feature state
- [x] Fetch and filter actions

**Files:**
- `frontend/src/stores/featuresStore.ts`

### Task 6.4: Create FeatureBrowser Component
- [x] Search input
- [x] Sort selector
- [x] Grid layout
- [x] Pagination

**Files:**
- `frontend/src/components/features/FeatureBrowser.tsx`

### Task 6.5: Create FeatureCard Component
- [x] Display feature index
- [x] Display label
- [x] Display top tokens
- [x] Display statistics

**Files:**
- `frontend/src/components/features/FeatureCard.tsx`

### Task 6.6: Create FeatureDetailModal
- [x] Statistics section
- [x] Top tokens section
- [x] Example activations
- [x] Token highlighting

**Files:**
- `frontend/src/components/features/FeatureDetailModal.tsx`

---

## Phase 7: Labeling UI

### Task 7.1: Create TokenHighlight Component
- [x] Parse context string
- [x] Highlight target token
- [x] Color by activation strength

**Files:**
- `frontend/src/components/features/TokenHighlight.tsx`

### Task 7.2: Create ExtractionJobCard
- [x] Show job progress
- [x] Show feature count
- [x] Cancel button

**Files:**
- `frontend/src/components/features/ExtractionJobCard.tsx`

### Task 7.3: Create StartLabelingButton
- [x] Template selector
- [x] Feature selection
- [x] Start labeling action

**Files:**
- `frontend/src/components/labeling/StartLabelingButton.tsx`

### Task 7.4: Create TemplatesPanel
- [x] List templates
- [x] Create/edit modal
- [x] Set default template

**Files:**
- `frontend/src/components/panels/LabelingPromptTemplatesPanel.tsx`

---

## Phase 8: Batch Extraction & Live Metrics (Added Jan 2026)

### Task 8.1: Add IncrementalTopKHeap Statistics
- [x] Add get_stats() method to IncrementalTopKHeap class
- [x] Return features_in_heap count
- [x] Return heap_examples_count total

**Files:**
- `backend/src/services/extraction_vectorized.py`

### Task 8.2: Implement Time-Based Progress Emission
- [x] Add 2-second emission interval
- [x] Include heap stats in progress payload
- [x] Cap progress at 99% during extraction

**Files:**
- `backend/src/services/extraction_service.py`

### Task 8.3: Batch Extraction API
- [x] Create batch extraction endpoint
- [x] Support multi-SAE selection
- [x] Link jobs via batch_id
- [x] Queue jobs sequentially

**Files:**
- `backend/src/api/v1/endpoints/extractions.py`
- `backend/src/schemas/extraction.py`

### Task 8.4: NLP Continuation Support
- [x] Add continue_to_nlp flag
- [x] Trigger labeling on extraction completion
- [x] Chain tasks via Celery

**Files:**
- `backend/src/workers/extraction_tasks.py`

### Task 8.5: Frontend Progress Types
- [x] Add features_in_heap to progress type
- [x] Add heap_examples_count to progress type
- [x] Update WebSocket handler

**Files:**
- `frontend/src/types/extraction.ts`
- `frontend/src/stores/extractionsStore.ts`

### Task 8.6: Live Progress Charts
- [x] Create ExtractionProgressChart component
- [x] Display features_in_heap graph
- [x] Display heap_examples_count graph
- [x] Real-time WebSocket updates

**Files:**
- `frontend/src/components/features/ExtractionProgressChart.tsx`

### Task 8.7: Batch Extraction UI
- [x] Add multi-select for SAEs
- [x] Add continue_to_nlp toggle
- [x] Display batch progress
- [x] Show job queue status

**Files:**
- `frontend/src/components/features/StartExtractionModal.tsx`
- `frontend/src/components/features/BatchExtractionProgress.tsx`

### Task 8.8: Save as Template Feature
- [x] Add "Save as Template" button in extraction config
- [x] Pre-fill template form with current config
- [x] Navigate to templates panel after save

**Files:**
- `frontend/src/components/features/StartExtractionModal.tsx`
- `frontend/src/stores/extractionTemplatesStore.ts`

---

## Relevant Files Summary

### Backend
| File | Purpose |
|------|---------|
| `backend/src/models/feature.py` | Feature models |
| `backend/src/models/labeling_job.py` | Labeling job model |
| `backend/src/schemas/feature.py` | Feature schemas |
| `backend/src/schemas/extraction.py` | Extraction schemas |
| `backend/src/services/extraction_service.py` | Extraction logic |
| `backend/src/services/openai_labeling_service.py` | AI labeling |
| `backend/src/workers/extraction_tasks.py` | Celery tasks |
| `backend/src/api/v1/endpoints/features.py` | API routes |

### Frontend
| File | Purpose |
|------|---------|
| `frontend/src/types/features.ts` | TypeScript types |
| `frontend/src/api/features.ts` | API client |
| `frontend/src/stores/featuresStore.ts` | Zustand store |
| `frontend/src/components/features/FeatureBrowser.tsx` | Browser |
| `frontend/src/components/features/FeatureCard.tsx` | Card |
| `frontend/src/components/features/FeatureDetailModal.tsx` | Modal |
| `frontend/src/components/features/TokenHighlight.tsx` | Highlight |

---

## Phase 9: Enhanced Per-Feature Labeling (Mar–Apr 2026)

### Task 9.1: Database — EnhancedLabelingJob
- [x] Alembic migration — `enhanced_labeling_jobs` table + `enhanced_llm` to `label_source` enum
- [x] `EnhancedLabelingJob` SQLAlchemy model with status, phase, method, endpoint, model, workers columns
- [x] Migration for `method` column (default `openai_compatible`)

**Files:** `backend/alembic/versions/`, `backend/src/models/enhanced_labeling_job.py`

### Task 9.2: Backend Service — EnhancedLabelingService
- [x] Two-pass strategy with `ThreadPoolExecutor` for parallel Pass-1 workers
- [x] Pass-1: per-example LLM summarization ("what is this token doing in context?")
- [x] Pass-2: synthesis across all summaries → structured label
- [x] Uses official OpenAI Python SDK (`OpenAI(api_key=..., base_url=...)`)
- [x] Reasoning-model detection (gpt-5*, o1*, o3*, o4*) → `max_completion_tokens` budget (16K synthesis)
- [x] `BadRequestError` from SDK: surface verbatim, no retry

**Files:** `backend/src/services/enhanced_labeling_service.py`

### Task 9.3: Backend — Celery Task + API
- [x] `enhanced_label_feature_task` Celery task with DatabaseTask base
- [x] DB-first approach: commit job record before dispatching Celery task
- [x] Queue routing: `enhanced_labeling` queue
- [x] Cleanup task for stuck queued/running jobs
- [x] API: `POST /features/{id}/label/enhanced` — reads settings for method/endpoint/model
- [x] API: `GET /features/{id}/label/enhanced/latest` — restore state on modal reopen
- [x] WebSocket emission: `enhanced_labeling:progress`, `completed`, `failed`

**Files:** `backend/src/workers/enhanced_labeling_tasks.py`, `backend/src/api/v1/endpoints/enhanced_labeling.py`

### Task 9.4: Frontend — Hook + UI
- [x] `useEnhancedLabeling.ts` hook: manages job lifecycle, WebSocket subscription, re-subscribe on reconnect
- [x] `FeatureDetailModal.tsx`: sparkle (✨) button, phase display, progress counter, auto-populate edit form on completion
- [x] Settings Labeling tab: `enhanced_labeling_method` (OpenAI vs OpenAI-Compatible), `enhanced_labeling_openai_model`, Fetch Models button, `enhanced_labeling_max_workers`

**Files:** `frontend/src/hooks/useEnhancedLabeling.ts`, `frontend/src/components/features/FeatureDetailModal.tsx`, `frontend/src/components/panels/SettingsPanel.tsx`

### Future / Nice-to-have
- [ ] Edge-tier analysis: separate pass-1 summarization for low-activation tail examples
- [ ] Confidence gating: auto-accept labels above threshold, flag below for review
- [ ] Batch enhanced labeling: run enhanced labeling across a filtered subset of features

---

## Phase 10: Star Color System (Mar 2026)

### Task 10.1: Database
- [x] `star_color` VARCHAR(20) column on features table (Alembic migration)
- [x] `star_color` included in all `FeatureResponse` / `FeatureDetailResponse` constructors

**Files:** `backend/alembic/versions/s6t7u8v9w0x1_add_star_color_to_features.py`, `backend/src/services/feature_service.py`

### Task 10.2: Backend Logic
- [x] Backend endpoint sets `star_color='purple'` when enhanced labeling job queued
- [x] Celery task sets `star_color='aqua'` on completion
- [x] Bulk labeling service: guard in all 3 persist loops — skip features where `star_color='aqua'`
- [x] `PATCH /features/{id}/star_color` endpoint for frontend to set yellow/null

**Files:** `backend/src/api/v1/endpoints/enhanced_labeling.py`, `backend/src/workers/enhanced_labeling_tasks.py`, `backend/src/services/labeling_service.py`

### Task 10.3: Frontend
- [x] `setStarColor(featureId, color)` in Zustand featuresStore
- [x] `patchFeatureLocally(featureId, fields)` — synchronous patch across `featuresByExtraction`, `featuresByTraining`, `selectedFeature` without network call
- [x] Star color rendered on feature list rows and Feature Detail modal header
- [x] `useEnhancedLabeling.ts`: calls `setStarColor('purple')` on start, `setStarColor('aqua')` on completion
- [x] Mount effect: syncs store to DB star color when modal opens for in-flight job

**Files:** `frontend/src/stores/featuresStore.ts`, `frontend/src/hooks/useEnhancedLabeling.ts`

---

## Phase 11: OpenAI API Integration & SDK Standardization (Apr 2026)

### Task 11.1: Settings — OpenAI API Key & Enhanced Labeling Method
- [x] API Keys tab: `openai_api_key` stored encrypted (AES-256-GCM)
- [x] Labeling tab: `enhanced_labeling_method` (openai | openai_compatible) + `enhanced_labeling_openai_model`
- [x] Fetch Models button on Labeling tab: calls `POST /labeling/models/openai` with stored key → populates dropdown
- [x] `/labeling/models/openai` endpoint falls back to DB AppSetting when `api_key` not in request body

**Files:** `backend/src/api/v1/endpoints/labeling.py`, `frontend/src/components/panels/SettingsPanel.tsx`

### Task 11.2: Enhanced Labeling → OpenAI SDK
- [x] `EnhancedLabelingService.__init__` creates `OpenAI(api_key, base_url)` instead of raw httpx
- [x] `_call_llm()` uses `self._openai.chat.completions.create()` — SDK handles per-model parameter differences
- [x] Reasoning model detection: `gpt-5*`, `o1*`, `o3*`, `o4*` → `max_completion_tokens=16000` (synthesis)
- [x] `BadRequestError` surfaces verbatim without retry; transient errors retry with backoff
- [x] Empty-content detection (reasoning budget exhausted) surfaces clear diagnostic

**Files:** `backend/src/services/enhanced_labeling_service.py`

### Task 11.3: Bulk Labeling — miLLM Pre-Loading
- [x] `ensure_model_loaded()` in `src/utils/millm_utils.py` — shared utility
- [x] Called in `labeling_service.py` before OPENAI_COMPATIBLE inference loop
- [x] Called in `enhanced_labeling_tasks.py` before `EnhancedLabelingService` is constructed

**Files:** `backend/src/utils/millm_utils.py`, `backend/src/services/labeling_service.py`, `backend/src/workers/enhanced_labeling_tasks.py`

---

## Phase 12: Context-Aware Labeling Template (Apr 2026)

### Task 12.1: Design
- [x] System message: explicitly deprioritizes prime token; instructs model to read full passages and find shared semantic pattern
- [x] User prompt: shows `{examples_block}` (full context windows); asks "what is semantically happening?"; cross-example question forces pattern-level thinking
- [x] 3 negative counter-examples included via `include_negative_examples=True`

### Task 12.2: Implementation
- [x] `template_type='mistudio_context'` — uses `{examples_block}` formatter (prefix <<prime>> suffix)
- [x] `is_system=True` — visible to all users, cannot be deleted
- [x] Seed script: `backend/scripts/seed_context_aware_template.py`
- [x] Seeded to both K8s and Docker Compose production databases via REST API

**Files:** `backend/scripts/seed_context_aware_template.py`

### Task 12.3: Documentation
- [x] Template description explains intent and best-fit models (GPT-4o / GPT-5)
- [x] Updated Feature Discovery PRD with FR-13 requirements

---

## Phase 13: Notes UX — Markdown Rendering (Apr 2026)

### Task 13.1: Install Dependencies
- [x] `npm install react-markdown remark-gfm`
- [x] Added to `package.json` as runtime dependencies

### Task 13.2: Notes Rendering Component
- [x] Replace `<p className="whitespace-pre-wrap">{notes}</p>` with `<ReactMarkdown remarkPlugins={[remarkGfm]}>`
- [x] Custom component renderers for `table`, `th`, `td`, `tr`, `p`, `code`, `pre`, `ul`, `li`, `strong`, `hr` — all dark-theme slate Tailwind classes
- [x] Container: `max-h-96 overflow-y-auto` — notes scroll inside modal, don't push other sections off-screen

**Files:** `frontend/src/components/features/FeatureDetailModal.tsx`

### Task 13.3: Settings Scroll Fix
- [x] `window.scrollTo(0, 0)` in `SettingsPanel` `useEffect` on mount — page no longer opens scrolled to bottom

**Files:** `frontend/src/components/panels/SettingsPanel.tsx`

---

*Related: [PRD](../prds/004_FPRD|Feature_Discovery.md) | [TDD](../tdds/004_FTDD|Feature_Discovery.md) | [TID](../tids/004_FTID|Feature_Discovery.md)*
