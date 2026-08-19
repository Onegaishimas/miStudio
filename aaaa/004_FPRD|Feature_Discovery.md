# Feature PRD: Feature Discovery

**Document ID:** 004_FPRD|Feature_Discovery
**Version:** 1.5 (doc refresh — endpoints & data model)
**Last Updated:** 2026-07-11
**Status:** Implemented
**Priority:** P0 (Core Feature)

> **Reference sections corrected 2026-07-11** — §2.11–2.13 and §7 (WebSocket) are
> accurate; but §5 (endpoints) and §6.1/§6.2 (Feature/FeatureActivation data
> model) had drifted. See the **Doc-Refresh Corrections** appendix at the end.

---

## 1. Overview

### 1.1 Purpose
Enable users to extract, browse, and label interpretable features from trained SAEs, including automated labeling via GPT-4o.

### 1.2 User Problem
Researchers need to understand SAE features but face challenges with:
- Extracting features from large datasets efficiently
- Finding patterns across thousands of features
- Creating meaningful labels for interpretation
- Organizing features for analysis

### 1.3 Solution
A comprehensive feature discovery system with batch extraction, statistical analysis, dual labeling (semantic + category), and GPT-4o auto-labeling.

---

## 2. Functional Requirements

### 2.1 Feature Extraction
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-1.1 | Extract top activations per feature | Implemented |
| FR-1.2 | Batch processing with GPU optimization | Implemented |
| FR-1.3 | Configurable activation threshold | Implemented |
| FR-1.4 | Context window capture (tokens before/after) | Implemented |
| FR-1.5 | Token filtering during extraction | Implemented |
| FR-1.6 | Real-time progress via WebSocket | Implemented |
| FR-1.7 | Live progress metrics (samples/second, ETA) | Implemented |
| FR-1.8 | Time-based progress emission (every 2 seconds) | Implemented |
| FR-1.9 | Features in heap count for progress graphs | Implemented |
| FR-1.10 | Heap examples count for collection rate graphs | Implemented |

### 2.2 Feature Browser
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-2.1 | List features with pagination | Implemented |
| FR-2.2 | Search by label, category, statistics | Implemented |
| FR-2.3 | Sort by activation frequency, max, mean | Implemented |
| FR-2.4 | Filter by label status, category | Implemented |
| FR-2.5 | View feature detail modal | Implemented |

### 2.3 Feature Statistics
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-3.1 | Activation frequency | Implemented |
| FR-3.2 | Max/mean activation value | Implemented |
| FR-3.3 | Interpretability score | Implemented |
| FR-3.4 | Token distribution analysis | Implemented |

### 2.4 Dual Labeling System
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-4.1 | Semantic label (free-text description) | Implemented |
| FR-4.2 | Category label (hierarchical taxonomy) | Implemented |
| FR-4.3 | Manual label editing | Implemented |
| FR-4.4 | Label confidence score | Implemented |
| FR-4.5 | Label source tracking (manual/auto/imported) | Implemented |

### 2.5 Auto-Labeling (Sub-feature)
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-5.1 | GPT-4o integration via OpenAI API | Implemented |
| FR-5.2 | Batch auto-labeling with progress | Implemented |
| FR-5.3 | Configurable prompt templates | Implemented |
| FR-5.4 | Labeling prompt template management | Implemented |
| FR-5.5 | Stop/resume labeling job | Implemented |

### 2.6 Extraction Templates (Sub-feature)
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-6.1 | Save extraction configuration | Implemented |
| FR-6.2 | Load template to populate form | Implemented |
| FR-6.3 | Template favorites and usage count | Implemented |
| FR-6.4 | Import/export templates (JSON) | Implemented |
| FR-6.5 | "Save as Template" button in extraction modal | Implemented |
| FR-6.6 | Auto-generated template names from config | Implemented |

### 2.7 Labeling Prompt Templates (Sub-feature)
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-7.1 | Create custom labeling prompts | Implemented |
| FR-7.2 | Variable substitution in prompts | Implemented |
| FR-7.3 | System/user prompt separation | Implemented |
| FR-7.4 | Template favorites | Implemented |

### 2.8 NLP Analysis (Sub-feature - Added Dec 2025)
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-8.1 | Part-of-speech tagging for tokens | Implemented |
| FR-8.2 | Lemmatization for semantic grouping | Implemented |
| FR-8.3 | Named entity recognition | Implemented |
| FR-8.4 | BPE token reconstruction | Implemented |
| FR-8.5 | spaCy integration | Implemented |

### 2.9 Local LLM Integration (Sub-feature - Added Dec 2025)
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-9.1 | Ollama backend for auto-labeling | Implemented |
| FR-9.2 | Configurable LLM provider selection | Implemented |
| FR-9.3 | Same prompt template compatibility | Implemented |
| FR-9.4 | Offline operation support | Implemented |

### 2.10 Batch Extraction (Sub-feature - Added Jan 2026)
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-10.1 | Select multiple SAEs for batch extraction | Implemented |
| FR-10.2 | Sequential processing (one SAE at a time) | Implemented |
| FR-10.3 | Auto-continue after NLP analysis completion | Implemented |
| FR-10.4 | Batch progress tracking (position/total) | Implemented |
| FR-10.5 | Continue batch even if one job's NLP fails | Implemented |
| FR-10.6 | Batch ID linking related extraction jobs | Implemented |

### 2.11 Enhanced Per-Feature Labeling (Sub-feature - Added March 2026)

The highest-quality labeling path. Two-pass LLM interpretation of a single feature triggered from the Feature Detail modal via the sparkle (✨) button.

| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-11.1 | Two-pass strategy: per-example summaries then synthesis | Implemented |
| FR-11.2 | Parallel Pass-1 workers (configurable count via Settings) | Implemented |
| FR-11.3 | Support OpenAI API (`api.openai.com`) with stored API key | Implemented |
| FR-11.4 | Support any OpenAI-compatible endpoint (miLLM, Ollama, vLLM) | Implemented |
| FR-11.5 | Support reasoning-class models (gpt-5*, o1*, o3*, o4*) with correct token budgets | Implemented |
| FR-11.6 | Writes `name`, `category`, `description`, `notes` (markdown synthesis) | Implemented |
| FR-11.7 | Live progress via WebSocket (phase + examples_completed) | Implemented |
| FR-11.8 | Prevent duplicate active jobs per feature | Implemented |
| FR-11.9 | Restore in-progress job state when modal reopens | Implemented |
| FR-11.10 | Auto-populate edit form with completed label | Implemented |
| FR-11.11 | miLLM model pre-loading before inference calls | Implemented |
| FR-11.12 | Cleanup task for stuck queued/running jobs | Implemented |

**Pass 1 (per-example summarization):**
- For each of up to 20 activation examples, asks: *"What is this token doing in THIS specific context?"*
- Workers run in parallel (default 8)
- Result: list of one-sentence observations, one per example

**Pass 2 (synthesis):**
- Feeds all per-example summaries + prime token frequency distribution
- Asks: *"What single unifying concept explains all examples?"*
- Output: `{name, category, description, notes}` where `notes` contains the reasoning paragraph + a markdown table of per-example summaries

### 2.12 Star Color System (Sub-feature - Added March 2026)

Visual indicator on every feature card and the feature detail modal tracking labeling lifecycle.

| Color | Meaning | Transition |
|-------|---------|-----------|
| None (unstarred) | No special status | — |
| Yellow ⭐ | Manually starred by user | User clicks star |
| Purple 🔮 | Enhanced labeling in-flight | Set at job queue time |
| Aqua 🔵 | Enhanced labeling completed | Set on job completion — permanent |

**Key behavior:**
- Aqua is the only permanent state — never downgraded
- Bulk auto-labeling jobs skip features with `star_color='aqua'` (guard in all three persist loops)
- Star color persists in DB; frontend syncs from DB on modal mount
- `patchFeatureLocally()` in Zustand updates feature list rows and open modal simultaneously on WebSocket completion event

| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-12.1 | star_color column on features table (yellow/purple/aqua/null) | Implemented |
| FR-12.2 | Purple set when enhanced labeling job starts | Implemented |
| FR-12.3 | Aqua set when enhanced labeling job completes | Implemented |
| FR-12.4 | Aqua features skipped in bulk labeling jobs | Implemented |
| FR-12.5 | Star color included in all feature API responses | Implemented |
| FR-12.6 | Live update of feature row + modal on job completion | Implemented |

### 2.13 Context-Aware Labeling Template (Sub-feature - Added April 2026)

A new system template for bulk labeling that shifts focus from prime-token naming to semantic pattern identification.

| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-13.1 | Template uses `mistudio_context` type (full context windows) | Implemented |
| FR-13.2 | System message explicitly deprioritizes the prime token | Implemented |
| FR-13.3 | User prompt asks model to identify shared semantic pattern across ALL examples | Implemented |
| FR-13.4 | Includes 3 negative (low-activation) counter-examples | Implemented |
| FR-13.5 | Output `specific` slug names the pattern, not the token | Implemented |
| FR-13.6 | Template marked `is_system=true` — visible to all users, cannot be deleted | Implemented |

---

## 3. Feature Detail Modal

### 3.1 Components
- **Summary**: Feature index, activation stats, labels, label source badge
- **Enhanced Labeling Panel**: Sparkle button (✨), progress display (phase + examples), completed label auto-populate
- **Notes Section**: Collapsible, renders markdown (tables, lists, bold) via react-markdown + remark-gfm; bounded to max-h-96 with scroll
- **Top Activations**: Examples with context highlighting
- **Token Analysis**: Distribution of activating tokens
- **Edit Labels**: Manual label input (auto-populated from enhanced labeling on completion)

### 3.2 Token Highlighting
```
Context: "The cat sat on the [mat] and looked around"
                                 ^^^
         Activation: 4.52 at position 6
```

---

## 4. User Interface

### 4.1 Features Panel
```
┌─────────────────────────────────────────────────────────────┐
│ Features                    [Extract] [Auto-Label] [Export] │
├─────────────────────────────────────────────────────────────┤
│ SAE: [gemma-2b-layer12 ▾]  Search: [________]  Filter: [▾] │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Feature #42                                    [Edit]   │ │
│ │ Label: "expressions of romantic love"                   │ │
│ │ Category: Emotion > Positive > Love                     │ │
│ │ Freq: 0.023% | Max: 8.4 | Mean: 2.1                    │ │
│ │ Top tokens: love, heart, romance, dear, beloved         │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Feature #156                                   [Edit]   │ │
│ │ Label: [unlabeled]                                      │ │
│ │ Freq: 0.045% | Max: 6.2 | Mean: 1.8                    │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Extraction Modal
- SAE selection
- Dataset/tokenization selection
- Activation threshold
- Context window size
- Token filters
- Template load/save

### 4.3 Auto-Label Modal
- Feature selection (all, unlabeled, custom range)
- Prompt template selection
- Batch size configuration
- Progress display
- Stop/resume controls

---

## 5. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/features` | GET | List features with filtering |
| `/api/v1/features/{id}` | GET | Get feature details |
| `/api/v1/features/{id}` | PATCH | Update labels |
| `/api/v1/features/extraction` | POST | Start extraction job |
| `/api/v1/features/extraction/{id}` | GET | Get extraction status |
| `/api/v1/features/labeling` | POST | Start auto-labeling |
| `/api/v1/features/labeling/{id}` | GET | Get labeling status |
| `/api/v1/features/{id}/label/enhanced` | POST | Start enhanced two-pass labeling job |
| `/api/v1/features/{id}/label/enhanced/latest` | GET | Get latest enhanced labeling job for a feature |
| `/api/v1/features/export` | POST | Export to JSON |
| `/api/v1/extraction-templates` | GET/POST | Template CRUD |
| `/api/v1/labeling-prompt-templates` | GET/POST | Prompt CRUD |
| `/api/v1/labeling/models/openai` | POST | Fetch available models from any OpenAI-compatible endpoint |

---

## 6. Data Model

### 6.1 Feature Table
```sql
CREATE TABLE features (
    id UUID PRIMARY KEY,
    training_id UUID REFERENCES trainings(id),
    external_sae_id UUID REFERENCES external_saes(id),
    neuron_index INTEGER NOT NULL,
    name VARCHAR(500),           -- Feature slug label
    category VARCHAR(255),       -- Category label
    description TEXT,            -- Long-form description
    notes TEXT,                  -- Markdown-formatted synthesis notes (enhanced labeling)
    label_source VARCHAR(50),    -- auto, user, llm, local_llm, openai, enhanced_llm
    star_color VARCHAR(20),      -- null, 'yellow', 'purple', 'aqua'
    is_favorite BOOLEAN,
    statistics JSONB,            -- frequency, max, mean, interpretability
    top_tokens JSONB,
    nlp_analysis JSONB,          -- pre-computed spaCy analysis
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### 6.4 EnhancedLabelingJob Table (Added March 2026)
```sql
CREATE TABLE enhanced_labeling_jobs (
    id VARCHAR(255) PRIMARY KEY,   -- elj_{neuron_index}_{timestamp_ms}
    feature_id VARCHAR(255) REFERENCES features(id) ON DELETE CASCADE,
    status VARCHAR(50),            -- queued, running, completed, failed
    phase VARCHAR(50),             -- pass1, pass2, null
    method VARCHAR(50),            -- openai, openai_compatible
    endpoint VARCHAR(500),
    model VARCHAR(255),
    workers INTEGER,
    examples_total INTEGER,
    examples_completed INTEGER,
    pass1_summaries JSONB,
    raw_synthesis TEXT,
    error_message TEXT,
    celery_task_id VARCHAR(255),
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

### 6.2 FeatureActivation Table
```sql
CREATE TABLE feature_activations (
    id UUID PRIMARY KEY,
    feature_id UUID REFERENCES features(id) ON DELETE CASCADE,
    activation_value FLOAT NOT NULL,
    token VARCHAR(255),
    token_id INTEGER,
    context_before TEXT,
    context_after TEXT,
    position INTEGER,
    sample_index INTEGER,
    created_at TIMESTAMP
);
```

### 6.3 ExtractionJob Table
```sql
CREATE TABLE extraction_jobs (
    id UUID PRIMARY KEY,
    sae_id UUID,  -- training_id or external_sae_id
    dataset_id UUID REFERENCES datasets(id),
    config JSONB NOT NULL,
    status VARCHAR(50),
    progress FLOAT,
    features_found INTEGER,
    error_message TEXT,
    created_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

---

## 7. WebSocket Channels

| Channel | Events | Payload |
|---------|--------|---------|
| `extraction/{id}` | `extraction:progress` | See detailed payload below |
| `extraction/{id}` | `extraction:completed` | `{features_count, nlp_status?}` |
| `extraction/{id}` | `extraction:failed` | `{error_message}` |
| `labeling/{id}` | `progress` | `{progress, labeled_count}` |
| `labeling/{id}` | `results` | `{feature_id, label, confidence}` |
| `labeling/{id}` | `completed` | `{total_labeled}` |
| `enhanced_labeling/{job_id}` | `enhanced_labeling:progress` | `{job_id, phase, examples_completed, examples_total}` |
| `enhanced_labeling/{job_id}` | `enhanced_labeling:completed` | `{job_id, name, category, description, notes}` |
| `enhanced_labeling/{job_id}` | `enhanced_labeling:failed` | `{job_id, error_message}` |

### 7.1 Extraction Progress Payload (Jan 2026)
```json
{
  "extraction_id": "uuid",
  "status": "extracting",
  "sae_id": "uuid",
  "progress": 0.45,
  "features_extracted": 7372,
  "total_features": 16384,
  "current_batch": 23,
  "total_batches": 50,
  "samples_processed": 23000,
  "total_samples": 50000,
  "samples_per_second": 125.5,
  "eta_seconds": 216,
  "features_in_heap": 14521,
  "heap_examples_count": 72605
}
```

---

## 8. Key Files

### Backend
- `backend/src/services/extraction_service.py` - Extraction logic
- `backend/src/services/feature_service.py` - Feature management
- `backend/src/services/labeling_service.py` - Labeling orchestration (bulk)
- `backend/src/services/openai_labeling_service.py` - OpenAI SDK integration (bulk)
- `backend/src/services/enhanced_labeling_service.py` - Two-pass enhanced labeling (per-feature)
- `backend/src/workers/extraction_tasks.py` - Celery extraction task
- `backend/src/workers/enhanced_labeling_tasks.py` - Celery enhanced labeling task
- `backend/src/workers/cleanup_stuck_enhanced_labeling.py` - Stuck job cleanup
- `backend/src/api/v1/endpoints/features.py` - Feature API routes
- `backend/src/api/v1/endpoints/enhanced_labeling.py` - Enhanced labeling API routes
- `backend/src/models/enhanced_labeling_job.py` - EnhancedLabelingJob model

### Frontend
- `frontend/src/components/panels/FeaturesPanel.tsx` - Main panel
- `frontend/src/components/features/FeatureDetailModal.tsx` - Detail view (+ enhanced labeling panel + markdown notes)
- `frontend/src/components/features/StartExtractionModal.tsx` - Extraction config
- `frontend/src/components/features/TokenHighlight.tsx` - Context display
- `frontend/src/components/panels/ExtractionTemplatesPanel.tsx` - Templates
- `frontend/src/components/panels/LabelingPromptTemplatesPanel.tsx` - Prompts
- `frontend/src/hooks/useEnhancedLabeling.ts` - Enhanced labeling lifecycle hook
- `frontend/src/stores/featuresStore.ts` - Zustand store (incl. `patchFeatureLocally()`, `setStarColor()`)

---

## 9. Dependencies

| Feature | Dependency Type |
|---------|-----------------|
| SAE Training | Provides trained SAE |
| SAE Management | Can extract from external SAEs |
| Dataset Management | Provides extraction data |
| Model Steering | Features used for steering |
| Neuronpedia Export | Features exported |

---

## 10. Testing Checklist

- [x] Extract features from trained SAE
- [x] Extract from external SAE
- [x] Feature browser pagination
- [x] Search features by label
- [x] Filter by statistics
- [x] Manual label editing
- [x] Auto-labeling with GPT-4o
- [x] Labeling prompt templates
- [x] Extraction templates
- [x] Token highlighting
- [x] Export features to JSON
- [x] Batch extraction with multiple SAEs (Jan 2026)
- [x] Live progress metrics display (Jan 2026)
- [x] Save as Template from extraction modal (Jan 2026)
- [x] Sequential batch processing with NLP continuation (Jan 2026)
- [x] Labeling: drag-to-resize results window (Feb 2026)
- [x] Labeling: maximize/restore toggle (Feb 2026)
- [x] Labeling: configurable NLP analysis per template (Feb 2026)
- [x] Labeling: configurable batch_size 1-100 (Feb 2026)
- [x] Labeling: configurable max_tokens 50-8000 (Mar 2026)
- [x] Labeling: configurable api_timeout 30-600s (Mar 2026)
- [x] Labeling: Fetch Models button for OpenAI endpoint (Mar 2026)
- [x] Labeling: reasoning model support with think tag stripping (Mar 2026)
- [x] Labeling: OpenAI-compatible endpoint support (e.g., miLLM, Ollama) (Mar 2026)
- [x] Labeling: model/layer/hook display on job cards (Mar 2026)
- [x] Enhanced per-feature labeling: two-pass LLM strategy from Feature Detail modal (Mar 2026)
- [x] Enhanced labeling: OpenAI API support with stored API key (Apr 2026)
- [x] Enhanced labeling: reasoning model support (gpt-5, o1, o3, o4) with max_completion_tokens (Apr 2026)
- [x] Enhanced labeling: OpenAI SDK replaces hand-rolled httpx (Apr 2026)
- [x] Star color system: yellow/purple/aqua tracking labeling lifecycle (Mar 2026)
- [x] Bulk labeling guards: skip aqua features (Apr 2026)
- [x] Feature notes: collapsible, markdown-rendered (react-markdown + remark-gfm) with bounded scroll (Apr 2026)
- [x] Context-Aware Labeling template: semantic pattern focus vs prime-token naming (Apr 2026)
- [x] Settings-driven enhanced labeling: method (OpenAI vs OpenAI-Compatible), model, Fetch Models button (Apr 2026)
- [ ] Enhanced labeling: edge-tier analysis and open-question generation
- [ ] Enhanced labeling: confidence threshold to auto-accept vs. flag for review

---

## Doc-Refresh Corrections (2026-07-11)

Authoritative reference, verified against the code. Supersedes §5 and §6.1/§6.2.

### Data model (real)
- **`features`** — PK `id String` (`feat_{training}_{neuron}` / `feat_sae_{sae}_{neuron}`);
  FKs `training_id` (nullable), `external_sae_id` (nullable), `extraction_job_id`,
  `labeling_job_id`; `neuron_index`; `name`, `category`, `description`,
  `label_source` (enum `label_source_enum`: `auto|user|llm|local_llm|openai|enhanced_llm`);
  **top-level indexed stats** `activation_frequency`, `interpretability_score`,
  `max_activation`, `mean_activation` (NOT nested in a `statistics` JSONB);
  `is_favorite`, `star_color` (`yellow|purple|aqua|null`), `notes`,
  `example_tokens_summary` (JSONB), `nlp_analysis` (JSONB), `nlp_processed_at`,
  timestamps + `labeled_at`.
- **`feature_activations`** — composite PK `(id BigInteger, feature_id)` (range-
  partitioned by feature_id); `sample_index`, `max_activation`, `tokens` (JSONB),
  `activations` (JSONB), `prefix_tokens`/`prime_token`/`suffix_tokens`,
  `prime_activation_index`, `created_at`. *(Not `activation_value`/`token`/
  `context_before`/`context_after`.)*
- **`enhanced_labeling_jobs`** — as documented in §6.4 (accurate).

### API endpoints (real)
Feature/analysis routes (router mounted **without prefix** — absolute paths):
`GET /extractions` · `DELETE /extractions/{id}` · `GET /trainings/{tid}/features` ·
`GET /extractions/{eid}/features` · `GET/PATCH /features/{id}` ·
`POST /features/{id}/favorite` · `POST /features/{id}/star`
(`?star_color=yellow|purple|aqua`) · `GET /features/{id}/examples` ·
`GET /features/{id}/token-analysis` · `GET /features/{id}/logit-lens` ·
`GET /features/{id}/correlations` · `GET /features/{id}/ablation` ·
`POST /features/{id}/analyze-nlp` · `GET /features/{id}/nlp-analysis` ·
`POST /extractions/{id}/analyze-nlp` · `.../cancel-nlp` · `.../reset-nlp`.
Enhanced labeling: `POST /features/{id}/label/enhanced` ·
`GET /features/{id}/label/enhanced/latest`.
Bulk labeling lives in a **separate `labeling` router**. Extraction is *started*
via Feature 002's `POST /models/{id}/extract-activations` and Feature 005's
`POST /saes/{id}/extract-features` (see Extraction Architecture note below).

### Extraction architecture (cross-feature)
Two distinct pipelines share the word "extraction":
`ActivationExtraction` (model→raw activations, Feature 002, `/models/*`) →
`ExtractionJob` (SAE→features, Feature 004/005, `/saes/*`). Different tables,
tasks, and status enums.

---

*Related: [Project PRD](000_PPRD|miStudio.md) | [TDD](../tdds/004_FTDD|Feature_Discovery.md) | [TID](../tids/004_FTID|Feature_Discovery.md)*
