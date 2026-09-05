# Codebase Findings — Knowledge Graph Analysis (2026-05-28)

Surfaced by the understand-anything knowledge graph analysis. These are real issues
identified when 853-node, 1483-edge graph was assembled from 591 files across 35
analysis batches. The 119 "dropped edges" (imports that referenced paths that didn't
exist) are the primary signal source — each represents a place where multiple
independent LLM analyses reached for a file they expected to exist.

---

## Category 1: Files That Exist But Were Missed in the Scan

Four files exist on disk but were not assigned to any analysis batch, so they have
no nodes in the knowledge graph. They will appear in the next incremental update;
note them here for awareness.

| File | Lines | What it is |
|------|-------|------------|
| `backend/src/services/model_service.py` | 415 | Model CRUD service (distinct from model_loader.py) |
| `frontend/src/components/extractionTemplates/ExtractionTemplateForm.tsx` | 580 | Extraction template creation form |
| `frontend/src/components/extractionTemplates/ExtractionTemplateCard.tsx` | 180 | Template display card |
| `frontend/src/components/extractionTemplates/ExtractionTemplateList.tsx` | 195 | Template list component |

**Action:** These will be included automatically in the next `/understand` run
(incremental update will pick them up). No code changes needed.

---

## Category 2: Architectural Naming & Organization Issues

Each of these was inferred by 10–30 independent LLM analyses that reached for a
file they expected to exist based on naming conventions. The more analyses that
expected the same missing path, the stronger the signal.

---

### 2.1 DB Session Management in Wrong Module (HIGH — 30+ references)

**Expected:** `backend/src/db/session.py`  
**Actual:** `backend/src/core/database.py` + `backend/src/core/deps.py`

The `db/` directory exists but only contains `schema_validator.py`. The actual
database session factory (`AsyncSessionLocal`) and the `get_db` FastAPI dependency
live in `core/`, which is the wrong conceptual home — `core/` is for app-wide
infrastructure (config, celery, encryption), not DB session management.

**Impact:** Every service that needs a DB session imports from `core/deps.py` or
`core/database.py`, mixing database concerns into the application core layer.

**Fix options:**
- Move `AsyncSessionLocal` and `get_db` to `backend/src/db/session.py`
- Update all `from ..core.database import AsyncSessionLocal` and
  `from ..core.deps import get_db` imports across the codebase

---

### 2.2 SAE Model Has Misleading "External" Prefix (MEDIUM — 10+ references)

**Expected:** `backend/src/models/sae.py`  
**Actual:** `backend/src/models/external_sae.py`

"External" is an implementation detail (distinguishing user-imported SAEs from
miStudio-trained SAEs) that has leaked into the model file name. Every service that
manages SAEs imports from `external_sae.py`, but they're often just "SAEs" from the
consumer's perspective.

**Fix options:**
- Rename `external_sae.py` → `sae.py` and update all imports
- OR keep `external_sae.py` but add a `sae.py` barrel that re-exports:
  `from .external_sae import ExternalSAE`

---

### 2.3 Extraction Model Split Across Two Files (MEDIUM — 10+ references)

**Expected:** `backend/src/models/extraction.py`  
**Actual:** `backend/src/models/activation_extraction.py` + `backend/src/models/extraction_job.py`

Services that work with extractions need to import from two separate files with
inconsistent naming (`activation_extraction` vs `extraction_job`). The `activation_`
prefix on one file is an internal detail.

**Fix options:**
- Create `backend/src/models/extraction.py` as a barrel re-exporting both:
  ```python
  from .activation_extraction import ActivationExtraction, ExtractionStatus
  from .extraction_job import ExtractionJob
  ```
- OR rename for consistency: `activation_extraction.py` → `extraction.py`,
  `extraction_job.py` → `extraction_job.py` (keep, it's clearly named)

---

### 2.4 Settings Service Has Verbose Inconsistent Name (LOW-MEDIUM — 5+ references)

**Expected:** `backend/src/services/settings_service.py`  
**Actual:** `backend/src/services/app_setting_service.py`

Every other service follows `{domain}_service.py` naming (`labeling_service.py`,
`training_service.py`, etc.). `app_setting_service.py` breaks this convention with
the `app_` prefix and singular `setting` instead of `settings`.

**Fix:** Rename `app_setting_service.py` → `settings_service.py` and update imports
in the settings endpoint and anywhere else that imports it.

---

### 2.5 `featuresStore.ts` Violates Single Responsibility (HIGH — 91 extraction refs)

**Expected:** `frontend/src/stores/extractionsStore.ts` (separate store)  
**Actual:** Extraction state merged into `frontend/src/stores/featuresStore.ts`

`featuresStore.ts` has 91 references to "extraction" — it manages both feature data
AND activation extraction jobs. These are separate concerns: extractions are the
computational jobs that produce activations; features are the interpretable units
extracted from those activations.

**Impact:** The store is 640 lines, hard to reason about, and anything that only
needs extraction state must import the full features store.

**Fix:** Split `featuresStore.ts`:
- `featuresStore.ts` — feature browsing, selection, labeling state
- `extractionsStore.ts` — extraction job state, progress, results

---

### 2.6 No Barrel Export for Frontend Types (LOW — 8+ references)

**Expected:** `frontend/src/types/index.ts`  
**Actual:** Types scattered across individual files (`model.ts`, `feature.ts`,
`training.ts`, `labeling.ts`, `steering.ts`, `sae.ts`, `system.ts`, etc.)

Without a barrel, consumers must know exactly which type file to import from. This
is a discoverability issue — the LLMs couldn't figure out the right import path
without one, and new developers face the same problem.

**Fix:** Create `frontend/src/types/index.ts` that re-exports everything:
```typescript
export * from './model';
export * from './feature';
export * from './training';
export * from './labeling';
// etc.
```

---

### 2.7 No Dedicated Extraction API Module (LOW — 5+ references)

**Expected:** `frontend/src/api/extractions.ts`  
**Actual:** Extraction API calls split across `frontend/src/api/models.ts` and
inline in `featuresStore.ts`

The models API (`models.ts`) handles extraction because extractions are scoped to
models, but this means extraction consumers import from `models.ts` even when they
don't care about models.

**Fix:** Create `frontend/src/api/extractions.ts` with dedicated extraction API
functions, and have `models.ts` re-export or delegate to it.

---

### 2.8 Labeling Prompts Have No API Module or Store (LOW — 3+ references)

**Expected:** `frontend/src/api/labelingPrompts.ts` + `frontend/src/stores/labelingPromptsStore.ts`  
**Actual:** Prompts functionality embedded in `LabelingPromptTemplatesPanel.tsx`

The backend has a full `labeling_prompt_templates` endpoint with CRUD operations,
but the frontend has no dedicated API module or store for it — the panel component
talks directly to the API. This means the panel is the only consumer; reusing
prompt state elsewhere requires importing the panel.

**Fix:** Extract `frontend/src/api/labelingPrompts.ts` and either a dedicated
store or fold into `labelingStore.ts` as a prompts sub-slice.

---

## Summary

| # | Issue | Severity | Effort |
|---|-------|----------|--------|
| 2.1 | DB session in wrong module (`core/` vs `db/`) | High | Medium |
| 2.5 | `featuresStore.ts` does too much (extraction + features) | High | Medium |
| 2.2 | `external_sae.py` misleading name | Medium | Low |
| 2.3 | Extraction model split inconsistently | Medium | Low |
| 2.4 | `app_setting_service.py` inconsistent naming | Low-Med | Low |
| 2.6 | No `types/index.ts` barrel | Low | Low |
| 2.7 | No dedicated `api/extractions.ts` | Low | Low |
| 2.8 | Labeling prompts have no API/store layer | Low | Low |

The highest-leverage fixes are **2.1** (DB session location) and **2.5**
(featuresStore SRP) because they're the most-referenced gaps and affect the most
files. Both are medium-effort refactors that would meaningfully improve navigability.

---

*Generated from knowledge graph analysis — `/understand` run on 2026-05-28*  
*Graph: 853 nodes, 1483 edges, 591 files analyzed*  
*Source: 119 dropped edges from merge-batch-graphs.py*
