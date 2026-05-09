# Project PRD: MechInterp Studio (miStudio)

**Document ID:** 000_PPRD|miStudio
**Version:** 3.0 (Security, Enhanced Labeling & Production Hardening)
**Last Updated:** 2026-04-26
**Status:** Active

---

## Executive Summary

MechInterp Studio (miStudio) is an open-source platform for Sparse Autoencoder (SAE) research that provides an end-to-end workflow for training SAEs, discovering interpretable features, and applying feature-based steering to transformer models. The platform has crossed 610+ commits, ships as a public release (v0.5.0), and is deployed on Kubernetes with a full CI/CD pipeline including Docker Scout, CodeQL, and supply-chain attestations.

### Project Metrics (as of April 2026)
- **Total Commits:** 610+
- **Database Migrations:** 60+
- **Backend Services:** 40+
- **Frontend Components:** 95+
- **Development Period:** October 2025 – Present
- **Release:** v0.5.0 (public, Apache 2.0)
- **Deployment:** Kubernetes (primary) + Docker Compose (secondary) + GCP

---

## 1. Vision & Goals

### 1.1 Vision
Democratize mechanistic interpretability research by providing a comprehensive, user-friendly workbench that enables researchers to train SAEs, discover features, and steer model behavior without requiring ML infrastructure expertise.

### 1.2 Goals
| Goal | Description | Status |
|------|-------------|--------|
| **Accessibility** | Make SAE research accessible to researchers without DevOps expertise | Achieved |
| **End-to-End Workflow** | Support complete pipeline from data to steering | Achieved |
| **Interoperability** | Compatible with HuggingFace, Neuronpedia, SAELens | Achieved |
| **Real-time Feedback** | Immediate progress updates for long-running operations | Achieved |
| **LLM-Assisted Interpretation** | AI-powered per-feature labeling (single and bulk) | Achieved |
| **Production Security** | Non-root containers, CVE scanning, supply-chain attestations | Achieved |
| **Scalability** | Multi-GPU monitoring, per-GPU job routing, per-GPU WebSocket metrics | Achieved |

### 1.3 Success Criteria
- [x] Users can download datasets from HuggingFace and tokenize them
- [x] Users can download and quantize models from HuggingFace
- [x] Users can train SAEs with multiple architectures (Standard, JumpReLU, TopK, Skip, Transcoder, Standard-Anthropic)
- [x] Users can extract and browse interpretable features
- [x] Users can apply feature-based steering with comparison mode
- [x] Users can export to Neuronpedia-compatible format and push to a local Neuronpedia instance
- [x] All long-running operations provide real-time progress via WebSocket
- [x] Users can trigger two-pass enhanced LLM labeling per feature from the Feature Detail modal
- [x] Bulk labeling runs against OpenAI API or any OpenAI-compatible local LLM (miLLM, Ollama, vLLM)
- [x] Platform ships with hardened security posture (non-root containers, CodeQL, supply-chain attestations)
- [x] API keys and sensitive settings are encrypted at rest (AES-256-GCM)
- [x] Users can monitor each GPU individually (VRAM, utilization, temperature, power) via per-GPU WebSocket channels
- [x] Users can route extraction jobs to a specific GPU via `gpu_id` parameter
- [ ] Users can distribute SAE training across multiple GPUs simultaneously (DDP — planned)

---

## 2. Feature Inventory

### 2.1 MVP Features (Implemented)

| # | Feature | Description | Status |
|---|---------|-------------|--------|
| 1 | Dataset Management | HuggingFace download, tokenization, statistics | Complete |
| 2 | Model Management | Model download, quantization, architecture viewer | Complete |
| 3 | SAE Training | Multi-architecture training with real-time metrics | Complete |
| 4 | Feature Discovery | Extraction, labeling, auto-labeling, enhanced labeling, search | Complete |
| 5 | SAE Management | Trained & external SAE management, format conversion | Complete |
| 6 | Model Steering | Feature interventions, combined multi-feature, comparison, export | Complete |
| 7 | Neuronpedia Export | Community-format export + push to local Neuronpedia instance | Complete |
| 8 | System Monitoring | GPU/CPU/Memory/Disk/Network monitoring | Complete |
| 9 | Settings & Configuration | Encrypted API keys, endpoints, labeling defaults | Complete |
| 10 | Multi-GPU Scalability | Per-GPU monitoring, job routing, aggregated/per-GPU views | Partially Complete (DDP training planned) |

### 2.2 Template Systems (Sub-features)

| Template Type | Parent Feature | Purpose |
|---------------|----------------|---------|
| Training Templates | SAE Training | Save/load training configurations |
| Extraction Templates | Feature Discovery | Save/load extraction configurations |
| Labeling Prompt Templates | Feature Discovery | Customize auto-labeling prompts (multiple types including context-aware) |
| Prompt Templates | Model Steering | Save/load steering experiment prompts |

### 2.3 Enhanced Labeling (Major Sub-feature — Added March 2026)

Two-pass LLM interpretation of individual features, distinct from bulk auto-labeling:

| Pass | What Happens | Purpose |
|------|-------------|---------|
| Pass 1 | Per-example summarization (parallel, N workers) | "What is this token doing in THIS specific context?" |
| Pass 2 | Synthesis across all summaries | "What single concept unifies all examples?" |

Tracked via star color on every feature card:
- **Yellow star:** manually starred
- **Purple star:** enhanced labeling in-flight
- **Aqua star:** enhanced labeling completed (permanent, protected from bulk overwrite)

---

## 3. Feature Details

### 3.1 Dataset Management

**Purpose:** Ingest and prepare training data for SAE training.

**Capabilities:**
- Download datasets from HuggingFace Hub
- Tokenize with configurable parameters (max_length, stride, truncation)
- Multi-tokenization support (create multiple tokenizations per dataset)
- Token filtering (minimum length, special tokens, stop words)
- Statistics visualization (vocabulary distribution, sequence lengths)
- Sample browser with pagination
- Real-time progress via WebSocket
- Bytes-safe sample handling (HuggingFace binary data)

**Key Files:**
- Backend: `dataset_service.py`, `tokenization_service.py`, `dataset_tasks.py`
- Frontend: `DatasetsPanel.tsx`, `DownloadForm.tsx`, `TokenizationStatsModal.tsx`

**API Endpoints:**
- `GET/POST /api/v1/datasets` - CRUD operations
- `POST /api/v1/datasets/{id}/download` - Start download
- `POST /api/v1/datasets/{id}/tokenize` - Start tokenization
- `GET /api/v1/datasets/{id}/statistics` - Get tokenization stats

---

### 3.2 Model Management

**Purpose:** Download and manage transformer models for analysis.

**Capabilities:**
- Download models from HuggingFace Hub
- Support for gated models with HF token authentication (key stored encrypted in Settings)
- Quantization options (4-bit, 8-bit via bitsandbytes)
- Model architecture viewer (layers, parameters)
- Dynamic architecture discovery via `discover_transformer_structure()` — any transformer model works without whitelisting
- Memory estimation before download
- Real-time download progress via WebSocket

**Key Files:**
- Backend: `model_service.py`, `model_tasks.py`, `layer_discovery.py`
- Frontend: `ModelsPanel.tsx`, `ModelDownloadForm.tsx`, `ModelPreviewModal.tsx`

**API Endpoints:**
- `GET/POST /api/v1/models` - CRUD operations
- `POST /api/v1/models/{id}/download` - Start download
- `GET /api/v1/models/{id}/architecture` - Get architecture info

---

### 3.3 SAE Training

**Purpose:** Train Sparse Autoencoders on model activations.

**SAE Architectures (6 paper-grounded frameworks):**
| Architecture | Description | Key Feature |
|-------------|-------------|-------------|
| Standard (Anthropic) | L1 sparsity, gated variant | Anthropic-style training |
| JumpReLU | Gemma Scope-style with differentiable L0 | State-of-the-art sparsity control |
| TopK | OpenAI-style guaranteed sparsity | Exact-K active features |
| Skip | Residual connections | Better reconstruction |
| Transcoder | Layer-to-layer mapping | Activation transcoding |
| Standard (EleutherAI) | Standard with EleutherAI conventions | Community-compatible |

**Key Implementation Details:**
- JumpReLU uses sigmoid STE for differentiable L0, count-based (not fraction-based) per Gemma Scope paper
- `sparsity_coeff` is paper-scale (default 1e-3); separate from L1 `l1_alpha`
- Sparsity warmup (10K steps for JumpReLU), EMA dead neuron detection

**Capabilities:**
- Real-time metrics streaming (loss, L0, L1, reconstruction error, FVU)
- Checkpoint management with configurable intervals
- Dead neuron detection and optional resampling
- Training templates for reproducibility
- Retry failed trainings with same configuration
- Bulk delete for cleanup
- Multi-extraction cached activations training support

**Key Files:**
- Backend: `training_service.py`, `sparse_autoencoder.py`, `jumprelu_sae.py`, `training_tasks.py`
- Frontend: `TrainingPanel.tsx`, `StartTrainingModal.tsx`, `TrainingCard.tsx`
- Templates: `TrainingTemplatesPanel.tsx`, `TrainingTemplateForm.tsx`

**API Endpoints:**
- `GET/POST /api/v1/trainings` - CRUD operations
- `GET /api/v1/trainings/{id}/metrics` - Get training metrics
- `GET /api/v1/trainings/{id}/checkpoints` - List checkpoints
- `POST /api/v1/trainings/{id}/stop` - Stop training
- `POST /api/v1/trainings/{id}/retry` - Retry failed training

---

### 3.4 Feature Discovery

**Purpose:** Extract, analyze, and interpret features from trained SAEs.

**Capabilities:**
- Batch extraction with GPU optimization
- Context window capture (tokens before/after activation)
- Token filtering during extraction
- Feature search by label, category, statistics, activation ranges
- Example export to JSON
- NLP analysis of top-activating tokens (spaCy)
- BPE token reconstruction for human-readable text

**Labeling Methods:**
| Method | Provider | Trigger | Notes |
|--------|---------|---------|-------|
| Manual | User | Edit form in modal | Full CRUD |
| Bulk Auto-Labeling | OpenAI API / miLLM / Ollama | Start Labeling job | configurable prompt template |
| Enhanced Per-Feature | OpenAI API / miLLM / Ollama | Sparkle button in Feature Detail modal | Two-pass, highest quality |

**Enhanced Per-Feature Labeling (Added March 2026):**
The highest-quality labeling path. Two-pass strategy:
- **Pass 1 (parallel):** For each activation example, asks: "What is this token doing in THIS specific context?" Runs N workers concurrently (configurable).
- **Pass 2 (synthesis):** Feeds all per-example summaries and asks: "What is the unifying concept?" Produces `name`, `category`, `description`, `notes` (reasoning + per-example table in markdown).
- Triggered via the sparkle (✨) button in the Feature Detail modal.
- Works against any OpenAI-compatible endpoint (miLLM, Ollama, vLLM) or the real OpenAI API.
- Supports reasoning-class models (`gpt-5*`, `o1*`, `o3*`, `o4*`) via `max_completion_tokens` — no `temperature` parameter.
- Uses the official OpenAI Python SDK (eliminates model-specific parameter hand-rolling).
- Star color tracks status: purple (in-flight) → aqua (completed, protected from bulk overwrite).
- Feature row and modal live-update on WebSocket completion event without page reload.

**Bulk Labeling improvements (March–April 2026):**
- miLLM model pre-loading before inference (prevents 503s after server restart)
- Bulk jobs skip features already labeled by enhanced labeling (aqua star guard)
- Context-Aware template available: uses full `prefix << prime >> suffix` context windows; instructs the model to find the shared semantic PATTERN across all examples, not just name the prime token

**Labeling Prompt Templates:**
- Multiple template types: `legacy` (token stats), `mistudio_context` (full context windows), `anthropic_logit`, `eleutherai_detection`
- Context-Aware Labeling template (system template, April 2026): designed to produce semantic pattern labels rather than prime-token labels
- Configurable: temperature, max_tokens, prime_token_marker, include prefix/suffix, negative examples

**Star Color System:**
- Null: unstarred
- Yellow (⭐): manually starred by user
- Purple: enhanced labeling in-flight
- Aqua (🔵): enhanced labeling completed — never downgraded, protected from bulk overwrite

**Key Files:**
- Backend: `extraction_service.py`, `feature_service.py`, `labeling_service.py`, `openai_labeling_service.py`, `enhanced_labeling_service.py`, `enhanced_labeling_tasks.py`
- Frontend: `FeaturesPanel.tsx`, `FeatureDetailModal.tsx`, `StartExtractionModal.tsx`
- Hooks: `useEnhancedLabeling.ts`
- Templates: `ExtractionTemplatesPanel.tsx`, `LabelingPromptTemplatesPanel.tsx`

**API Endpoints:**
- `GET /api/v1/features` - List features with filtering
- `PATCH /api/v1/features/{id}` - Update labels
- `POST /api/v1/features/extraction` - Start extraction job
- `POST /api/v1/features/labeling` - Start auto-labeling job
- `POST /api/v1/features/{id}/label/enhanced` - Start enhanced two-pass labeling
- `GET /api/v1/features/{id}/label/enhanced/latest` - Get latest enhanced labeling job

**WebSocket Channel:**
- `enhanced_labeling/{job_id}` — progress, completed, failed events

---

### 3.5 SAE Management

**Purpose:** Manage both trained and external SAEs.

**SAE Sources:**
- **Trained:** SAEs trained within miStudio (linked to training record)
- **HuggingFace:** Download from model hub (HF token from Settings if gated)
- **Gemma Scope:** Pre-trained Google SAEs (special download flow)
- **Batch downloads:** Multi-select SAEs from HuggingFace in a single operation

**Format Support:**
- **Community Standard:** SAELens-compatible (cfg.json + sae_weights.safetensors)
- **miStudio Native:** Internal format with extended metadata
- Automatic format detection and conversion
- Batch extraction support (multiple SAEs from one dataset pass)

**Key Files:**
- Backend: `sae_manager_service.py`, `huggingface_sae_service.py`, `sae_converter.py`
- Frontend: `SAEsPanel.tsx`, `SAECard.tsx`, `DownloadFromHF.tsx`

**API Endpoints:**
- `GET/POST /api/v1/saes` - CRUD operations
- `POST /api/v1/saes/download-hf` - Download from HuggingFace
- `POST /api/v1/saes/{id}/convert` - Convert format
- `POST /api/v1/saes/batch-download` - Multi-select batch download

---

### 3.6 Model Steering

**Purpose:** Control model behavior via feature interventions.

**Steering Types:**
- **Activation:** Add/subtract feature directions to the residual stream
- **Suppression:** Reduce specific feature activations toward zero

**Capabilities:**
- Multi-feature selection (select multiple features for steering)
- Combined multi-feature generation (apply all features in a single pass) ✅ Complete
- Strength sweep (test multiple intensities in one run)
- Comparison mode (steered vs. unsteered side-by-side)
- Neuronpedia-compatible calibration
- Prompt templates for repeatable experiments
- Async Celery execution with GPU isolation (prevents CUDA re-initialization conflicts)
- Zombie process detection for steering workers

**Implementation Notes:**
- Steering migrated from synchronous API to async Celery tasks with GPU isolation
- Dynamic layer discovery replaces hardcoded architecture if/elif chains
- Any transformer model (Llama, Gemma, LFM2, GraniteMoEHybrid) works without code changes

**Key Files:**
- Backend: `steering_service.py`, `forward_hooks.py`, `steering_tasks.py`, `layer_discovery.py`
- Frontend: `SteeringPanel.tsx`, `FeatureBrowser.tsx`, `ComparisonResults.tsx`, `SelectedFeatureCard.tsx`
- Templates: `PromptTemplatesPanel.tsx`, `PromptListEditor.tsx`

**API Endpoints:**
- `POST /api/v1/steering/generate` - Generate with steering (async)
- `POST /api/v1/steering/compare` - Compare steered vs. baseline
- `POST /api/v1/steering/sweep` - Multi-strength test
- `POST /api/v1/steering/combined` - Combined multi-feature generation

---

### 3.7 Neuronpedia Export & Push

**Purpose:** Share SAE findings with the research community.

**Export Contents:**
- Feature activation examples (top activating tokens with context)
- Logit lens data (promoted/suppressed tokens per feature)
- Activation histograms
- Feature explanations (name + description combined as "name: description")
- SAELens-compatible weights

**Neuronpedia Local Push (Added Jan 2026):**
- Direct push to a local Neuronpedia instance via async Celery task
- WebSocket progress tracking
- Job tracked in DB for Active Operations monitor
- Handles FK constraint ordering (Model before Source)
- Polling fallback after browser refresh

**Key Files:**
- Backend: `neuronpedia_export_service.py`, `logit_lens_service.py`, `neuronpedia_local_service.py`, `neuronpedia_push_tasks.py`
- Frontend: `ExportToNeuronpedia.tsx`

**API Endpoints:**
- `POST /api/v1/neuronpedia/export` - Start export job
- `GET /api/v1/neuronpedia/export/{id}` - Get job status
- `GET /api/v1/neuronpedia/export/{id}/download` - Download archive
- `POST /api/v1/neuronpedia/push` - Push to local Neuronpedia instance

---

### 3.8 System Monitoring

**Purpose:** Track resource utilization during operations.

**Metrics:**
| Category | Metrics |
|----------|---------|
| GPU | Utilization %, memory used/total, temperature, power draw |
| CPU | Per-core utilization % |
| Memory | RAM used/total, swap used/total |
| Disk | Read/write I/O rates (MB/s) |
| Network | Upload/download I/O rates (MB/s) |

**Implementation:**
- WebSocket streaming (2-second intervals via Celery Beat)
- Fallback to HTTP polling on WebSocket disconnect
- 1-hour rolling history with chart visualization
- Combined GPU utilization + temperature chart

**Key Files:**
- Backend: `system_monitor_service.py`, `system_monitor_tasks.py`, `websocket_emitter.py`
- Frontend: `SystemMonitor.tsx`, `UtilizationChart.tsx`, `useSystemMonitorWebSocket.ts`

**WebSocket Channels:**
- `system/gpu/{id}` - Per-GPU metrics
- `system/cpu` - CPU metrics
- `system/memory` - Memory metrics
- `system/disk` - Disk I/O
- `system/network` - Network I/O

---

### 3.9 Settings & Configuration Panel (Added Feb 2026)

**Purpose:** Manage API keys, endpoints, and application defaults from the UI — no server restarts or env var editing required.

**Tabs:**
| Tab | What It Configures |
|-----|-------------------|
| **Endpoints** | OpenAI-Compatible endpoint + model (used by all labeling paths), Ollama URL override, saved endpoint bookmarks |
| **API Keys** | OpenAI API key, HuggingFace token — both AES-256-GCM encrypted at rest |
| **Labeling** | Default batch size, max examples per feature, Enhanced Labeling method (OpenAI vs OpenAI-Compatible) + model, max parallel workers |
| **Display** | Theme preferences |

**Security:**
- All sensitive values encrypted before storage using AES-256-GCM with HKDF key derivation
- Masked display (e.g. `sk-...XXXX`) returned to frontend — real plaintext never sent to client after initial save
- `decrypt_value()` gracefully handles legacy plaintext rows (logs warning, returns as-is)
- Critical bug fixed (April 2026): upsert endpoint no longer commits the masked display string back to the DB

**Fetch Models Buttons:**
- Endpoints tab: queries the configured OpenAI-Compatible endpoint for available models
- Labeling tab (Enhanced Labeling → OpenAI method): queries `api.openai.com/v1/models` using the stored API key — populates a dropdown for model selection

**Key Files:**
- Backend: `settings.py` endpoint, `app_setting_service.py`, `encryption.py`
- Frontend: `SettingsPanel.tsx`, `useSettingsStore.ts`
- DB Model: `AppSetting` (key, value, is_sensitive, category)

**API Endpoints:**
- `GET /api/v1/settings` - List all settings
- `PUT /api/v1/settings` - Upsert a setting (encrypts if `is_sensitive=true`)
- `DELETE /api/v1/settings/{key}` - Remove a setting
- `POST /api/v1/labeling/models/openai` - Fetch available models from any OpenAI-compatible endpoint

---

### 3.10 Multi-GPU Scalability (Partially Complete)

**Purpose:** Enable distributed training and enhanced multi-GPU monitoring.

**Implemented (Dec 2025):**
- Per-GPU metrics collection via `GPUMonitorService` (`pynvml`-based enumeration)
- Real-time per-GPU WebSocket channels: `system/gpu/{gpu_id}` — utilization, VRAM, temperature, power
- Aggregated vs. per-GPU comparison view in System Monitor (commit 8cbe31c)
- `gpu_id` parameter for routing extraction jobs to specific GPUs
- GPU validation: error if requested GPU index exceeds `torch.cuda.device_count()`
- Emergency cleanup iterates all GPUs on worker shutdown
- GPU watchdog Celery task monitors per-device processes
- API endpoints: `/api/v1/system/gpu-list`, `/system/gpu/{gpu_id}`, `/system/gpu-metrics`, `/system/gpu-processes`

**Still Planned:**
- Distributed SAE training across multiple GPUs simultaneously (PyTorch DDP + NCCL)
- Data parallelism with gradient synchronization
- Automatic batch size scaling per GPU count
- Memory-based GPU recommendation

---

## 4. Technology Stack

### 4.1 Backend
| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.12 | Runtime |
| FastAPI | 0.100+ | REST API framework |
| PostgreSQL | 14+ | Primary database |
| Redis | 7+ | Message broker & cache |
| Celery | 5.x | Distributed task queue |
| SQLAlchemy | 2.0 | ORM with async support |
| Alembic | 1.x | Database migrations (60+) |
| PyTorch | 2.0+ | ML framework |
| Transformers | 5.x | HuggingFace models |
| huggingface-hub | 1.x | Hub API client |
| bitsandbytes | 0.41+ | Quantization |
| Socket.IO | 5.x | WebSocket server |
| OpenAI SDK | 1.x | OpenAI API + compatible endpoints |
| spaCy | 3.x | NLP analysis |
| cryptography | 46+ | AES-256-GCM key encryption |

### 4.2 Frontend
| Technology | Version | Purpose |
|-----------|---------|---------|
| React | 18+ | UI framework |
| TypeScript | 5.x | Type safety |
| Vite | 6.x | Build tool |
| Zustand | 4.x | State management |
| Tailwind CSS | 3.x | Styling (slate dark theme) |
| Recharts | 2.x | Data visualization |
| Lucide React | - | Icon library |
| Socket.IO Client | 4.x | WebSocket client |
| react-markdown | 9.x | Markdown rendering (feature notes) |
| remark-gfm | 4.x | GitHub-flavored markdown tables |

### 4.3 Infrastructure
| Technology | Purpose |
|-----------|---------|
| Kubernetes (primary) | Production orchestration (namespace: mistudio) |
| Docker Compose v2 | Development + secondary deployment |
| Nginx (unprivileged) | Reverse proxy — runs as uid 101, port 8080 |
| Celery Beat | Scheduled tasks (monitoring, cleanup) |
| GitHub Actions | CI: backend tests, frontend CI, Docker image builds |
| Docker Scout | Image vulnerability scanning (supply-chain policies) |
| CodeQL | Static application security testing (via hitsainet default setup) |

---

## 5. Architecture Highlights

### 5.1 WebSocket-First Real-time Updates
All long-running operations emit progress via WebSocket for immediate UI feedback:
- Channel pattern: `{entity_type}/{entity_id}`
- Automatic fallback to HTTP polling on disconnect
- Celery tasks emit via internal HTTP endpoint (`/api/internal/ws/emit`)

### 5.2 Celery Task Queue
Background processing for CPU/GPU-intensive operations:
- Queues: `high_priority`, `datasets`, `processing`, `training`, `extraction`, `sae`, `low_priority`
- Priority routing for training vs. extraction vs. labeling
- Celery Beat for periodic system monitoring and cleanup tasks (stuck job detection)

### 5.3 Architecture-Agnostic Model Support
`discover_transformer_structure()` in `layer_discovery.py` dynamically inspects loaded models:
- No hardcoded architecture whitelists — any transformer works
- Forward hooks placed dynamically based on discovered layer structure
- Supports LFM2 (Liquid Foundation Models), GraniteMoEHybrid, Llama, Gemma, Phi, Mistral, etc.

### 5.4 SAELens Compatibility
Community Standard format ensures interoperability:
- `cfg.json` with SAELens-compatible configuration
- `sae_weights.safetensors` for model weights
- Automatic format detection and conversion

### 5.5 Security Architecture
- **At-rest encryption:** AES-256-GCM with HKDF-SHA256 key derivation for all sensitive settings
- **Path injection prevention:** `resolve_user_path()` performs string-only normalization + containment check against trusted roots before any filesystem operation
- **Non-root containers:** Frontend (`nginx-unprivileged`, uid 101, port 8080); Backend entrypoint drops privileges to `mistudio` user after init
- **Supply-chain security:** SLSA provenance, SBOM, Docker Scout scanning on all image builds
- **Static analysis:** CodeQL with `security-extended` queries runs on every push to main (via hitsainet public repo)

---

## 6. External Integrations

| Integration | Purpose | Status |
|-------------|---------|--------|
| HuggingFace Hub | Dataset/model/SAE downloads | Complete |
| Neuronpedia | Export format + local instance push | Complete |
| SAELens | Weight format compatibility | Complete |
| OpenAI API | GPT-4o/GPT-5 auto-labeling (bulk + enhanced) | Complete |
| miLLM | Local GPU LLM server (OpenAI-compatible) | Complete |
| Ollama | Local LLM auto-labeling | Complete |
| vLLM | OpenAI-compatible inference (supported via endpoint config) | Complete |
| spaCy | NLP analysis for features | Complete |

---

## 7. Data Storage

### 7.1 Database Schema
- **25+ SQLAlchemy models** across core entities, templates, and settings
- **60+ Alembic migrations** for schema evolution
- JSONB columns for flexible metadata storage
- Key tables added since v2.1: `enhanced_labeling_jobs`, `app_settings`, `neuronpedia_pushes`

### 7.2 File Storage
- Local filesystem at configurable `DATA_DIR`
- Organized by entity type: `models/`, `datasets/`, `saes/`, `exports/`
- Safetensors format for model/SAE weights

---

## 8. Development & Deployment

### 8.1 Development Setup
```bash
# Add domain to hosts
sudo bash -c 'echo "127.0.0.1 mistudio.hitsai.local" >> /etc/hosts'

# Start all services
./start-mistudio.sh

# Access at http://mistudio.hitsai.local
```

### 8.2 Service Components
1. Docker Compose (PostgreSQL, Redis, Nginx)
2. Backend (FastAPI on port 8000)
3. Frontend (Nginx unprivileged on port 8080, mapped from external 80)
4. Celery Worker (background tasks — shares backend pod in K8s)
5. Celery Beat (scheduled tasks — shares backend pod in K8s)

### 8.3 Kubernetes Deployment
- **Host:** mcs-lnxgpu01 (192.168.244.61), GPU: NVIDIA RTX 3090 24GB
- **Namespace:** `mistudio`
- **Public URL:** `https://mistudio.hitsai.net` (via Cloudflare)
- **K8s URL:** `http://k8s-mistudio.hitsai.local`
- **Deploy command:** `k8s_deploy` (helper in `scripts/k8s-helpers.sh`)

### 8.4 CI/CD Pipeline
1. Push to `main` on `Onegaishimas/miStudio` (private)
2. `sync-to-clean.yml` mirrors to `hitsainet/miStudio` (public, filtered)
3. `docker-images.yml` builds and pushes `hitsai/mistudio-backend:latest` and `hitsai/mistudio-frontend:latest` with SLSA provenance + SBOM
4. Docker Scout scans each image on push
5. CodeQL Default Setup scans the public repo on each push

---

## 9. Related Documents

| Document | Path | Description |
|----------|------|-------------|
| Architecture Decision Record | `0xcc/adrs/000_PADR\|miStudio.md` | Technical decisions |
| Developer Guide | `0xcc/docs/Developer_Guide.md` | Implementation details |
| Feature PRDs | `0xcc/prds/001-009_FPRD\|*.md` | Individual feature specs |
| Technical Design Docs | `0xcc/tdds/*.md` | Design specifications |
| Implementation Docs | `0xcc/tids/*.md` | Implementation guidance |
| Task Lists | `0xcc/tasks/*.md` | Development tracking |

---

## 10. Recent Improvements (March–April 2026)

### 10.1 Enhanced Per-Feature Labeling
Complete two-pass LLM interpretation system triggered per-feature from the Feature Detail modal. Uses parallel per-example summarization then synthesizes a structured label with name, category, description, and markdown-formatted notes (reasoning + per-example table). Supports OpenAI API and any OpenAI-compatible local server.

### 10.2 OpenAI API Integration for Labeling
Both enhanced and bulk labeling can now target `api.openai.com` directly. API key stored encrypted in Settings → API Keys. Reasoning-class models (`gpt-5*`, `o1*`, `o3*`, `o4*`) automatically use `max_completion_tokens` with appropriate budgets. Official OpenAI Python SDK replaces hand-rolled httpx to avoid per-model parameter quirks.

### 10.3 Context-Aware Labeling Template
New `mistudio_context` template that shifts the bulk labeling frame from "what token does this feature fire on?" to "what semantic pattern is common across ALL activation contexts?". Uses full context windows (prefix/prime/suffix), includes counter-examples, and instructs the model to find the shared meaning rather than name the prime token.

### 10.4 Settings & API Key Management
DB-backed application settings with AES-256-GCM encryption. Settings Panel with Endpoints, API Keys, Labeling, and Display tabs. Critical encryption bug fixed: upsert endpoint previously wrote the masked display string back over the ciphertext on every save.

### 10.5 Security Hardening
Resolved all Dependabot CVEs, addressed all CodeQL findings (path injection, stack-trace exposure, supply-chain attestations). Frontend switched to non-root `nginx-unprivileged` base image. `resolve_user_path()` performs string-only validation before touching the filesystem.

### 10.6 Feature Notes UX
Feature detail modal notes section renders as markdown (react-markdown + remark-gfm), with proper table rendering for the per-example summary table generated by enhanced labeling. Bounded to `max-h-96` with scroll. Collapsible.

### 10.7 v0.5.0 Public Release
- Apache 2.0 license
- Versioning system (`VERSION` file, `/api/v1/version` endpoint)
- GitHub Actions test and build pipeline
- Docker Scout image scanning integration
- Public deployment at `mistudio.hitsai.net`

---

## 11. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-05 | Initial project vision and feature breakdown |
| 2.0 | 2025-12-05 | MVP complete — reflects actual implementation |
| 2.1 | 2025-12-16 | Post-MVP: NLP analysis, Ollama integration, infrastructure improvements |
| 3.0 | 2026-04-26 | Enhanced labeling, OpenAI API integration, context-aware labeling template, Settings Panel, security hardening, v0.5.0 public release, K8s production deployment, CI/CD pipeline |

---

*Generated: 2026-04-26*
*MechInterp Studio — v0.5.0 Production Release*
