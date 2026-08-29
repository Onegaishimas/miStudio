# Project PRD: MechInterp Studio (miStudio)

**Document ID:** 000_PPRD|miStudio
**Version:** 3.13 (Feature 30 — Labeling Prompt-Template Optimization: scoped non-persisting trials over a fixed panel, ranked by automated detection score, In Progress; Features 22–29 — J-Space capability family: a training-free, model-agnostic second dictionary substrate (Jacobian lens) reading what a model is poised to say per layer and position, annotating SAE dictionaries with workspace membership, and emitting a runtime watchlist for miLLM; opened by BRD-MIS-JSPACE-001 v0.3, Planned)
**Last Updated:** 2026-07-26
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
- [x] Settings panel is optionally PIN-protected against local-network access (PBKDF2-SHA256 hash, bypass via env var for recovery)
- [x] Users can monitor each GPU individually (VRAM, utilization, temperature, power) via per-GPU WebSocket channels
- [x] Users can route extraction jobs to a specific GPU via `gpu_id` parameter
- [ ] Users can distribute SAE training across multiple GPUs simultaneously (DDP — planned)
- [ ] A circuit self-calibrates its usable steering band (onset + correctness cliff) against generated falsifiable probes, and ships that band in its contract so a served dial cannot reach the nonsense zone (Feature 20 — planned)

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
| 11 | MCP Server & Feature Groups | MCP tools for agentic analysis/steering + cross-feature grouping (API/UI) | Complete |
| 12 | Steering UX Enhancements | Blended/Compare toggle, up to 20 features, frequency-based auto-baseline strength, compact tiles | ✅ Complete |
| 13 | Clusters UX & Trustworthy Blended Results | "Feature Groups" → "Clusters" (UI), cluster-labeled steering results, visible all-members-applied verification | ✅ Complete |
| 14 | Cluster Strength Budget Model | Principled combined-strength model: frequency-derived budget, similarity-weighted allocation, resultant-norm gain, pin+rebalance, intensity dial | ✅ Complete |
| 15 | Cluster Authoring & Portable Definitions | Named/narrated cluster profiles with tuned strengths; standardized JSON export/import (the mobile artifact for the MILLM/MCP/Open WebUI arc) | ✅ Complete |
> **Status reconciled 2026-08-23 (MIS-E2E-011).** Rows 16–24 read "Planned"
> while the work had shipped — nine rows, some of them closed for over a month.
> Status was verified against the code rather than against another document:
> each row's primary module was checked to exist (e.g. row 22 →
> `ml/jlens_fitter.py`, row 24 → `components/panels/JLensPanel.tsx`), and rows
> 25–29 stay Planned because theirs do not.
>
> **This table is the authority on product status.** `CLAUDE.md` carries a
> narrative of the current arc and will always be more detailed and less
> durable; where the two disagree, this row wins and CLAUDE.md is the thing to
> correct.

| 16 | Multi-SAE Cross-Layer Steering | Features steered through their OWN layer's SAE; per-layer budgets + global λ; hazard detection v2 (validated-effect-size-driven) | ✅ Shipped — circuits arc, 2026-07-20 |
| 17 | Capture, Statistical Mining & Attribution | Position-carrying sparse capture; PMI/null-model/held-out statistics; feature- AND cluster-level (supernode) mining; Tier-2 gradient-attribution re-ranking; Tier-2.5 readiness | ✅ Shipped — circuits arc, 2026-07-20 |
| 18 | Intervention, Validation & Faithfulness | Directional-subtraction intervention (error-preserving); effect-size-vs-null edge validation with manifests; circuit-level faithfulness; heuristic-ablation remediation | ✅ Shipped — circuits arc, 2026-07-20 |
| 19 | Circuit Review, Ladder & Portability | Evidence-ladder claims discipline; edge typing (computed/persistence/attention-mediated); review/promotion; `mistudio.circuit-definition/v1` + per-layer v1 projection | ✅ Shipped — circuits arc, 2026-07-20 |
| 20 | Circuit Strength Calibration | Automated search for the usable steering band — onset (min influence above none) and the correctness cliff (max before facts break) — judged against generated falsifiable probes; ships an intensity band + default into the circuit contract so a served dial cannot reach the nonsense zone | ✅ Shipped — 2026-07-22, hardware E2E on k8s |
| 21 | Training Finalization & Checkpoint Lifecycle | Stop-and-finalize: rebuild a stopped run's SAEs from a checkpoint and write the Community Standard export, so a halted training's weights stay importable instead of being forfeited; plus configurable checkpoint retention (keep best + last N steps) that reclaims the unbounded checkpoint footprint, disabled and dry-run by default | Implemented |
| 22 | J-Space Phase 0: Lens Fitting, Validation & Gate | Fit a Jacobian lens for any loadable model using miStudio's own architecture discovery; validate it against six acceptance classes before any consumer sees it (upstream loading fails silently); reproduce the source paper's six evaluation distributions; produce a per-model workspace-band report and record an explicit GO/NO-GO | ✅ Shipped — files `021_*`, 2026-08-10 |
| 23 | J-Space Readout Substrate & Wire Format | Logit-lens readout path (J=I, no artifact required) plus the artifact class, three distinct readout modes (full ranked / probe / sparse decomposition by gradient pursuit), recipe provenance, and a readout stream mirroring Neuronpedia's lens wire format so the viewer is driven by either source without a translation layer | ✅ Shipped — files `022_*`, 2026-07-30 |
| 24 | J-Lens Readout Viewer | Position × layer readout panel: top-token grid with hover, pinned-token rank heatmap, rank-vs-layer trajectories, per-position and per-layer readouts, Jacobian/logit/diff lens modes, model-derived layer bands, and the interpretability caveats that stop an uninterpretable readout being read as a null result | ✅ Shipped — files `023_*`, 2026-07-31 |
| 25 | SAE Workspace Annotation & Weight-Space Readouts | Annotate every dictionary feature with dual geometric (lens kurtosis) and behavioral (motor vs workspace) classification; raise a label-disagreement queue where the auto-label and the lens readout diverge; project SAE decoders, transcoder pairs and attention Q/K/V/O through the lens with no activations at all | Planned |
| 26 | J-Space Intervention Engine Extension | Paired-run execution with clamping (the capability every mediation analysis needs), projective ablation, dynamic top-k workspace ablation with clean-pass exclusion, and scale-aware lens-coordinate swap — each with a mandatory size-matched random-direction control | Planned |
| 27 | J-Space Claims Discipline | Assign J-space evidence to rungs on the existing evidence ladder and enforce that a readout is never presented in causal language; state that absence of a signal is not evidence of absence; hold framing to functional and mechanistic terms | Planned |
| 28 | J-Space Contracts & Neuronpedia Conformance | Additive interchange kinds for the lens artifact, workspace annotation and watchlist, with projections so miLLM-as-shipped works unchanged; Track A supplies a conformant artifact to a local Neuronpedia instance by mounted directory; Track B exports workspace annotations through the existing feature upload path | Planned |
| 29 | J-Space Runtime Handoff & Full MCP Parity | FULL MCP parity across the J-space surface — every capability reachable in the workbench is reachable by an agent, with tools shipped alongside the feature that creates them and each covered by the reachability harness; plus author, validate, version and export watchlists — named concept sets with detection thresholds that miLLM evaluates per token at the cost of an inner product — plus a validated reference evaluation-awareness watchlist, MCP surface, and cost-envelope reporting for every J-space operation class | Planned |
| 30 | Labeling Prompt-Template Optimization | Iteratively improve feature-labeling prompt templates by running variants over a FIXED feature panel without writing any label, and ranking them by an automated detection score rather than by reading every label; panel identity is content-addressed so two runs are comparable by construction, and a comparison refuses a verdict it cannot support | In Progress |

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
- WebSocket streaming (2-second intervals from `services/background_monitor.py`, an asyncio task in the FastAPI process — **not** Celery Beat; MIS-E2E-156)
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
- **Settings panel PIN protection (May 2026):** optional PBKDF2-SHA256 PIN gate prevents unauthorised access to the Settings panel from the local network. PIN stored as hash (not encrypted value) in `app_settings` under key `settings_pin_hash`. Session unlocked in `sessionStorage` — one entry per browser session. Bypass via `MISTUDIO_BYPASS_PIN=true` env var requires server filesystem access, making it a safe recovery mechanism without needing DB access.

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
- `GET /api/v1/settings/pin/status` - Check if PIN is configured and whether bypass is active
- `POST /api/v1/settings/pin/verify` - Verify PIN (returns `{valid: bool}`)
- `POST /api/v1/settings/pin/set` - Set or change PIN (requires current PIN when changing, waived when bypass active)

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

### 3.11 MCP Server & Cross-Feature Grouping (Complete)

**Purpose:** Make miStudio agent-native — expose the post-extraction workflow (analyze → group → steer → relabel) to MCP-capable AI clients (Claude Code, Codex, etc.), and add a first-class cross-feature grouping capability usable from both agents and the frontend. Derived from `0xcc/brds/miStudio-MCP-Server-BRD.md` (BRD-MIS-MCP-001).

**Capabilities:**
- MCP server (official `mcp` Python SDK, streamable-HTTP transport on port 8765 + stdio dev mode) shipped as a separate `mcp-server` container reusing the backend image; talks to the backend exclusively via `/api/v1`
- ~24 MCP tools across gated categories (`read`, `groups`, `steering`, `labeling`, `experiments`, `jobs`, `admin`) — async 202 jobs translated to start-tool + status-polling tools
- Bearer-token auth (`MCP_AUTH_TOKEN`, LAN-reachable by default); category exposure via `MCP_TOOL_CATEGORIES`; destructive tools off by default
- **Cross-feature grouping (new REST capability):** precompute job builds a token→feature inverted index + context-similarity subgroups (TF-IDF/cosine over prime-token contexts); endpoints for groups, by-token search, and seed-feature related lookup
- **Feature Groups frontend view:** browse groups by top activating token, filter by label/category/star, jump to feature detail, hand selected members to Steering
- Steering guardrails: concurrency cap, max_new_tokens ceiling, and an operator-approval mode (durable approval queue surfaced in the UI)
- Agent label write-back with `mcp_agent` provenance and aqua-star protection (409 unless `override_protected=true`)

**Key Files:**
- Backend: `src/mcp_server/` package, `feature_grouping_service.py`, `feature_grouping_tasks.py`, `api/v1/endpoints/feature_groups.py`
- Frontend: `FeatureGroupsPanel.tsx`, `featureGroupsStore.ts`, `useFeatureGroupsWebSocket.ts`
- Docs: `010_FPRD|MCP_Server.md` → FTDD → FTID → FTASKS

---

### 3.12 Steering UX Enhancements (✅ Implemented & Deployed 2026-07-15)

**Purpose:** Bring empirically-validated steering capabilities (from this session's MCP experiments,
experiment `c4a273f1`) into the Steering UI — blended multi-feature steering at scale, frequency-derived
starting strengths, and a usable layout for many features.

**Capabilities:**
- **Blended vs Compare toggle** — Blended sums all selected features in one pass (`/steering/async/combined`); Compare steers each separately (`/steering/async/compare`)
- **Up to 20 features** selectable (was 4) — backend already blends N features; 4 was a UI/schema limit
- **Auto-baseline strength** from activation frequency: `S = clamp(2.9 − 2.6·freq, 1.0, 3.0)`, with a default-10 fallback and an auto/default badge; `max_activation` shown for context but not used (unit-norm decoder)
- **Compact selected-feature tiles** (~half height) retaining every control, and a 20-entry color palette
- Feature Groups → Steering hand-off preserves per-member stats so baselines auto-compute

**Key Files (implemented):**
- Backend: `schemas/steering.py` (max_length 4→20, dropped compare unique-color validator, color Literal widened 4→20), `schemas/sae.py` + `saes.py` (activation_frequency)
- Frontend: `utils/steeringStrength.ts`, `types/steering.ts` (palette + fields), `SelectedFeatureCard.tsx` (compact), `SteeringPanel.tsx` (Blended|Compare toggle), `FeatureSelector.tsx`, `ComparisonPreview.tsx`, `steeringStore.ts`, `featureGroupsStore.ts`, `FeatureGroupsPanel.tsx`, `FeatureBrowser.tsx`, `GroupMembersTable.tsx`
- Tests: `backend/tests/unit/test_steering_schema.py` (9), `frontend/src/utils/steeringStrength.test.ts` (4), `steeringStore.test.ts` (auto-baseline + MAX-20)
- Docs: `011_FPRD|Steering_UX.md` → FTDD → FTID → FTASKS
- Deployed to K8s 2026-07-15; E2E-verified (compact tiles, 3/20 header, Blended|Compare toggle, default badges, Auto preset) — `0xcc/caps/miStudio_Steering_Panel-CompactTiles_20260715.png`

---

### 3.13 Clusters UX & Trustworthy Blended Results (✅ Complete 2026-07-16)

**Purpose:** Establish *clusters* — sets of features that fire together and share a meaning — as the
product's primary steering primitive, and make combined ("Blended") steering results trustworthy.
From BRD-MIS-CLUSTERS-001 (BR-001, BR-002, BR-003, BR-011).

**Capabilities:**
- **Terminology:** the user-facing term becomes **Clusters** everywhere (nav, panels, copy); "Feature
  Groups" no longer appears in the UI. Backend/API/data names (`feature_groups` et al.) are unchanged this
  increment (recorded as future work). The pre-existing per-feature NLP "Semantic Clusters" section is
  relabeled to avoid collision.
- **Cluster identity through the hand-off:** steering selections that originate from a single cluster carry
  the cluster's identity (display token, group id) into the steering configuration.
- **Trustworthy result labels:** combined-steering outputs are titled by the cluster (authored name →
  display token → "Blended (N features)"), never by a lone top-feature index.
- **Verifiable combination:** the result surface shows the full applied-features summary
  (`features_applied` already returned by the combined endpoint) so users can confirm every member
  contributed its assigned strength.

**Docs:** `012_FPRD|Clusters_UX.md` → FTDD → FTID → FTASKS. ADR: IDL-28.

---

### 3.14 Cluster Strength Budget Model (✅ Complete 2026-07-16)

**Purpose:** Replace guessed starting strengths for cluster steering with a principled, outcome-grounded
model — the user must never start from a useless point. From BRD-MIS-CLUSTERS-001 (BR-004, BR-005, BR-006).

**Capabilities:**
- **Total influence budget** derived from the cluster's aggregate activation frequency via the empirically
  fit solo law (`B_dir = clamp(a − b·f_eff, m, M)`, f_eff = similarity-weighted mean member frequency).
- **Similarity-weighted allocation** (`wᵢ = sᵢ/Σsⱼ`): equal similarity ⇒ equal strengths; members most
  representative of the cluster's meaning carry more of the budget.
- **Exact resultant-norm gain** `G = ‖Σσᵢwᵢdᵢ‖` (server-side, from the SAE decoder) scales the budget so
  the *injected vector* — not the naive sum — matches the validated solo magnitude; coherence flags warn on
  near-cancellation, and low-cohesion clusters fall back to per-feature solo baselines.
- **Budget-preserving rebalance:** manually editing one member pins it and redistributes the remainder.
- **Master cluster-intensity dial** (λ ∈ [0,2]) scaling the whole cluster — the dial semantics that Open
  WebUI inherits in the future integration arc.
- **Empirical validation protocol:** MCP-driven sweeps on real clusters gate the formula constants
  (per-SAE-namespaced config) before the model is trusted.

**Docs:** `013_FPRD|Cluster_Strength_Model.md` → FTDD → FTID → FTASKS. ADR: IDL-29.

---

### 3.15 Cluster Authoring & Portable Definitions (✅ Complete 2026-07-16)

**Purpose:** Capture a tuned cluster as a first-class, mobile artifact — named, narrated, strength-tuned —
exportable as standardized JSON that later travels across the miStudio↔MILLM ecosystem (MILLM import,
unified MCP server, Open WebUI dial — all future scope, separate BRD). From BRD-MIS-CLUSTERS-001 (BR-007
through BR-010).

**Capabilities:**
- **Cluster profiles:** name + narrative + persisted per-member tuned strengths, stored decoupled from the
  recomputable grouping index (recompute never destroys tuned profiles).
- **Portable JSON cluster definitions** (`mistudio.cluster-definition/v1`): versioned, self-describing,
  consumer-neutral — members, strengths, budget + formula parameters, intensity range, model/SAE references,
  provenance. Single-cluster and multi-cluster bundles.
- **Round-trip fidelity:** export → import reproduces the identical steering configuration.
- **Future arc (vision, not this increment):** MILLM imports the same definition for live-chat steering; a
  single MCP server spans both products with health-gated tool sets; Open WebUI exposes the cluster
  intensity dial in real chat sessions; a marketplace for trading cluster definitions.

**Docs:** `014_FPRD|Cluster_Definitions.md` → FTDD → FTID → FTASKS. ADR: IDL-30.

---

### 3.16 Multi-SAE Cross-Layer Steering (Planned)

**Purpose:** Make cross-layer combined steering real. Hooks already register per-layer, but every hook
shares the single loaded SAE's decoder — features placed on other layers would be steered with directions
from the wrong layer's basis. From BRD-MIS-CIRCUITS-001 (BR-001..004) as amended by BRD-MIS-CIRCUITS-002
(BR-024 hazard grounding).

**Capabilities:**
- **Per-layer SAE application:** a combined generation loads every SAE referenced by the configuration and
  steers each feature through the SAE trained on its own layer; configurations spanning layers are never
  silently served through one decoder.
- **Per-layer budgets + global λ:** the established strength model (IDL-29) runs independently per layer;
  one global intensity dial; formula id `freq-budget/sim-alloc/per-layer@1`. Joint cross-layer calibration
  explicitly deferred.
- **Hazard detection v2 (BR-024):** compounding/cancellation warnings consume **measured validated edge
  effect sizes** as the primary signal (quantifying expected double-counting); where no validated edge
  exists, heuristic signals remain but are labeled `heuristic` per the evidence-ladder discipline. Warnings
  surface, never silently correct.
- **Trust across layers:** member-contribution verification and circuit-titled results extend IDL-28's
  guarantees to multi-layer runs.

**Docs:** `015_FPRD|MultiSAE_Steering.md` → FTDD → FTID → FTASKS. ADR: IDL-31.

---

### 3.17 Capture, Statistical Mining & Attribution (Planned)

**Purpose:** Build the evidence base and mine it soundly. Extraction discards the per-token SAE code
matrix; naive lag-0 co-activation ranked by a residual weight prior would surface base-rate noise and
echoes. From BRD-MIS-CIRCUITS-001 (BR-005..008) as amended by BRD-MIS-CIRCUITS-002 (BR-015, BR-016,
BR-022, BR-023).

**Capabilities:**
- **Position-carrying sparse capture:** per-token, multi-layer, above-threshold feature activations with
  token positions, SAE reconstruction-error norms, and optional per-head attention artifacts (top-k keys)
  — Tier-2.5-ready by construction; cost-estimated before launch; managed GPU task.
- **Statistically sound mining (BR-015):** PMI/lift with base-rate correction; minimum support; a
  within-document circular-shift null model; Benjamini–Hochberg FDR; per-document 80/20 discovery/held-out
  split with the **held-out replication rate reported first-class** per run.
- **Two granularities (BR-016):** feature-level and **cluster-level (supernode)** mining — curated cluster
  profiles as units (max-over-members activation, cohesion-gated), the recommended default for seeded mode;
  drill-down refines a cluster edge to member pairs.
- **Tier-2 gradient attribution (BR-022):** one forward + one backward per prompt with stop-gradient
  through the SAE reconstruction error; re-ranks candidates before ablation sampling; the **survival-rate
  uplift vs co-activation-only ranking is a measured, reported number**.
- **Tier-2.5 readiness (BR-023):** lag-0 disclosed as the deliberately-limited Tier-1 signal; the
  attention-mediated cross-position mining design (Appendix A.8) is a named deliverable; no schema
  migration will be needed to enable it.

**Docs:** `016_FPRD|Circuit_Discovery.md` → FTDD → FTID → FTASKS. ADR: IDL-32, IDL-36.

---

### 3.18 Intervention, Validation & Faithfulness (Planned)

**Purpose:** Make "causally validated" mean what the interpretability literature expects. The prior
ablation surface fabricated its numbers; CIRCUITS-001 required real intervention; CIRCUITS-002 specifies
it. From BRD-MIS-CIRCUITS-001 (BR-009) as amended by BRD-MIS-CIRCUITS-002 (BR-017, BR-018, BR-019).

**Capabilities:**
- **Intervention engine v2 (BR-017):** directional subtraction — remove the feature's realized
  contribution `(a_u − a_base)·W_dec[:,i]` from the residual stream, **preserving the SAE reconstruction
  error term**; zero vs corpus-mean baseline as a recorded parameter; every run persists a reproducible
  **validation manifest**.
- **Edge validation criterion (BR-018):** standardized effect size vs a shuffled-pair null + sign
  consistency across evaluation prompts; failures recorded as tested-and-failed, never silently dropped.
- **Circuit-level faithfulness (BR-019):** at promotion — necessity (whole-circuit ablation) and a
  tractable sufficiency approximation (top-k non-member ablation); scores displayed and carried in the
  contract. Badge, not gate.
- **Heuristic remediation:** the fabricated-ablation surface is relabeled "impact estimate (statistical —
  no inference)" or removed; no surface presents a fabricated number as causal.

**Docs:** `017_FPRD|Circuit_Validation.md` → FTDD → FTID → FTASKS. ADR: IDL-34.

---

### 3.19 Circuit Review, Evidence Ladder & Portability (Planned)

**Purpose:** Turn discovered circuits into first-class, honestly-labeled, portable artifacts. From
BRD-MIS-CIRCUITS-001 (BR-010..014) as amended by BRD-MIS-CIRCUITS-002 (BR-020, BR-021, BR-025, BR-026).

**Capabilities:**
- **Evidence ladder (BR-026):** one shared rung model — mined → attribution-supported → causally
  validated → faithfulness-tested — machine-readable in the contract, rendered in every UI surface,
  returned by every MCP tool; a circuit's rung = min over member edges; causal language forbidden below
  rung 2. Subsumes the "unvalidated" badge.
- **Edge typing (BR-020, BR-021):** every edge is typed `computed` | `persistence` | `attention_mediated`
  (schema-ready); the weight prior becomes an **echo detector** inside the classifier — never a standalone
  ranking booster; persistence edges are de-ranked from default views but queryable, steerable, always
  typed; low prior + high association is recognized as the computed-edge signature.
- **Review & promotion:** evidence-rich review (statistics, attribution, validation status, manifests);
  edit/name/narrate; promotion to loadable multi-layer steering profiles (badge, not gate); **per-layer
  member caps (BR-025)**.
- **Portable contract:** `mistudio.circuit-definition/v1` — NEW kind; per-layer SAE refs; members keyed to
  layers (feature refs or cluster refs); `edges[]` with type, rung, evidence, attribution scores, and
  manifest references; position/attention fields present from day one; discovery provenance; lossless
  round-trip. Amendments land **before freeze**.
- **Projection to today's runtime (BR-014):** per-layer cluster-definition/v1 slices carrying the parent's
  rung and a partial-rendering marker — current single-SAE consumers (miLLM) work unchanged.
- **Future arc (vision, separate BRDs):** miLLM circuits runtime (BRD-MILLM-CIRCUITS-001); Tier-2.5
  mining fast-follow; Tier-3 attribution graphs; substrate pilot (CIRCUITS-002 Addendum B — research
  track, no PPRD row) seeding BRD-MIS-SUBSTRATE-001 on GO.

**Docs:** `018_FPRD|Circuit_Portability.md` → FTDD → FTID → FTASKS. ADR: IDL-33, IDL-35.

---

### 3.20 Circuit Strength Calibration (Planned)

**Purpose:** Find the strength at which a circuit is actually *usable* when served, automatically. The
circuits arc (16–19) discovers, validates, and makes a circuit portable — but the per-member steering
strengths it ships are **uncalibrated placeholders**. Taking the first real circuit through to serving
exposed the gap concretely: a hand-run single-prompt strength sweep declared a "usable ceiling" that,
when actually served, produced fluent-but-false output. The strength that reads as on-theme is well past
the strength at which the answer stops being *true*. Calibration is the discovery-plane act that closes
this — and per the plane split, it belongs in miStudio (which runs the model to learn), not in miLLM
(which runs the model to serve).

**The two thresholds (found by opposite tests):**
- **Onset — min influence above none.** The smallest dial where output *measurably diverges* from the
  unsteered baseline. A *difference* test (embedding/distribution drift vs baseline crossing a noise
  floor); no semantic judge needed. Below onset the circuit is inert.
- **Correctness cliff — max before facts break.** The largest dial where output is *still correct*. A
  *property* test: an LLM judge scores each generation "still true / degrading / broken" against
  **falsifiable probes**. This is the threshold a perplexity/theme metric cannot see — the observed
  cliff sat between two adjacent dial steps, one giving a correct answer with light humor, the next
  inventing a plainly false claim in the same confident tone.

**Capabilities:**
- **Adaptive search, not a fixed grid.** Bisection locates onset (walk up until divergence crosses the
  floor) and the cliff (binary-search between last-correct and first-broken), so it finds the band
  wherever it sits and at whatever width — directly handling that optimum starting points differ per
  circuit and the usable increments vary. A fixed linear sweep steps over narrow, off-center bands (the
  hand sweep sampled only the collapsed region and missed the entire 0.4–0.6 usable band).
- **Probes generated from the circuit's feature labels.** No human authoring required. The generator
  targets **neutral factual topics the steering should NOT touch** (so degradation shows up as the
  circuit's tint corrupting unrelated facts, which is detectable), not topics *about* the circuit's
  concept (whose "right answer" is not falsifiable). Bands from generated probes are marked
  **provisional** — honest about confidence, since generated probes are the weakest link.
- **Ship the band, not a point.** Calibration writes `{onset, sweet_spot, cliff}` with per-step evidence
  into the circuit contract, clamps `intensity_range` to `[onset, cliff]` so a served dial **physically
  cannot** reach the nonsense zone, and defaults `intensity` to the sweet-spot. Snapshot + parameters
  persist like faithfulness (badge, not gate).
- **Provisional across the plane boundary.** Strength measured against miStudio's model instance may sit
  slightly off from what miLLM actually serves (SAE attach, inference backend), which is *why* a
  measure-only sweep misled. The band is a grounded **starting point**; the probes travel in the
  contract so a one-shot re-verify at serve time is cheap when wanted. Recorded tech debt, not a claim
  of perfect cross-plane transfer.

**Reuses:** the existing strength-sweep engine (§3.6 steering) as the raw generation loop; the
faithfulness service's "run the model to measure a circuit property, persist a manifest + snapshot"
pattern (§3.18); the enhanced-labeling LLM-judge plumbing (§2.3) for cliff scoring.

**Docs:** `019_FPRD|Circuit_Calibration.md` → FTDD → FTID → FTASKS. ADR: IDL-37 (proposed).

---

### 3.21 Training Finalization & Checkpoint Lifecycle (Implemented)

**Purpose:** make a stopped training's SAE usable, and stop checkpoints growing without bound.

**The problem this closes.** Cancelling a training set `status=cancelled` and
returned. That skipped the training loop's finalize block — the step that writes
`community_format/`, which is the *only* artifact downstream consumers read
(`sae_manager_service` scans it; circuit capture, Neuronpedia export and the
analysis service all load through `load_sae_auto_detect`). A stopped run
therefore left behind intact, often excellent checkpoints that nothing in the
product could consume, and the UI offered only **Retry** — which creates a
brand-new training from step 0 and orphans them.

This was not hypothetical. `train_969e90af` (granite-4.1-8b, JumpReLU, layers
34/35/36, 4096→32768) was stopped at step 10,300 after its FVU flattened at
**0.065 with zero dead neurons** — a good SAE by every metric the run reported,
and completely unusable. The product's own manual compounded it, promising
*"Stop: Gracefully end training (saves final checkpoint)"*.

Separately, training writes a checkpoint every `checkpoint_interval` steps and
nothing ever removed them: 423 checkpoint rows across 18 trainings, ~24 steps
each, **78 GB** under `/data/trainings` on a volume at 81% capacity.

**Capabilities:**

- **Stop & Finalize** — stop a run *and* write its Community Standard export in
  one action, so the SAE stays importable.
- **Finalize** — the rescue path for runs stopped before this existed
  (including FAILED runs, behind an explicit confirmation): rebuild from the
  newest complete checkpoint and export.
- **Honest status** — finalizing sets `status=COMPLETED` so the import path
  unlocks, but `progress`/`current_step` stay truthful and a new
  `finalized_from_step` column records where the run actually stopped. The UI
  badges *"Finalized early @ N"*. A run halted at 10k of 50k never presents as a
  finished 50k-step training.
- **Checkpoint retention** — keep the best checkpoint plus the newest N steps,
  with a read-only preview and a manual per-training prune. **Disabled by
  default, and dry-run when first enabled**, because deletion is irreversible.
- **Checkpoint deletion** — `DELETE /trainings/{id}/checkpoints/{ckpt_id}`,
  which the frontend had been calling against a route that did not exist.

**Normative behaviour:**

- Retention selects whole **steps**, never individual rows. A multi-layer
  training writes one row per `(step, layer, hook)` sharing one
  `checkpoint_{step}/` directory; deleting a subset leaves an unloadable
  checkpoint.
- Finalize reads SAE dimensions from **checkpoint tensor shapes**, not stored
  hyperparameters, which the training task corrects in memory and never persists.
- The export is **atomic** (staged then swapped) and only proceeds from a step
  verified **complete**; the completeness check fails **closed**.
- Pruning never touches `is_best`, the newest step, a training in an active
  state, or a checkpoint younger than the configured minimum age.

**Cross-plane:** none. This is entirely miStudio-side; the artifact it produces
is the existing `community_format/` export that miLLM and Neuronpedia already
consume.


### 3.22 J-Space Phase 0: Lens Fitting, Validation & Gate (Planned)

**Purpose:** produce a trustworthy Jacobian lens for *any* model the workbench can load, and
decide — on evidence — whether the workspace claim set holds at the scales we serve.

**Why fitting, not downloading.** Pre-fitted lenses exist upstream for 36 models. The reference
model here, `LFM2.5-1.2B-Instruct`, is not among them, and neither is most of what this workbench
loads. A capability that only works for pre-fitted models is not the capability. Acquisition stays
as a cost optimisation when a conformant lens exists for the *exact* weights — a distinction that
matters: the workbench holds `gemma-2-2b-it`, whose weights differ from the `gemma-2-2b` base model
the upstream lens is fitted for, so downloading would have silently supplied an invalid artifact.

**Capabilities:**

- **Model-agnostic fitting** through miStudio's own `discover_transformer_structure`, not an
  upstream fitter's layout auto-detection. Convergence-based stopping from a floor of 100 prompts;
  corpus-slice parallelism with merge rather than splitting the model.
- **Six-class validation** before any consumer sees an artifact — structural, naming, envelope,
  semantic, cross-implementation, round-trip.
- **Replication** of the source paper's six evaluation distributions against the vendored upstream
  harness, reported per distribution rather than pooled.
- **Workspace-band report** deriving *that model's own* sensory / workspace / motor boundaries.
- **Recorded GO / NO-GO / GO-AT-LARGER-SCALE**, published either way.

**Normative behaviour:**

- Validation is **mandatory and explicit**, because upstream lens loading is best-effort: a
  malformed artifact does not fail at deploy time, it fails at request time inside the webapp. The
  round-trip check must be an actual request, never inferred from a clean startup log.
- The envelope check derives its bound from the model's **own** `d_model`, `n_vocab`, `n_layers`.
  For the reference model that is ~134 MB required against ~4.3 GB materialised — a ratio of ~32×,
  not the ~111× that holds for a 256k-vocabulary model. The rule is portable; the arithmetic is not.
- Layer bands are **never inherited**. The source paper's L38–92 figures are Sonnet-4.5's, and the
  product makes porting them impossible by construction.
- Next-token agreement is **not** a quality metric anywhere — the lens is deliberately worse on it
  than the logit lens through most of the network. A gate that rewards it is a defect.

**Cross-plane:** none. Artifacts are consumed locally and, optionally, by a local Neuronpedia
instance via a mounted directory (§3.28).

### 3.23 J-Space Readout Substrate & Wire Format (Planned)

**Purpose:** make "what is this model poised to say here" a cheap, routine query.

**Capabilities:**

- **Logit-lens path (`J = I`) first**, as a complete shippable capability requiring **no artifact**.
  Everything above it — viewer, analysis suite, interventions — is built and validated against it,
  and the Jacobian substitutes at a single call site with no consumer change.
- **Three distinct readout modes**: full ranked over the vocabulary; single-direction probe; sparse
  non-negative decomposition by gradient pursuit.
- **Artifact class** with full recipe provenance — target layer, attention-gradient treatment,
  target-position scope, aggregation, corpus, convergence criterion, library commits.
- **Readout stream mirroring Neuronpedia's lens wire format**, so a miStudio stream and a
  Neuronpedia stream are interchangeable at the client.

**Normative behaviour:**

- **Never materialise `W_U J`.** Token directions are synthesised on demand and cached by working
  set. This is a hard constraint with a CI envelope guard, not an optimisation.
- Occupancy, variance and J-space/non-J-space splits come from **sparse decomposition only**.
  Top-k by inner product is not a substitute — on an overcomplete non-orthogonal frame it gives a
  different and more redundant answer.
- Decomposition is non-unique by construction, so solver parameters and control seeds are
  **mandatory provenance**; figures lacking them are invalid.
- The residual stream is captured at the **decoder-layer output**, never at a discovered
  normalisation module — on a hybrid model the module a naive search calls "residual" is a
  post-attention RMSNorm, and reading there fails silently with plausible-looking numbers.

**Cross-plane:** the wire format is shared with Neuronpedia's webapp and, later, miLLM.

### 3.24 J-Lens Readout Viewer (Planned)

**Purpose:** a first-class panel for reading a model layer by layer and position by position.

**Capabilities:** prompt strip with per-token selection; a top-ranked-token grid over
(position × layer) with hover for the full top-k; a rank heatmap for pinned tokens; full ranked
readouts at a selected position or layer; rank-vs-layer trajectories; and a lens-mode control
offering Jacobian, logit and diff.

**Normative behaviour:**

- Bands are drawn from **that model's** band report, or not at all. No default L40/L90.
- The layer axis comes from the stream's `layers_by_type`, never a hardcoded count.
- Jacobian and diff modes are **visibly disabled with a stated reason** until a validated artifact
  exists. Logit data is never rendered under a Jacobian label.
- Early-layer readouts are marked as *expected to be uninterpretable*, and the panel states that a
  readout resisting interpretation is not a null result — it may be averaging noise, a multi-token
  concept, or genuine content we cannot yet name.
- Every readout carries its evidence rung. A readout is rung 0 and is not a causal claim.

**Cross-plane:** the viewer is driven by either a miStudio or a Neuronpedia stream without a
translation layer.

### 3.25 SAE Workspace Annotation & Weight-Space Readouts (Planned)

**Purpose:** give the accumulated dictionary — labels, clusters, validated circuits — a
reportability dimension without retraining anything.

**Capabilities:** per-feature workspace classification; a label-disagreement queue; and weight-space
readouts for SAE decoders, transcoder encoder/decoder pairs, and attention Q/K/V/O matrices.

**Normative behaviour:**

- Classification uses **two independent fields**, never one score: a geometric field (lens kurtosis
  against a covariance-matched null) and a behavioural field separating **motor** from **workspace**
  features. Motor features share high kurtosis with workspace features; a single "workspace score"
  would present output-driving features as reportable ones, on exactly the dimension a user would
  trust it for.
- The disagreement queue exists because the failure is documented and consequential: example-driven
  labelling called a fabricated-content detector "technical exposition". Suppressing that feature
  cut the model's stated recognition of an artificial scenario from 28/50 to 10/50.
- Weight-space readouts need **no activations and no corpus** — the cheapest high-value capability
  in the family.
- Annotations reference their lens artifact by identity and refuse to validate against a different
  one; a recipe change produces a new artifact version rather than mutating one.

### 3.26 J-Space Intervention Engine Extension (Planned)

**Purpose:** make causal claims about the workspace substantiable rather than merely assertable.

**Capabilities:** paired-run execution with clamping; additive steering along a lens direction;
projective ablation; dynamic top-k workspace ablation with clean-pass exclusion; and lens-coordinate
swap.

**Normative behaviour:**

- **Paired-run with clamping is the highest-priority capability here.** Without it the product can
  execute interventions but cannot reproduce a single mediation analysis — which is where the
  evidential weight sits, and therefore where credibility on causal claims sits.
- Every run executes against a **size-matched random-direction control** at the same layers and
  positions, reported alongside by default. A run without its control is invalid, because the
  interpretation *is* the gap between the two.
- Projective ablation is recorded as distinct from negative-strength additive steering.
- Dynamic top-k ablation **excludes tokens present in the clean pass's top output candidates**, so
  the intervention targets internal reasoning rather than the report of it.
- Band presets are **scale-aware**: on smaller models coordinate swaps oversteer and need fewer
  layers selected, so presets are not uniform across scales.

### 3.27 J-Space Claims Discipline (Planned)

**Purpose:** keep a tool that surfaces what a model is "thinking" from becoming a tool that
overclaims.

**Normative behaviour:**

- J-space evidence takes rungs on the **existing** evidence ladder — readout (lowest, explicitly not
  causal), probe-threshold crossing, decomposition membership, intervention-with-control, and
  mediation (highest). No surface may present a lower rung in a higher rung's language.
- Badge-not-gate throughout: low-rung evidence is surfaced *with its rung*, not suppressed.
- **Absence of a signal is not evidence of absence.** Sufficiently automatic computation proceeds
  without engaging the workspace, and a concept without a single-token name may not surface even
  when represented. Every monitoring, auditing and screening surface states this.
- Product copy, labels, documentation and export metadata **do not assert or imply** that a served
  model has subjective experience. Framing is functional and mechanistic. This is a shipping
  requirement with reviewer sign-off, not a disclaimer — mis-framing is the fastest route to having
  serious researchers discount the tool.

### 3.28 J-Space Contracts & Neuronpedia Conformance (Planned)

**Purpose:** freeze a surface the runtime consumer can build against, and conform to upstream
representations rather than paralleling them.

**Capabilities:** additive interchange kinds for the lens artifact, the workspace annotation, the
readout record and the watchlist, each with a projection to an existing kind; **Track A**, supplying
a conformant artifact to a local Neuronpedia instance through a mounted directory; and **Track B**,
exporting workspace annotations through the existing feature/explanation upload path.

**Normative behaviour:**

- Existing kinds are **not mutated**. Where a new kind carries content a shipped consumer could use,
  a projection is provided so miLLM-as-shipped works unchanged, marked as a partial rendering.
- **There is no J-lens upload path, and none will be built.** Upstream's entire J-lens database
  footprint is two tables persisting shared analysis sessions; the lens is compute-on-demand from a
  mounted artifact. Track A is a directory and one environment variable.
- Track B placement is confirmed against the running instance's actual schema before building.
- Where an upstream representation cannot carry something we need, the gap is **proposed upstream**
  rather than worked around locally.

### 3.29 J-Space Runtime Handoff & Full MCP Parity (Planned)

**Purpose:** turn interpretability into runtime observability — per-token probes against named
concepts, at the cost of an inner product — and make every J-space capability reachable by an agent,
not just by a human at the panel.

**Capabilities:** watchlist authoring, validation, versioning and export; a validated reference
evaluation-awareness watchlist; **full MCP parity** across the J-space surface; and cost envelopes
for every J-space operation class.

**Full MCP parity means what it says.** Every capability a user can reach in the workbench is
reachable by an agent: readout in all three modes, artifact listing and validation status, band
reports, workspace annotations, weight-space component readouts, interventions with their mandatory
matched control, and watchlists. Tools ship **alongside the feature that creates the capability**
rather than batched here at the end — a readout tool is useful the moment a readout exists.

Every tool is covered by the reachability harness (`tests/unit/test_reachability.py`), asserting
presence in the **live registry and the built server**, with payload and call count asserted. This
is not ceremony: this repo once shipped 16 fully-implemented, unit-tested, documented
`millm_circuit_*` tools that were never registered, so the suite was green and the docs said ✅
while no agent could call the feature.

**Normative behaviour:**

- A watchlist references its lens artifact and layer band **by identity** and refuses to validate
  against a model or artifact it was not authored for.
- The reference watchlist's thresholds are **rederived** for the target model, never inherited.
- The MCP readout tool ships **normative interpretation guidance** as part of the deliverable, not
  as documentation: treat the output as a bag of related ideas rather than prose, look for token
  families rather than exact terms, discount single-layer noise in favour of content recurring
  across layers, and cite a specific layer line verbatim so claims can be checked.
- Operations that cannot fit the GPU envelope at the chosen scale **fail with a stated reason**
  rather than degrading silently.

**Cross-plane:** this increment owns authoring, validation and export only. Runtime evaluation and
streaming are `BRD-MILLM-JSPACE-001`.


### 3.30 Labeling Prompt-Template Optimization (In Progress)

**Purpose:** make feature labeling something that can be *improved on purpose*. Label quality is
set almost entirely by the prompt template, and until now there was no way to change a template and
find out whether the change helped — short of relabeling an entire extraction and reading the
results by hand.

**Capabilities:** scoped trial runs over an explicit feature panel; non-persisting execution, so a
trial never overwrites the labels it is comparing against; content-addressed panel identity;
automated detection scoring; and paired comparison of two trials with an honest verdict.

**Why a trial cannot write labels.** Running five template variants over a panel would otherwise
stomp the user's real labels five times, and the fifth variant would be scored against features the
first four had already rewritten. The measurement has to leave the thing it measures untouched.

**Why the panel is content-addressed.** `panel_id = sha256(extraction_job_id | sorted feature ids)`.
Equal ids *prove* an identical, order-independent, extraction-bound feature set, so a comparison can
refuse a mismatch instead of trusting that two runs happened to use the same features.

**Why the template must be the only variable.** Example order was shuffled with an unseeded global
RNG, so two runs over one panel saw the same examples in a different order — the prompt differed and
the template was not the only thing that changed. A trial seeds the shuffle per
`(panel_id, feature_id)` and records a fingerprint over the examples actually shown, so isolation is
checkable rather than merely intended.

**Why the ruler is not adjustable.** The detection scoring prompt is a pinned, versioned constant
owned by the scorer — never a `labeling_prompt_templates` row. The template under test varies; the
instrument measuring it must not, or scores across trials are not comparable.

**Normative behaviour:**

- A trial writes labels ONLY to `labeling_trial_runs`. No `features` row is modified, and the run
  asserts this rather than assuming it.
- A comparison over a different panel is REFUSED; a comparison with zero overlapping scored
  features returns no verdict and a reason. Scoring nothing is not scoring.
- A detection score is reported only when the judge passes a sanity gate. A judge that cannot detect
  a literal token it was told to look for cannot grade explanations, and the result is reported as
  `judge_unreliable` — never as a low score attributed to the label. A weak judge scoring a good
  prompt badly would send a user off rewriting prompts that were already fine.
- Negatives are drawn from other features and are never described as non-activating. There is no
  encode-on-text service, so the only defensible claim is that a passage falls below the target
  feature's stored-example threshold; that threshold is recorded per feature.
- Panel-level differences are reported with a confidence interval and the minimum difference the
  panel could have detected. A three-point gap over a thirty-feature panel is not a result.

**Depends on:** Feature 4 (Feature Discovery & Browser) for extractions and features.

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
| Celery Beat | Scheduled tasks: stuck-job janitors, checkpoint pruning, the GPU watchdog, the steering reconciler. *(System monitoring is NOT one — see MIS-E2E-156.)* |
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
- Celery Beat for periodic cleanup (stuck-job detection, checkpoint pruning, GPU watchdog). System monitoring runs as an asyncio task inside the API process instead — `services/background_monitor.py` (MIS-E2E-156).

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
- **Settings panel PIN gate:** optional PBKDF2-SHA256 PIN (600k iterations, random salt) protects the Settings panel from local-network access; env-var bypass (`MISTUDIO_BYPASS_PIN=true`) provides a recovery path that requires server filesystem access
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
| Feature PRDs | `0xcc/prds/001-010_FPRD\|*.md` | Individual feature specs |
| Business Requirements | `0xcc/brds/*.md` | Business-level enhancement requests (feeds FPRDs) |
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

| 3.12 | 2026-07-31 | Added Features 22–29 from BRD-MIS-JSPACE-001 v0.3 (Planned) — the JSPACE family: a training-free, MODEL-AGNOSTIC second dictionary substrate. Phase 0 fitting/validation/gate (§3.22); readout substrate, three modes and the upstream lens wire format (§3.23); the J-Lens readout viewer (§3.24); SAE workspace annotation and weight-space readouts (§3.25); intervention-engine extension with mandatory matched controls (§3.26); claims discipline on the existing evidence ladder (§3.27); additive contracts plus the two-track Neuronpedia conformance — artifact mount, not upload (§3.28); and the runtime watchlist handoff to BRD-MILLM-JSPACE-001 (§3.29). v0.3 of the BRD inverts BR-031 to CONSTRUCTION-first (pre-fitted lenses cover 36 models; the reference model is not among them), sets the reference model to LFM2.5-1.2B-Instruct, and adds BR-032 requiring architecture resolution through discover_transformer_structure rather than any whitelist. Hybrid architectures are first-class: the reference model interleaves 10 conv with 6 attention layers, so frozen-Q/K is undefined on 10 of 16 and attention-broadcast metrics are computable on 6 — recorded per layer, never averaged over the subset that qualified. |
| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-05 | Initial project vision and feature breakdown |
| 2.0 | 2025-12-05 | MVP complete — reflects actual implementation |
| 2.1 | 2025-12-16 | Post-MVP: NLP analysis, Ollama integration, infrastructure improvements |
| 3.0 | 2026-04-26 | Enhanced labeling, OpenAI API integration, context-aware labeling template, Settings Panel, security hardening, v0.5.0 public release, K8s production deployment, CI/CD pipeline |
| 3.1 | 2026-05-09 | Settings panel PIN protection (PBKDF2-SHA256 gate + env-var bypass); multi-GPU doc corrections (Phases 1 & 2 retrospectively marked complete) |
| 3.2 | 2026-07-12 | Added Feature 11: MCP Server & Cross-Feature Grouping (Planned, from BRD-MIS-MCP-001) — §3.11, document chain 010_FPRD/FTDD/FTID/FTASKS |
| 3.3 | 2026-07-12 | Feature 11 implemented: MCP server (33 tools, bearer auth, approval mode), cross-feature grouping (index + REST + Feature Groups UI), mcp_agent provenance, deployment (compose profile + k8s) |
| 3.4 | 2026-07-15 | Added Feature 12: Steering UX Enhancements (Planned) — §3.12, doc chain 011_FPRD/FTDD/FTID/FTASKS |
| 3.5 | 2026-07-15 | Feature 12 implemented & deployed: Blended\|Compare toggle, up to 20 features, frequency auto-baseline (default fallback), compact tiles, 20-color palette. K8s-deployed + E2E-verified. |
| 3.11 | 2026-07-26 | Feature 21 — Training Finalization & Checkpoint Lifecycle (row 21, §3.21); stop-and-finalize + configurable checkpoint retention — Implemented |
| 3.9 | 2026-07-19 | Circuits arc re-planned against BRD-MIS-CIRCUITS-001 + BRD-MIS-CIRCUITS-002 consumed as one unit (002 amends 001; conflicts resolve to 002). Four features (16–19, §3.16–3.19): multi-SAE steering w/ hazard-v2; capture + sound statistics (PMI/null/FDR/held-out) + cluster-level supernode mining + Tier-2 attribution + Tier-2.5 readiness; intervention engine v2 + effect-size-vs-null validation + faithfulness; evidence ladder + edge typing + contract (pre-freeze amendments) + projection. Substrate pilot (002 Addendum B) recorded as research track — no PPRD row (seed: BRD-MIS-SUBSTRATE-001.seed.md). |
| 3.7 | 2026-07-16 | Features 13–15 COMPLETE: implemented, 3× review iterations each (28/28/15 findings), deployed via GitOps, Playwright E2E-verified (profile-titled Blended results, applied-features count, budget bar/λ dial, low-cohesion gate, profile save/load/import/export). 013 empirical validation EXECUTED — hard gate PASSED after fitting gain exponent γ=0 (the 1/G boost overdrove ~2×; `0xcc/docs/cluster-strength-validation.md`). IDL-29 step-5 amendment: B = B_dir/max(G,floor)^γ, default γ=0. HF-as-marketplace research recorded (`0xcc/docs/hf-marketplace-cluster-definitions-research.md`) for the follow-on BRD. |
| 3.6 | 2026-07-16 | Added Features 13–15 from BRD-MIS-CLUSTERS-001 (Planned): Clusters UX & trustworthy blended results (§3.13), cluster strength budget model (§3.14), cluster authoring & portable JSON definitions (§3.15). MILLM/unified-MCP/Open WebUI integration recorded as future arc (separate BRD). |

---

*Generated: 2026-07-12*
*MechInterp Studio — v0.5.0 Production Release*
