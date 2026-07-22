yes
# Project: MechInterp Studio (miStudio)

## Current Status
- **Phase:** ✅ **miLLM CIRCUIT CONSOLIDATION CLOSED (2026-07-21)** — the cross-repo increment BRD-MILLM-CIRCUITS-002 (miLLM features 016–020) landed its MCP half HERE: the **16 `millm_circuit_*` tools** in `backend/src/mcp_server/tools/millm_circuits.py`, now REGISTERED (they were the increment's signature defect — fully implemented, unit-tested and documented while never registered with the server, so every test passed by importing the module directly). 293 findings across 5 features × 3 review rounds. **Durable deliverable: the reachability rule** — *a capability is not shipped until a test FAILS when its wiring is removed* — enforced by `backend/tests/unit/test_reachability.py` (registry / built-server / caller shapes, payload AND call-count asserted) and written into this file's Code Quality Checklist. Also hardened here: the causal-language copy audit (`SURFACES` was hand-maintained at 5 files while 16 circuit modules went unaudited — now discovered, 5→17), and `millm_client.py` failure paths (a 200 HTML page from a misrouted ingress used to reach the agent as an empty SUCCESS, so it would read 'nothing is steering' and activate into a contention; `test_millm_client_failure_paths.py` is new — the client had no test file at all). Review records live in the miLLM repo: `0xcc/reviews/review_feature020_R{1,2,3}_2026-07-21.md`.
- **Phase (prior):** ✅ **CIRCUITS ARC INCREMENT CLOSED (2026-07-20)** — features 015/016/017/018 all IMPLEMENTED + each ran a THREE-round review cycle (~250+ findings total). The arc is closed end-to-end (UI+API+MCP): **discover (016) → validate (017, rung-2 ES) → make portable (018 contract + evidence ladder) → steer (015) with the compounding/cancellation hazard QUANTIFIED from the causally-validated effect size** (heuristic weight-prior fallback always labeled `heuristic`, never causal). "Steer this circuit" button bridges a promoted circuit into steering. Per-feature review records `.claude/context/sessions/review_feature01{5,6,7,8}_R{1,2,3}_2026-07-*.md`. Also this increment: mcslab.io domain purged; frontend vitest baseline 98→0 (surfaced a real useTrainingWebSocket resubscribe bug). **Recorded tech debt (follow-on BRD):** cluster-granularity hazards/steering (feature-level v1), two-SAE GPU generation run + VRAM<200MB (only unproven FPRD §8 criteria — GPU close-out on k8s host). Next natural step: BRD-MILLM-CIRCUITS-001 (multi-SAE serving) or Tier-2.5 attention-mediated mining fast-follow.
- **Phase (prior):** Post-increment sensing enhancements (2026-07-17) — 4 goal items shipped; review rounds in progress (R1 done: 23 findings/17 fixed)
- **Last Session:** July 17, 2026 — sensing enhancements per /goal: span highlighting (context_parts {before,span,after} via PREFIX decodes — SP-safe; migration 010; <mark> in detail), history dedup (LCP boundary; ANY re-arm clears history; truncation caps at last-reported; never-shrink guard; SENSING_DEDUP_HISTORY), quorum default = ALL sensable members, min_k runtime override (PUT /api/sensing/{id}/config, sensing_overrides stripped on export, panel input + reset, sensable-ceiling validation, millm_sensing_config MCP tool). Review R1: 23 findings/17 fixed (`0xcc/reviews/review_sensing_enhancements_2026-07-17.md`). Suites: backend 1108 / admin-ui 209.
- **Current Task:** None — **BRD-MIS-CLUSTERS-001 increment CLOSED (2026-07-16)**. Features 012/013/014 implemented, 3× review iterations each (28/28/15 findings), GitOps-deployed, Playwright E2E-verified (profile-titled Blended results + applied-count, budget bar/λ dial, low-cohesion gate, profile save/load/import/export). **013 validation gate PASSED after fitting γ=0** — the 1/G gain boost overdrove ~2× on all test clusters; B = B_dir/max(G,floor)^γ with default γ=0 (IDL-29 step-5 amendment; full data `0xcc/docs/cluster-strength-validation.md`). Review records: `.claude/context/sessions/review_feature01{2,3,4}_*_2026-07-16.md`.
- **Active Work:** None. Next natural step: follow-on BRD (MILLM import / unified MCP / Open WebUI / HF-marketplace publishing — research ready at `0xcc/docs/hf-marketplace-cluster-definitions-research.md`)
- **Circuits arc doc chain COMPLETE (2026-07-19):** BRD-MIS-CIRCUITS-001 + BRD-MIS-CIRCUITS-002 (rigor supplement: evidence ladder, statistics, attribution, intervention v2, faithfulness, Tier-2.5 readiness — consumed as ONE unit, 002 wins conflicts; Appendix A is normative math) → PPRD v3.9 (rows 16–19, §3.16–3.19) + PADR v3.0 (IDL-31..36) + four feature chains: 015 MultiSAE_Steering (hazards-v2), 016 Circuit_Discovery v2.0 (capture+stats+granularities+attribution), 017 Circuit_Validation (intervention/ES-vs-null/faithfulness/manifests), 018 Circuit_Portability (ladder/edge-typing/contract/projection — SEQUENCED FIRST: its ladder enum + contract gate the increment; then 016 → 017 → 015-hazards). SUBSTRATE pilot = research track only (BRD-MIS-SUBSTRATE-001.seed.md; no PPRD row). Next: execute 018 Phases 1–2 via 007_process-task-list.md.
- **New BRD (2026-07-15):** BRD-MIS-CLUSTERS-001 — rename Feature Groups→Clusters (UI), verify+trustworthy combined-strength steering, principled budget model (frequency-derived total budget, similarity-weighted allocation, budget-preserving rebalance), cluster authoring (name+narrative+tuned strengths), portable JSON cluster-definition export/import. **miStudio-only this increment**; MILLM import + unified MCP + Open WebUI captured as future_considerations for a follow-on BRD. Locked decisions: two-BRDs split, UI-only rename, sim-weighted allocation, marketplace=vision.
- **Deferred (separate initiative, awaiting user sign-off):** CI/CD → miLLM-style selective rebuilds + ArgoCD Image Updater (plan at `0xcc/plans/CICD-ArgoCD-ImageUpdater-Migration.md`, open decisions in §5 — NOT started, do not interleave with feature work)
- **Completed:**
  - Feature 011 IMPLEMENTED & DEPLOYED (2026-07-15): Steering UX — Blended|Compare segmented toggle (combinedMode boolean; /combined vs /compare), up to 20 features (was 4; backend max_length 4→20 both paths + color Literal widened 4→20 + dropped compare unique-color validator), 20-color purge-safe palette, frequency auto-baseline `S=clamp(2.9−2.6·freq,1,3)` with default-10 fallback + auto/default badge (`computeBaselineStrength` util), applyAutoBaseline + "Auto" apply-to-all preset, compact SelectedFeatureCard tiles (p-3→p-2, one-line header, additional-strengths behind expander), SAEFeatureSummary.activation_frequency, Feature Groups selection-map widened to carry stats through the hand-off. Commits 53f2245 (docs+caps) + e959ce5 (impl). CI green, k8s-deployed, Playwright E2E-verified (3/20 header, toggle, default badges, Auto preset — `0xcc/caps/miStudio_Steering_Panel-CompactTiles_20260715.png`). Doc chain 011_FPRD/FTDD/FTID/FTASKS + PPRD v3.5 (row 12 ✅, §3.12) + PADR v2.8 (IDL-27) ✅
  - Feature 010 IMPLEMENTED (2026-07-12): grouping data layer (4 tables + mcp_agent enum), FeatureGroupingService (TF-IDF context subgroups), Celery job + WS channel, 6 REST endpoints + approvals API, aqua-star 409 guard, MCP server (backend/src/mcp_server/, 33 tools, streamable-HTTP :8765, bearer auth, category gating, approval mode), compose profile `mcp` + k8s mistudio-mcp, Feature Groups panel + ApprovalsBanner, manual pages (mcp-server, feature-groups, API/WS reference) ✅
  - Feature 010 doc chain (2026-07-12): BRD-MIS-MCP-001 → PPRD v3.3 (row 11, §3.11), PADR v2.6 (IDL-26), 010_FPRD/FTDD/FTID/FTASKS ✅
  - Docusaurus manual overhaul (2026-07-11): 19→34 pages (Concepts, Quickstart, API/WS/data-model Reference, FAQ, landing page) — live ✅
  - Feature-by-feature review (001–009) + synthesis in `0xcc/reviews/` ✅
  - Remediated all review findings: P0 (broken extraction imports), P1/P2/P3, + 3 deferred schema items ✅
  - Alembic multi-head merge (cd6c46abac48) + celery_task_id + training_metrics unique constraint + NeuronpediaPushJob ORM model ✅
  - Fixed pre-existing test flakiness (conftest enum isolation + parallel-mock test) ✅
  - Enhanced per-feature two-pass LLM labeling ✅
  - OpenAI API integration (enhanced + bulk labeling) ✅
  - OpenAI SDK standardization in EnhancedLabelingService ✅
  - Context-Aware Labeling template (semantic pattern focus) ✅
  - Settings Panel — encrypted API keys (AES-256-GCM), Fetch Models ✅
  - Security hardening — path injection, stack-trace exposure, non-root containers ✅
  - Supply-chain security — CodeQL, Docker Scout, SLSA provenance ✅
  - Feature notes markdown rendering (react-markdown + remark-gfm) ✅
  - v0.5.0 public release (Apache 2.0, CI/CD, K8s deployment) ✅
  - 0xcc documentation updated to v3.0 (PPRD, PADR, FPRD, FTASKS) ✅
  - Docusaurus manual updated with enhanced labeling docs + 12 screenshots ✅
  - Settings panel PIN protection — PBKDF2-SHA256 gate + MISTUDIO_BYPASS_PIN recovery ✅
  - Multi-GPU doc corrections — Phases 1 & 2 retrospectively marked complete ✅
  - Full end-to-end security review (multi-agent) — 7 findings identified and documented ✅
- **Test Status:** 995 passed, 4 skipped (conditional on external dependencies)
- **Services Status:** K8s (mistudio namespace) ✅, Docker Compose (192.168.244.222) ✅
  - Backend (port 8000) ✅, Frontend (nginx-unprivileged, port 8080→80) ✅
  - PostgreSQL ✅, Redis ✅, Celery Worker ✅, Celery Beat ✅, Nginx ✅
- **K8s Manifest:** Restored and in sync at /home/sean/app/k8s-mistudio.hitsai.local/mistudio-deployment.yaml
- **Pending (deferred):**
  - Backend non-root container (entrypoint refactor + K8s fsGroup — its own session)
  - Pytest 9 bump for miLLM (pre-existing test env issues, not blocking)
  - Multi-GPU distributed training (DDP/NCCL) — monitoring + job routing already complete since Dec 2025

## PRIMARY UI/UX REFERENCE

Key aspects (load full file only when needed):
- UI/UX design patterns and visual style
- Component layouts and interactions
- User workflows and navigation
- API contracts and data structures
- Feature completeness and behavior

All implementation MUST match the Mock UI specification exactly.

## Application Startup

### Complete Startup (All Services)
```bash
# ONE COMMAND to start everything:
./start-mistudio.sh

# This starts (in order):
# 1. Docker services (PostgreSQL, Redis, Nginx)
# 2. Celery worker
# 3. Backend (FastAPI on port 8000)
# 4. Frontend (Vite on port 3000)
#
# Access at: http://mistudio.hitsai.local
```

**IMPORTANT**: Before first run, add domain to /etc/hosts:
```bash
sudo bash -c 'echo "127.0.0.1  mistudio.hitsai.local" >> /etc/hosts'
```

### Stop All Services
```bash
./stop-mistudio.sh
```

## Standard Development Workflow

### Bug Fix / Feature Workflow (The Normal Pattern)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. IDENTIFY & FIX                                                       │
│     - Find the bug/implement feature                                    │
│     - Run type-check: cd frontend && npm run type-check                 │
│     - Run build: npm run build                                          │
│                                                                          │
│  2. COMMIT & PUSH                                                        │
│     - git add <files>                                                   │
│     - git commit -m "fix/feat: description"                             │
│     - git push origin main                                              │
│                                                                          │
│  3. WAIT FOR CI                                                          │
│     - Sync workflow (~1 min)                                            │
│     - Frontend build (~2-3 min)                                         │
│     - Backend build (~9 min) ← PyTorch/ML deps take longer              │
│                                                                          │
│  4. DEPLOY TO K8S (use k8s helper function)                             │
│     - k8s_deploy                                                        │
│                                                                          │
│  5. VERIFY                                                               │
│     - Test at http://k8s-mistudio.hitsai.local                             │
└─────────────────────────────────────────────────────────────────────────┘
```

**CRITICAL: NO LOCAL DOCKER BUILDS. The CI/CD pipeline is fully automated.**

### K8s Helper Commands

**Load helpers at session start:**
```bash
source scripts/k8s-helpers.sh
```

**Available commands after sourcing:**
| Command | Description |
|---------|-------------|
| `k8s_check` | Check DockerHub image timestamps (verify CI completed) |
| `k8s_deploy` | Full deploy: pull + restart + wait + verify |
| `k8s_status` | Quick pod status |
| `k8s_logs [n]` | Backend logs (default 50 lines) |
| `k8s_logs_celery` | Celery worker logs |
| `k8s_gpu` | GPU utilization |
| `k8s "cmd"` | Run any command on k8s host |

**Typical deploy sequence after push:**
```bash
source scripts/k8s-helpers.sh  # Load helpers
k8s_check                       # Wait for timestamps > push time
k8s_deploy                      # Pull + restart + verify
```

### K8s Environment
| Setting | Value |
|---------|-------|
| Host | 192.168.244.61 (mcs-lnxgpu01) |
| Namespace | mistudio |
| Domain | k8s-mistudio.hitsai.local |
| GPU | NVIDIA RTX 3090 (24GB) |
| Manifest | k8s/mistudio-deployment.yaml |

### Service Status Check
```bashPlease
# Check all services:
docker ps  # Should show: mistudio-postgres, mistudio-redis, mistudio-nginx
lsof -i :8000  # Backend should be running
lsof -i :3000  # Frontend should be running
pgrep -f celery  # Celery worker should be running

# Access points:
# - Main app: http://mistudio.hitsai.localplease
# - Frontend direct: http://localhost:3000
# - Backend direct: http://localhost:8000
# - API docs: http://localhost:8000/docs
```

## Quick Resume Commands

### Lean Session Start (Recommended)
```bash
# Minimal context loading - most efficient approach
"Please help me resume where I left off"
# This automatically loads: CLAUDE.md + session_state.json

# Load specific current work area only when needed:
# 0xcc/tasks/[current-task-file].md  # The specific task being worked on
```

### On-Demand Loading Strategy
⚠️ **IMPORTANT**: The following files are LARGE (40k+ chars) and should ONLY be loaded when you encounter specific questions. **DO NOT load them automatically at session start.**

```bash
# Load when UI/styling question arises (207k chars):
# 0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx

# Load when business context/feature priority question arises (54k chars):
# 0xcc/prds/000_PPRD|miStudio.md

# Load when architectural decision question arises (72k chars):
# 0xcc/adrs/000_PADR|miStudio.md

# Load when design clarification needed:
# 0xcc/tdds/[feature]_FTDD.md

# Load when implementation guidance needed:
# 0xcc/tids/[feature]_FTID.md
```

### Research Integration
```bash
# Use MCP ref server for contextual research (when available)
/mcp ref search "[context-specific query]"
```

## Housekeeping Commands
```bash
"Please create a checkpoint"        # Save complete state
"Please help me resume"            # Restore context for new session
"My context is getting too large"  # Clean context, restore essentials
"Please save the session transcript" # Save session transcript
"Please show me project status"    # Display current state
```

## Project Standards

### Technology Stack

**Backend:**
- Python 3.11+, FastAPI, PostgreSQL 14+, Redis 7+, Celery
- PyTorch 2.0+, HuggingFace (transformers, datasets), bitsandbytes
- TensorRT for Jetson optimization

**Frontend:**
- React 18+ with TypeScript, Vite, Zustand
- Tailwind CSS (slate dark theme per Mock UI)
- Lucide React icons, D3.js + Recharts
- Socket.IO for real-time updates

**Infrastructure:**
- Docker Compose for development (nginx, postgres, redis, backend, frontend, celery)
- Nginx reverse proxy (port 80, future HTTPS on 443)
- Base URL: http://mistudio.hitsai.local
- systemd for production (Jetson)
- Local filesystem storage (/data/)

### Coding Standards

**Python:**
- Formatter: Black (line length 100)
- Linter: Ruff
- Type Checker: MyPy (strict)
- Docstrings: Google style

**TypeScript:**
- Formatter: Prettier
- Linter: ESLint (Airbnb)
- All components strictly typed

### Naming Conventions

**Python:** `snake_case` functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
**TypeScript:** `camelCase` functions, `PascalCase` components/types, `UPPER_SNAKE_CASE` constants

### Testing

**Backend:** pytest (>80% coverage target)
**Frontend:** Vitest + React Testing Library
**E2E:** Playwright for critical paths

### Git Workflow

**Branches:** `main` (production), `develop` (integration), `feature/*`, `bugfix/*`
**Commits:** Conventional commits (`feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`)
**Review:** All code reviewed, tests pass, no coverage decrease

### File Organization

**Backend:** `app/api/`, `app/services/`, `app/ml/`, `app/db/`, `app/workers/`
**Frontend:** `src/components/`, `src/stores/`, `src/services/`, `src/hooks/`, `src/types/`

### Error Handling

- Backend: FastAPI HTTPException with proper status codes
- Frontend: Try-catch with axios error handling
- Structured error responses with `error.code`, `error.message`, `error.details`

### API Design

- RESTful conventions (GET, POST, PUT, PATCH, DELETE)
- Response format: `{ data, meta }` or `{ error }`
- Status codes: 200, 201, 202, 400, 404, 409, 429, 500, 503
- Pagination: `?page=1&limit=50`
- WebSocket: Socket.IO with rooms per training job

### UI/UX Standards

#### **PRIMARY REFERENCE:** `0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx`

- Dark theme: slate color palette (bg-slate-950, 900, 800)
- Emerald accents: buttons, success states
- Tailwind utility classes matching Mock UI exactly
- Functional components with TypeScript
- Zustand for global state, local state for UI

### Database Schema

- PostgreSQL with JSONB for flexible metadata
- Time-series metrics in dedicated tables with indexes
- Partitioned tables for large data (feature_activations)
- Foreign keys with CASCADE for data integrity

### Edge Optimization

- Mixed precision training (FP16)
- Gradient accumulation for large effective batches
- Memory-mapped files for datasets/activations
- TensorRT optimization for Jetson inference
- INT8/INT4 quantization via bitsandbytes

### Deployment

**Development:** Docker Compose (nginx, postgres, redis, backend, frontend, celery)
**Production:** systemd service on Jetson with Docker Compose + nginx reverse proxy
**Base URL:** http://mistudio.hitsai.local (port 80)
**Future HTTPS:** Port 443 with SSL certificate
**Alternative:** Native installation (Nginx + PostgreSQL + Redis + Python + Node.js)

## AI Dev Tasks Framework Workflow

### Document Creation Sequence
1. **Project Foundation**
   - `000_PPRD|[project-name].md` → `0xcc/prds/` (Project PRD)
   - `000_PADR|[project-name].md` → `0xcc/adrs/` (Architecture Decision Record)
   - Update this CLAUDE.md with Project Standards from ADR

2. **Feature Development** (repeat for each feature)
   - `[###]_FPRD|[feature-name].md` → `0xcc/prds/` (Feature PRD)
   - `[###]_FTDD|[feature-name].md` → `0xcc/tdds/` (Technical Design Doc)
   - `[###]_FTID|[feature-name].md` → `0xcc/tids/` (Technical Implementation Doc)
   - `[###]_FTASKS|[feature-name].md` → `0xcc/tasks/` (Task List)

### Instruction Documents Reference
- `0xcc/instruct/001_create-project-prd.md` - Creates project vision and feature breakdown
- `0xcc/instruct/002_create-adr.md` - Establishes tech stack and standards
- `0xcc/instruct/003_create-feature-prd.md` - Details individual feature requirements
- `0xcc/instruct/004_create-tdd.md` - Creates technical architecture and design
- `0xcc/instruct/005_create-tid.md` - Provides implementation guidance and coding hints
- `0xcc/instruct/006_generate-tasks.md` - Generates actionable development tasks
- `0xcc/instruct/007_process-task-list.md` - Guides task execution and progress tracking
- `0xcc/instruct/008_housekeeping.md` - Session management and context preservation

## Document Inventory

### Project Level Documents
- ✅ 0xcc/prds/000_PPRD|miStudio.md (Project PRD - Completed 2025-10-05)
- ✅ 0xcc/adrs/000_PADR|miStudio.md (Architecture Decision Record - Completed 2025-10-05)
- ✅ 0xcc/prds/BRD-MIS-CLUSTERS-001.md (Incremental BRD — Feature Clusters & portable combined-strength steering, 2026-07-15)
- ✅ PPRD v3.6 (rows 13–15, §3.13–3.15) + PADR v2.9 (IDL-28 Clusters terminology/labeling, IDL-29 cluster strength budget model, IDL-30 portable cluster definitions) — 2026-07-16

### Feature 012 — Clusters UX & Trustworthy Blended Results (✅ Complete 2026-07-16)
- ✅ 0xcc/prds/012_FPRD|Clusters_UX.md · ✅ 0xcc/tdds/012_FTDD|Clusters_UX.md · ✅ 0xcc/tids/012_FTID|Clusters_UX.md · ✅ 0xcc/tasks/012_FTASKS|Clusters_UX.md

### Feature 013 — Cluster Strength Budget Model (✅ Complete 2026-07-16, validation gate PASSED γ=0)
- ✅ 0xcc/prds/013_FPRD|Cluster_Strength_Model.md · ✅ 0xcc/tdds/013_FTDD|Cluster_Strength_Model.md · ✅ 0xcc/tids/013_FTID|Cluster_Strength_Model.md · ✅ 0xcc/tasks/013_FTASKS|Cluster_Strength_Model.md

### Feature 014 — Cluster Authoring & Portable Definitions (✅ Complete 2026-07-16)
- ✅ 0xcc/prds/014_FPRD|Cluster_Definitions.md · ✅ 0xcc/tdds/014_FTDD|Cluster_Definitions.md · ✅ 0xcc/tids/014_FTID|Cluster_Definitions.md · ✅ 0xcc/tasks/014_FTASKS|Cluster_Definitions.md

> **⚠️ FILE-NUMBER ↔ PPRD-ROW OFFSET (read before adding a feature).** For features **1–14** the
> file number equals the PPRD inventory row (file `014_` = row 14). Starting with the circuits arc a
> **+1 offset** exists: **file number N = PPRD row N+1** for features 15 onward — file `015_` = row 16,
> `018_` = row 19, `019_` = row 20. The PPRD inventory (§2.1) and detail sections (§3.x) are the
> authority on the product feature number; filenames keep the `015_`… sequence. This is deliberate (kept
> to avoid renaming shipped files); do NOT "fix" a doc by renumbering it. The next feature is file `020_`
> = PPRD row 21.

### Circuits arc — BRDs + Features 015–018 (= PPRD rows 16–19) (Planned 2026-07-19)
- ✅ 0xcc/prds/BRD-MIS-CIRCUITS-001.md · ✅ 0xcc/prds/BRD-MIS-CIRCUITS-002.md (supplement; Appendix A normative) · 📌 0xcc/prds/BRD-MIS-SUBSTRATE-001.seed.md (seed only)
- ✅ 015_FPRD/FTDD/FTID/FTASKS|MultiSAE_Steering · ✅ 016_…|Circuit_Discovery (v2.0) · ✅ 017_…|Circuit_Validation · ✅ 018_…|Circuit_Portability
- ✅ PPRD v3.9 (§3.16–3.19) · ✅ PADR v3.0 (IDL-31..36)

### Feature 020 (= files 019_*) — Circuit Strength Calibration (doc chain COMPLETE, impl PLANNED 2026-07-21)
- ✅ 019_FPRD/FTDD/FTID/FTASKS|Circuit_Calibration — **files are `019_*`, product feature is PPRD row 20** (the +1 offset above; not an error). The arc's next step. Grounded in the served-circuit finding: placeholder strengths shipped fluent-but-FALSE at "usable" 1.40; usable band was ~0.4–0.6 effective.
- ✅ PPRD v3.10 (row 20, §3.20) · ✅ PADR v3.1 (IDL-37: two-detector usable-band search — onset by output-drift, correctness cliff by LLM judge on generated NEUTRAL-topic falsifiable probes; adaptive bisection; additive nullable `calibration` block clamps intensity_range to [onset,cliff]; badge not gate; provisional cross-plane)
- Next: execute 019 Phase 1 (schema/contract) via 007_process-task-list.md. Contract crosses to miLLM (additive nullable) — schema-sync + vendored-identity guards are acceptance-blocking.

### Feature Documents
*[Add as features are identified and developed]*

**Example format:**
- ❌ 0xcc/prds/001_FPRD|Feature_A.md (Feature PRD)
- ❌ 0xcc/tdds/001_FTDD|Feature_A.md (Technical Design Doc)
- ❌ 0xcc/tids/001_FTID|Feature_A.md (Technical Implementation Doc)
- ❌ 0xcc/tasks/001_FTASKS|Feature_A.md (Task List)

### Status Indicators
- ✅ **Complete:** Document finished and reviewed
- ⏳ **In Progress:** Currently being worked on
- ❌ **Pending:** Not yet started
- 🔄 **Needs Update:** Requires revision based on changes

## Housekeeping Status
- **Last Checkpoint:** [Date/Time] - [Brief description]
- **Last Transcript Save:** [Date/Time] - [File location in 0xcc/transcripts/]
- **Context Health:** Good/Moderate/Needs Cleanup
- **Session Count:** [Number] sessions since project start
- **Total Development Time:** [Estimated hours]

## Task Execution Standards

### Completion Protocol
- ✅ One sub-task at a time, ask permission before next
- ✅ Mark sub-tasks complete immediately: `[ ]` → `[x]`
- ✅ When parent task complete: Run tests → Stage → Clean → Commit → Mark parent complete
- ✅ Never commit without passing tests
- ✅ Always clean up temporary files before commit

### Commit Message Format
```bash
git commit -m "feat: [brief description]" -m "- [key change 1]" -m "- [key change 2]" -m "Related to [Task#] in [PRD]"
```

### Test Commands
*[Will be defined in ADR, examples:]*
- **Frontend:** `npm test` or `npm run test:unit`
- **Backend:** `pytest` or `python -m pytest`
- **Full Suite:** `[project-specific command]`

## Code Quality Checklist

### Before Any Commit
- [ ] All tests passing
- [ ] No console.log/print debugging statements
- [ ] No commented-out code blocks
- [ ] No temporary files (*.tmp, .cache, etc.)
- [ ] Code follows project naming conventions
- [ ] Functions/methods have docstrings if required
- [ ] Error handling implemented per ADR standards

### Reachability (a shipping gate, not a style preference)

**A capability is not shipped until a test FAILS when its wiring is removed.**

Before marking any user-facing capability complete — an MCP tool, a REST route, a
panel, a store action — delete the line that REGISTERS it, run the suite, and
require a red. Green means the capability is unreachable in production, untested,
or both.

This repo's MCP server is the cautionary case: the 16 `millm_circuit_*` tools
were fully implemented, unit-tested and documented in `docs/mcp-contract.md`
while never registered with the server. Every test passed by importing the tool
module directly, so the suite was green and the docs said ✅ while no agent could
call the feature. See `backend/tests/unit/test_reachability.py` — the harness that
now guards it, and the shape to copy for new surfaces.

- [ ] Assert presence in the **live registry**, never that the module imports
- [ ] Assert the **payload and the call count** — "was called" passes against a
      call sending the wrong arguments
- [ ] When a review round fixes an unreachable capability, mutate the new wiring
      as a negative control to prove the guard bites

### File Organization Rules
*[Will be defined in ADR, examples:]*
- Place test files alongside source files: `Component.tsx` + `Component.test.tsx`
- Follow directory structure from ADR
- Use naming conventions: `[Feature][Type].extension`
- Import statements organized: external → internal → relative
- Framework files in `0xcc/` directory, project files in standard locations

## Context Management

### Session End Protocol
```bash
# 1. Update CLAUDE.md status section
# 2. Create session summary
"Please create a checkpoint"
# 3. Commit progress
git add .
git commit -m "docs: completed [task] - Next: [specific action]"
```

### Context Recovery (If Lost)
```bash
# Mild context loss - files to reference if needed:
# CLAUDE.md
# 0xcc/session_state.json
ls -la 0xcc/*/
# 0xcc/instruct/[current-phase].md

# Severe context loss - files to reference if needed:
# CLAUDE.md
# 0xcc/prds/000_PPRD|[project-name].md
# 0xcc/adrs/000_PADR|[project-name].md
ls -la 0xcc/*/
# 0xcc/instruct/
```

### Resume Commands for Next Session
```bash
# Standard resume sequence
"Please help me resume where I left off"
# Files are automatically loaded from context - no need to manually load
# Specific next action: [detailed action]
```

## Progress Tracking

### Task List Maintenance
- Update task list file after each sub-task completion
- Add newly discovered tasks as they emerge
- Update "Relevant Files" section with any new files created/modified
- Include one-line description for each file's purpose
- Distinguish between framework files (0xcc/) and project files (src/, tests/, etc.)

### Status Indicators for Tasks
- `[ ]` = Not started
- `[x]` = Completed
- `[~]` = In progress (use sparingly, only for current sub-task)
- `[?]` = Blocked/needs clarification

### Session Documentation
After each development session, update:
- Current task position in this CLAUDE.md
- Any blockers or questions encountered
- Next session starting point
- Files modified in this session (both 0xcc/ and project files)

## Implementation Patterns

### Real-time Updates Architecture
The application uses a consistent WebSocket-first approach for all real-time updates:

**WebSocket Channels Pattern:**
- Channel naming: `{entity_type}/{entity_id}/{event_type}` or `{entity_type}/{entity_id}`
- Event types: `progress`, `metrics`, `status`, etc.
- All channels use Socket.IO rooms for pub/sub

**Current WebSocket Implementations:**
1. **Training Progress** - Channel: `training/{training_id}`, Events: `progress`, `completed`, `failed`
2. **Extraction Progress** - Channel: `extraction/{extraction_id}`, Events: `progress`, `completed`, `failed`
3. **Model Download Progress** - Channel: `model/{model_id}`, Events: `download_progress`, `download_completed`, `download_failed`
4. **Dataset Progress** - Channel: `dataset/{dataset_id}`, Events: `progress`, `completed`, `failed`
5. **System Monitoring** - Channels:
   - `system/gpu/{gpu_id}` - Per-GPU metrics (utilization, memory, temperature, power)
   - `system/cpu` - CPU utilization metrics
   - `system/memory` - RAM and Swap usage
   - `system/disk` - Disk I/O rates
   - `system/network` - Network I/O rates
   - Event type: `metrics` (emitted every 2 seconds via Celery Beat)

**WebSocket Fallback Pattern:**
- Frontend hooks automatically detect WebSocket connection state
- Stores implement automatic fallback to HTTP polling when WebSocket disconnects
- Polling stops automatically when WebSocket reconnects
- Example: `systemMonitorStore.setIsWebSocketConnected()` manages fallback logic

**Backend Emission Pattern:**
- All WebSocket emissions use `backend/src/workers/websocket_emitter.py`
- Celery tasks emit updates via internal HTTP endpoint: `POST /api/internal/ws/emit`
- Emission functions: `emit_training_progress()`, `emit_gpu_metrics()`, etc.
- Celery Beat scheduler handles periodic emissions (system monitoring)

**Frontend Subscription Pattern:**
- React hooks manage channel subscriptions: `useTrainingWebSocket()`, `useSystemMonitorWebSocket()`, etc.
- Hooks subscribe to channels on mount, unsubscribe on unmount
- Event handlers update Zustand stores
- Stores provide data to components via selectors

### Error Handling
*[Will be defined in ADR - placeholder for standards]*
- Use project-standard error handling patterns from ADR
- Always handle both success and failure cases
- Log errors with appropriate level (error/warn/info)
- User-facing error messages should be friendly

### Testing Patterns
*[Will be defined in ADR - placeholder for standards]*
- Each function/component gets a test file
- Test naming: `describe('[ComponentName]', () => { it('should [behavior]', () => {})})`
- Mock external dependencies
- Test both happy path and error cases
- Aim for [X]% coverage per ADR standards

## Debugging Protocols

### When Tests Fail
1. Read error message carefully
2. Check recent changes for obvious issues
3. Run individual test to isolate problem
4. Use debugger/console to trace execution
5. Check dependencies and imports
6. Ask for help if stuck > 30 minutes

### When Task is Unclear
1. Review original PRD requirements
2. Check TDD for design intent
3. Look at TID for implementation hints
4. Ask clarifying questions before proceeding
5. Update task description for future clarity

## Feature Priority Order
*From Project PRD - Core Features (P0):*

**MVP Features (Must Have):**
1. Dataset Management Panel (P0) - HuggingFace integration, local ingestion ✅
2. Model Management Panel (P0) - Model downloads, quantization, architecture viewer ✅
3. SAE Training System (P0) - Sparse autoencoder training with real-time monitoring ✅
4. Feature Discovery & Browser (P0) - Extract and analyze features from trained SAEs ✅
5. Model Steering Interface (P0) - Feature-based interventions and comparative generation ✅

**Secondary Features (P1):**
6. Training Templates & Presets - Save/load training configurations ✅
7. Extraction Templates - Preset activation extraction configs ✅
8. Steering Presets - Save/load steering configurations
9. Advanced Visualizations - UMAP, correlation heatmaps
10. Feature Analysis Tools - Logit lens, ablation studies
11. Checkpoint Auto-Save - Automatic training checkpoints
12. Dataset Statistics Dashboard - Detailed dataset metrics

**Future Features (P3):**
13. Multi-Model Comparison
14. Export & Reporting
15. Collaborative Features
16. Advanced Circuit Analysis

## Session History Log

### Session 1: 2025-10-05 - Project Foundation
- **Accomplished:**
  - Created 0xcc framework directory structure (prds, adrs, tdds, tids, tasks, docs, transcripts, checkpoints, scripts)
  - Created comprehensive Project PRD (000_PPRD|miStudio.md) based on Mock UI specification
  - Updated CLAUDE.md with project name, status, and UI reference priority
  - Established Mock UI as PRIMARY reference for all implementation
- **Next:** Create Architecture Decision Record using 0xcc/instruct/002_create-adr.md
- **Files Created:**
  - 0xcc/prds/000_PPRD|miStudio.md (14,000+ lines)
  - Updated CLAUDE.md with project context
- **Duration:** ~2 hours
- **Key Decision:** Mock-embedded-interp-ui.tsx is the authoritative UI/UX specification

### Session 2: 2025-10-18 - SAE Training Feature Implementation & Bug Fixes
- **Accomplished:**
  - Fixed critical API configuration bug (same-origin requests through nginx proxy)
  - Fixed WebSocket configuration to use proper WS_URL and WS_PATH
  - Fixed hardcoded `/data` path in training worker to use `settings.data_dir`
  - Fixed Models dropdown showing blank (changed `model.model_id` to `model.name`)
  - Reordered training configuration fields: Dataset → Model → Architecture (consistent with data flow)
  - Added delete functionality for completed/failed training jobs with confirmation
  - Tested backend API endpoints, database schema, training creation, and Celery worker
  - Successfully ran test training job (100 steps, final loss: 112.93)
- **Tests Completed:**
  - ✅ Backend API endpoints accessible
  - ✅ Database tables exist with correct schemas (trainings, training_metrics, checkpoints)
  - ✅ Training creation via API (fixed permission denied error)
  - ✅ Celery worker processes training tasks successfully
- **Files Modified:**
  - `frontend/src/config/api.ts` - Changed API_BASE_URL and WS_URL to empty string
  - `frontend/src/api/websocket.ts` - Added WS_URL and WS_PATH configuration
  - `backend/src/workers/training_tasks.py` - Fixed hardcoded data path
  - `frontend/src/components/panels/TrainingPanel.tsx` - Fixed model display and field order
  - `frontend/src/components/training/TrainingCard.tsx` - Added delete functionality
- **Duration:** ~4 hours
- **Key Fixes:** API configuration for nginx proxy, data directory permissions, UI/UX improvements

### Session 3: 2025-10-18/19 - SAE Training UX & System Monitor Improvements
- **Accomplished:**
  - **Training Feature Enhancements:**
    - Fixed retry button functionality (implemented retryTraining store method)
    - Added bulk delete with checkbox selection for training jobs
    - Added compact hyperparameters display in training tiles
    - Implemented detailed hyperparameters modal with organized sections
    - Changed icon from Info to Sliders for better affordance
    - Added human-readable model/dataset names (lookup from stores)
    - Added completion timestamp and calculated training duration
    - Implemented config persistence after job start for easy iteration
  - **System Monitor Improvements:**
    - Fixed time range to 1 hour view only (removed TimeRangeSelector)
    - Overlaid GPU temperature on utilization chart with dual Y-axis
    - Combined 3 charts into 2-column grid layout
    - Ensured always-current data on page visit
    - Updated chart title to "Utilization & Temperature"
    - Added proper units to tooltip (% vs °C)
- **Files Modified:**
  - `frontend/src/components/training/TrainingCard.tsx`
  - `frontend/src/components/panels/TrainingPanel.tsx`
  - `frontend/src/stores/trainingsStore.ts`
  - `frontend/src/components/SystemMonitor/SystemMonitor.tsx`
  - `frontend/src/components/SystemMonitor/UtilizationChart.tsx`
  - `frontend/src/hooks/useHistoricalData.ts`
- **Duration:** ~3 hours
- **Key Improvements:** Enhanced UX for training iteration, cleaner System Monitor with efficient layout

### Session 4: 2025-10-21 - Training Templates Feature Implementation
- **Accomplished:**
  - **Complete Training Templates Frontend Implementation:**
    - Created TrainingTemplateForm.tsx with comprehensive validation (16 hyperparameter fields)
    - Created TrainingTemplateCard.tsx with action buttons and template details display
    - Created TrainingTemplateList.tsx with search, pagination, and empty states
    - Rebuilt TrainingTemplatesPanel.tsx with full CRUD workflow
    - Implemented collapsible Advanced Settings section in form
    - Added Export/Import functionality with JSON file handling
    - Added Favorites management (toggle and filter by favorite)
    - Implemented Duplicate functionality with "(Copy)" suffix
    - Added notification system with success/error messages and auto-dismiss
    - Implemented modal-based editing with overlay
    - Added comprehensive client-side validation for all fields
- **Pattern Study:**
  - Studied ExtractionTemplatesPanel.tsx (359 lines) for architecture patterns
  - Studied ExtractionTemplateCard.tsx (162 lines) for card layout patterns
  - Studied ExtractionTemplateForm.tsx (400 lines) for form validation patterns
  - Studied ExtractionTemplateList.tsx for search and pagination patterns
- **Files Created:**
  - `frontend/src/types/trainingTemplate.ts` - TypeScript type definitions
  - `frontend/src/api/trainingTemplates.ts` - API client functions
  - `frontend/src/stores/trainingTemplatesStore.ts` - Zustand state management
  - `frontend/src/components/trainingTemplates/TrainingTemplateForm.tsx` - Comprehensive form component
  - `frontend/src/components/trainingTemplates/TrainingTemplateCard.tsx` - Display card component
  - `frontend/src/components/trainingTemplates/TrainingTemplateList.tsx` - List component with search
- **Files Replaced:**
  - `frontend/src/components/panels/TrainingTemplatesPanel.tsx` - Main orchestration panel (replaced placeholder)
- **Backend Files (Previously Complete):**
  - Database migration, SQLAlchemy model, Pydantic schemas, service layer, and API endpoints already implemented
- **Duration:** ~3 hours
- **Key Achievement:** Production-ready Training Templates feature with full CRUD, matching ExtractionTemplates quality and patterns

### Session 5: 2025-10-22 - System Monitoring WebSocket Migration & Architecture Review
- **Accomplished:**
  - **Architecture Review:**
    - Conducted comprehensive multi-agent review of progress/resource monitoring architecture
    - Identified inconsistency: Job progress uses WebSocket consistently, system monitoring uses polling
    - Created detailed review document with findings from 4 agent perspectives (Product, QA, Architect, Test)
    - Generated prioritized task list (9 major tasks, 79 sub-tasks, 110-144 hours estimated)
  - **System Monitoring WebSocket Migration (HP-1):**
    - Added 6 new WebSocket emission functions to `websocket_emitter.py` for system metrics
    - Created new Celery Beat task for periodic system metrics collection (every 2 seconds)
    - Defined WebSocket channel naming conventions for system monitoring:
      - `system/gpu/{gpu_id}` - Per-GPU metrics
      - `system/cpu` - CPU utilization
      - `system/memory` - RAM and Swap
      - `system/disk` - Disk I/O rates
      - `system/network` - Network I/O rates
    - Created `useSystemMonitorWebSocket.ts` React hook for channel subscriptions
    - Updated `systemMonitorStore.ts` with WebSocket integration and automatic polling fallback
    - Updated `SystemMonitor.tsx` component to use WebSocket-first with polling fallback
    - Configured Celery Beat scheduler with system monitoring task
    - Added `system_monitor_interval_seconds` configuration setting (default: 2s)
  - **Bug Fixes:**
    - Fixed console spam from 404 errors on extraction endpoint (now returns 200 with null data)
    - Updated frontend to handle new extraction endpoint response format
  - **Documentation:**
    - Added comprehensive Real-time Updates Architecture section to CLAUDE.md
    - Documented WebSocket channel patterns, fallback logic, emission patterns, and subscription patterns
- **Files Created:**
  - `.claude/context/sessions/review_progress_monitoring_architecture_2025-10-22.md` - Architecture review document
  - `0xcc/tasks/SUPP_TASKS|Progress_Architecture_Improvements.md` - Implementation task list
  - `backend/src/workers/system_monitor_tasks.py` - Celery Beat task for metrics collection
  - `frontend/src/hooks/useSystemMonitorWebSocket.ts` - WebSocket subscription hook
- **Files Modified:**
  - `backend/src/workers/websocket_emitter.py` - Added system metrics emission functions
  - `backend/src/core/config.py` - Added system_monitor_interval_seconds setting
  - `backend/src/core/celery_app.py` - Added beat schedule, routing, autodiscovery
  - `frontend/src/stores/systemMonitorStore.ts` - Added WebSocket integration
  - `frontend/src/components/SystemMonitor/SystemMonitor.tsx` - Integrated WebSocket hook
  - `backend/src/api/v1/endpoints/models.py` - Fixed extraction endpoint 404 response
  - `frontend/src/stores/modelsStore.ts` - Updated to handle new extraction endpoint format
  - `CLAUDE.md` - Added Real-time Updates Architecture documentation
- **Duration:** ~5 hours
- **Key Achievement:** Achieved architectural consistency by migrating system monitoring from polling to WebSocket-first pattern, matching the approach used for all job progress tracking

### Session 6: 2025-12-16 - Integration Test Fixes & Dataset Samples Bug Fix
- **Accomplished:**
  - **Integration Test Suite Fixes (15 tests fixed):**
    - Fixed `test_websocket_emission_integration.py` - Updated event name assertions to use namespaced events (`extraction:progress`, `extraction:failed`)
    - Fixed `test_dataset_cancellation.py` - Removed invalid `tokenized_path` attribute (moved to DatasetTokenization model), corrected PROCESSING status behavior (raw files preserved for retry)
    - Fixed `test_dataset_workflow.py` - Removed all `tokenized_path` references from DatasetUpdate calls
    - Fixed `test_dual_labels.py` - Added `pytestmark` to skip when OPENAI_API_KEY not configured
    - Fixed `test_training_workflow.py` - Fixed `delete_training` return type handling (returns dict, not boolean)
    - Fixed `test_vectorization_manual.py` - Added `pytestmark` to skip when no completed training exists
  - **Dataset Samples Endpoint Bug Fix:**
    - Fixed 500 Internal Server Error when fetching dataset samples
    - Root cause: HuggingFace datasets (e.g., The Pile) contain `bytes` objects in fields like `repetitions`
    - Added `sanitize_value()` function to recursively convert bytes to strings
    - Handles nested dicts, lists, and tuples
    - Uses UTF-8 decoding with Latin-1 fallback for any byte sequence
- **Key Technical Insights:**
  - Dataset model vs DatasetTokenization model: `tokenized_path` is stored in DatasetTokenization, not Dataset
  - WebSocket event naming: Events are namespaced (e.g., `extraction:progress` not just `progress`)
  - `cancel_dataset_download`: DOWNLOADING status deletes raw files, PROCESSING status preserves them for retry
  - `delete_training` service returns `{"deleted": True, ...}` dict, not boolean
- **Files Modified:**
  - `backend/tests/integration/test_websocket_emission_integration.py` - Event name assertions
  - `backend/tests/integration/test_dataset_cancellation.py` - Model attributes and behavior fixes
  - `backend/tests/integration/test_dataset_workflow.py` - Removed tokenized_path
  - `backend/tests/integration/test_dual_labels.py` - Added skip marker
  - `backend/tests/integration/test_training_workflow.py` - Return type handling
  - `backend/tests/integration/test_vectorization_manual.py` - Added skip marker
  - `backend/src/api/v1/endpoints/datasets.py` - Added sanitize_value() for bytes handling
- **Commits:**
  - `2980033` - test: fix 15 failing tests across integration test suite
  - `3ef63fa` - fix(api): handle bytes data in dataset samples endpoint
- **Duration:** ~2 hours
- **Key Achievement:** Restored test suite health with 887 passing tests, fixed critical API bug affecting dataset sample viewing

### Session 7: 2026-01-02 to 2026-01-21 - Steering Migration & January Documentation Update
- **Accomplished:**
  - Migrated steering from synchronous API to async Celery tasks with GPU isolation
  - Added zombie process detection for steering workers
  - Fixed WebSocket timeout issues for long steering operations
  - Comprehensive January documentation update across PRDs, TDDs, TIDs
  - Added multi-extraction cached activations training support
  - Enhanced labeling with configurable batch size and NLP analysis per template
- **Files Modified:** Multiple across backend/src/workers/, frontend/src/components/steering/, 0xcc/ docs
- **Duration:** ~20 sessions

### Session 8: 2026-01-22 to 2026-01-26 - Neuronpedia Push & LFM2 Support
- **Accomplished:**
  - Implemented direct push to local Neuronpedia instance (async Celery, WebSocket progress)
  - Added LFM2 (Liquid Foundation Model) architecture support
  - Added layer discovery, extraction hooks for LFM2
  - GCP deployment configuration (Docker Compose, Neuronpedia domain, Ollama)
  - Combined multi-feature steering mode implemented
  - Upgraded transformers to 4.57.6
- **Files Created:** backend/src/services/neuronpedia_local_service.py, backend/src/workers/neuronpedia_push_tasks.py
- **Duration:** ~5 sessions

### Session 9: 2026-01-31 to 2026-02-07 - Dynamic Layer Discovery & Architecture Agnosticism
- **Accomplished:**
  - Replaced hardcoded architecture whitelists with dynamic discover_transformer_structure()
  - Frontend SUPPORTED_ARCHITECTURES whitelist removed
  - Steering service refactored to use dynamic discovery
  - Multi-select SAE downloads from HuggingFace
  - Test suite expanded to 961 tests
- **Key Decision:** Any transformer model can now be used without code changes
- **Duration:** ~4 sessions

### Session 10: 2026-02-13 to 2026-02-18 - JumpReLU L0 Fixes & SAE Framework Expansion
- **Accomplished:**
  - Fixed JumpReLU L0 loss: non-differentiable → sigmoid STE, fraction-based → count-based
  - Expanded SAE architectures from 4 to 6 paper-grounded frameworks
  - Added TopK (OpenAI) and Standard (Anthropic) architectures
  - Framework-aware configuration with paper-grounded defaults
  - Added activation normalization modes (constant_norm_rescale, anthropic_rescale, none)
- **Key Achievement:** SAE training now matches paper implementations exactly
- **Duration:** ~5 sessions

### Session 11: 2026-02-20 to 2026-03-08 - Labeling Enhancements & Settings Panel
- **Accomplished:**
  - Labeling: drag-to-resize results, maximize/restore, configurable max_tokens, Fetch Models button
  - Reasoning model support (think tag stripping for labeling)
  - DB-backed application settings with AES-256-GCM encryption
  - Settings Panel with tabbed interface (Endpoints, API Keys, Labeling, Display)
  - Sidebar navigation replacing horizontal tabs
  - HF upload path improvements (latent width, layer_XX structure)
- **Duration:** ~8 sessions

### Session 12: 2026-03-08 to 2026-03-21 - Bug Fixes, Monitoring & Model Loader
- **Accomplished:**
  - Fixed probe monitoring activations (tensor dimension, service initialization, WebSocket emission)
  - Fixed FastAPI validation error display in UI
  - Handle unclosed think tags from reasoning models
  - Compact extraction card tiles
  - Batch size wiring through labeling service
  - Integration with miLLM for labeling via OpenAI-compatible endpoint
- **Duration:** ~6 sessions

### Session 13: 2026-03-22 to 2026-04-26 — Enhanced Labeling, Security Hardening & Production Release
- **Accomplished:**
  - **Enhanced Per-Feature Labeling (major new feature):** Two-pass LLM labeling triggered from Feature Detail modal — Pass 1 parallel per-example summarization, Pass 2 synthesis. WebSocket progress, auto-populate edit form, live Zustand patch on completion.
  - **Star Color System:** yellow (starred), purple (in-flight), aqua (completed, permanent, protected from bulk overwrite)
  - **OpenAI API Integration for Labeling:** both enhanced and bulk labeling now target api.openai.com. API key stored AES-256-GCM encrypted in Settings → API Keys. Reasoning-class models (gpt-5, o1, o3, o4) auto-detected and use `max_completion_tokens`.
  - **OpenAI SDK Standardization:** EnhancedLabelingService refactored from hand-rolled httpx to official OpenAI Python SDK — eliminates per-model parameter whack-a-mole.
  - **Settings Panel:** Encrypted API keys (openai_api_key, hf_token), endpoint management, Labeling defaults, Fetch Models buttons. Critical encryption bug fixed: upsert endpoint no longer commits masked display string back over ciphertext.
  - **Context-Aware Labeling Template:** New system template using full context windows; instructs model to find shared semantic PATTERN across all examples rather than naming prime token. Seeded to production.
  - **Security Hardening:** Resolved all Dependabot CVEs and CodeQL findings — path injection (resolve_user_path), stack-trace exposure (6 endpoints), supply-chain attestations (SLSA mode=max), non-root frontend (nginx-unprivileged uid 101 port 8080).
  - **Feature Notes UX:** react-markdown + remark-gfm renders synthesis markdown tables and paragraphs. Max-height + scroll. Settings page scroll-to-top on mount.
  - **v0.5.0 Public Release:** Apache 2.0, GitHub Actions CI/CD, Docker Scout scanning, CodeQL via hitsainet Default Setup.
  - **K8s Production:** Kubernetes deployment (mcs-lnxgpu01), Cloudflare → mistudio.hitsai.net. K8s manifest restored and cleaned from placeholder-secrets incident.
  - **miLLM GraniteMoEHybrid Fix:** KV cache bug fixed for granite-4.0-micro hybrid models; monkey-patched `_update_mamba_mask` for attention-only configs.
  - **Documentation Update (this session):** PPRD v3.0, PADR v2.4 (IDL-18 through IDL-24), Feature Discovery FPRD v1.4 and FTASKS v1.3 with all post-March 2026 phases.
- **Duration:** ~30 sessions over 5 weeks

*[Add new sessions as they occur]*

## Research Integration

### MCP Research Support
When available, the framework supports research integration via:
```bash
# Use MCP ref server for contextual research
/mcp ref search "[context-specific query]"

# Research is integrated into all instruction documents as option B
# Example: "🔍 Research first: Use /mcp ref search 'MVP development timeline'"
```

### Research History Tracking
- Research queries and findings captured in session transcripts
- Key research decisions documented in session state
- Research context preserved across sessions for consistency

## Quick Reference

### 0xcc Folder Structure
```
project-root/
├── CLAUDE.md                       # This file (project memory)
├── 0xcc/                           # XCC Framework directory
│   ├── adrs/                       # Architecture Decision Records
│   ├── docs/                       # Additional documentation
│   ├── instruct/                   # Framework instruction files
│   ├── prds/                       # Product Requirements Documents
│   ├── tasks/                      # Task Lists
│   ├── tdds/                       # Technical Design Documents
│   ├── tids/                       # Technical Implementation Documents
│   ├── transcripts/                # Session transcripts
│   ├── checkpoints/                # Automated state backups
│   ├── scripts/                    # Optional automation scripts
│   ├── session_state.json          # Current session tracking
│   └── research_context.json       # Research history and context
├── src/                            # Your project code
├── tests/                          # Your project tests
└── README.md                       # Project README
```

### File Naming Convention
- **Project Level:** `000_PPRD|ProjectName.md`, `000_PADR|ProjectName.md`
- **Feature Level:** `001_FPRD|FeatureName.md`, `001_FTDD|FeatureName.md`, etc.
- **Sequential:** Use 001, 002, 003... for features in priority order
- **Framework Files:** All in `0xcc/` directory for clear organization
- **Project Files:** Standard locations (src/, tests/, package.json, etc.)

### Emergency Contacts & Resources
- **Framework Documentation:** 0xcc/instruct/000_README.md
- **Current Project PRD:** 0xcc/prds/000_PPRD|miStudio.md
- **PRIMARY UI REFERENCE:** 0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx
- **Tech Specification:** 0xcc/project-specs/core/miStudio_Specification.md
- **Tech Standards:** 0xcc/adrs/000_PADR|miStudio.md
- **Housekeeping Guide:** 0xcc/instruct/008_housekeeping.md

---

**Framework Version:** 1.1
**Last Updated:** 2026-04-26
**Project Started:** 2025-10-05
**Project:** MechInterp Studio (miStudio)
**Structure:** 0xcc framework with MCP research integration