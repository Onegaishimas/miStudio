# Cross-Feature Review Synthesis — miStudio (Features 001–009)

**Compiled:** 2026-07-11
**Inputs:** [001](001_REVIEW|Dataset_Management.md) · [002](002_REVIEW|Model_Management.md) · [003](003_REVIEW|SAE_Training.md) · [004](004_REVIEW|Feature_Discovery.md) · [005](005_REVIEW|SAE_Management.md) · [006](006_REVIEW|Model_Steering.md) · [007](007_REVIEW|Neuronpedia_Export.md) · [008](008_REVIEW|System_Monitoring.md) · [009](009_REVIEW|Multi_GPU_Scalability.md) + [this session's Celery/Monitor review](../../.claude/context/sessions/review_celery_monitor_operations_2026-07-10.md)
**Purpose:** the holistic layer the per-feature reports can't see — recurring patterns worth fixing **once across the app**, genuine cross-feature interdependencies, and a whole-app documentation-refresh plan.

---

## 0. Executive Summary

Nine features, ~40k lines of backend service code reviewed end-to-end. **The research-critical core is excellent** — SAE architectures (003), logit lens (007), dynamic layer discovery (002/006), and the enhanced-labeling subsystem (004) are all paper-faithful and well-built. **The defects cluster into a small number of repeating patterns**, which is good news: fixing each pattern once closes findings across many features.

The single highest-value structural fix is untangling the **"extraction" architecture** (spans 002/004/005) — a naming collision that has *already caused a P0*. The single highest-value documentation fix is a **systematic FPRD reference-section refresh** — every feature's data-model/endpoint/channel tables have drifted, in a consistent and mechanically-fixable way.

---

## 1. Recurring Patterns → Batch Fixes

These are the "fix once, resolve many" opportunities. Ordered by leverage.

### BATCH-1 · The "extraction" architecture tangle *(002-F1 P0, 004-F5, 005-F3, 002-F6)*
**The single biggest structural issue in the app.** There are **two distinct extraction pipelines** with colliding names:
- **Stage A — model → raw activations:** `ActivationExtraction` model, `model_tasks.extract_activations`, `ExtractionStatus` enum, routed under `POST /models/{id}/extract-activations` (Feature 002).
- **Stage B — SAE → interpretable features:** `ExtractionJob` model, `extraction_tasks.extract_features_from_sae`, a *second* `ExtractionStatus` enum, routed under **three** entry points: `/saes/{id}/extract-features`, `/saes/batch-extract-features` (005), and the 004 service layer.

Consequences already realized: **002's `cancel_extraction`/`retry_extraction` import the wrong `extraction` module and are 100% broken (P0)** — the collision directly caused it. FR-verification requires cross-feature grepping.

**Batch fix:**
1. Rename one family — recommend `ActivationExtraction` → `ActivationCapture` (+ `ActivationCaptureStatus`), leaving `ExtractionJob` = feature extraction. Or the reverse; the point is to end the collision.
2. Fix 002-F1's three imports (`....models.extraction` → `....models.activation_extraction`) — **do this immediately regardless**, it's a live P0.
3. Document the two-stage pipeline (model → activations → SAE → features) once, in both FPRD 002 and 004.
4. Consolidate the 3 Stage-B entry points or document why they differ.

### BATCH-2 · "Cancel/delete doesn't revoke the running Celery task" *(001-F4, 002-F3, 005-F2)*
The **same bug in three features**: the cancel/delete endpoint flips DB status and deletes files but does **not** revoke the in-flight Celery task, racing cleanup against still-running work.
- **001** (dataset download-cancel): `task_id` *is* in metadata but unused.
- **002** (model download-cancel): task_id not stored at all.
- **005** (SAE delete during extraction): no chain to the (working) cancel-extraction path.

**Contrast:** the paths that *do* revoke correctly — tokenization-cancel (001), extraction-cancel (002/005), training-stop (003), steering-cancel (006) — show the right pattern exists. **Batch fix:** a shared `revoke_and_cleanup(task_id)` helper; store the Celery task id at dispatch everywhere (most already do); call revoke before file deletion in every cancel/delete path.

### BATCH-3 · FPRD reference-section drift *(every feature)*
**All nine** FPRDs have stale §Data-Model / §API-Endpoints / §WebSocket-Channels sections, drifting in *consistent, mechanical* ways:
- **PK types:** every table documented as `UUID` is actually a prefixed `String` (`m_`, `train_`, `feat_`, `sae_`, `tok_`, `elj_`, `push_`) — except `prompt_templates`, `neuronpedia_export_jobs` (real UUIDs).
- **Status enums:** documented values rarely match (`stopped`→`cancelled`, `pending/failed`→`downloading/error`, `ingesting` doesn't exist, etc.).
- **Endpoints:** `/{id}/download` is really `/download`; several documented routes don't exist (`/models/preview`, `/trainings/{id}/retry`, `/steering/generate`, `/steering/calibrate`, `/features/extraction`, `/saes/{id}/convert`); large real route surfaces are undocumented.
- **WS channels:** documented `{entity}/{id}` + un-namespaced events; real is `{entity}s/{id}/progress` + namespaced `entity:event`.

**Batch fix:** see §3 (Doc-Refresh Plan) — this is a single systematic pass, not nine independent rewrites, because the transformations are uniform.

### BATCH-4 · Zero/thin test coverage on whole subsystems *(006, 007, 008, 005, 009)*
- **006 Steering:** zero tests (2639-line service, a known concurrency bug).
- **007 Neuronpedia:** zero tests (research-critical logit lens unverified).
- **008 Monitoring:** zero tests — *directly why this session's P0 403-bug shipped and hid for ~2 months.*
- **005 SAE:** 2 tests for ~2800 lines (delete semantics, HF/Gemma parsing untested).
- **009 Multi-GPU:** no gpu_id-validation test.

**Contrast:** 001/003/004 are well-tested (003 especially). **Batch fix priority (by risk):** (1) a `BackgroundMonitor` emit test — would have caught this session's P0 and prevents regression; (2) logit-lens correctness (research validity); (3) steeringStore resolver-lifecycle (catches 006-F1). Establish the pattern that **every WebSocket emit path and every cancel path gets one integration test** — those are exactly the untested seams where this session's bugs lived.

### BATCH-5 · Debug `print()` / secret-adjacent logging *(001, 002, 003, 004, 005)*
`print("[X DEBUG]…")` and token-length logging in request/hot paths across five features. **Batch fix:** sweep to `logger.debug`; remove token-length logs (001-F5). Trivial, mechanical, do in one PR.

### BATCH-6 · Model/SAE delete doesn't guard referencing rows *(002/003-F6, 005-F1)*
Deleting a model that a training references → raw FK `IntegrityError`→500 (should be 409). SAE soft-delete orphans features. **Batch fix:** a pre-delete "in-use" check pattern (training→model, feature→SAE, training→extraction) returning 409 with the referencing ids, mirroring the *working* pattern in `list_model_extractions`' `can_delete` logic (002) and training's `used_by_trainings` (also 002/003).

### BATCH-7 · WebSocket channel-name mismatches = silent progress failure *(008 P0 this session, 007-F2)*
This session's P0 was a channel/token mismatch silently freezing the Monitor page. 007 has a doc-vs-code channel-name mismatch (`neuronpedia-push` vs `neuronpedia/push`). **This class of bug is invisible until a user notices "progress stopped."** **Batch fix:** a single source-of-truth channel-name constants module shared by emitter + frontend, plus one test per channel asserting emit→receive round-trips. Ties to BATCH-4.

### BATCH-8 · REST verb inconsistency for cancel/delete *(001-F9, 007-F4, general)*
Cancel is variously `POST /{id}/cancel`, `DELETE /{id}/cancel`, `POST /{id}/control`. Delete is `DELETE /{id}` vs `POST /delete` (batch). Low severity but pervasive. **Batch fix:** adopt a convention (e.g., `POST /{id}/cancel` for actions, `DELETE /{id}` for removal, `POST /{entity}/batch-delete` for batch) and align opportunistically during the doc refresh.

---

## 2. Genuine Interdependencies (not just shared patterns)

| Interdependency | Features | Note |
|---|---|---|
| **Two-stage extraction pipeline** | 002 → 004/005 | model→activations (002) feeds SAE→features (004/005). The naming collision (BATCH-1) obscures a *real* data-flow dependency that should be a documented architecture, not an accident. |
| **`layer_discovery.py` is shared infra** | 002, 003, 006 | Model arch introspection, SAE training hook placement, and steering injection all depend on it. It's the app's best-abstracted component — the one thing correctly *not* duplicated. Protect it (it has implicit test coverage via features but no direct test). |
| **GPU as a contended resource** | 003, 006, 008, 009 | Training (default GPU), steering (hard-pinned GPU 0), extraction (gpu_id-routable), monitoring (observes all). No scheduler arbitrates — steering on GPU 0 can collide with training on GPU 0. 009's `gpu_metrics`+`job_id` table (never built) was the intended correlation layer. |
| **task_queue federation** | this session → all | The Active/Failed Operations federation I built reads `trainings`, `extraction_jobs`, `labeling_jobs`, `neuronpedia_pushes`. It's coupled to each table's schema — esp. the model-less raw-SQL `neuronpedia_pushes` (007-F1), which will break silently if that INSERT's columns change. |
| **Settings → labeling/enhanced-labeling** | 004, Settings | Enhanced + bulk labeling both resolve endpoint/model/key from the encrypted AppSetting store. A settings schema change ripples into both labeling paths. |
| **Star-color lifecycle** | 004 internal | enhanced-labeling job (purple→aqua) ↔ bulk-labeling aqua-guard ↔ feature row ↔ modal. Correctly implemented; the tightest-coupled *working* subsystem. |

---

## 3. Documentation-Refresh Plan (the goal's priority)

The doc gaps are **uniform and mechanically fixable**. Do this as one systematic pass, not nine rewrites.

### 3.1 Priority order (by staleness × importance)
1. **FPRD 008 (System Monitoring) — CRITICAL.** Describes a *deleted* Celery-Beat architecture (`system_monitor_tasks.py` removed this session). The entire §3.1 diagram, §7 key files, and §6 events are wrong. Biggest single rewrite. **Rewrite to the `BackgroundMonitor` asyncio reality.**
2. **FPRD 004 (Feature Discovery) — HIGH.** §5 endpoints almost entirely fictional; §6.1/§6.2 data model wrong. (Keep §2.11–2.13 + §7 — accurate.)
3. **FPRD 001 (Dataset) — HIGH.** Predates the per-model multi-tokenization redesign; nearly every reference section wrong.
4. **FPRD 009 (Multi-GPU) — HIGH (accuracy, not volume).** Fix the FR-1.3 training-GPU overclaim; mark §6/§8 as Planned not current.
5. **FPRD 002, 003, 005, 006, 007 — MEDIUM.** Data-model/endpoint/channel corrections; keep the accurate parts (003 §3 frameworks, 007 export half + logit-lens, 006 Architecture Notes).

### 3.2 Mechanical transformations to apply app-wide
Apply these uniformly to every FPRD §Data-Model / §API / §WebSocket section:
- **PK type:** `UUID` → `String(255)` with the real prefix (`m_`/`train_`/`feat_`/`sae_`/`tok_`/`elj_`/`push_`). List real prefixes per table.
- **Status enums:** copy the actual enum members from the model file verbatim.
- **Endpoints:** regenerate the endpoint table from the router (`grep '@router'`) rather than hand-maintaining. Delete fictional routes; add the real surface.
- **WS channels:** `{entity}/{id}` → `{entity}s/{id}/progress`; events → namespaced `entity:event`. Reference a single channel-constants source (BATCH-7).
- **Key files:** verify each path exists (`test -f`); several point at deleted/renamed/never-existed files (`jumprelu_sae.py`, `StartTrainingModal.tsx`, `export_tasks.py`, `ml/sae_converter.py`, `system_monitor_tasks.py`, `TokenizationStatsModal.tsx`, `useDatasetWebSocket.ts`).

### 3.3 Structural doc additions
- **New: "Extraction Architecture" section** (in 002 + 004, or a shared ADR) documenting the two-stage pipeline + the model/task/route map (resolves the BATCH-1 confusion at the doc layer).
- **Promote 007's push half** from §1.4 narrative into §5/§6 tables.
- **Promote 006's Architecture Notes** (Celery async) into the main body; retire the obsolete §5/§6.
- **Note the cross-feature routing couplings** (extraction under /models + /saes; multi-GPU under /system) so they're intentional, not surprising.

### 3.4 A note on process
The drift is systemic because the FPRDs are hand-maintained prose that the code outgrew. Consider a lightweight **doc-generation step** for the mechanical sections (endpoint tables from routers, data-model tables from models) so §5/§6/§7 are generated, not hand-written — the prose sections (Overview, FRs, Architecture) stay human. This prevents re-drift after this refresh.

---

## 4. Consolidated Backlog (severity-ranked, cross-feature)

### P0 — do immediately
- **002-F1** — `cancel_extraction`/`retry_extraction` broken imports (`....models.extraction` → `....models.activation_extraction`). Two dead endpoints. One-line ×3. *(This is the only open P0; this session's monitoring P0 is already fixed+deployed.)*

### P1
- **006-F1** — steering resolver-singleton orphan-hang (concurrent steering hangs the first op).
- **BATCH-2** — cancel/delete-doesn't-revoke (001-F4, 002-F3, 005-F2) — shared fix.
- **004-F1** — enhanced-labeling FAILED-then-autoretry (user sees non-final failures).

### P2 — correctness / robustness
- **005-F1** — SAE soft-delete orphans features + deletes files (hybrid delete).
- **003-F1** — `create_training` has no validation (contradicts FR-6.5 "Implemented").
- **004-F2/F3** — atomic duplicate-job guard + star_color enum validation.
- **007-F1** — `neuronpedia_pushes` needs an ORM model (protects the task_queue federation).
- **007-F2 / BATCH-7** — WS channel-name mismatches (silent-failure class).
- **001-F1/F2/F3** — tokenization-cancel `NameError`, status-enum regex, lock-release-on-failure.
- **002-F5** — `redownload_model` path resolution (stale-file bug on native deploys).
- **003-F2** — `TrainingMetric` missing unique constraint (metric duplication).
- **BATCH-6** — model/SAE delete guards.
- **006-F2/F3** — steering worker config comments + redelivery risk.
- **009-F1** — FR-1.3 training-GPU overclaim (doc, but material).

### P3 — quality / tech-debt (mostly batchable)
- **BATCH-3** doc refresh · **BATCH-4** test coverage · **BATCH-5** debug logging · **BATCH-8** REST verbs.
- **003-F3** — JumpReLU docstring (fraction→count) — misleads sparsity tuning.
- **003-F4** — triplicated `normalize()` in SAE classes.
- God-files: `features.py` (1085/21-route), `steering_service.py` (2639), `community_format.py` (1233).
- **009-F3** — GPU-metrics persistence + job correlation (real anticipated capability, never built).

---

## 5. What's Genuinely Good (worth protecting)

Not everything is a finding. These are load-bearing and correct — don't let refactors regress them:
- **SAE architectures (003)** — all 6 paper-faithful; JumpReLU STE + count-based L0 are subtle and right.
- **`layer_discovery.py` (002)** — the best abstraction in the app; architecture-agnostic, clean fallbacks.
- **Enhanced-labeling subsystem (004)** — two-pass service, star lifecycle, aqua-guard-in-all-three-loops all match docs exactly.
- **Logit lens (007)** — correct `W_dec @ W_U.T` with proper transpose handling.
- **`forward_hooks.py` (006)** — GPU-memory-explicit, fail-loud, context-managed.
- **Test suites for 003 & 004** — the model to extend to 005/006/007/008.
- **This session's Celery/Monitor fixes (008)** — the monitoring code is now in its best state; only the doc lags.

---

## 6. Suggested Execution Sequence

1. **P0 now:** 002-F1 (3 imports). ~10 min.
2. **Batch quick-wins:** BATCH-5 (logging sweep), 003-F3/F7 (docstrings), 001-F1 (`NameError`), 003-F7 f-string. One small PR.
3. **BATCH-2** (cancel-revoke shared helper) + **BATCH-6** (delete guards) — one PR, resolves ~6 findings.
4. **BATCH-1** (extraction rename + pipeline doc) — one focused PR; highest structural value.
5. **P1 remainder:** 006-F1 (resolver), 004-F1 (autoretry).
6. **Doc refresh (§3):** 008 first, then 004/001/009, then the rest — ideally with the generated-section tooling so it doesn't re-drift.
7. **BATCH-4 test seams:** BackgroundMonitor emit, logit-lens correctness, steering resolver — the three that map to real shipped/latent bugs.

---

*This synthesis + the nine per-feature reports live in `0xcc/reviews/`. All are inside `0xcc/` and therefore excluded from the public hitsainet mirror.*
