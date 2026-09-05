yes
# Project: MechInterp Studio (miStudio)

## Current Status
- **Phase:** ✅ **FEATURE 30 — LABELING PROMPT-TEMPLATE OPTIMIZATION (2026-08-29)** — files `029_*`, PPRD row 30 (v3.13), PADR **IDL-48**. Improve labeling prompt templates by trial and error: run a template over a FIXED feature panel, write **no label**, and rank variants by an automated **detection score**. All 5 phases shipped + **3 review rounds**. Backend **3728 passed / 27 skipped / 0 failed**; frontend 1327 (untouched). **58 mutation controls, all verified biting.**
  - **Impl:** `models/labeling_trial_run.py`, `LabelingMode` + 3 columns on `labeling_jobs` (migration `d7e3a91c04b8`), `services/detection_metrics.py`, `services/labeling_detection_scorer.py`, `services/labeling_trial_service.py`, `workers/labeling_tasks.label_features_trial`, `workers/cleanup_stuck_labeling.py`, 4 REST operations, **5 MCP tools** in the `labeling` category.
  - **Four structural blockers found, all verified in code:** labeling could not be scoped; every path wrote to `features`; the **detection path had never executed** (3 independent breaks behind unit tests that called the formatter directly while the fixture exercising the real dispatch was defined and never used); and example order was shuffled with an **unseeded** global RNG, so the template was never the only variable.
  - **⚠ THE DURABLE LESSON OF THIS ARC — a fix was worse than the bug it replaced in EACH of the three rounds:** R1's `MIN_FEATURES_FOR_VERDICT` moved a false-positive from n=2 to n=8 while its companion change *removed the only visible warning* from that branch (`candidate_better / mde: None / reason: None`). R2's deterministic interleave stopped single-class batches and made the ground truth a **constant** — at the module defaults every batch was `[1,0,1,0,1,0,1,0,1,0]`, so a judge with an alternation bias scores 1.0 on every label under every template. R3's stratification then mis-aligned buckets against how the consumer slices. **Round 3 is not optional; the fixes are where the defects were.**
  - **Also durable:** a *correct* judge failed my own sanity gate (an all-positive first batch answered all-1 read as degenerate) — caught only because the gate had **never once passed in a test**; my "real database" tests were **silently skipping** on a hardcoded extraction id, so three controls reported as verified asserted nothing; and a mutation `count=1` matched the WRONG function, which revealed the three pre-existing MCP tools had no payload assertion at all.
  - **Nine mutation controls needed more than one attempt** (C1, C12, C19, C36 ×2, C41b, C46, C47, C14 ×2). A control that "survives" is as often a bad fixture as a real gap — verify the mutation LANDED and that the fixture sits in the region where the mutation can change the answer.
  - Reviews: `.claude/context/sessions/review_feature030_R{1,2,3}_2026-08-29.md` (24 + 11 + 20 findings).
  - **Manual is LIVE** at `https://onegaishimas.github.io/miStudio/core-workflow/labeling-trials` (content + all 3 images verified 200). ⚠ **The canonical `hitsainet.github.io` mirror is STALE**: `sync-to-clean` pushes with an action token, and GitHub does not fire further workflows on such a push, so the mirror's `deploy-manual.yml` never runs on a sync. It must be dispatched manually — **which needs admin rights on `hitsainet/miStudio`**. This affects every manual change, not just this one.
  - Manual: NEW `manual/docs/core-workflow/labeling-trials.md` (registered in `sidebars.ts`) with **3 Playwright captures**; extended `reference/api/features-labeling.md` and `reference/data-model.md`. **Playwright caught a doc error before publication** — the page said Settings, the panel is under Templates → Labeling Templates.
  - **Tracked debt:** the hard-negative donor pool is the 20 lexicographically-first ids (deterministic, but a biased sample — `md5(feature_id || salt)` would be both); `feature_token_index` has no unique constraint on `(extraction_id, feature_id, token_rank)`, so a failed grouping run's rows could duplicate donors; the template editor's hint names only `{tokens_table}` while the backend also accepts `{examples_block}`.
- **Phase (prior):** ✅ **E2E AUDIT REMEDIATION — WAVES 0–9 (2026-08-24).** The twelve-phase assessment (`0xcc/audits/E2E-2026-08/`, 166 findings) is remediated: **all 13 P0s closed and verified live**, Waves 1–7 complete, hardware acceptance done for the reachable items, and **all 13 surviving audit mutations re-run and killed** — five were still alive, including the hook-target regression that made steered output byte-identical to the baseline. Tracker: `0xcc/tasks/AUDIT_TASKS|E2E_Remediation_2026-08.md`, ~220 negative controls (C1–C224). **Live-verified on k8s:** backend 4/4, `/api/v1/settings` returns 8 of 9 rows with `settings_pin_hash` withheld (the row exists and is filtered, not deleted), destructive routes 404 on unknown ids, `/system/restart` 403, a 2 MB import 413, and a real steered generation. **The public mirror is verified clean: exactly 1 commit, none of the six excluded paths in its history.** **The GPU node's SSH password WAS rotated** — the disclosure in published history is mitigated. (This line read "still unrotated" until 2026-09-05 while §"NEVER disable password authentication" below said the opposite; the contradiction was repeated back to the user as an open risk. Rotation is the mitigation for disclosure; disabling password auth is not, and remains forbidden.) **Durable lesson: a fix is the most dangerous code in a review** — three of my own fixes broke production this arc (celery beat crashloop, every steering generation, and a guard that quoted the defect it removed twelve times).
- **Durable review lesson from this arc:** *a fix is the most dangerous code in a review.* Round 3 existed only to re-review rounds 1 and 2 and found **both** headline fixes unprotected — the unit-norm regression test could not fail, because the fixture's `W_U` was `torch.eye(...)`, already unit-norm, so both behaviours are identical on it. **A test written to pin a fix can inherit the very trap the fix addresses.** Also: a source-scrape guard fails OPEN (twice this arc, including inside the reachability guard itself); verify a mutation LANDED before concluding it survived; and never run a mutating agent concurrently with a reading one — a reviewer read an in-flight mutation out of the tree and reported it as a committed defect.
- **Phase (prior):** ⏳ **J-SPACE ARC (2026-07-31)** — BRD-MIS-JSPACE-001 v0.3, files `021_`–`028_` = PPRD rows 22–29, PADR IDL-40..46. A **second interpretability substrate** alongside SAEs: a training-free dictionary (the Jacobian lens) that reads what a model is *poised to say* at every layer and token position. **Shipped so far: files `022_*` (readout substrate) and `023_*` (the J-Lens viewer)** — 3 review rounds each, 26 findings, 39 mutation controls, verified on two real architectures (LFM2 hybrid + gemma-2-2b dense). The load-bearing product rule is **BR-002**: the published sensory/workspace/motor band boundaries were measured on one specific model, so miStudio draws **no bands at all** unless a band report exists for the model in front of you — there is no default band constant anywhere, by construction. **Next: files `021_*`** (Phase-0 fitting, validation suite, replication and gate) — which is what binds `/jlens/readout` (**since bound — the 501 is gone**) and lights up the Jacobian and Diff modes. Frontend suite **1007/1007, fully green**.
- **Phase (prior):** ✅ **FEATURE 21 — TRAINING FINALIZATION & CHECKPOINT LIFECYCLE (2026-07-26)** — files `020_*` / PPRD row 21 / PADR IDL-39. Stopping a training **forfeited its SAE**: cancel skipped the loop's finalize block, so `community_format/` — the only artifact downstream reads — was never written, and the UI offered only "Retry" (restart from step 0). Found on `train_969e90af` (granite-4.1-8b, JumpReLU L34/35/36) stopped at step 10,300 with **FVU 0.065 / zero dead neurons**: a good SAE, unusable. Separately checkpoints were never pruned (423 rows / 18 trainings / **78 GB**). Shipped: **Stop & Finalize** + **Finalize** (rebuild SAEs from a checkpoint, atomic staged export, reusing the success path's writer), honest `finalized_from_step` (status=COMPLETED unlocks import but progress stays truthful + amber "Finalized early @ N" badge), and **step-granular checkpoint retention** shipped **disabled + dry-run**. Also fixed a pre-existing bug: `DELETE /checkpoints/{id}` was called by the frontend and did not exist. **3 review rounds = 104 findings** (35 + 41 + 28), **24 mutation controls verified biting** — each previously left the suite green. Durable lessons: *Celery routes match the TASK NAME, so a short name silently uses the default queue*; *`hp['hidden_dim']` is corrected in memory and never persisted — read dims from checkpoint tensors*; *a boolean setting must fail to its DEFAULT, not to False (for `dry_run`, False means delete)*; *unlink before committing the row, or a failed unlink strands the file forever since planning is row-driven*. 141 backend + 105 frontend tests. PR #2, branch `feat/checkpoint-lifecycle`. **Hardware acceptance outstanding** — finalize `train_969e90af` on k8s after deploy.
- **Phase (prior):** ✅ **STEERED TRANSCRIPT RECORDER CLOSED (2026-07-22)** — BRD-MIS-RECORDER-001 / PADR IDL-38. Instrument-not-judge: records `(dial, prompt, unsteered, steered)` transcripts for a **circuit, cluster, OR ad-hoc feature set** over MCP (`record_steering_samples` / `get_steering_samples`), for a strong model (Opus) to read afterward — no weak in-loop judge. One unified GPU steering core (`steering_core.build_steer_generator`) + three per-type resolvers; calibration refactored onto it. New `steering_record_runs` marker table (migration `5cede2a1b3f7`) under the single-GPU guard; `steering_samples` manifest kind carries the generated TEXT; calibration manifests now carry a required `transcripts` field (back-compat: `validate_payload` runs only on write). **Full doc chain + 3 review rounds + hardware E2E across ALL THREE artifact types on k8s.** **The hardware round found FOUR bugs 4 static rounds + the unit suite all missed — every one a "fixtures agree by construction" trap:** (1) the unified core hooked the discovered `"residual"` module = a post-attn RMSNorm on LFM2, so the steering vector was **renormalized away — steered==unsteered at every dial**; fix = hook the WHOLE decoder-layer output `structure.layers_module[L]` (resid_post), the serve-matching point (91b5a6c). (2/3/4) the cluster path was broken for **all 35 existing profiles**: they store `sae_id` but `model_id=None`, and members carry no per-member `layer` — both must be derived from the `ExternalSAE` row (4b25b2c, 40ebc50); and the recorder had a **parallel `_artifact_model_id` copy** that a single-site fix missed → "Model None not found" (7b0fa56, + a consistency guard so the two resolution paths can't silently diverge again). E2E: circuit `vman_425ea6375805` (strong 0.4-humor/0.6-collapse dose-response), feature `vman_4f8f0d5c45df` (**byte-identical to the circuit path** — proves one-core+resolvers), cluster `vman_eb77e66b9405`/`vman_f9b13b2710e8` (real dose-dependent steer; subtle at 0.4 because "Verified & Trustworthy" is a weak 0.1–0.2 cluster, clear at 1.5–2.0). Calibration re-run **unregressed** (still `judge_unreliable` on the weak k8s judge). Every fix has a mutation-verified guard; CI green on all four (Backend Tests success). **Durable lessons** in memory `steering-hook-target-whole-layer.md` + `cluster-profile-persisted-shape.md`. Commits 91b5a6c→7b0fa56.
- **Phase (prior):** ✅ **FEATURE 20 CIRCUIT STRENGTH CALIBRATION CLOSED (2026-07-22)** — files `019_*` (the arc's +1 file/row offset; PPRD row 20, IDL-37). Automated usable-band search: ONSET (output-drift, no judge) + CORRECTNESS CLIFF (LLM judge on generated NEUTRAL-topic falsifiable probes), bisection, clamps the served dial to [onset,cliff]; badge-not-gate. Full stack: schema `calibration` block (migration `ca11b7a7e020`) + probe generator + search + service/task/manifest/reproduce + endpoint + MCP (`calibrate_circuit_strength`, `reproduce_calibration`) + Calibration UI. **RAN END-TO-END ON k8s** (crc_124fd83d1f2a, k8s miLLM judge) — pipeline works, judge-sanity gate + greedy generation + badge-not-gate all verified on hardware. **Review: 3 rounds (15+8+6) + a hardware round (3) = 32 findings, all fixed.** The R2 round (framed "runs on real hardware") caught a FATAL `get_hookable_module` arg-order crash 3 static rounds missed; the hardware run then caught a weak-judge/noisy-floor interaction (→ judge-sanity gate + greedy generation). **Durable lessons** in memory `circuit-strength-calibration-feature20.md`: GPU bugs only hardware finds; a weak judge fails the UNSTEERED baseline (report `judge_unreliable`, never a false `no_band`); greedy generation for a stable floor + reproducibility. **Tracked follow-up (not a blocker):** the 1.2B model this k8s serves is too weak to judge itself — validating the band near ground-truth needs a stronger judge. Commits 33e5b07→ac86c6e.
- **Phase (prior):** ✅ **miLLM CIRCUIT CONSOLIDATION CLOSED (2026-07-21)** — the cross-repo increment BRD-MILLM-CIRCUITS-002 (miLLM features 016–020) landed its MCP half HERE: the **16 `millm_circuit_*` tools** in `backend/src/mcp_server/tools/millm_circuits.py`, now REGISTERED (they were the increment's signature defect — fully implemented, unit-tested and documented while never registered with the server, so every test passed by importing the module directly). 293 findings across 5 features × 3 review rounds. **Durable deliverable: the reachability rule** — *a capability is not shipped until a test FAILS when its wiring is removed* — enforced by `backend/tests/unit/test_reachability.py` (registry / built-server / caller shapes, payload AND call-count asserted) and written into this file's Code Quality Checklist. Also hardened here: the causal-language copy audit (`SURFACES` was hand-maintained at 5 files while 16 circuit modules went unaudited — now discovered, 5→17), and `millm_client.py` failure paths (a 200 HTML page from a misrouted ingress used to reach the agent as an empty SUCCESS, so it would read 'nothing is steering' and activate into a contention; `test_millm_client_failure_paths.py` is new — the client had no test file at all). Review records live in the miLLM repo: `0xcc/reviews/review_feature020_R{1,2,3}_2026-07-21.md`.
- **Phase (prior):** ✅ **CIRCUITS ARC INCREMENT CLOSED (2026-07-20)** — features 015/016/017/018 all IMPLEMENTED + each ran a THREE-round review cycle (~250+ findings total). The arc is closed end-to-end (UI+API+MCP): **discover (016) → validate (017, rung-2 ES) → make portable (018 contract + evidence ladder) → steer (015) with the compounding/cancellation hazard QUANTIFIED from the causally-validated effect size** (heuristic weight-prior fallback always labeled `heuristic`, never causal). "Steer this circuit" button bridges a promoted circuit into steering. Per-feature review records `.claude/context/sessions/review_feature01{5,6,7,8}_R{1,2,3}_2026-07-*.md`. Also this increment: mcslab.io domain purged; frontend vitest baseline 98→0 (surfaced a real useTrainingWebSocket resubscribe bug). **Recorded tech debt (follow-on BRD):** cluster-granularity hazards/steering (feature-level v1), two-SAE GPU generation run + VRAM<200MB (only unproven FPRD §8 criteria — GPU close-out on k8s host). Next natural step: BRD-MILLM-CIRCUITS-001 (multi-SAE serving) or Tier-2.5 attention-mediated mining fast-follow.
- **Phase (prior):** Post-increment sensing enhancements (2026-07-17) — 4 goal items shipped; review rounds in progress (R1 done: 23 findings/17 fixed)
- **Last Session:** July 17, 2026 — sensing enhancements per /goal: span highlighting (context_parts {before,span,after} via PREFIX decodes — SP-safe; migration 010; <mark> in detail), history dedup (LCP boundary; ANY re-arm clears history; truncation caps at last-reported; never-shrink guard; SENSING_DEDUP_HISTORY), quorum default = ALL sensable members, min_k runtime override (PUT /api/sensing/{id}/config, sensing_overrides stripped on export, panel input + reset, sensable-ceiling validation, millm_sensing_config MCP tool). Review R1: 23 findings/17 fixed (`0xcc/reviews/review_sensing_enhancements_2026-07-17.md`). Suites: backend 1108 / admin-ui 209.
- **Current Task:** None — Feature 30 closed. Natural next step: run a real trial pair on the L46 panel against a judge strong enough to pass the sanity gate (the 1.2B model this cluster serves will not).
- **Current Task (prior):** None — **BRD-MIS-CLUSTERS-001 increment CLOSED (2026-07-16)**. Features 012/013/014 implemented, 3× review iterations each (28/28/15 findings), GitOps-deployed, Playwright E2E-verified (profile-titled Blended results + applied-count, budget bar/λ dial, low-cohesion gate, profile save/load/import/export). **013 validation gate PASSED after fitting γ=0** — the 1/G gain boost overdrove ~2× on all test clusters; B = B_dir/max(G,floor)^γ with default γ=0 (IDL-29 step-5 amendment; full data `0xcc/docs/Archive/cluster-strength-validation.md`). Review records: `.claude/context/sessions/review_feature01{2,3,4}_*_2026-07-16.md`.
- **Active Work:** Feature 30 (see Phase). Panel for experiments: 30 L46 features stratified by activation shape, scratchpad `L46_prompt_panel.md`. ⚠ L46's median **document**-frequency is 0.2023, so cross-feature negatives are ~20% contaminated — prefer low-frequency features when finalising the panel (L45 is ~4× cleaner at 0.056).
- **Active Work (prior):** None. Next natural step: follow-on BRD (MILLM import / unified MCP / Open WebUI / HF-marketplace publishing — research ready at `0xcc/docs/Archive/hf-marketplace-cluster-definitions-research.md`)
- **Circuits arc doc chain COMPLETE (2026-07-19):** BRD-MIS-CIRCUITS-001 + BRD-MIS-CIRCUITS-002 (rigor supplement: evidence ladder, statistics, attribution, intervention v2, faithfulness, Tier-2.5 readiness — consumed as ONE unit, 002 wins conflicts; Appendix A is normative math) → PPRD v3.9 (rows 16–19, §3.16–3.19) + PADR v3.0 (IDL-31..36) + four feature chains: 015 MultiSAE_Steering (hazards-v2), 016 Circuit_Discovery v2.0 (capture+stats+granularities+attribution), 017 Circuit_Validation (intervention/ES-vs-null/faithfulness/manifests), 018 Circuit_Portability (ladder/edge-typing/contract/projection — SEQUENCED FIRST: its ladder enum + contract gate the increment; then 016 → 017 → 015-hazards). SUBSTRATE pilot = research track only (BRD-MIS-SUBSTRATE-001.seed.md; no PPRD row). Next: execute 018 Phases 1–2 via 008_process-task-list.md.
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
- **Test Status:** backend **3728 passed / 0 failed / 27 skipped**, frontend **1327 passed / 0 failed** (both re-measured 2026-08-29, after Feature 30)
  (measured 2026-08-24, after the E2E remediation). *Do not hand-maintain this line* — it once read
  "995 passed" while the real figure was ~3.3x that, and a second line 38 rows above disagreed with
  it (MIS-E2E-163). Re-measure before quoting:
  `cd backend && pytest tests/ --no-cov -q` · `cd frontend && npx vitest run`
  A frontend **test-file** type-check also exists now (`npm run type-check:test`); it reports 432
  pre-existing errors, ratcheted down-only by `src/test/typeCheckRatchet.test.ts`. It is deliberately
  NOT in the CI gate — see MIS-E2E-021.
- **Services Status:** K8s (mistudio namespace) ✅, Docker Compose (192.168.244.222) ✅
  - Backend (port 8000) ✅, Frontend (nginx-unprivileged, port 8080→80) ✅
  - PostgreSQL ✅, Redis ✅, Celery Worker ✅, Celery Beat ✅, Nginx ✅
- **K8s Manifest:** `k8s/base/` (kustomize). The standalone `k8s/mistudio-deployment.yaml` was DELETED (MIS-E2E-144): it was a stale second copy that `k8s_deploy` re-applied, reverting the queue-split and SQL-echo fixes.
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
cp .env.example .env      # first run only
docker compose up -d
```

`docker-compose.yml` declares **10** services. Everything runs in containers.

**⚠ `./start-mistudio.sh` is NOT the general startup path (MIS-E2E-162).** It
hardcodes `PROJECT_ROOT=/home/x-sean/app/miStudio` under `set -e`, so it aborts
on any other clone; it starts only five containers from `docker-compose.dev.yml`
and runs the backend, Celery and the frontend **on the host** from a
`backend/venv/` it never creates; and it serves `dev-mistudio.hitsai.local`, not
the domain this file used to name beside it — so the `/etc/hosts` line below was
for a URL the script never served. It is a development convenience for one
machine. The four other repo shell scripts hardcode the same home directory.

**Before first run**, add the domain to /etc/hosts:
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
| Manifest | k8s/base/ (kustomize — what ArgoCD deploys) |

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

## Reaching the GPU node (agents: read this before any `ssh`)

**Authentication is key-based. There is no password anywhere in this repo, and there must never be
one again.**

```bash
source scripts/k8s-helpers.sh   # then: k8s_status, k8s_logs, k8s "any command"
```

`k8s()` is one line and deliberately plain:

```bash
k8s() { ssh -o BatchMode=yes "${K8S_USER}@${K8S_HOST}" "$1"; }
```

- `K8S_HOST` defaults to `192.168.244.61`, `K8S_USER` to `sean`; override by exporting them.
- `BatchMode=yes` makes it **fail instead of prompting**. That is intentional: a hang waiting on a
  password prompt inside an agent looks like a dead command, and a fallback to password auth is the
  thing that was removed.
- `StrictHostKeyChecking=no` is gone too. It accepted any host key, so the connection was
  unauthenticated in *both* directions.

**One-time setup, per machine:** `ssh-copy-id ${K8S_USER}@${K8S_HOST}`

### Certificate auth (live since 2026-08-24) — how to onboard an agent

The node trusts a **user CA**. Any key that CA signs is admitted as `sean`; no server change is
needed per agent, and every certificate carries a hard expiry. That expiry is the point: an
`authorized_keys` entry is permanent by default, a certificate is temporary by default.

Trust is established **per-user, not system-wide** — a `cert-authority` line in
`~sean/.ssh/authorized_keys`, not `TrustedUserCAKeys` in `sshd_config`. That needs no root, no
`sshd` reload, and therefore carries no risk of locking anyone out. It is scoped to this one
account, which is exactly the scope wanted.

```
cert-authority ssh-ed25519 AAAA…  miStudio user CA (workstation-held)
```

**The CA private key lives at `~/.ssh/mistudio_user_ca` on the operator workstation only.** Never in
this repo, never on a server — only the `.pub` is copied anywhere. Anyone holding it can mint access
to the node indefinitely.

#### Signing a certificate for a new agent

One command on the workstation that holds the CA. Nothing to do on the server.

```bash
ssh-keygen -s ~/.ssh/mistudio_user_ca \
  -I "some-agent@its-host" \      # key id — appears in the node's auth log
  -n sean \                       # principal: the username it may log in as
  -V -5m:+30d \                   # validity; -5m absorbs clock skew
  -z 3 \                          # serial — INCREMENT; revocation keys on it
  -O clear -O permit-pty \        # drop agent/port/X11 forwarding
  /path/to/agent_id_ed25519.pub
```

That writes `agent_id_ed25519-cert.pub` beside the public key. The agent copies its private key **and
the `-cert.pub`** into `~/.ssh/`; OpenSSH finds the cert automatically from the `-cert.pub` naming.
No `ssh-copy-id`, no touching the server.

`-O clear -O permit-pty` matters: the default extensions include agent and port forwarding, which an
automation certificate has no business carrying.

#### Verified, not assumed (2026-08-24)

A throwaway key **absent from `authorized_keys`** was refused bare, then admitted once signed —
nothing else changed between the two attempts. An **expired** cert and a cert for the **wrong
principal** were both refused. So the expiry and principal guarantees are enforced, not merely
configured.

#### Revoking one agent

Does not need the CA private key, and does not invalidate anyone else:

```bash
ssh-keygen -k -f ~/.ssh/mistudio_revoked -u -s ~/.ssh/mistudio_user_ca -z <serial> /dev/null
# copy to the node, then (needs root):  RevokedKeys /etc/ssh/mistudio_revoked
```

Without root, revoke by removing the `cert-authority` line and re-adding it after re-issuing the
certs you still want — blunt, but it needs no privilege.

### 🔒 NEVER disable password authentication (standing user directive, 2026-08-24)

`PasswordAuthentication` on `192.168.244.61` **stays `yes`. No agent may turn it off**, and no agent
may suggest it as a hardening step. This is not a default to be improved on — it is a decision.

Password auth is the **break-glass path**. Key and certificate auth both have failure modes that end
with nobody able to reach the GPU node: a certificate expires on its own schedule, a key can be lost
with its workstation, and `~/.ssh/authorized_keys` can be truncated by a bad script. Password auth
is what remains when those fail.

This is not hypothetical here. On 2026-08-24 the MIS-E2E-143 fix removed the committed password from
`scripts/k8s-helpers.sh` and installed no key, so the helper was dead the first time an incident
needed it — during a live outage. **Removing a credential path without a proven replacement is
itself an outage.** The password has since been rotated, which is the correct mitigation for
disclosure; disabling it is not.

If you want the node harder, do it somewhere that does not remove the last way in: fail2ban, a
firewall rule limiting port 22 to the LAN, or `AllowUsers`. Not `PasswordAuthentication no`.

### If `ssh` returns `Permission denied (publickey)`

Your key is not in the node's `~/.ssh/authorized_keys`. **Do not reach for `sshpass`, and do not
add a password to any file.** Run `ssh-copy-id` once (it will prompt interactively), or ask the
user to add your public key. This happened on 2026-08-24: the audit removed the committed password
without installing a key, so `k8s-helpers.sh` was dead the first time an incident needed it —
removing a credential and leaving no working path is its own outage.

### What NOT to do

| Don't | Why |
|---|---|
| `sshpass -p …` | Puts a credential on a command line, visible in `ps` to every process on the host |
| `K8S_PASS=…` in any tracked file | `test_no_dumps_or_secrets_tracked.py` fails the build, and the mirror publishes it permanently |
| `-o StrictHostKeyChecking=no` | Accepts any host key; the server is then unauthenticated to you |
| `pkill -f steering@` | Pattern-kills any matching process on the host. Use the tracked PIDs in `api/v1/endpoints/steering.py` |

Deploys are GitOps: push to `main` → mirror sync → image build → ArgoCD. `k8s_deploy` is
break-glass only. See `0xcc/plans/CICD-Runbook.md`.

## Quick Resume Commands

### Lean Session Start (Recommended)
```bash
# Minimal context loading - most efficient approach
"Please help me resume where I left off"
# This loads: CLAUDE.md. (`0xcc/session_state.json` does NOT exist — MIS-E2E-010.
# It was documented as auto-loaded and never created.)

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
> **These were all off by one (MIS-E2E-155).** `001_generate-brd.md` was added at
> the front of the sequence and this list was never renumbered, so every entry
> named a real file that performs a DIFFERENT action — `007` was described as
> "process task list" while `007_generate-tasks.md` generates them, and
> `008_housekeeping.md` does not exist at all. Following any of these by number
> ran the wrong step. Verified against `ls 0xcc/instruct/`.

- `0xcc/instruct/000_README.md` - How the document chain fits together
- `0xcc/instruct/001_generate-brd.md` - Frames an increment as business requirements
- `0xcc/instruct/002_create-project-prd.md` - Creates project vision and feature breakdown
- `0xcc/instruct/003_create-adr.md` - Establishes tech stack and standards
- `0xcc/instruct/004_create-feature-prd.md` - Details individual feature requirements
- `0xcc/instruct/005_create-tdd.md` - Creates technical architecture and design
- `0xcc/instruct/006_create-tid.md` - Provides implementation guidance and coding hints
- `0xcc/instruct/007_generate-tasks.md` - Generates actionable development tasks
- `0xcc/instruct/008_process-task-list.md` - Guides task execution and progress tracking

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
> to avoid renaming shipped files); do NOT "fix" a doc by renumbering it. The next feature is file `021_`
> = PPRD row 22.

### Circuits arc — BRDs + Features 015–018 (= PPRD rows 16–19) (Planned 2026-07-19)
- ✅ 0xcc/prds/BRD-MIS-CIRCUITS-001.md · ✅ 0xcc/prds/BRD-MIS-CIRCUITS-002.md (supplement; Appendix A normative) · 📌 0xcc/prds/BRD-MIS-SUBSTRATE-001.seed.md (seed only)
- ✅ 015_FPRD/FTDD/FTID/FTASKS|MultiSAE_Steering · ✅ 016_…|Circuit_Discovery (v2.0) · ✅ 017_…|Circuit_Validation · ✅ 018_…|Circuit_Portability
- ✅ PPRD v3.9 (§3.16–3.19) · ✅ PADR v3.0 (IDL-31..36)

### Feature 020 (= files 019_*) — Circuit Strength Calibration (✅ COMPLETE — doc chain + implementation, closed 2026-07-22)
- ✅ 019_FPRD/FTDD/FTID/FTASKS|Circuit_Calibration — **files are `019_*`, product feature is PPRD row 20** (the +1 offset above; not an error). The arc's next step. Grounded in the served-circuit finding: placeholder strengths shipped fluent-but-FALSE at "usable" 1.40; usable band was ~0.4–0.6 effective.
- ✅ PPRD v3.10 (row 20, §3.20) · ✅ PADR v3.1 (IDL-37: two-detector usable-band search — onset by output-drift, correctness cliff by LLM judge on generated NEUTRAL-topic falsifiable probes; adaptive bisection; additive nullable `calibration` block clamps intensity_range to [onset,cliff]; badge not gate; provisional cross-plane)
- Next: execute 019 Phase 1 (schema/contract) via 008_process-task-list.md. Contract crosses to miLLM (additive nullable) — schema-sync + vendored-identity guards are acceptance-blocking.

### Steered Transcript Recorder — BRD-MIS-RECORDER-001 (✅ CLOSED 2026-07-22)
- ✅ 0xcc/prds/BRD-MIS-RECORDER-001.md · ✅ PADR IDL-38 (v3.2 → corrected: whole-layer resid_post hook target, not "residual"). No FPRD/FTASKS — BRD-driven increment.
- Impl: `steering_core.py` (unified core + resolvers), `steering_recorder_service.py`, `steering_record_run.py` (migration `5cede2a1b3f7`), `circuit_record_tasks.py`, `POST /circuits/steering-samples`, MCP `record_steering_samples`/`get_steering_samples`, `steering_samples` manifest kind. 3 review rounds + hardware E2E (circuit+cluster+feature). Commits 91b5a6c→7b0fa56.

### Feature 21 (= files 020_*) — Training Finalization & Checkpoint Lifecycle (✅ Implemented 2026-07-26)
- ✅ 020_FPRD/FTDD/FTID/FTASKS|Checkpoint_Lifecycle — **files are `020_*`, product feature is PPRD row 21** (the +1 offset; not an error)
- ✅ PPRD v3.11 (row 21, §3.21) · ✅ PADR v3.3 (IDL-39: finalize-from-checkpoint; step-granular retention; unlink-before-commit; fail-closed completeness)
- ✅ Manual: NEW `manual/docs/core-workflow/training-lifecycle.md` (registered in `sidebars.ts`); corrected `sae-training.md` (its "Stop … saves final checkpoint" line was **factually wrong** and is what cost a real run); extended `reference/api/trainings.md`, `reference/data-model.md`, `reference/websocket-channels.md`, `advanced/external-saes.md`
- Impl: `training_finalize_service.py`, `checkpoint_retention.py`, `training_finalize_tasks.py`, `prune_checkpoints.py`, migration `b4d19f0c73ae`. Commits 8d44fa4→54197af.
- **Tracked debt:** retention is row-driven so orphaned checkpoint dirs are never reclaimed; finalize/prune create no `task_queue` row so they don't show in Active Operations; prune results aren't polled by the UI; destructive routes are unauthenticated and the Storage tab isn't PIN-gated.

### J-Space arc — BRD-MIS-JSPACE-001 + Features 022–029 (= files `021_`–`028_`) (2026-07-30/31)

A second interpretability substrate alongside SAEs: a **training-free dictionary** (the Jacobian
lens) reading what a model is *poised to say* at every layer and position. 32 business requirements.
**Files `021_`–`028_` = PPRD rows 22–29** (the +1 offset). **Numbering ≠ build order:** built
`022 → 023 → 021 → 026 → 024/025/027 → 028`.

- ✅ BRD v0.3 · PPRD v3.12 (rows 22–29, §3.22–§3.29) · PADR v3.4 (IDL-40..46)
- **BRD v0.3 amendments** driven by *"work with ANY model, LFM2 first"*: BR-031 **inverted**
  (construction is the primary path, acquisition opportunistic — LFM2 has no pre-fitted lens and
  neither does most of what this shop runs); BR-032 added (model-agnostic via
  `discover_transformer_structure`, decoder-layer capture, **per-layer** applicability — the
  reference model is hybrid, so frozen-Q/K is undefined on 10 of its 16 layers); BR-027 broadened to
  **full MCP parity**.

#### Feature 023 (= files `022_*`) — J-Space Readout Substrate (✅ Implemented 2026-07-30)
- ✅ 022_FPRD/FTDD/FTID/FTASKS|JSpace_Readout_Substrate
- Impl: `schemas/jlens.py`, `services/jlens_readout_service.py`, `api/v1/endpoints/jlens.py`,
  `tests/unit/test_jlens_{readout,model_agnostic,reachable}.py`
- 3 review rounds, **11 findings**, **24 mutation controls**. Verified on **two real architectures**
  in the backend pod — LFM2.5-1.2B (hybrid, 6/16 attention layers) and gemma-2-2b-it (dense, 26/26);
  `"The capital of France is"` → `' Paris'` at the final layer on both.
- Durable lessons: *normalise with the model's OWN final norm, never L2*; *the product of per-field
  bounds permitted 102M readouts / 8.4 GB residuals — bound the envelope, not just the fields*;
  *a `mode="after"` validator that pydantic's type coercion pre-empts can never fire*.

#### Feature 024 (= files `023_*`) — J-Lens Readout Viewer (✅ Implemented 2026-07-31)
- ✅ 023_FPRD/FTDD/FTID/FTASKS|JLens_Readout_Viewer · Manual: NEW
  `manual/docs/core-workflow/jlens.md` (registered in `sidebars.ts`)
- Ports `0xcc/brds/JSpacePanel.jsx` (the BR-010 interaction spec) onto a **live logit-lens readout**.
  Nav entry **"J-Lens" immediately before Steering**. Impl: `types/jlens.ts`, `api/jlens.ts`,
  `stores/jlensStore.ts`, `components/jlens/*`, `panels/JLensPanel.tsx`, `config/panels.ts`,
  `feature-jlens` chunk. Commits b201ce4→88b01cd.
- **The three constants in the mock are gone and none has a replacement:** `LAYERS` (21 at 0,5,…,100)
  → `meta.layers_by_type`; `BAND {40, 90}` → **nothing** (those are the source paper's Sonnet-4.5
  figures; BR-002 requires porting be impossible by construction, so bands render only from a band
  report); `TOP_N = 8` → `meta.top_n`. Fixture generator deleted, with a source-level guard.
- Also folded the **triplicated panel-id list** (two `ActivePanel` unions + a hardcoded
  `validPanels`) into one registry at `config/panels.ts` — the missed third copy produced a panel
  that worked when clicked and vanished on reload, with no type error anywhere.
- 3 review rounds, **15 findings**, **15 mutation controls**. **Two mutations initially survived**,
  both because fixtures agreed by construction (one axis shared across lens types; an evenly spaced
  axis where category and numeric positions coincide).
- **Pre-existing, now fixed:** 4 tests pinned to single-theme Tailwind classes that went dual-mode,
  and `DatasetsPanel.handleDownload` rejecting **unhandled** under a test named *"should handle
  download errors gracefully"*. Frontend suite **1007/1007, 0 unhandled errors** — fully green.
- ~~**Tracked debt:** `/jlens/readout` returns **501**~~ — **RESOLVED**; the endpoint is bound. Original note: until feature 022 (files `021_*`) binds model
  resolution; band rendering is unreachable from the panel until a band report exists; MCP parity is
  owned by files `028_*`.

#### Feature 022 (= files `021_*`) — J-Space Fitting, Validation & the Phase-0 Gate (✅ Shipped 2026-08-10 — see Current Status; the outstanding list below is historical)
- ✅ 021_FPRD/FTDD/FTID/FTASKS|JSpace_Fitting_And_Gate
- **Shipped:** `ml/jlens_fitter.py` (model-agnostic fit; freezing by patching the OPERATIONS, not the
  modules), `services/jlens_validation.py` (the six BR-030 classes, fail-closed), `ml/jlens_metrics.py`
  (seven band metrics + both null controls), `services/jlens_band_report.py` (boundaries + GO/NO-GO
  gate). 52 tests; backend suite 2138 green. Commits 9ab2330→56c3c20.
- **The fitter as first committed COULD NOT HAVE RUN** — one forward-mode pass per input dimension is
  millions of passes on the reference model. Freezing makes the map **affine**, so `J[:,i] = fn(e_i) −
  fn(0)` comes out of one batched call per chunk. The jvp version survives as `jacobian_by_jvp` and a
  test asserts the two agree, so the fast path's assumption is *verified*, not asserted.
- **`affine_residual` refuses a fit whose freeze leaked.** An incomplete freeze yields a matrix of the
  right shape and size that passes STRUCTURAL/NAMING/ENVELOPE and reads out plausible nonsense; fit
  time is the only point where it is detectable.
- Durable lessons: *`_norm_modules` must be `endswith("norm")`, not `contains` — a substring match
  captured a decoder block named `NormedBlock`, and freezing a decoder block replaces it with an
  elementwise rescaling*; *replay the layer's recorded kwargs — a decoder layer called with hidden
  states alone silently takes a default path and fits a lens for a model that was never run that way*;
  *a mutation that "survives" may simply never have applied — check the edit landed before concluding*.
- **Artifact lifecycle + MCP parity also shipped:** `services/jlens_artifact_service.py` (stage →
  validate → commit → serve), the `/jlens/artifacts` list and `/jlens/artifacts/{slug}/validate`
  endpoints, and the **`jlens` MCP category** (`list_jlens_artifacts`, `validate_jlens_artifact`)
  registered and covered by a reachability harness **written before the tools**. Commits
  7a2956a→HEAD.
- **The filesystem is the registry, not a DB table** (PADR IDL-46). A J-lens artifact is consumed by
  MOUNTING a conformant directory; there is no upload path, so a DB row as source of truth would
  invent a second registry that can silently disagree with the one the consumer reads. This removed
  Phase 1 (ORM/migration) from the critical path. Stage-then-commit: `<slug>.staging` is excluded
  from discovery, because a half-written artifact in the mounted directory gets served.
- Durable lessons: *the envelope check compares a FILE size to a TENSOR size — a flat overhead
  allowance then exceeded a small model's entire materialised dictionary, so it must be bounded at
  BOTH ends*; *artifacts load `weights_only=True` — an artifact is an untrusted file and the
  unrestricted loader executes pickled code*.
- **Outstanding:** Phase 4.2/4.3/4.5 (acquisition weight-identity check, Celery task, **readout
  binding — ~~`/jlens/readout` stays 501~~ **since resolved**), band-report/gate/fit endpoints + their MCP tools, Phase 7
  (UI), Phase 8.1/8.6 (two-architecture and **hardware acceptance on the local 3080 Ti**), review
  rounds 2–3. The cross-implementation and round-trip checks are written and tested but not yet
  wired to a live consumer; `ValidationReport` fails closed on a class that never ran, so nothing can
  publish on their absence.

### Feature 30 (= files `029_*`) — Labeling Prompt-Template Optimization (⏳ Phase 1 shipped 2026-08-29)
- ✅ PPRD v3.13 (row 30, §3.30) · ✅ PADR v3.6 (IDL-48: non-persisting trials; content-addressed panel identity; a PINNED scoring prompt — an editable ruler silently invalidates every prior score; a dedicated results table because the manifest path guard would discard a completed run on a corpus passage beginning with `/home/`; `judge_unreliable` over a false low score; negatives never claimed to be non-activating)
- ❌ 029_FPRD/FTDD/FTID/FTASKS|Labeling_Template_Optimization (BRD-driven increment; PPRD/PADR carry the contract) — **files are `029_*`, product feature is PPRD row 30** (the +1 offset; not an error)
- Impl so far: `models/labeling_trial_run.py`, `LabelingMode` + 3 columns on `models/labeling_job.py`, migration `d7e3a91c04b8`, `tests/unit/test_labeling_trial_foundations.py`
- ✅ Manual: `manual/docs/core-workflow/labeling-trials.md` + 3 Playwright captures, written only AFTER the REST/MCP surface existed — this repo's signature failure is documentation claiming ✅ over a capability no caller can reach.

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
# (0xcc/session_state.json — does not exist; see MIS-E2E-010)
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
   - Event type: `system:metrics` (emitted every 2 seconds by `services/background_monitor.py`, an asyncio task in the FastAPI process — **not** Celery Beat; MIS-E2E-156/-158)

**WebSocket Fallback Pattern:**
- Frontend hooks automatically detect WebSocket connection state
- Stores implement automatic fallback to HTTP polling when WebSocket disconnects
- Polling stops automatically when WebSocket reconnects
- Example: `systemMonitorStore.setIsWebSocketConnected()` manages fallback logic

**Backend Emission Pattern:**
- All WebSocket emissions use `backend/src/workers/websocket_emitter.py`
- Celery tasks emit updates via internal HTTP endpoint: `POST /api/internal/ws/emit`
- Emission functions: `emit_training_progress()`, `emit_gpu_metrics()`, etc.
- System-metric emission is an asyncio loop in the API process (`background_monitor.py`), started from the FastAPI lifespan. Celery Beat runs the janitors, the pruner, the GPU watchdog and the steering reconciler — not this.

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
- **Next:** Create Architecture Decision Record using 0xcc/instruct/003_create-adr.md
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
    - Created periodic system metrics collection (every 2 seconds) — originally a Celery Beat task; **replaced by an asyncio task in the API process on 2026-07-10** and `workers/system_monitor_tasks.py` deleted
    - Defined WebSocket channel naming conventions for system monitoring:
      - `system/gpu/{gpu_id}` - Per-GPU metrics
      - `system/cpu` - CPU utilization
      - `system/memory` - RAM and Swap
      - `system/disk` - Disk I/O rates
      - `system/network` - Network I/O rates
    - Created `useSystemMonitorWebSocket.ts` React hook for channel subscriptions
    - Updated `systemMonitorStore.ts` with WebSocket integration and automatic polling fallback
    - Updated `SystemMonitor.tsx` component to use WebSocket-first with polling fallback
    - Configured the scheduler (the system-monitoring entry has since been removed — see above)
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
  - ~~`backend/src/workers/system_monitor_tasks.py`~~ — **deleted 2026-07-10**; superseded by `services/background_monitor.py`
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
│   ├── (session_state.json)        # DOCUMENTED, NEVER CREATED — MIS-E2E-010
│   └── (research_context.json)     # DOCUMENTED, NEVER CREATED — MIS-E2E-010
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
- **Housekeeping Guide:** (removed — there is no 008_housekeeping.md; see 000_README.md)

---

**Framework Version:** 1.1
**Last Updated:** 2026-04-26
**Project Started:** 2025-10-05
**Project:** MechInterp Studio (miStudio)
**Structure:** 0xcc framework with MCP research integration