# miStudio End-to-End Assessment — Plan

## Context

miStudio has grown through ~29 PPRD feature rows, 28 document chains, and roughly
a year of increments (clusters → circuits → calibration → recorder → checkpoint
lifecycle → J-Space). Reviews have been **per-increment**: each arc got its own
three rounds, but nobody has ever walked the whole application at once. The gaps
that produces are already visible from a two-hour scan:

- **No application-layer authentication exists on any of the 270 REST routes** —
  including 26 `DELETE`s, `POST /system/restart`, and two routes that spawn and
  `pkill -9` OS processes. This turns out to be a deliberate posture (nginx plus
  network isolation is the control) — but **no PADR IDL says so**, and nobody has
  checked that the control actually holds.
- The **2026-07-11 review batch covered features 001–009 only.** Features 010,
  011, 019, 020, 024–028 have **never** been reviewed at all.
- The **PPRD status column contradicts the task lists** on at least six rows
  (rows 22/23/24 say "Planned" while FTASKS 021 is 40/41 done and 023 is 35/35).
- **348 task boxes are unchecked**, and 193 of them sit in six ad-hoc task files
  that have no `## Relevant Files` section — so that work is not traceable to
  code from the documents at all.
- `aaaa/` is a **tracked duplicate of the 0xcc doc chain** that the
  `sync-to-clean` workflow does *not* strip, while it does strip `0xcc/`.
- The four `/review` personas in `.claude/agents/*.md` describe **a different
  project** (Prisma schema, RTK Query, SAML auth, dated 2025-01-15).

This assessment walks the application end to end — frontend, backend, MCP server,
infrastructure, and the document chain — in **12 subsystem phases × 3 rounds ×
3 review commands**, records every defect and improvement into a single
append-only register, and emits one master remediation tasklist at the end.

**Decisions locked with the user:**

| Decision | Choice |
|---|---|
| Phasing | 12 subsystem phases (not per-feature) |
| Recording | New `0xcc/audits/E2E-2026-08/` |
| Verification depth | Static + doc conformance **+** run the suites **+** mutation controls **+** live k8s/GPU/Playwright |
| Fix policy | **Strict record-only.** Zero code changes during the assessment — even a live security hole is recorded, not fixed. |
| Auth posture | **Accepted architectural posture** (nginx + network isolation is the control). Recorded once, not per route; P05 spends its effort verifying the control actually holds. |
| Repo boundary | **miStudio only.** miLLM gets its own end-to-end assessment later. The miLLM contract is verified from this side — schema-sync guards, `docs/schemas/*.json`, the generated `mcp-contract.md`, `millm_client` failure handling. Suspected cross-repo drift is recorded as a finding to confirm during miLLM's own assessment, not chased now. |

### The surface being audited

| | Measured |
|---|---|
| Backend | 266 `.py` files, **99,521 lines** — `services/` 42,682 (43%), `api/v1/endpoints/` 16,071, `workers/` 15,087, `schemas/` 6,340, `ml/` 5,142, `models/` 3,083 |
| REST | **270 routes** across 25 modules (GET 115 · POST 117 · DELETE 26 · PATCH 10 · PUT 2), plus 4 app-level routes and a Socket.IO mount at `/ws` |
| MCP | **117 tools across 14 categories** in 16 modules (83 across 9 enabled by default; `admin` + 4 `millm_*` opt-in) |
| Migrations | 98 revisions, **single head** `c7e2a4f18b93`, one merge point |
| Backend tests | 164 files, **2,673 `def test_`** |
| Frontend | 295 `.ts(x)` files, **94,869 lines**; 20 stores, 13 registered panels, 26 API modules, 21 hooks |
| Frontend tests | 65 files, **1,211 cases**, 0 skipped |
| Manual | 47 Docusaurus pages |
| Docs | 197 files in `0xcc/` — 28 complete FPRD/FTDD/FTID/FTASKS chains, 46 PADR IDLs, 29 PPRD rows |

Note the counts differ from `CLAUDE.md`'s recorded 2461/1149 — parametrization and
collection make `def test_` ≠ collected tests. **`BASELINE.md` measures; it does not
quote `CLAUDE.md`.**

---

## Where the plan and the issues are recorded

Everything the assessment produces lands under one new directory, plus one
tasklist in the existing framework location.

```
0xcc/audits/E2E-2026-08/
  README.md                    Status board: phase × round grid, finding counts, how to read this
  PLAN.md                      This plan, committed into the repo as the audit's charter
  BASELINE.md                  Pre-audit ground truth (suites, lint, type-check, CI, alembic heads, live env)
  FINDINGS.md                  ── THE REGISTER ── append-only, one entry per finding, stable IDs MIS-E2E-###
  TRACEABILITY.md              Doc chain → code → test matrix (BRD/PPRD/PADR/FPRD/FTDD/FTID/FTASKS ↔ files ↔ tests)
  rounds/
    P01-R1-security-review.md  108 round records: 12 phases × 3 rounds × 3 commands
    P01-R1-code-review.md
    P01-R1-review.md
    P01-R2-...                 (etc.)
  mutations/
    P01-mutations.md           Per-phase mutation control log: line broken, suite result, restore verified

0xcc/tasks/
  AUDIT_TASKS|E2E_Remediation_2026-08.md   ── THE MASTER TASKLIST ── generated from FINDINGS.md at the end
```

**Naming note:** the tasklist uses the repo's existing **ad-hoc** task prefix
(`SUPP_TASKS|`, `BUGFIX_TASKS|`, `UI_TASKS|`) rather than claiming feature number
`029_`. Remediation is not a feature, and taking `029_` would corrupt the
documented file-number ↔ PPRD-row offset convention. Unlike the six existing
ad-hoc files, this one **will** carry a `## Relevant Files` section.

### The register entry schema (`FINDINGS.md`)

Every finding — bug, security issue, test gap, doc drift, or enhancement — is one
block. The schema is fixed so the tasklist can be generated mechanically from it.

```markdown
### MIS-E2E-042 — Circuit export drops the calibration block on re-export
- **Phase / Round:** P05 / R2
- **Source:** /code-review high
- **Severity:** P1
- **Type:** bug            (bug | security | test-gap | doc-drift | reachability | perf | debt | enhancement)
- **Location:** backend/src/api/v1/endpoints/circuits.py:214
- **Claim:** one sentence, the defect itself
- **Failure scenario:** concrete inputs/state → wrong output or crash
- **Evidence:** verified-by-mutation | verified-by-live-repro | verified-by-test | plausible (read-only)
- **Doc reference:** PADR IDL-33 / 017_FPRD §4.2   (or "none — undocumented behaviour")
- **Verification (R3):** CONFIRMED | PLAUSIBLE | REFUTED
- **Proposed remediation:** what to do, not done here
- **Effort:** S | M | L
```

### Severity rubric

| | Meaning |
|---|---|
| **P0** | Exploitable security hole, data loss, secret exposure, or a capability the product claims to ship that is unreachable in production |
| **P1** | Wrong results presented as correct (this repo's recurring "honesty" class), crash on a normal path, or a silent-failure path that goes quietly dark |
| **P2** | Broken edge case, a test gap on a load-bearing line, doc contradicts code |
| **P3** | Cleanup, perf, stale artifact, enhancement |

---

## Phase 0 — Baseline (before any review round)

Establishes the ground truth every later phase compares against. Written to
`BASELINE.md`. Nothing is fixed here either.

1. **Backend suite.** `cd backend && DATABASE_URL=postgresql+asyncpg://postgres:devpassword@localhost:5432/mistudio DATABASE_URL_SYNC=postgresql://postgres:devpassword@localhost:5432/mistudio ./venv/bin/python -m pytest tests/ --no-cov -q` — record pass/fail/skip counts and every failure verbatim. (`mistudio-postgres` is up on :5432, verified.)
2. **Frontend suite.** `cd frontend && npx vitest run` (note: `npm test` is watch mode), `npm run type-check`, `npm run lint`, `npm run build`.
3. **Manual site.** `cd manual && npm run typecheck && npm run build`.
4. **CI truth, not local truth.** `gh run list --limit 20` — per this repo's own recorded lesson, a green local suite has masked four red CI builds before.
5. **Schema state.** `alembic heads` (flag multiple heads), `alembic current` against the live DB.
6. **Live environment.** `k8s-mistudio.hitsai.local` responds 200 on `/` and `/api/v1/models` (verified). Capture pod status, image digests, `k8s_gpu`, and the MCP server's live tool count via `mistudio_howto`.
7. **Coverage snapshot.** Backend `pytest --cov` summary; note that the frontend has `@vitest/coverage-v8` installed but **no coverage block or thresholds** configured.

**Pre-recorded blocker for `/review` — WITHDRAWN at P01 R1.** The plan assumed
`/review` loads `.claude/agents/{architect,product_engineer,qa_engineer,test_engineer}.md`,
which do contain another project's content. It does not: `.claude/commands/review.md`
Step 1 loads **`.claude/context/agents/`**, a different directory whose four personas
are current to 2026-08-10 and carry the accumulated review lessons of this codebase.
No workaround is needed and none is applied from P01 R1 onward — the personas are
loaded as intended. The stale duplicates remain filed as `MIS-E2E-001`, downgraded
to P3, as a delete-the-decoy task.

---

## The 12 phases

Doc-chain conformance is checked *inside* the phase that owns the code, and rolled
up in P11. Each phase names its code paths, its owning documents, and the specific
hazards this repo's history says to look for there.

| # | Phase | Primary paths | Owning docs |
|---|---|---|---|
| **P01** | Data layer & migrations | `backend/src/models/`, `backend/src/db/`, `backend/alembic/versions/`, `docs/schemas/*.json` | PADR IDL-30, 33, 37, 39; data-model manual page |
| **P02** | Backend services — CRUD & lifecycle | `backend/src/services/` (datasets, models, saes, trainings, extractions, templates, settings) | FPRD 001–005, 007, 010; IDL-14, 20, 25 |
| **P03** | ML / GPU | `backend/src/ml/` (SAE architectures, `jlens_fitter`, `jlens_metrics`, `layer_discovery`, `steering_core`, `model_loader`, `forward_hooks`) | FPRD 003, 006, 015, 021, 022; IDL-2, 3, 13, 38, 40–43 |
| **P04** | Workers, Celery, task lifecycle | `backend/src/workers/`, `backend/src/core/celery_app.py` | IDL-5, 11, 12, 39; FPRD 019, 020 |
| **P05** | REST API surface & schemas | `backend/src/api/`, `backend/src/schemas/` | every FPRD's API section; manual `reference/api/*` (11 pages) |
| **P06** | MCP server | `backend/src/mcp_server/`, `docs/mcp-contract.md`, `backend/tests/unit/test_reachability.py` | FPRD 010, 028; IDL-26; `0xcc/brds/miStudio-MCP-Server-BRD.md` |
| **P07** | Frontend state layer | `frontend/src/{stores,api,hooks,contexts,utils}/` | FPRD 011–014, 023; IDL-1, 27 |
| **P08** | Frontend UI | `frontend/src/components/**`, `frontend/src/config/panels.ts` | Mock UI reference; FPRD 011, 012, 023; IDL-15, 24, 28 |
| **P09** | Realtime (WebSocket end to end) | `backend/src/workers/websocket_emitter.py`, socket server, `frontend/src/hooks/use*WebSocket*`, `contexts/WebSocketContext.tsx` | IDL-1, 12; manual `reference/websocket-channels.md` |
| **P10** | Infra & supply chain | `docker*/`, `k8s/`, `nginx/`, `.github/workflows/`, `scripts/`, `0xcc/plans/CICD-*` | IDL-23; `SECURITY.md` |
| **P11** | Documentation chain conformance | all of `0xcc/`, `manual/` (47 pages), `README.md`, `CLAUDE.md`, `docs/` | the whole chain |
| **P12** | Cross-cutting synthesis & live journeys | spans every layer | PPRD success criteria |

### What each phase specifically hunts

**P01 Data layer.** Single alembic head confirmed (`c7e2a4f18b93`, one merge point
at `cd6c46abac48`) and guarded by `test_alembic_single_head.py` — so the hunt is
elsewhere. Two revision-id conventions coexist and some hand-written ids
(`g3h4i5j6k7l8`) are **not valid hex**, which breaks any tooling that assumes it;
four standalone audit scripts sit unowned at `backend/` root (`audit_migrations.py`,
`check_migrations.py`, `check_type_mismatches.py`, `find_column_gaps.py`) — find out
whether anything runs them. Contract IDLs are the highest-value code-checkable claims:
IDL-30 `mistudio.cluster-definition/v1`, IDL-33 `mistudio.circuit-definition/v1`
(the edge object is spelled out at PADR:3113), IDL-37's `calibration{onset,
sweet_spot,cliff}`. Grep each declared field against the ORM models, the Pydantic
schemas, `docs/schemas/*.json`, and the migration that created the column. Check
alembic head count, down_revision chains, enum creation order, CASCADE rules, and
whether any migration is unreachable. Known repo hazard: *`hp['hidden_dim']` is
corrected in memory and never persisted*.

**P02 Backend services.** The largest and least-tested surface: 42,682 lines, of
which **~19,000 sit in 29 modules with no name-matching test file** — led by
`steering_service.py` (2,993, the biggest file in the repo) and
`extraction_service.py` (2,116). Settings encryption is here, and
`core/encryption.py:118` **fails open on `InvalidTag`**: any decryption failure,
including authentication failure, returns the raw stored bytes as plaintext, so
AES-GCM's integrity property is discarded. `resolve_user_path` (`config.py:404`)
is well-built but has **exactly one production caller** (`sae_manager_service.py:543`);
everything else uses `resolve_data_path`, which performs no containment check —
so the trust boundary rests on every caller classifying its own input correctly.
Also in scope: `backend/services/ollm_server/` — a **second FastAPI app, 1,221
lines, not wired into the router**, with its own `CORSMiddleware`.

**P03 ML / GPU.** The repo's hardest-won lessons live here: additive steering must
hook `structure.layers_module[L]` (resid_post), **not** the discovered `"residual"`
module which renormalizes the vector away; `_norm_modules` must be
`endswith("norm")`, not `contains`; artifacts must load `weights_only=True`;
JumpReLU L0 must stay count-based, and `sparsity_coeff` must never fall back to
`l1_alpha`. Verify GPU memory release and the `affine_residual` freeze-leak refusal.

**P04 Workers.** `task_routes` globs match the **task name**, so a short name
silently uses the default queue — check every registered task. Check every janitor
(`cleanup_stuck_*`) actually closes the state it was written for, that `dry_run`
fails to its *default* not to `False`, and that unlink happens before the row
commit. Check finalize/prune create `task_queue` rows (recorded debt says they
do not).

**P05 REST API.** All 270 routes enumerated. **Authorization is treated as an
accepted posture, not a per-route defect** (see the decision above): `core/deps.py`
supplies only `get_db` and there are zero `get_current_user`/`HTTPBearer`/`Security(`
hits under `backend/src/api/`, but nginx plus network isolation is the intended
control. That yields **one** architectural finding — plus a remediation task to
write it into the PADR, since **no IDL states it today** and an undocumented
security posture is indistinguishable from an oversight to the next reader.

The effort goes instead into **verifying the control actually holds**, which is the
part nobody has checked:

- `nginx/nginx.conf:54` derives `$cors_origin` dynamically and sets
  `Allow-Credentials: true` — a permissive origin map alongside credentials would
  make the whole posture moot. `nginx.docker.conf:59` hardcodes one origin instead.
- Nginx `deny all`s `/api/internal/` (correct), and those two routes are separately
  HMAC-gated with `compare_digest` and always 403 — verify both layers.
- The MCP server binds `0.0.0.0` by default and is "LAN-reachable by design", with
  `MCP_ALLOW_ANONYMOUS` able to drop its bearer check entirely.
- `backend/services/ollm_server/` installs its own `CORSMiddleware` — confirm
  `allowed_origins` is never `["*"]` beside credentials.
- Confirm k8s ingress and compose expose nothing that bypasses nginx, and that
  `MISTUDIO_BYPASS_PIN` is false in every shipped manifest.

Two specifics stay individually recorded because they escape the network boundary's
reasoning: `steering.py:394,625` runs **`pkill -9 -f steering@`** (killing any
process whose cmdline matches, not just its own workers) and `steering.py:433,532`
spawns a Celery worker via `Popen` — a pattern-kill and a process-spawn are
privilege operations regardless of who can reach the port. The Settings PIN
(PBKDF2-SHA256, 600k iterations — a well-built primitive) is also recorded, since it
is a **UI gate only**: `GET/PUT/DELETE /settings` never verify it, so it protects
nothing it appears to protect.

The rest of the phase is ordinary API review: 11 of 25 endpoint modules have no
matching test file (`features.py`, 1,095 lines / 21 routes, the largest);
status-code correctness; the `{data, meta}` / `{error}` envelope; and Pydantic
`alias` hazards (a validation alias renames on **output** too, which once
invalidated every exported document).

**P06 MCP server.** The repo's signature defect class. For every tool: assert
presence in the **live registry**, not that the module imports; assert the
**payload and the call count**; confirm `MCP_TOOL_CATEGORIES` in k8s/compose does
not override registration; regenerate `docs/mcp-contract.md` and diff it; exercise
`millm_client` failure paths (a 200 HTML page from a misrouted ingress once reached
the agent as an empty SUCCESS). The existing `test_reachability.py` (1,265 lines) is
genuinely strong — three shapes, parametrized off the registry, with a negative
control — and this phase's job is to find what it *doesn't* cover, not to re-praise
it. Already visible: `SERVER_INSTRUCTIONS` (`server.py:32`) claims *"92 tools across
13 categories"* when the truth is 117 across 14, and `contract.py:9` says *"58 native
tools"* when there are 85 — the generated contract is dynamic, so only the prose
lies. The audit wrapper reaches into the private `mcp._tool_manager`. This session
has the MCP tools connected live; they get called.

**P07 Frontend state.** 11 of 20 stores, 17 of 19 hooks and 18 of 22 API modules
have **no test file**; `steeringStore.ts` is 2,598 lines. 172 `any` in production
code with `no-explicit-any` set to `warn`, concentrated at the WebSocket payload
boundary. 33 `.then()` chains with no `.catch()` (≈14 in `CircuitsPanel.tsx` alone)
despite a tested `utils/fireAndForget.ts` helper existing and not being applied.

**P08 Frontend UI.** 11 of 17 panels untested, including `SettingsPanel` (1,368
lines). Conformance against the Mock UI reference. The causal-language copy audit
(`SURFACES` was once hand-maintained at 5 files while 16 circuit modules went
unaudited). Hardcoded internal hosts in `SettingsPanel.tsx` including the private
IP `192.168.244.61:8001`. Production sourcemaps on; 519 `console.*` calls shipped.

**P09 Realtime.** The single highest-risk privacy path in the product: a broadcast
payload once leaked user prompt text and two reading rounds missed it. Mutate
`include_context` on every emitter. Cross-check all 15 hook→channel→event mappings
against the backend emitter and against `manual/docs/reference/websocket-channels.md`.
Investigate the **duplicate Socket.IO client** (`api/websocket.ts` vs
`contexts/WebSocketContext.tsx`) — different transports, untested, would open a
second connection.

**P10 Infra.** The `sync-to-clean` workflow strips `0xcc/`, `.claude/`, `CLAUDE.md`,
`backups/`, `scripts/` — but **not `aaaa/`**, which is a tracked duplicate of the
doc chain. `backups/*.sql.gz` (5 DB dumps) are in git history. Audit k8s manifests,
secrets handling, compose files, the ArgoCD/Image-Updater GitOps path, and CI
workflow permissions.

**P11 Documentation chain.** Produces `TRACEABILITY.md`. Three divergence axes,
tested independently: (a) PPRD §2.1 Status vs FTASKS checkbox state — already
contradictory on ≥6 rows; (b) FTASKS `## Relevant Files` paths vs files that exist
on disk — 11 task files have no such section at all; (c) IDL decisions vs shipped
schema and tables. **Off-by-one hazard: PPRD row N ≠ chain number N for everything
from chain 009 upward** (row 9, Settings & Configuration, has no chain), so no
automated cross-reference may assume numeric equality. Also verify the manual's 47
pages against actual behaviour and check whether `000_REMEDIATION_STATUS.md`'s items
were ever converted to tasks.

**P12 Cross-cutting synthesis.** Auth/PIN/secrets/path-resolution/`torch.load`
swept across all layers at once. A **reachability sweep** of every user-facing
capability. Then the end-to-end journeys driven live on k8s with Playwright and the
MCP tools: dataset → train → finalize → extract → label → cluster → circuit
discover/validate/promote → calibrate → steer → export → J-Lens readout.

---

## The three rounds

Every phase runs the same three rounds. Findings from a round are recorded and
**left unfixed**, so round 2 and round 3 review the same tree round 1 did.

### Round 1 — Discovery

| Command | Invocation | Output |
|---|---|---|
| `/security-review` | fed the phase's synthetic full-file diff (below) | `rounds/PXX-R1-security-review.md` |
| `/code-review high <paths>` | path target, high effort | `rounds/PXX-R1-code-review.md` |
| `/review code` | with the phase's own context passed in `args` (the stale personas are bypassed, not repaired) | `rounds/PXX-R1-review.md` |

Plus the phase's doc-conformance pass and its slice of the suites.

**Making `/security-review` work on committed code.** That command reviews the
*pending changes on the current branch*; on a clean `main` there are none. Each
phase therefore generates a synthetic diff in which every line of the phase's files
appears as an addition, using git's empty-tree object:

```bash
git diff 4b825dc642cb6eb9a060e54bf8d69288fbee4904 HEAD -- <phase paths> \
  > "$SCRATCH/P05.diff"
```

`/code-review` needs no such treatment — it accepts a path target directly.

### Round 2 — Adversarial re-review + mutation controls

Re-runs the same three commands with an explicit adversarial frame: *round 1
marked these areas clean; attack that conclusion.* This repo's history is the
argument for it — a review round once recorded "Privacy holds" as **verified
clean** by reading, and a later round flipped one argument and leaked user prompt
text into a broadcast with the suite still 135/135 green.

Then the mutation pass, per the standing rule in the user's global `CLAUDE.md`:

1. Pick the phase's load-bearing lines — anything carrying user data off-box
   (broadcast/WS payloads, exports, logs), correctness or honesty guarantees,
   retention and deletion, authorization gates, and any path whose failure mode is
   going quietly dark rather than raising.
2. Back up the file → edit **one** line → **confirm the edit landed** → run the
   affected suite → restore → confirm `git diff` is clean before moving on.
3. A surviving mutation is a **test finding** (`type: test-gap`), recorded with the
   regression test as its remediation. No test is written during the audit.
4. Log every mutation — line, expected break, actual suite result, restore
   confirmed — in `mutations/PXX-mutations.md`.

**Serialization rule:** mutation work never runs concurrently with any reading
agent. A reviewer once read an in-flight mutation out of the tree and reported it as
a committed defect.

### Round 3 — Verification & closure

Because the fix policy is record-only, round 3 cannot re-review fixes. It instead
verifies the **findings**:

1. **Adversarial refutation.** Each R1/R2 finding gets an independent attempt to
   *refute* it. Verdict recorded as CONFIRMED, PLAUSIBLE, or REFUTED. Refuted
   findings stay in the register marked REFUTED — the register is append-only, so a
   later round cannot silently rediscover them.
2. **Reproduction.** Every CONFIRMED finding gets a concrete reproduction: a failing
   assertion, a live request/response, a mutation that survived, or a UI trace.
3. **Live verification** for this phase — k8s requests, MCP tool calls, Playwright,
   GPU checks. The repo's own record says a hardware round found four bugs that four
   static rounds and the whole unit suite missed.
4. **Severity and effort** assigned; the phase's `README.md` grid row is closed.

---

## Closing the assessment

After P12 R3:

1. **`FINDINGS.md` is frozen** — no new IDs.
2. **Generate `0xcc/tasks/AUDIT_TASKS|E2E_Remediation_2026-08.md`** from the
   register, in the framework's FTASKS shape (`0xcc/instruct/007_generate-tasks.md`):
   parent tasks grouped by subsystem and ordered P0 → P3, sub-tasks sized at about
   one commit each and citing their `MIS-E2E-###`, a **Category Checklist Results**
   section (data, API, UI, integration, error handling, testing, perf/security,
   config/deploy, docs — each yielding tasks or an explicit N/A with reason), a
   `## Relevant Files` section, and a final **Feature Acceptance** parent task.
3. **REFUTED findings** are excluded from the tasklist but stay in the register.
4. **Enhancements** are separated into their own parent task so remediation and
   improvement can be scheduled independently.
5. **Write a synthesis** into `README.md`: counts by severity, by subsystem, by
   type; which phases were cleanest; which review command found what (a
   command-effectiveness read-out that informs the next audit); and the durable
   lessons, following the shape of `review_jlens_enhancements_2026-08-10.md`.
6. **Hand off the cross-repo tail.** Any finding whose confirmation needs the miLLM
   side is collected into a short `MILLM-HANDOFF.md` so the later miLLM assessment
   starts with them rather than rediscovering them.
7. **Update `CLAUDE.md`** Current Status and the Document Inventory to point at the
   audit — the only file edit the whole assessment makes outside `0xcc/audits/`.

---

## Seed findings (already observed while planning)

These go into `FINDINGS.md` as `MIS-E2E-001`… before P01 begins, so they are not
rediscovered as if new.

| Provisional | Finding | Phase |
|---|---|---|
| Architecture | **No application-layer auth on any of the 270 REST routes** — accepted posture per the decision above, so recorded **once**. The finding is that **no IDL documents it**: an undocumented security posture is indistinguishable from an oversight. Remediation = a new PADR IDL stating the control and its boundary | P05 |
| **Security P1** | `POST /steering/reset` and `/exit-mode` run **`pkill -9 -f steering@`** (`steering.py:394,625`) — a *pattern* kill, so any process whose cmdline matches dies, not just its own workers; `POST /steering/enter-mode` **spawns a Celery worker** via `Popen` (`steering.py:433,532`). Recorded individually: process kill and spawn are privilege operations regardless of who can reach the port | P05 |
| **Security P1** | `decrypt_value` **fails open on `InvalidTag`** (`core/encryption.py:118`) — a tampered ciphertext is returned as plaintext, discarding AES-GCM's integrity guarantee | P02 |
| **Security P1** | The Settings PIN is a **UI gate only** — `GET/PUT/DELETE /settings` never verify it. `MISTUDIO_BYPASS_PIN` also lets `POST /settings/pin/set` change the PIN without the current one | P02/P05 |
| Security | `resolve_user_path` (the hardened, containment-checked resolver) has **exactly one production caller**; every other path uses the unchecked `resolve_data_path` | P02 |
| Security | `aaaa/` is a **tracked duplicate of the 0xcc doc chain** and is **not** in the `sync-to-clean` exclusion list — dev-internal docs publish to the public mirror while `0xcc/` is stripped | P10 |
| Security | Five `backups/*.sql.gz` database dumps are committed to git history | P10 |
| Security | Private IP `192.168.244.61:8001` and four internal hostnames hardcoded in `SettingsPanel.tsx` in a publicly mirrored repo | P08 |
| Bug | `/review`'s four personas in `.claude/agents/*.md` describe a different project (Prisma, RTK Query, SAML; dated 2025-01-15) | P11 |
| Doc-drift | `CLAUDE.md` and `/review` reference `0xcc/session_state.json` and `.claude/context/health/dashboard.md`; **neither exists** | P11 |
| Doc-drift | PPRD §2.1 Status contradicts FTASKS state on rows 19, 21, 22, 23, 24, 27 | P11 |
| Debt | 348 unchecked task boxes; 193 in six ad-hoc files with no `## Relevant Files` section | P11 |
| Debt | Features 010, 011, 019, 020, 024–028 have never been reviewed | P11 |
| Bug | Duplicate Socket.IO client: `api/websocket.ts` vs `contexts/WebSocketContext.tsx`, different transports, untested | P09 |
| Test-gap | **~19,000 untested backend service lines** — 29 of 76 service modules have no matching test file, led by `steering_service.py` (2,993) and `extraction_service.py` (2,116). 11 of 25 endpoint modules likewise | P02/P05 |
| Test-gap | 11/20 stores, 11/17 panels, 17/19 hooks, 18/22 API modules have no test file | P07/P08 |
| Doc-drift | MCP `SERVER_INSTRUCTIONS` (`server.py:32`) says *"92 tools across 13 categories"*; actual is **117 across 14** (83/9 default). `contract.py:9` says *"58 native tools"*; actual **85** | P06 |
| Doc-drift | `core/websocket.py:15` claims CORS is handled by `CORSMiddleware` in `main.py`; `main.py:85` explicitly says the opposite and installs none | P09 |
| Debt | `backend/services/ollm_server/` — a **second FastAPI app (1,221 lines)** not wired into the router, carrying its own `CORSMiddleware` | P02 |
| Debt | Four unowned migration-audit scripts at `backend/` root; two alembic id conventions, some ids not valid hex | P01 |
| Debt | `backend/.env` present in the working tree — confirm it is ignored and holds no live secret before any review record is shared | P10 |
| Debt | `npm run type-check` excludes test files; no `format:check` script; no vitest coverage thresholds despite `@vitest/coverage-v8` installed | P07 |
| Debt | Production sourcemaps enabled; 519 `console.*` calls ship to the browser | P08 |
| Debt | 172 `any` in production frontend code with `no-explicit-any` at `warn`; 33 `.then()` without `.catch()` | P07 |
| Debt | `manual/` runs React 19 while `frontend/` runs React 18 | P10 |

---

## Verification

The assessment is itself verifiable — these are the checks that prove it was done,
not just written:

1. **Round completeness.** `ls 0xcc/audits/E2E-2026-08/rounds/ | wc -l` returns
   **108** (12 phases × 3 rounds × 3 commands). Every file names its command, its
   scope, and either findings or an explicit "no findings, here is what was
   examined".
2. **Register integrity.** Every `MIS-E2E-###` is unique and contiguous; every entry
   has a Verification verdict; every CONFIRMED entry has a reproduction; every
   CONFIRMED, non-REFUTED entry appears in the tasklist exactly once.
3. **Tree is clean.** `git status --porcelain` shows only files under
   `0xcc/audits/E2E-2026-08/`, `0xcc/tasks/AUDIT_TASKS|…`, and `CLAUDE.md`. No
   mutation is left in the tree — proven by `git diff` after each mutation batch.
4. **Baseline holds.** The backend and frontend suites at the end of the assessment
   return the exact counts recorded in `BASELINE.md`. A drift means something
   changed during a record-only audit and must be explained.
5. **Live verification actually ran.** Each phase's R3 record cites at least one
   live artifact — an HTTP response from `k8s-mistudio.hitsai.local`, an MCP tool
   result, a Playwright screenshot in `0xcc/caps/`, or a `k8s_logs`/`k8s_gpu`
   capture.
6. **Mutation controls bit.** `mutations/*.md` records, for each mutation, that the
   edit landed, what the suite did, and that the restore was confirmed. Mutations
   that *survived* are cross-referenced to a `test-gap` finding.

## Cost and sequencing

12 phases × (3 review commands × 3 rounds + suites + mutations + live checks) is a
large body of work — realistically **several sessions**. The phase boundaries are
the checkpoints: each phase closes with its three round records, its mutation log,
and its register entries committed, so the assessment is resumable at a phase
boundary without re-reading anything.

Suggested order is P01 → P12 as listed: the data layer and services first (later
phases depend on understanding the schema), MCP and frontend in the middle, and
documentation (P11) and cross-cutting synthesis (P12) last, since both consume what
the earlier phases found.
