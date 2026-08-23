# BASELINE — measured ground truth before any review round

**Captured:** 2026-08-23 · **Commit:** `408102e` (clean `main`, `git status` empty)

Everything here was **measured**, not read from `CLAUDE.md`. Where the two
disagree, this file is right and the disagreement is itself recorded.

---

## 1. Environment

| | State |
|---|---|
| Repo | `/home/x-sean/app/miStudio`, branch `main`, working tree clean |
| Postgres | `mistudio-postgres` up 13 days (healthy), `0.0.0.0:5432` |
| Redis | **was not running** at the start of the audit — started via `docker compose -f docker-compose.dev.yml up -d redis` to obtain an honest baseline (see MIS-E2E-027) |
| Local backend (:8000) | not running |
| Local frontend (:3000) | not running |
| Live k8s | `http://k8s-mistudio.hitsai.local` — `/` 200, `/api/health` 200, `/api/v1/version` 200, `/api/v1/models` 200 |
| Live MCP | `http://mcp-mistudio.hitsai.local/mcp` — connected to this session, 116 tools |
| k8s shell access | **unavailable** — `ssh 192.168.244.61` rejects publickey; `scripts/k8s-helpers.sh` uses `sshpass` with `K8S_USER=sean` and a `$K8S_PASS` this session does not hold. Live verification therefore runs over **HTTP and the MCP server**, not `kubectl`. |
| Stray process | PID 1334, `pytest tests/unit`, **elapsed 13d 22h** — a leaked test run from a prior session, still resident. Not a code defect; noted because it may hold DB connections. |

## 2. Test suites

### Backend — `pytest tests/`
Command (the one this audit uses throughout):
```bash
cd backend && DATABASE_URL=postgresql+asyncpg://postgres:devpassword@localhost:5432/mistudio \
  DATABASE_URL_SYNC=postgresql://postgres:devpassword@localhost:5432/mistudio \
  ./venv/bin/python -m pytest tests/ --no-cov -q -p no:randomly
```

| | Value |
|---|---|
| Test files | 164 (`tests/unit` 144, `tests/integration` 15, `tests/api/v1/endpoints` 3, root 2) |
| `def test_` count | 2,673 |
| **Collected tests** | **2,883** (parametrization accounts for the gap) |
| **Result (Redis down)** | 2 failed — `test_datasets.py::TestDownloadDataset::{test_download_dataset_success,test_download_dataset_with_access_token}`, both `ConnectionRefusedError` from `celery.backends.redis` → **MIS-E2E-027** |
| **Result (Redis up)** | **0 failed.** Full run reached 100% with no `FAILED`/`ERROR` line. Confirms both failures above were purely environmental. |
| Skips | 27 observed — 15 `test_activation_service.py` (needs CUDA), 7 `test_causal_language_audit.py` (documented rung-2 exemptions, deliberately skipped rather than silently passed), 1 `test_jlens_readout.py` (needs a second device), 3 integration skips (no `OPENAI_API_KEY`, no completed training ×2) |

### Frontend — `npx vitest run`

| | Value |
|---|---|
| Test files | 65, all co-located `*.test.ts(x)` |
| Tests | **1,211 passed / 1,211** — 0 failed, 0 skipped, 0 `.only` |
| Duration | 21.65 s |
| **Gated by CI** | only 56 files / **882 tests**. Nine files carrying **329 tests (27%)** are excluded by `frontend-ci.yml` and all nine pass → **MIS-E2E-025** |

Note: `npm test` is **watch mode**; CI and this audit use `--run`.

### Manual — `manual/`
`npm run typecheck` → clean (exit 0). 47 doc pages (46 `.md` + `intro.mdx`).

## 3. Static gates

| Gate | Command | Result |
|---|---|---|
| Frontend types | `npx tsc --noEmit` | **clean** (exit 0) |
| Frontend lint | `npx eslint .` | **FAILS — 34 errors, 494 warnings** (exit 1) → MIS-E2E-024 |
| Frontend format | — | **no `format:check` script exists** → MIS-E2E-021 |
| Frontend coverage | — | **no coverage block, no thresholds** despite `@vitest/coverage-v8` installed → MIS-E2E-021 |
| Manual types | `npm run typecheck` | clean |

**Lint errors by rule:** `no-unused-vars` 15 · `no-unsafe-function-type` 5 ·
`no-useless-catch` 3 · `no-useless-escape` 3 · `no-constant-binary-expression` 2
(both **refuted** — deliberate `{false && …}` unmount assertions) ·
**`react-hooks/rules-of-hooks` 2** (→ MIS-E2E-023, a real bug) ·
`no-require-imports` 2 · `no-array-constructor` 1 · `prefer-const` 1.

**What CI actually runs:** `frontend-ci.yml` = `type-check` → `build` → `test`
(with 9 exclusions). **It never runs lint.** That is the mechanism by which a
Rules-of-Hooks violation survived a three-round review.

## 4. Schema state

| | Value |
|---|---|
| Revision files | 98 |
| `alembic heads` | **`c7e2a4f18b93` — single head** ✅ (guarded by `test_alembic_single_head.py`) |
| `alembic current` (local DB) | `c7e2a4f18b93 (head)` — up to date |
| Base | `118f85d483dd` (`create_datasets_table`) |
| Merge points | one — `cd6c46abac48`, `down_revision = ('f3a7b1c2d4e5', 't7u8v9w0x1y2')` |
| Id conventions | two, mixed; some hand-written ids (`g3h4i5j6k7l8`) are **not valid hex** → MIS-E2E-022 |

## 5. CI

Last 15 runs (`gh run list`): all **green or skipped**. Backend Tests 6m37s,
Frontend CI 1m31s, both on `8726648`. The two most recent commits are docs-only
and correctly skipped the build workflows via path filters.

**Green CI does not mean a green codebase here.** Lint is not run at all, and
27% of the frontend suite is excluded from the run that reports green.

## 6. Live surface

`GET /api/v1/version` → `{"version":"unknown","app":"miStudio"}` on the deployed
app, while `VERSION` in the repo reads `0.5.0` → **MIS-E2E-028**.

MCP live registry (`mistudio_howto(topic='tools')`): **116 tools / 14 categories** —
admin 2, circuits 24, experiments 3, groups 6, jlens 19, jobs 1, labeling 3,
millm_circuits 16, millm_clusters 6, millm_runtime 5, millm_sensing 5, profiles 4,
read 12, steering 10. The server's own `SERVER_INSTRUCTIONS` claims 92/13 →
MIS-E2E-017.

Deployed `MCP_TOOL_CATEGORIES` (`k8s/base/mcp.yaml`, the manifest ArgoCD applies):
`read,groups,steering,labeling,experiments,profiles,circuits,jlens,jobs,millm_runtime,millm_clusters,millm_sensing,millm_circuits`
— `admin` deliberately off (two irreversible delete tools). Confirmed against
this session's tool list: no `delete_extraction` / `delete_experiment` present. ✅

## 7. Counts vs `CLAUDE.md`

| | `CLAUDE.md` says | Measured |
|---|---|---|
| Backend tests | 2461 | 2,673 `def test_` |
| Frontend tests | 1149 | 1,211 cases / 65 files |

Both differ. `def test_` is not the collected-test count (parametrization), and
`CLAUDE.md` was written at an earlier commit. Not recorded as a finding —
recorded here so no later phase treats `CLAUDE.md`'s numbers as the baseline.

## 8. Deviations from a pristine baseline

Two environment actions were taken, both reversible, neither a repo change:

1. **Started `mistudio-redis`** via `docker-compose.dev.yml`. Without it two API
   tests fail for reasons unrelated to the code.
2. Nothing else. `git status --porcelain` is empty apart from
   `0xcc/audits/E2E-2026-08/`.
