# P10 — Infra & supply chain: all three rounds

**Phase:** P10 · **Date:** 2026-08-23
**Scope:** `.github/workflows/`, `k8s/`, `nginx/`, `docker-compose*.yml`,
`backend/Dockerfile`, `frontend/Dockerfile`, `scripts/`

Mutation log: `mutations/P10-mutations.md` (1 run, 0 survived).

## R1 — findings (6 register entries, 14 defects)

| Id | Sev | Claim |
|---|---|---|
| **MIS-E2E-143** | **P0** | The public mirror publishes the **full unfiltered history** — an SSH password, five DB dumps, and this audit's findings register |
| MIS-E2E-144 | P1 | `k8s_deploy` re-applies a stale manifest that **reverts two incident fixes**; the guard reads only `k8s/base` |
| MIS-E2E-145 | P1 | Postgres and Redis are RollingUpdate Deployments over `hostPath` — two pods, one data dir |
| MIS-E2E-146 | P1 | Compose publishes an unauthenticated Redis (the Celery broker) and a known-password Postgres on `0.0.0.0` |
| MIS-E2E-147 | P2 | Five defects: the compose frontend is unreachable; `k8s_deploy` reports success on failure; no `/ollama/` route; one-slot worker; a `server_name` typo |
| MIS-E2E-148 | P3 | A guard that `pytest.skip`s if a path moves; `/api` exposing `/api/internal/*` at the ingress; `apt-key adv` in the global keyring |

## MIS-E2E-143 — verified against the live public repository

`sync-to-clean.yml` checks out with `fetch-depth: 0`, `rm -rf`s the excluded paths,
makes **one** filter commit, and `git push --force`. That publishes the entire
unfiltered history with a cleanup commit on top, so the filter removes the files
from the **tip and from nowhere else**.

Confirmed via the GitHub API against `hitsainet/miStudio` (`"visibility": "public"`),
comparing the tip to its parent `ef270db`:

```
scripts/k8s-helpers.sh                      tip 404   history PRESENT  ← K8S_PASS=
backups/mistudio_db_20251218_035811.sql.gz  tip 404   history PRESENT
CLAUDE.md                                   tip 404   history PRESENT
0xcc/audits/E2E-2026-08/FINDINGS.md         tip 404   history PRESENT
scripts/backup-db.sh                        tip 404   history PRESENT
```

Three live exposures: an **SSH password** for the GPU node (`192.168.244.61`, user
`sean`, used with `StrictHostKeyChecking=no`); **five database dumps**; and **this
audit's own findings register** — a complete indexed inventory of 148 unremediated
defects including 12 P0s with reproductions, which reached the public repo through
the merge that deployed the Feature Detail fixes.

**Why reading the workflow does not reveal it.** The exclusion list is right, the
intent is right, and the `docs/schemas` and `mcp-contract.md` preservation is
deliberate and correct. The defect is not in *what* it removes; it is that
force-pushing a full history makes removal-at-the-tip meaningless. `docker-images.yml`
even documents the mechanism — *"HEAD~1 is the unfiltered source tip"* — as a
build-detection nuance rather than a disclosure.

This supersedes MIS-E2E-007 and MIS-E2E-008, both of which assumed the filter
worked.

## R2 — mutation

**M28 KILLED.** Stripping `circuits` and the `millm_*` categories from the deployed
`MCP_TOOL_CATEGORIES` fails two tests. This is the increment's real outage — 35 tools
registered, tested, documented and unreachable — and it is genuinely pinned. It is
also the only test in the entire audit that reads a k8s manifest and compares it to
the code registry, which is exactly why it catches a class nothing else can.

## R3 — verification

| Verdict | Ids |
|---|---|
| **CONFIRMED** | 143 (live public API), 144, 146, 147, 148 |
| **PLAUSIBLE** | 145 |
| **REFUTED** | none |

Two items the reviewer checked and cleared: `NEURONPEDIA_LOCAL_URL:
http://localhost:3001` is only used to build a browser-facing display URL and is
correct as written; and the CI torch-CPU install is not clobbered by
`requirements.txt`, since `torch==2.9.1` matches.

**One item promoted from P05.** That phase flagged as *"assumed, not verified"* that
`secret_key` is distinct per deployment — the k8s secret was not inspected.
`k8s/mistudio-secrets.yaml.example` is a template and the real secret is not in the
repo, so the HMAC derivation is sound. Recorded as checked.

## Phase closed

**6 findings** (MIS-E2E-143 … 148), **1 P0** and it is the audit's most severe.
Mutations: 1 run, 0 survived.

**The one sentence for the synthesis:** the filter that protects the public mirror
does exactly what its comments say and still publishes everything, because deleting
files from a tip commit does not remove them from a force-pushed history.
