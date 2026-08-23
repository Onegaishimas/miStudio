# miStudio End-to-End Assessment — E2E-2026-08

**Started:** 2026-08-23 · **Status:** ⏳ P00 ✅ · P01 ✅ · P02 ✅ · P03 ✅ · P04 ✅ · P05 ✅ · P06 ✅ · P07 ✅ · P08 ✅ · P09 ✅ · P10 ✅ closed · P11 next

> **⚠ MIS-E2E-143 — ACT NOW.** An SSH password, five database dumps and this
> register itself are published in a **public** GitHub repository. See FINDINGS.md.

An end-to-end assessment of the whole application — frontend, backend, MCP server,
infrastructure, and the `0xcc` document chain — in **12 subsystem phases × 3 rounds
× 3 review commands**. Every defect and improvement is recorded here and **fixed
later**: the assessment makes no code changes.

The charter is [PLAN.md](PLAN.md). Read it first.

## How to read this directory

| File | What it is |
|---|---|
| [PLAN.md](PLAN.md) | The approved charter — phases, rounds, decisions, methodology |
| [BASELINE.md](BASELINE.md) | Measured ground truth before any round ran: suites, lint, type-check, CI, migrations, live env |
| [FINDINGS.md](FINDINGS.md) | **The register.** Append-only. Every finding, one block, stable id `MIS-E2E-###` |
| [TRACEABILITY.md](TRACEABILITY.md) | Doc chain → code → test matrix (produced in P11) |
| [MILLM-HANDOFF.md](MILLM-HANDOFF.md) | Findings whose confirmation needs the miLLM repo, for its own later assessment |
| `rounds/PXX-RY-<command>.md` | 108 round records — one per phase × round × review command |
| `mutations/PXX-mutations.md` | Mutation control log: line broken, whether the edit landed, suite result, restore confirmed |

The single remediation tasklist is generated at the end, outside this directory, at
`0xcc/tasks/AUDIT_TASKS|E2E_Remediation_2026-08.md`.

## Ground rules

1. **Strict record-only.** No code is changed during the assessment — not even a
   live security hole. Rounds 2 and 3 must review the same tree round 1 did.
   **One deliberate exception, 2026-08-23:** the user found four defects in the
   live Feature Detail modal and asked for them to be fixed. They were
   (MIS-E2E-132…135, plus MIS-E2E-030/031 which they confirmed). Phases P09–P12
   therefore review a tree P01–P08 did not; the changed files are
   `models/{feature,feature_analysis_cache}.py`, `services/analysis_service.py`
   and `components/features/FeatureTokenAnalysis.tsx`.
2. **Mutations are always reverted**, and the revert is confirmed with `git diff`
   before moving on. A surviving mutation is a *test* finding, not a code finding.
3. **Mutation work never runs concurrently with a reading agent.** A reviewer once
   read an in-flight mutation out of the tree and reported it as a committed defect.
4. **The register is append-only.** A refuted finding is marked REFUTED and stays,
   so a later round cannot silently rediscover it.
5. **Verify a mutation landed** before concluding it survived.

## Phase × round status board

Legend: `·` not started · `~` in progress · `✅` closed

| # | Phase | R1 | R2 | R3 | Findings |
|---|---|:--:|:--:|:--:|---:|
| P00 | Baseline | ✅ | — | — | 28 |
| P01 | Data layer & migrations | ✅ | ✅ | ✅ | 26 |
| P02 | Backend services | ✅ | ✅ | ✅ | 24 |
| P03 | ML / GPU | ✅ | ✅ | ✅ | 13 |
| P04 | Workers & Celery | ✅ | ✅ | ✅ | 6 |
| P05 | REST API & schemas | ✅ | ✅ | ✅ | 15 |
| P06 | MCP server | ✅ | ✅ | ✅ | 7 |
| P07 | Frontend state layer | ✅ | ✅ | ✅ | 8 |
| P08 | Frontend UI | ✅ | ✅ | ✅ | 4 |
| — | *Out-of-band: user-reported modal defects (**fixed**)* | ✅ | — | ✅ | 4 |
| P09 | Realtime (WebSocket) | ✅ | ✅ | ✅ | 7 |
| P10 | Infra & supply chain | ✅ | ✅ | ✅ | 6 |
| P11 | Documentation chain | · | · | · | 0 |
| P12 | Cross-cutting & live journeys | · | · | · | 0 |

## Findings by severity

| P0 | P1 | P2 | P3 | Total |
|---:|---:|---:|---:|---:|
| **12** | 57 | 58 | 21 | **148** |

Verdicts so far: **15 CONFIRMED**, 39 pending (most pending belong to phases that
have not run yet). Severities are provisional until each finding's R3 verification.

## Mutation controls

| Phase | Run | Survived | Killed |
|---|---:|---:|---:|
| P01 | 5 | **3** | 2 |
| P02 | 4 | **3** | 1 |
| P03 | 6 | **3** | 3 |
| P04 | 2 | **1** | 1 |
| P05 | 2 | **1** | 1 |
| P06 | 2 | 0 | 2 |
| P07 | 1 | **1** | 0 |
| P08 | 2 | 0 | 2 |
| P09 | 2 | **1** | 1 |
| P10 | 1 | 0 | 1 |

A survival is a **test finding**, not a code finding. Every edit was confirmed to
land before the suite ran, and `git diff` verified clean after every restore.
