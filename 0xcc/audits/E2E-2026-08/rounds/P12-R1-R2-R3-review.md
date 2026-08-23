# P12 — Cross-cutting synthesis & live journeys

**Phase:** P12 · **Date:** 2026-08-23 · **The final phase.**

## Live verification

The live surface was probed on `k8s-mistudio.hitsai.local`. Every panel-backing
endpoint answers: `/`, `/api/health`, `/api/v1/{version,models,extractions,saes,circuits,jlens/artifacts,cluster-profiles,settings}` — all 200. The MCP
registry reports 116 tools across 14 categories.

**Two P0s were confirmed against the running deployment**, not merely at source:

- **MIS-E2E-165** — `GET /api/v1/settings` returns `settings_pin_hash` with
  `is_sensitive=False`, unmasked, 150 characters. The gate protecting the credential
  panel is served in the clear to any unauthenticated caller.
- **MIS-E2E-166** — destructive routes were probed with **nonexistent ids** (nothing
  deleted). None returned 401/403; all reached the handler. The accepted posture is
  real in production, which is what MIS-E2E-002's PADR-IDL task now has evidence for.

Also verified live this session, after the out-of-band fixes were deployed: the four
Feature Detail modal defects (MIS-E2E-030, 132–135) all return 200, and correlations
now draw peers from the same SAE rather than from other models.

## The cross-cutting sweep

Five classes were traced across every layer rather than within one.

### 1. A guard that exists and is not on the path
The dominant class, and it accounts for most of the P0s. `validate_llm_endpoint_url`
has two call sites and neither is the credential-bearing one (069) or `/bulk` (073) or
`judge_endpoint` (075). `resolve_user_path` is correctly built and has **one**
production caller — not either `rmtree` site (070, 071). The "never write the bearer
token to disk" rule is implemented, commented, and applied to the cURL branch but not
the Postman branch sixty lines below (072). The `is_system` guard is on update and
delete but not import (108). In every case the fix already exists in the codebase,
ten to sixty lines from where it is missing.

### 2. Fixed one representative, never generalized
Five instances, independently found: the SAE-basis fix applied to `combined` and not
compare/sweep (064); the Postman/cURL split (072); `looks_abandoned` in one janitor of
five (092); `asyncio.to_thread` on one emit call of fourteen (136); and the "Stop
saves final checkpoint" correction applied to one manual of two (149).

### 3. A guard whose scope is narrower than its claim
BR-002 says "no band constant **anywhere**" and scans two hardcoded modules (090). The
MCP reachability harness proves registration for all 116 tools and behaviour for 16
(119). `test_worker_queue_coverage` proves every queue has a consumer and never that a
task reaches one (097). `check_migrations.py` prints "All models match the database
schema" while comparing column names only (048). IDL-16's validator covers 17 tables
of 36 and cannot fail the boot (157).

### 4. The test schema is not the production schema
`create_all()` builds from the ORM, and the ORM diverged from the migrations in both
directions on a single table (031, 033). Three mutations confirmed the consequence:
constraints, cascades and dropped fields are all invisible to the suite (051, 052,
053). This is the root enabler behind the production 500 the user hit.

### 5. Wrong results presented as correct
The class this product can least afford. Coherence and behavioral scores are a
hardcoded `0.5` because a dependency is absent (063). "Converged" measures the
shrinkage of a running mean, not stabilisation (080). A published artifact labels a
positional spread as a residual (081). FVE is overstated 4.5× on a rank-deficient
basis (088). The Diff view shades agreement as disagreement (129). Correlations were
drawn from other models (133). A prune dialog says "report on" while deleting (128).

## Mutation testing — the whole-audit result

**29 mutations run across 8 phases; 14 survived.** Every edit was confirmed to land
before the suite ran, and `git diff` was verified clean after every restore.

The fifteen kills matter as much as the survivors, and are recorded per phase: the
expunge-before-mask encryption fix, the `NormedBlock` lesson, BR-002's AST guard,
BR-030 fail-closed, the calibration clamp, the byte-exact schema-sync guard, the
plain-alias guard, the one fixed janitor, the internal-token header, the MCP
registration harness, the deployed-categories guard, and the panel registry.

Three results are worth carrying forward as method:

- **A control can be correct and completely unprotected.** `_SENSITIVE_KEYS` was
  listed under "verified clean" by a security review that described it accurately.
  Removing a key left 40 tests green.
- **A test can sample only where the behaviour cannot differ.** The auto-baseline
  formula's own test file exercises freq 0 (slope × 0) and the clamped extremes; the
  arithmetic that would distinguish 2.6 from 2.4 lives in a comment.
- **Mutating *toward* a fix is informative.** Applying MIS-E2E-137's correction changed
  nothing in the suite, which says the fix would not stay fixed.

## Corrections made to this audit's own findings

Recorded because a register that overstates is as useless as one that misses.

| Finding | Correction |
|---|---|
| MIS-E2E-001 | Wrong about impact — `/review` loads a *different* directory whose personas are current. P2 → P3. |
| MIS-E2E-014 | Import graph traced: genuinely dead code. P2 → P3. |
| MIS-E2E-023 | Over-stated — the crash is unreachable, held off by one backend expression. P1 → P2. |
| MIS-E2E-011 | Undercount — 13 PPRD rows contradict, not 6. |
| MIS-E2E-072 | Confirmed in code, **not materialised**: the `tmp_api` sweep found 27 files, 9 Postman collections, zero `Authorization` headers. |
| Panel-registry hypothesis | Refuted outright — coverage is registry-derived and general. |
| `steering.*` mis-routing | My own false positive: I probed with the function name, not the registered task name. |

## Live journeys not driven

Stated plainly rather than implied. The end-to-end UI journey (dataset → train →
extract → label → cluster → circuit → steer → export) was **not** driven with
Playwright, and the GPU-dependent claims — MIS-E2E-082's patch leak, MIS-E2E-083's
off-distribution capture — were not exercised on hardware. `ssh` to the GPU node is
unavailable in this session (BASELINE §1), and those are precisely the findings this
repo's history says only hardware confirms. They are recorded PLAUSIBLE for that
reason and carried into the remediation tasklist as hardware-acceptance items rather
than presented as verified.

## Phase closed — and the audit with it

**2 findings** (MIS-E2E-165, 166), both live confirmations of earlier work.

**Final: 166 findings — 13 P0, 62 P1, 68 P2, 23 P3.** Twelve phases, 36 round records,
29 mutations, 8 mutation logs, one traceability matrix.
