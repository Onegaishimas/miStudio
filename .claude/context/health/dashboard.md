# Project Health Dashboard — miStudio

**Last updated:** 2026-08-24
**Updated by:** the E2E audit remediation (`0xcc/audits/E2E-2026-08/`)

> **This file did not exist until now (MIS-E2E-010).** Four slash commands —
> `/health`, `/review`, `/smart-clear` and this directory's own template —
> referenced `.claude/context/health/dashboard.md`, and `/review` Step 5
> instructs the reviewer to *update* it. Only `assessment_template.md` was ever
> created. A command that reads a missing file either fails or silently skips;
> either way the health step never ran.
>
> Created rather than de-referenced, because the commands want a living
> dashboard and there was a template for one. Keep it honest: every number here
> should be measured, and say when it was.

## Suites

| Suite | Result | Measured |
|---|---|---|
| Backend (`pytest tests/ --no-cov -q`) | **3247 passed / 0 failed** (27 skipped) | 2026-08-24 |
| Frontend (`npx vitest run`) | **1239 passed / 0 failed** | 2026-08-24 |
| Types (`npx tsc --noEmit`) | clean | 2026-08-24 |
| Lint (`npx eslint .`) | **0 errors**, 492 warnings | 2026-08-24 |

Baseline at the start of the remediation: backend 2883, frontend 1211.

**CI now runs all of it.** Until 2026-08-23 the frontend workflow excluded nine
test files (329 tests, 27% of the suite) and never ran lint at all — which is
how two `react-hooks/rules-of-hooks` violations shipped (MIS-E2E-024/-025).

## Open risk

| Area | State |
|---|---|
| Audit remediation | Waves 1–5 closed; Wave 6 (docs) nearly closed; Waves 7–9 open |
| Surviving audit mutations | 6 of 14 killed (M2, M3, M5, M13, M22, cache divergence) |
| Hardware acceptance | **Not run** — needs GPU-node access (Task 15) |
| SSH credential | ⚠️ **Rotation outstanding** — the GPU node password is published in mirror history |

## Standing rules this project learned the hard way

- **A capability is not shipped until a test FAILS when its wiring is removed.**
  A grep for a caller is the weaker form and is not sufficient.
- **Mutate to verify.** Break a load-bearing line, run the suite, revert —
  confirm the edit LANDED before concluding a mutation survived.
- **A surviving mutation is a TEST finding**, not a code finding.
- **Fix the siblings.** "Fixed one representative, never generalized" is this
  codebase's most repeated defect — five independent instances in one audit.
- **Parse, don't grep.** A guard that greps source fails OPEN when the layout
  changes, and cannot tell a fix's explanatory comment from the defect it
  describes. Five source-scrape guards in this audit failed open.
- **Derive from the registry, never a hand-list.** Three guards had scope
  narrower than their claim.

## How to refresh this file

```bash
cd backend && DATABASE_URL=... pytest tests/ --no-cov -q
cd frontend && npx vitest run && npx tsc --noEmit && npx eslint .
```

Then update the table above **with the date you measured**. A number with no
date is the defect this file was created to stop.
