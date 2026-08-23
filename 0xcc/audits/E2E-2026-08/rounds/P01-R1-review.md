# P01 R1 — /review (multi-agent)

**Phase:** P01 Data layer & migrations · **Round:** 1 · **Date:** 2026-08-23
**Scope arg:** `code` · **Commit:** `408102e`

## A correction to the plan, made at the start of this round

PLAN.md pre-recorded a blocker: that `/review` loads four persona files containing
another project's content. **It does not.** `.claude/commands/review.md` Step 1
loads `@.claude/context/agents/*.md` — a *different directory* from
`.claude/agents/`, and those four personas are current to 2026-08-10, carrying this
codebase's accumulated review lessons. They were loaded as intended and are the
reason this round looked where it did.

The stale duplicates are real but inert: `.claude/agents/{architect,product_engineer,
qa_engineer,test_engineer}.md`, dated 2025-01-15, grep-negative for `miStudio`,
`SAE` and `steering`. MIS-E2E-001 has been rewritten and downgraded to P3, and the
withdrawal is recorded in PLAN.md rather than silently edited.

## What the (real) personas directed this round to check

The QA persona's standing priority — *"claim-versus-behaviour on anything carrying a
guarantee word"* — turned out to be exactly the right lens for a data layer whose
audit tooling makes claims. Two of this round's three findings are guarantee words
that do not hold.

## Findings (3)

| Id | Sev | Claim |
|---|---|---|
| MIS-E2E-048 | P2 | `check_migrations.py` prints **"All models match the database schema"** while comparing column names only — it printed that on a DB with five proven constraint divergences |
| MIS-E2E-049 | P3 | `find_column_gaps.py` contradicts it, reporting 11 present columns as missing, because its `create_table` regex truncates at the first `)` |
| MIS-E2E-050 | P2 | `manual/docs/reference/data-model.md` omits 11 of 38 tables, including `checkpoints` — which its own ER diagram draws — and the entire Clusters subsystem |

## The four scripts, run (MIS-E2E-022 discharged)

All four executed against `mistudio-postgres`. What they say:

| Script | Result |
|---|---|
| `audit_migrations.py` | Lists migrations by date; one file dated `unknown`. Ends by recommending `check_migrations.py` as *"the authoritative check"* — which is MIS-E2E-048 |
| `check_migrations.py` | **"✅ No migration gaps found! All models match the database schema."** — false, see MIS-E2E-048 |
| `check_type_mismatches.py` | 25 columns checked, all ✅. Genuinely correct as far as it goes — it verifies varchar widths on id/FK columns |
| `find_column_gaps.py` | 11+ false positives on `trainings` alone — see MIS-E2E-049 |

So of four unowned scripts: one is a pointer to a broken one, one overclaims green,
one cries wolf, and one (`check_type_mismatches.py`) is small and correct. The
recommendation in MIS-E2E-022 sharpens: keep `check_type_mismatches.py`, fix or
narrow `check_migrations.py` and wire it into CI, delete the other two.

## IDL conformance — checked field by field

**IDL-37 (calibration): the write path conforms; the read path discards.**
Clause 4 requires the calibration block to clamp `budget.intensity_range` to
`[onset, cliff]` and default `budget.intensity` to the sweet spot.
`circuit_service.py:375-376` does exactly that — `budget["intensity_range"] =
[onset, cliff]`, `budget["intensity"] = sweet` — and refuses an inverted band first.
**Conformant.** But clause 5's stated purpose is that *"the probe set travels in the
contract so a one-shot re-verify at serve time is cheap"*, and import drops the whole
block (MIS-E2E-037). The decision is implemented on the way out and defeated on the
way in. Recorded against MIS-E2E-037 rather than as a new id.

**IDL-33 (circuit-definition): conformant on structure.** `saes[]`, `members[]` with
required `layer` and `member_kind`, the full edge shape including `type`, `rung`,
`tested_and_failed`, `coactivation`, `weight_prior`, `attribution`,
`validation_manifest_ref` and nullable `position` — all present in
`schemas/circuit_definition.py` and in the published JSON Schema, and the two are
byte-synced by an executed test. Per-layer member caps enforced. `circuits` table
with `crc_` ids and JSONB members/edges/budget/faithfulness present. The gap is not
structural: clause 1 declares `rung: 0..3` as a field and nothing verifies an
asserted value (MIS-E2E-035).

**IDL-30 (cluster-definition):** structurally conformant; `cluster_profiles` table
present with the documented shape.

**A chain worth naming.** `_dial_within_range`'s docstring says the calibration
guarantee *"rests on `intensity_range` being the true envelope AND `intensity`
sitting inside it"* — it enforces the second and assumes the first.
`circuit_service.py:370-373` then says *"The intensity∈range invariant is now
enforced by `CircuitBudget` itself"* and delegates to it. Two sites lean on a
validator whose own precondition the published schema leaves unbounded. Recorded as
strengthening evidence on MIS-E2E-034.

## Test strategy for the data layer

The structural problem is stated in MIS-E2E-031/033 and is worth restating as a
strategy conclusion rather than a defect: **the unit suite builds its schema from
`Base.metadata.create_all()`, so it tests a schema that differs from production in
five known ways.** Every constraint added by migration only is invisible to it, and
every constraint declared on a model but dropped from the database is *present* for
it. No amount of additional test-writing fixes this while the two schemas can
diverge — the fix is a diff guard, and it belongs in P01's remediation regardless of
which individual constraints are wrong today.

Second: `tests/api/v1/endpoints/` holds only three files for 25 endpoint modules, and
two of the tests in it reach a real Redis (MIS-E2E-027). Data-layer coverage is
mostly indirect, through service tests.

## Verified clean — R2 must attack these

- **`check_type_mismatches.py` is correct.** 25 id/FK columns, all `character varying`
  at the declared widths. No `String(255)` vs `String(36)` mismatch anywhere it looks.
- **The ER diagram's relationships are right** even where the tables behind them are
  undocumented — the `trainings ||--o{ checkpoints` edge matches the real FK.
- **IDL-33's "additive-only family rule"** holds: `circuit-definition/v1` is a new
  kind, not a mutation of `cluster-definition/v1`, and the seven shared `$defs` are
  byte-identical between the two published files.
- **`_dial_within_range` rejects a malformed range** (wrong length, inverted) rather
  than clamping it — the shape check is strict and correct. Only the *element bounds*
  are missing.
