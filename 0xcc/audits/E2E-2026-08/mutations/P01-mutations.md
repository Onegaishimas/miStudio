# P01 — mutation control log

**Phase:** P01 Data layer & migrations · **Round:** 2 · **Date:** 2026-08-23

Discipline (global `CLAUDE.md`): back up the file → edit **one** line → **confirm the
edit landed** → run the affected suite → restore → confirm `git diff` clean. No
reading agent ran concurrently with any mutation. Harness:
`$SCRATCH/mutate.sh`, which aborts with a distinct message if the edit does not
land, so a non-applying patch can never be mistaken for a surviving mutation.

`rc=0` means the suite stayed green, which means the mutation **survived**, which is
a **test finding**.

| # | Target | Mutation | Suite | Landed | Result |
|---|---|---|---|---|---|
| M1 | `schemas/circuit_definition.py:176-179` | Invert the calibration clamp: `intensity = hi` when below `lo`, `lo` when above `hi` | `tests/unit -k 'circuit or calibration or budget'` | ✅ | **KILLED** — 2 failures |
| M2 | `db/schema_validator.py:20` | Remove `"models"` from `REQUIRED_TABLES` | `tests/unit -k 'schema or validator or startup'` (155 tests) | ✅ | **SURVIVED** → MIS-E2E-051 |
| M3 | `api/v1/endpoints/circuits.py:201-202` | Drop `"faithfulness"` from the import dict — the exact shape of the real calibration bug | `tests/unit -k 'circuit'` | ✅ | **SURVIVED** → MIS-E2E-052 |
| M4 | `schemas/circuit_definition.py:225` | Change a published field `description` string | `test_circuit_definition_schema_sync.py` | ✅ | **KILLED** — 1 failure |
| M5 | `models/feature.py` | Flip all 3 `ondelete="CASCADE"` → `"RESTRICT"` | `tests/unit -k 'feature or delete or cascade or training'` (211 tests) | ✅ | **SURVIVED** → MIS-E2E-053 |

**3 of 5 survived.** All five edits were confirmed to land before the suite ran, and
`git diff` was verified clean after each restore.

## What the two kills tell us

**M1 and M4 are the honest half of this round.** They confirm that two of R1's
"verified clean" claims hold under mutation rather than merely under reading:

- The calibration **clamp direction** is genuinely protected —
  `TestBudgetInvariantIsBackwardCompatible` has two tests that fail the moment the
  direction inverts.
- The **schema-sync guard genuinely bites**, down to a single description string.
  R1 claimed the published JSON Schemas are byte-in-sync with the pydantic
  contracts; M4 proves the test that claims it would notice if they were not.

That matters because this repo's history is full of guards that pass because their
fixture makes both behaviours identical. These two do not.

## What the three survivals tell us

They are not three independent gaps — they are one gap seen three ways. **Nothing in
the data layer's test suite asserts anything about the shape of the schema or about
what crosses a boundary.** It tests behaviour computed *in Python*, over a schema it
built itself from the ORM.

- M2: the startup schema validator can be emptied and no test notices.
- M3: a field can vanish on import and no test notices.
- M5: every delete rule on the busiest table can be inverted and no test notices.

M5 is the sharpest, because it explains MIS-E2E-033. Three foreign keys are declared
on ORM models and absent from the database. The suite could not have caught that,
because it never exercises a cascade at all — and it builds its own schema from the
very ORM declarations that are wrong.

## Equivalent mutants — deliberately not chased

None encountered this round. M2's removal of a single table is not equivalent (the
validator's output genuinely changes); M5 changes real DDL in the test schema.
