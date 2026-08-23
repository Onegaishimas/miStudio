# P01 R2 — adversarial re-review

**Phase:** P01 · **Round:** 2 · **Date:** 2026-08-23
**Frame:** *R1 marked these areas clean. Attack that conclusion.*

Mutation controls for this round are logged separately in
`mutations/P01-mutations.md` (5 run, 3 survived, 2 killed).

## Attacks made on R1's "verified clean" list

### 1. "Published schemas are byte-in-sync with the pydantic contracts" — **HOLDS**
Attacked by mutation M4: changed a single `description` string on
`CircuitCalibration.sweet_spot`. `test_published_schema_matches_pydantic_contract`
**failed**. The guard bites at string granularity. R1's claim was verified by
executing the test; R2 verified the test itself can fail. Clean.

### 2. "Alembic graph is sound — single head" — **HOLDS**
Two independent methods agree: the reviewer's corrected parse of all 98
`down_revision` values, and `alembic heads` run directly against the database
(`c7e2a4f18b93 (head)`, and `alembic current` matches). Worth keeping the reviewer's
own note: its **first** pass used a regex that missed `down_revision = 'x'` with
spaces and reported **19 false heads**. A source-scrape that fails on an unexpected
layout, caught in the act — and the reason the claim was re-derived from `alembic`
itself rather than from source text.

### 3. "Naive vs tz-aware `DateTime` is inconsistent but currently safe" — **HOLDS**
R1 explicitly deferred this judgement to R2. Attacked in three steps:

- **Which columns are naive?** The circuits/clusters/manifests family and only it —
  confirmed against the live database: `circuits.created_at`,
  `circuits.updated_at`, `cluster_profiles.*`, `validation_manifests.created_at` are
  `timestamp without time zone`, while `datasets.*`, `features.*`, `trainings.*` are
  `timestamp with time zone`.
- **Are the naive values actually UTC?** Yes. Every one is defaulted by
  `datetime.utcnow`, which is naive-but-UTC. No `datetime.now()` anywhere in these
  models — that would have been the real bug.
- **Does the import path corrupt a foreign offset?** This was the sharpest
  hypothesis: if `replace(tzinfo=None)` runs *before* a UTC conversion, a `+05:00`
  timestamp lands five hours wrong. It does not. `circuit_service.py:251` reads
  `authored.astimezone(timezone.utc).replace(tzinfo=None)` — convert first, then
  strip. Correct order.

**Attack refuted.** Recorded in full so R3 does not re-run it. One residual filed as
MIS-E2E-054: `utcnow()` is deprecated in this Python (3.12.3) at 37 sites, and the
obvious replacement would break these five tables.

### 4. "`ON DELETE RESTRICT` FKs have explicit pre-checks" — **UNDERMINED**
The pre-checks exist, as R1 said. But mutation M5 shows **no test exercises any
delete rule at all** — flipping all three `CASCADE` declarations on the `Feature`
model to `RESTRICT` left 211 tests green (MIS-E2E-053). So the pre-checks are
present and unverified, and the same blindness is what let three declared foreign
keys go missing from the database entirely (MIS-E2E-033). R1's claim is true as
written and much weaker than it reads.

### 5. "`check_type_mismatches.py` is correct" — **HOLDS**
Re-run; 25 id/FK columns, all `character varying` at the declared widths. It is the
only one of the four unowned scripts that both works and claims no more than it
checks.

## New findings this round

| Id | Sev | Source | Claim |
|---|---|---|---|
| MIS-E2E-051 | P2 | M2 | The startup schema validator has **zero** tests and its boolean result is only logged |
| MIS-E2E-052 | P1 | M3 | **Any** field can be dropped on circuit import with the suite green — the systemic cause of MIS-E2E-037 |
| MIS-E2E-053 | P1 | M5 | **No test exercises any delete cascade** — which is why MIS-E2E-033 was invisible |
| MIS-E2E-054 | P3 | attack #3 | `datetime.utcnow()` deprecated, 37 sites; the naive fix breaks five tables |

## The shape of this phase

R1 found defects by reading. R2 found that **the reading could not have been
enough**: the three surviving mutations say the data layer's suite tests behaviour
computed in Python over a schema it builds itself, and asserts nothing about the
schema's shape or about what crosses a boundary. Every R1 finding that involves a
constraint, a cascade or a dropped field is a consequence of that one gap.

Two R1 claims survived direct attack (schema sync, datetime safety), which is worth
stating as plainly as the failures: they are protected, not merely unexamined.
