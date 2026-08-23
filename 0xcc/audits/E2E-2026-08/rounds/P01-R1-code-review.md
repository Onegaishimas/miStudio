# P01 R1 — /code-review high

**Phase:** P01 Data layer & migrations · **Round:** 1 · **Date:** 2026-08-23
**Command:** `/code-review high backend/src/models backend/src/db docs/schemas`
**Commit:** `408102e` (clean)

## Scope

`backend/src/models/` (33 ORM files, 3,083 lines), `backend/src/db/` (2 files,
213 lines), `docs/schemas/` (2 published JSON Schemas). `backend/alembic/versions/`
(98 revisions) reviewed as graph structure rather than file-by-file.

**Scope note recorded by the command:** `git diff origin/main...HEAD` and
`git diff HEAD` are both empty, so it reviewed the target paths as-is. This is
the expected mode for this audit — see PLAN.md, "Making /security-review work on
committed code".

## Findings (4)

| Id | Sev | Claim |
|---|---|---|
| MIS-E2E-029 | P1 | `circuit_runs.bytes_total` is 32-bit; a >2 GiB capture store overflows on the final commit, poisons the session so the error handler also fails, and leaks the store |
| MIS-E2E-030 | P1 | `_cache_analysis` blind-INSERTs against a unique constraint; after expiry or under concurrency, logit lens / correlations / ablation 500 **permanently** for that feature |
| MIS-E2E-031 | P1 | Two unique constraints exist in migrations but not on the ORM models, so `create_all` gives the unit suite a schema **without** them — which is why 030 has no failing test |
| MIS-E2E-032 | P1 | `REQUIRED_TABLES` covers 15 of 36 tables and logs "Schema validation passed" on a database missing `circuits`, `steering_record_runs`, `dismissed_operations`… |

The 029/030/031 cluster is the interesting one: 031 is the reason 030 is invisible
to the suite. A test written for `_cache_analysis` today would pass against a
schema that lacks the constraint the production database enforces.

## Verified clean (recorded so R2 knows what to attack)

Round 2 must treat every line below as a claim to be refuted, not as settled.

- **Published schemas are in sync with the pydantic contracts.** Both
  `test_circuit_definition_schema_sync.py` and `test_cluster_definition_schema_sync.py`
  were executed, not just read. The seven `$defs` shared between
  `circuit-definition-v1.json` and `cluster-definition-v1.json` are identical —
  no mirror drift.
- **Alembic graph is sound.** 98 revisions, single head `c7e2a4f18b93`, no
  dangling `down_revision`, exactly one branch and it merges at `cd6c46abac48`.
- **Enum conventions agree** across models, migrations and the conftest fixture
  (`modelstatus`, `extractionstatus`, `label_source_enum`, `analysis_type_enum`).
- **`ON DELETE RESTRICT` FKs have explicit pre-checks** rather than surfacing raw
  `IntegrityError`s — `cluster_profiles.sae_id`, `trainings.model_id`,
  `labeling_jobs.prompt_template_id`.
- **Naive vs tz-aware `DateTime` inconsistency is currently safe.** Circuits,
  clusters and manifests use naive columns while the rest are tz-aware; the
  Python comparison helper normalises, the import path strips `tzinfo`, and
  Postgres defaults to UTC. Inconsistent, not broken. **Deliberately not filed** —
  R2 should decide whether that judgement holds.

## Not covered by this command

Contract-IDL conformance (IDL-30/33/37 field-by-field against the ORM and the
migration that created each column), CASCADE semantics under real deletes, and
the four unowned migration-audit scripts (MIS-E2E-022). Those belong to
`/review` and the doc-conformance pass in this same round.
