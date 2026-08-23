# P01 R1 — /security-review

**Phase:** P01 Data layer & migrations · **Round:** 1 · **Date:** 2026-08-23
**Commit:** `408102e` (clean)

## Scope and mechanism

`git diff origin/main...HEAD` and `git diff HEAD` are both empty — this is an audit
of committed code, so the command had nothing to review. Per PLAN.md the phase was
scoped explicitly instead, over `backend/src/models/`, `backend/src/db/`,
`backend/alembic/versions/` (98 files) and `docs/schemas/*.json`. Synthetic
full-file diffs were generated at `$SCRATCH/P01.diff` (5,559 lines) and
`$SCRATCH/P01-migrations.diff` (7,225 lines).

Two independent deep passes ran: one over the published JSON Schemas and their
pydantic contracts, one over the full migration graph.

## Findings (14)

| Id | Sev | Claim |
|---|---|---|
| MIS-E2E-033 | P1 | **The ORM declares three FKs the database does not have** — verified by querying the live DB |
| MIS-E2E-034 | P1 | `intensity` escapes its published `maximum: 2.0` via the unbounded `intensity_range` clamp; schema-valid in → schema-invalid out |
| MIS-E2E-035 | P1 | The evidence **rung is self-asserted** by the imported document, `validation_manifest_ref` is never dereferenced, and the rung gates activation |
| MIS-E2E-037 | P1 | `calibration` is silently dropped on circuit import — the honesty marker goes, the clamped dial stays |
| MIS-E2E-038 | P1 | A cleanup migration removes **every** occurrence of the prime token from context and does not recompute `prime_activation_index`; downgrade is a no-op |
| MIS-E2E-041 | P1 | Plaintext OpenAI keys can persist forever, and `decrypt_value`'s fail-open is what makes the gap undetectable |
| MIS-E2E-036 | P2 | Circuit import size cap skipped when `Content-Length` is absent (cluster endpoint gets this right) |
| MIS-E2E-039 | P2 | A downgrade deletes all user tokenizations under a comment claiming it deletes none |
| MIS-E2E-040 | P2 | A migration drops four data-bearing columns with no copy into their replacements |
| MIS-E2E-042 | P2 | `extra="allow"` on `MemberMeta`/`MemberExample` persists unbounded attacker keys into JSONB and re-exports them |
| MIS-E2E-043 | P2 | The import path bypasses the length guards the create path has, ten lines away |
| MIS-E2E-045 | P2 | Two migrations overwrite system templates with no `is_system` guard, discarding user edits; `downgrade` is `pass` |
| MIS-E2E-046 | P2 | Deleting a training destroys user-curated labels, notes and aqua stars with no preview |
| MIS-E2E-044 | P3 | Published schemas carry `hitsai.net` as their canonical `$id`, on the public mirror |
| MIS-E2E-047 | P3 | Two template-seeding migrations build SQL by f-string with unescaped numerics (repo-controlled source; not exploitable today) |

## A methodological caveat that was discharged

The migration audit ended with: *"this was a static read. I did not execute
`alembic upgrade head` against a scratch database to confirm the constraint names…
If you want §1 nailed down, `\d features` on the production DB settles it."*

It was settled, against the running `mistudio-postgres`:

```
FK: features_extraction_job_id_fkey    ON features
FK: fk_features_external_sae_id        ON features
FK: fk_extraction_jobs_external_sae_id ON extraction_jobs
```

Three FKs declared on the ORM (`features.training_id` CASCADE,
`extraction_jobs.training_id` CASCADE, `features.labeling_job_id` SET NULL);
**zero present in the database**. The same query confirmed
`uq_feature_analysis_cache_feature_type` and `uq_features_extraction_neuron` **do**
exist in the DB while being absent from the ORM — the exact inverse, and the reason
MIS-E2E-031 has no failing test. MIS-E2E-033 is therefore CONFIRMED, not plausible.

## Verified clean — R2 must attack these, not re-derive them

Recorded deliberately. This repo's history says a "verified clean by reading" entry
is where the next round finds the bug.

**Migration graph.** All 98 `down_revision` values parsed programmatically: exactly
one root (`118f85d483dd`), exactly one head (`c7e2a4f18b93`), one branch point at
`l8m9n0o1p2q3` properly rejoined by `cd6c46abac48`. No missing parents, no orphans.
*Method note worth keeping:* the reviewer's first pass used a regex that missed
`down_revision = 'x'` with spaces and reported **19 false heads**. The corrected run
is what stands — a live instance of "a source-scrape guard fails open on an
unexpected layout".

**Security-relevant `server_default`s.** Every boolean defaulting `true` is a
content-filter or prompt-formatting flag, never an access gate, and `true` is the
conservative direction for all of them. Every actual gate defaults closed:
`circuits.promoted` false, `agent_approval_requests.status` `'pending'`,
`app_settings.is_sensitive` false, `save_poor_quality_labels` false,
`save_requests_for_testing` false, `auto_retried` false, `stale` false. No
`dry_run`/`enabled`/`allow`/`public`/`bypass`/`approved`/`validated` column exists
in any migration.

**SQL injection at migration time.** Every f-string interpolated into SQL takes its
value from a hardcoded literal tuple or a loop counter (`range(16)` partition
names, a literal 4-tuple of column names). Every data-bearing write with a variable
uses bound parameters. The sole exception is the template-seeding pair, filed as
MIS-E2E-047.

**`alter_column` type narrowing.** All 60+ calls read. Every upgrade-path change
**widens**. No `Text`→`String(n)`, no `BigInteger`→`Integer`, no `JSON`→`String`
anywhere. The only narrowings are in downgrade bodies, where Postgres errors rather
than truncating.

**Ciphertext→plaintext conversion.** No migration converts an encrypted column to
plaintext or backfills plaintext over ciphertext. `app_settings.value` is created
`Text` and stays `Text`; the AES-GCM envelope is entirely service-layer. The
historical defect this project recorded (the upsert committing a masked display
string over ciphertext) was app-layer and is not present in any migration. The gap
is the *absence* of a backfill — MIS-E2E-041.

**FK/ondelete ORM-vs-migration comparison: 22 of 24 pairs agree.** The two that do
not are MIS-E2E-033. Notably safe by design: `circuits`, `validation_manifests`,
`steering_record_runs`, `neuronpedia_export_jobs`, `dismissed_operations` and
`agent_approval_requests` carry **no foreign keys at all** — evidence-bearing
artifacts use soft references (`models/circuit.py:59` is commented *"SOFT ref —
runs are prunable"*), so no wide delete can cascade into causal-validation evidence.
That is a deliberate and correct decision.

**Contract defences that do bite.** Edge direction (`up.layer < down.layer`), edge
endpoints must reference declared members, per-layer member cap, duplicate-feature
and duplicate-SAE rejection, calibration `onset ≤ sweet_spot ≤ cliff`,
`neuronpedia` http(s)-only, `kind`/`schema_version` as `const` with explicit
pre-validation rejection, `_validate_bounds` rejecting `feature_idx >= n_features`
against the bound SAE. `ProfileMember.strength` `[-300,300]`, `sign` enum `{1,-1}`,
`EvidenceRung` enum `{0,1,2,3}`, array caps (edges ≤200, saes ≤16, members ≤20,
bundle ≤50, top_tokens ≤10).

**Path traversal via document ids: not live.** All four `get_sae_storage_path`
callers traced; none receives an imported definition's id. The import path uses the
document's id only as a dict key and binds to a local row id. `hook_type`,
`mistudio_sae_id` and friends accept `"../../../../etc/passwd"` and an embedded NUL,
but nothing joins them to a path. Hardening concern for external consumers, not a
miStudio vulnerability. One partial guard worth noting: `source_hint`'s
`no_local_paths` validator only rejects a **leading** `/`, `~`, `..` or `:\`, so
`"hf:../../../../.."` passes — and that field exists to be dereferenced.

**Info-leakage sweep of the published schemas.** Regex over both files for
`hitsai.local`, `mcslab`, RFC1918 ranges, `localhost`, `/home/`, `/data/`, emails,
`api_key`/`token`/`secret`/`password`/`bearer`, `hf_*`, `sk-`: **only `hitsai.net`
in `$id` matched** (MIS-E2E-044). No `examples` key exists in either file; every
`default` is a neutral constant; no real model or dataset names anywhere.

**Top-level extra keys are not exploitable in the importer.** `extra="ignore"`
verified to drop them on `ClusterDefinitionV1`, `ProfileMember` and all
absent-`additionalProperties` models; nothing does `Model(**doc)`;
`from_definition` stores only enumerated fields. The exposure is confined to the two
`extra="allow"` models — MIS-E2E-042.

**miLLM-side containment (verified from this side, per the audit's repo boundary).**
`millm/core/steering_range.py:declared_intensity_range` intersects any authored
range with `[0,2]`, normalizes swapped pairs, degrades malformed content to `None`,
and NaN is guarded at `millm/ml/circuit_steering.py:400-410`. This is why
MIS-E2E-034 does not cause overdrive in production. Recorded for MILLM-HANDOFF.md.

## Not covered

CASCADE behaviour under a real delete on live data (P12), and whether the four
unowned migration-audit scripts (MIS-E2E-022) would have caught any of the above.
