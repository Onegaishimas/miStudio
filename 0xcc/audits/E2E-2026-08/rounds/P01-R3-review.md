# P01 R3 — verification & closure

**Phase:** P01 Data layer & migrations · **Round:** 3 · **Date:** 2026-08-23

R3 verifies **findings**, not fixes — the audit is record-only, so there are no
fixes to re-review. Each R1/R2 finding was given an independent attempt at
refutation; survivors got a reproduction.

## Reproductions obtained this round

### MIS-E2E-034 — reproduced end to end
Executed against the real contract module and the real published schema file:
```
published schema accepts {intensity:1.0, intensity_range:[8,10]}: YES
pydantic result: intensity = 8.0        (the field declares le=2.0)
re-export FAILS its own published schema: 8.0 is greater than the maximum of 2.0
intensity_range=[-500,-400] -> intensity = -400.0
intensity_range=[1e9,1e9]   -> intensity = 1000000000.0
```
A schema-valid document in, a schema-invalid document out. **CONFIRMED.**

### MIS-E2E-035 — confirmed without touching production
`displayed_rung()` is `circuit_rung([e.rung for e in self.edges])`; `circuit_rung`
returns `min(int(r) for r in edge_rungs)`. Executed: `circuit_rung([3,3,3]) -> 3`.
The MIN aggregation is a *good* conservative choice — it is the values being
aggregated that are self-asserted. Separately confirmed that
`validation_manifest_ref` is **written and never read**: set at
`circuit_service.py:183` and `circuit_intervention_service.py:512`, and neither the
import endpoint nor `circuit_service.py` imports `ValidationManifest` at all.
**CONFIRMED.**

### MIS-E2E-037 — confirmed at source
`CircuitService.create` reads `data.get("calibration")`; the import endpoint's dict
passes every contract field except that one. The service is fully wired to carry it.
**CONFIRMED.**

### MIS-E2E-031 + MIS-E2E-033 — confirmed from both ends
`Base.metadata` is the authority on what `create_all` gives the tests. Queried
directly and compared to `pg_constraint` on the live database:
```
features — ORM (test schema): external_sae_id→CASCADE, extraction_job_id→CASCADE,
                              labeling_job_id→SET NULL, training_id→CASCADE
features — DB  (production):  fk_features_external_sae_id,
                              features_extraction_job_id_fkey
feature_analysis_cache — ORM: unique=NONE   DB: uq_feature_analysis_cache_feature_type
features               — ORM: unique=NONE   DB: uq_features_extraction_neuron
```
**One table's test schema differs from production in both directions at once** — it
has two foreign keys production lacks, and lacks a unique constraint production has.
**Both CONFIRMED.**

## Refutations — attacks that failed, recorded so they are not re-run

- **The naive/tz-aware datetime hypothesis** (R2 attack #3). The sharpest version —
  that `replace(tzinfo=None)` runs before a UTC conversion, so a `+05:00` foreign
  timestamp lands five hours wrong — is **false**. `circuit_service.py:251` reads
  `astimezone(timezone.utc).replace(tzinfo=None)`. Correct order. R1's judgement
  stands.
- **The two `no-constant-binary-expression` lint errors** (MIS-E2E-024 sub-item) are
  **not** defects: `{false && <Component/>}` in a rerender is a deliberate unmount
  assertion. Recorded as REFUTED inside that finding.

## Live verification

Deliberately limited. The three contract findings (034/035/037) were reproducible by
executing the contract modules locally, which is **decisive without writing to
production** — importing a fabricated circuit into the live k8s deployment would
create a real row in a real database. That was not done and is not necessary: the
defect is in the parsing layer, and the parsing layer is the same code.

Live checks that *were* run: `pg_constraint` and `information_schema` queries against
`mistudio-postgres`; `alembic heads`/`current`; all four migration-audit scripts;
`GET /api/v1/version` on `k8s-mistudio.hitsai.local`; the MCP registry via
`mistudio_howto(topic='tools')`.

## Verdict summary for P01

26 findings raised across R1 and R2. Verdicts:

| Verdict | Count | Ids |
|---|---|---|
| **CONFIRMED** | 11 | 031, 032, 033, 034, 035, 037, 048, 049, 051, 052, 053, 054 |
| **PLAUSIBLE** (read-only, strong) | 13 | 029, 030, 036, 038, 039, 040, 041, 042, 043, 044, 045, 046, 047 |
| **REFUTED** | 0 (1 sub-item, inside 024) | — |
| Deferred to another phase | 2 | 050 → P11, and the P00 seeds owned elsewhere |

The PLAUSIBLE set is not weak — most are single-line reads of unambiguous code
(a 32-bit column, a missing `ON CONFLICT`, a header-conditional guard). They are
marked PLAUSIBLE rather than CONFIRMED because no reproduction was run, and under
record-only several of them (a destructive downgrade, a >2 GiB capture) cannot be
reproduced without doing the damage.

## Phase closed

Findings: **26** (MIS-E2E-029 … 054). Mutations: **5 run, 3 survived, 2 killed**
(`mutations/P01-mutations.md`). Tree verified clean after every mutation.

**The one sentence for the synthesis:** the data layer's suite tests behaviour
computed in Python over a schema it builds itself from the ORM — so it cannot see a
constraint, a cascade, or a field that crosses a boundary, and every structural
finding in this phase is a consequence of that.
