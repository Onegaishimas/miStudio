# Review — Feature 30 (Labeling Prompt-Template Optimization), Round 1 of 3

**Date:** 2026-08-29 · **Scope:** code · **Phases covered:** 1 (schema + defect fixes), 2 (detection scorer)
**Suite at review time:** backend 3651 passed / 27 skipped / 0 failed

> The four persona files in `.claude/agents/` are stale (Prisma/RTK Query/SAML, dated
> 2025-01-15 — MIS-E2E-001). They were NOT loaded as fact; the four perspectives were
> applied to miStudio's actual code.

## Findings verified directly against PostgreSQL

### R1-01 · MEDIUM · `labeling_detection_scorer.py` `_EASY_NEGATIVES_SQL`
`ORDER BY md5(fa.feature_id || :salt || fa.sample_index::text)` concatenates without a
delimiter, so distinct pairs collide. Verified on Postgres:

```
md5('a1' || '' || '23') = md5('a12' || '' || '3')  ->  true
```

Two different (feature, sample) pairs receive the same sort key, biasing which passages are
drawn as easy negatives. Not fatal — the draw is still arbitrary — but it is not the uniform
sample the code claims. **Fix:** insert a separator that cannot occur in an id.

### R1-02 · MEDIUM · `sample_negatives`, silent total failure
`fa.sample_index <> ALL(:exclude_samples)` returns **no rows at all** if the bound array
contains a single NULL. Verified:

```
x <> ALL(ARRAY[1, NULL])   ->  keeps 0 of 3 rows
x <> ALL(ARRAY[]::int[])   ->  keeps 3 of 3 rows   (so the [-1] sentinel is unnecessary)
```

Consequence: zero negatives → every `ba_*` is None → the feature is silently unscored, and a
panel could quietly shrink to nothing while reporting success. **Not reachable from the DB
today** (`feature_activations.sample_index` is NOT NULL in dev and prod, 0 NULL rows), but it
is caller-reachable and the failure is silent. **Fix:** filter `None` out of `exclude_samples`.

### R1-03 · MEDIUM · hard negatives collapse onto a few donors
`_HARD_NEGATIVES_SQL` picks donor features, then orders the union of their passages by
`max_activation DESC`. Measured on a real L46 feature:

```
donors available: 10  |  distinct donors among the 5 negatives drawn: 2
```

The highest-activating donors monopolise the draw, so "hard negatives" test a narrower slice
than intended and the score is noisier than the item count suggests. **Fix:** round-robin
across donors (`ROW_NUMBER() OVER (PARTITION BY feature_id ...)`) before applying the limit.

### R1-04 · MEDIUM (process) · the migration round-trip proved less than claimed
`d7e3a91c04b8` was round-tripped against the dev database — which has **0 rows** in
`labeling_jobs`. Production has **23**. An empty-table round-trip cannot detect a NOT NULL
added without a default, which is precisely the failure that crashlooped this project's
backend once already.

Re-verified in an isolated database using the migration's own DDL, with data present:

| claim | result |
|---|---|
| `ADD COLUMN IF NOT EXISTS` re-run | NOTICE, no error — idempotent |
| delete the labeling job | trial row **survives**, `labeling_job_id` → NULL |
| delete the extraction | trial row cascades away |

The migration is correct. The **test** was weak, and would have stayed green over a real
defect. Recorded as a finding against the verification, not the code.

### R1-05 · LOW · dead sentinel
`exclude = list(exclude_samples) or [-1]` — unnecessary given R1-02's empty-array result.
Harmless; simplify or keep as explicit intent.

## Mutation controls run this round

C1–C3 (Phase 1) and C4–C12 (Phase 2) all verified biting. **Two survived on first run and
were fixed as test findings, not code findings:**

- **C1** initially killed only a source-scrape assertion; the behavioural test rebuilt the job
  id inline and proved nothing about production. Rewritten to drive the real `start_labeling`.
- **C12** survived entirely: a mutation wrapping the prime token in `<<>>` was silently
  cleaned up by a downstream `_MARKER_CHARS` scrubber — **two independent guards masked it**.
  The scrubber was removed (it also mangled legitimate corpus text such as `**bold**`) and
  replaced with an exact-equality assertion, plus a new control C12b for an activation-value
  leak. Both now bite.

This is the second time in this project that a scrubber/fallback has hidden a mutation. The
durable rule: **assert the invariant ("this function adds nothing"), never the cleanup.**

## Carried forward to Round 2

- Verify `run_gate`'s rate/count arithmetic (`parse_failures += rate` then `/ total_batches`).
- `sample_negatives` has no test against a real database — the SQL above was exercised by hand.
- Whether `perfect_judge` makes the harness-leakage assertion vacuous by construction.

---

# Round 1 — agent findings and fixes

Two review agents (correctness/QA, test-quality) plus direct verification. **24 findings.**
All HIGH and MEDIUM findings fixed this round. Suite after fixes: **3678 passed / 0 failed**.

## The three that mattered most

**A correct judge failed my own gate.** With the module's defaults an unshuffled first batch is
all positives; a correct judge answers all-1; `is_degenerate` read that as a broken judge and
refused the panel. Compounding it, the null control's *correct* answer is all-zero (its label
describes nothing present) and that was counted as degeneracy too. Verified live: a judge scoring
1.0 on the literal oracle and 0.5 on the null control was rejected as `judge_degenerate`.
Fixed by giving `is_degenerate` the truth vector, and by measuring degeneracy on the literal
control only. **The gate now passes a capable judge and still rejects an always-1 judge.**

**No judge fixture read the explanation.** Every judge computed its answer from the passages
alone, so removing `{explanation}` from the prompt — which destroys the entire feature — broke
no test, and the gate's two controls produced byte-identical output whichever way round they were
handed. Rebuilt the fixtures around a concept/token split (`locomotion` vs the surface word
`running`) and added an explanation-sensitive judge. Measured:

| label | BA | ba_hard | ba_easy |
|---|---|---|---|
| concept | 1.00 | 1.00 | 1.00 |
| token-only | 0.75 | **0.50** | 1.00 |
| empty | 0.50 | 0.50 | 0.50 |

That table is the feature working: the `ba_easy − ba_hard` gap measures how much of a label's
apparent quality is just naming the surface token. It was previously untested.

**My "real database" tests were silently skipping.** `test_detection_negative_sampling.py` keyed
off a hardcoded production extraction id absent from dev, so all three data tests SKIPPED and
asserted nothing while the suite reported green — the same fail-open shape as a source scrape,
but quieter, because `-q` hides skips. Rebuilt with a hermetic in-transaction fixture. **C13,
C15 and C16 had all been "verified" against nothing; they bite now.**

## Also fixed

| id | finding |
|---|---|
| H1 | `score_panel(gate=None)` returned `scored: True` from an unvetted judge — reachable by forgetting an argument. An absent gate is now a failed gate. |
| H2 | A feature scored from mostly-failed batches still contributed a number: 2 of 3 batches unparseable and the third all-1 gives BA 0.5, indistinguishable from a vague label. Added per-feature coverage floors. |
| H3 | `negative_ceiling` was the module's stated validity argument and was never computed. Now implemented and returned. |
| H6 | `minimum_detectable_effect` returned **0.0** for zero-variance deltas — published as "resolves 0.000 points", a claim of infinite resolution. Returns None. A verdict also required ≥8 overlapping features; a bootstrap over 2 identical deltas certified +0.001 as a win. |
| M2 | `run_gate` summed per-run *rates* and divided by a *run* count, so the smaller literal control was over-weighted and an empty run diluted the rate toward zero. Pools batch counts now. |
| M3 | `judge_reliable` was True for a judge that produced nothing parseable. |
| M5 | `compare_panels` hid asymmetric dropout — a template that gave up on the hardest third won on the easy rest invisibly. Reports `dropped`/`baseline_total`/`candidate_total`. |

## Test defects fixed (found by mutation, not reading)

- **C19 survived**: the coverage-floor test passed because its last batch happened to be
  single-class, not because the floor worked. Items are now interleaved.
- **H4**: `assert len(out.split()) <= 25` was constant-true — every fixture token was one word,
  so `"".join` produced a single word and the whole truncation block could be deleted.
- **H5**: `panel_id_for` never asserted that *different feature sets differ*. Hashing only the
  extraction id passed every existing assert, which would have collapsed every panel in an
  extraction and destroyed `compare`'s ability to refuse a mismatch.
- **M5(test)**: the junk-guard test asserted `!= COMPLETED`, satisfied by LABELING and QUEUED, so
  deleting the FAILED transition survived. Asserts `== FAILED` plus a recorded reason.
- **M7**: model and migration are two independent declarations of the same FK, and the deployed
  database follows the **migration**. Now cross-checked.
- **M9**: the mix-disclosure guard was a four-word blacklist. The prompt's actual contract
  (object shape, count, order, numbering) is pinned directly.

## Mutation controls: 27 run, 27 biting

C1–C27. **Three survived on first run** (C1 behaviourally, C12, C19) and were fixed as *test*
findings. C24 — dropping `{explanation}` — went from killing **0** tests to killing **6**.

## Carried to Round 2

- Neither service module is imported by any production code yet (Phase 4 wiring).
- `DetectionScoringError` declared, never raised.
- `bootstrap_ci` seeding contract untested; `confusion`'s length-mismatch raise untested directly.
- `MIN_FEATURE_ITEMS_SCORED = 4` boundary never approached.
