# Review — Feature 30, Round 3 of 3

**Date:** 2026-08-29 · **Scope:** the Round 2 fixes, plus Phases 3–4 as landed
**Suite after:** backend **3718 passed / 27 skipped / 0 failed**

## The headline: a fix made things worse, for the third round running

Round 2 replaced `assemble_items`'s plain shuffle with deterministic largest-remainder
interleaving, to stop single-class batches. It worked — and introduced a strictly worse defect.

The within-class order stayed seeded, but **the class pattern became a pure function of
`(n_pos, n_neg)`**. At the module's own defaults (10 positives, 5+5 negatives, batch 10), *every
batch of every feature of every panel of every trial* carried ground truth:

```
[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
```

A judge with any alternation prior scores balanced accuracy **1.0 on every label under every
template**, and two templates that genuinely differ come back `indistinguishable`. The old shuffle
made the pattern unguessable; the fix made it a constant.

**Fix:** stratify ACROSS batches, shuffle WITHIN each. Both properties now hold and are pinned:

| ratio | single-class batches / 300 seeds |
|---|---|
| 10/5/5, 10/1/1, 5/1/1, 20/2/2, 3/3/3, 25/3/3 | **0** |

and 39 distinct truth patterns across 40 features (1 would mean guessable).

An intermediate version failed too: equal-sized buckets over 12 items with batch_size 10 gave 6+6
while the consumer sliced 10+2, and the 2-item tail was single-class. Buckets must match how the
consumer slices, and the minority class must be dealt first — filling proportionally rounds a small
bucket's minority share to zero.

## Other HIGH findings

**The gate accepted a truthy non-boolean `passed`.** Validation ran only when `gate.get("passed")`
was truthy, and never checked the type or consistency. A gate round-tripped through JSON as
`passed: "false"` — truthy in Python — carrying
`failures: ["judge_unreliable", "harness_leakage"]` **authorised scoring**, and the result echoed
the failure list it had just ignored. Shape and provenance are now validated unconditionally;
`passed` must be a bool and must not coexist with failures.

**The MDE was being used as a significance test.** `abs(mean_delta) < mde` sat *before* the interval
checks. MDE is 2.80 SE while a 95% interval excludes zero at ~1.96 SE, so every genuine effect in
that 43% band was discarded. Verified: **+0.046 with a 95% interval of [0.011, 0.082] was reported
`indistinguishable`.** The MDE is now reported, never used to decide.

**`MIN_MEANINGFUL_DELTA` relocated rather than removed the Round-2 defect** — it certified +0.021
with `mde: None, reason: None`. The root cause is that a bootstrap over identical deltas yields a
point interval that excludes zero by construction. Zero-variance cases now take a separate branch
that states the caveat in the record instead of leaving `reason` empty.

## MEDIUM

| id | finding |
|---|---|
| M6 | `judge_reliable` — an affirmative claim — was True for a judge that answered uniformly on 40% of the batches where it said anything. `judge_degenerate` now counts against it; `harness_leakage` deliberately does not, since it indicts the harness. |
| M12 | The migration-agreement guard was `col in sql`, fail-open for **8 of 15** columns whose names also appear in indexes, FKs, `COMMENT ON` or the docstring. Deleting `ADD COLUMN IF NOT EXISTS mode VARCHAR(16),` left it green while `upgrade()` would die on the comment referring to that column. Now anchored to a column definition. |
| M9 | The donor pool is still the 20 lexicographically-first ids. Determinism was the right call, but `ORDER BY feature_id` is a biased deterministic sample. Recorded; the md5 tiebreak fixed the final selection, not the pool. |
| L3 | `run_gate([])` diagnosed `control_unscorable` for a control that did not exist, having never called the judge. Refuses now. |
| M4/M5 | `MIN_FEATURE_ITEMS_SCORED` was logically subsumed by `MIN_ITEMS_PER_CLASS` and could never be decisive; two refusal blocks below it had become unreachable. Removed. |

## A precondition the fixture hunt exposed

`run_gate`'s degeneracy check is **inert on grouped control items**. Measured: a judge answering
all-1 on 40% of literal batches reported `degenerate_rate 0.0` on grouped items and `0.4` on the
same items assembled. Controls must come from `assemble_items`; now documented in the function.

## Mutation controls: 48 run, 48 biting

C1–C48. **Nine needed more than one attempt** (C1, C12, C19, C36 ×2, C41b, C46, C47, C14 ×2).
Notable failures of the controls themselves this round:

- **C41b applied to the wrong function.** A `count=1` replace matched `update_feature_label`'s
  identical line first, broke it, and nothing went red — which revealed that the three
  **pre-existing** MCP tools had no payload assertion at all. Now covered.
- **C46 and C47 were no-ops on their fixtures.** C46's judge tripped *both* failure modes so
  dropping one was invisible; C47's effect sat above the MDE so the mutation could not change the
  verdict. Both fixtures rebuilt to land in the discriminating region.
- **C14 twice.** First it never executed the module's SQL; then it used a non-empty salt, which is
  itself a separator, so the guard could not fail. It bites only with an empty salt — which also
  established that the separator guards a *caller*, not today's call site.

## Phases 3–4, as landed

`labeling_trial_service.py` (panel resolution, frozen template copy, non-persisting runner with a
`Session.dirty` guard, comparison), the trial Celery task (fully-qualified name → `processing`
queue, verified via the live router), four REST operations (verified in the OpenAPI schema), and
**five MCP tools verified through a real `build_server`**, absent when the category is disabled,
each with method, path, call-count and payload asserted.
