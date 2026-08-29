# Review — Feature 30, Round 2 of 3

**Date:** 2026-08-29 · **Scope:** the Round 1 FIXES
**Suite after:** backend **3692 passed / 27 skipped / 0 failed**

Round 2 existed to re-review Round 1's fixes, on this project's standing rule that *a fix is the
most dangerous code in a review*. That rule paid: **the headline Round 1 fix did not work, and one
part of it made the failure less visible than the bug it replaced.**

## The finding that matters most — a fix that made things worse

Round 1 added `MIN_FEATURES_FOR_VERDICT = 8` to stop `compare_panels` certifying noise, and made
`minimum_detectable_effect` return `None` instead of `0.0` at zero variance.

Neither addressed the actual defect. A percentile bootstrap over *identical* deltas reproduces them
exactly, so the interval never straddles zero **at any n**. Verified at exactly the new threshold:

```
compare_panels({f0..f7: 0.500}, {f0..f7: 0.501})
  -> verdict: candidate_better
     mean_delta: 0.001
     minimum_detectable_effect: None
     reason: None
```

Before Round 1 that branch at least published `mde: 0.000`, which reads as obviously wrong. The
zero-variance fix replaced it with `None`, and the `candidate_better` branch sets `reason = None` —
so the record became *a win, with no resolution estimate and no explanation*. **The fix removed the
only visible signal from the exact branch where the number is bogus.** The regression test pinning
it used n=2, so it pinned the threshold rather than the property; adding six features to that same
fixture leaves it green while certifying +0.001 as a win.

**Fix:** an absolute floor, `MIN_MEANINGFUL_DELTA = 0.02`, plus a requirement that the effect exceed
the panel's own resolution when that is estimable. Both sides now behave:

| deltas | verdict |
|---|---|
| uniform +0.001 | `indistinguishable`, with a reason |
| uniform +0.400 (also zero variance) | `candidate_better` — a real result is not suppressed |
| noisy +0.15 over 20 features | `candidate_better` |

## Other HIGH findings

**A hand-built `{"passed": True}` authorised scoring.** `gate.get("passed")` was the whole check —
no structural validation, and no binding to the ruler the gate was measured under, so a gate cached
from `detection/v0` authorised scoring under `v1` while the result claimed v1 provenance. Worse,
**my own Round 1 regression test gated with `coin_judge` and scored with `perfect_judge`** — the test
written to pin the fix demonstrated the API's most dangerous affordance. Now validated, and the
ruler version must match.

**`MIN_FEATURE_ITEMS_SCORED = 4` certified nothing.** 3 positives + 1 negative passes a bare count
floor, but TNR from a single item is quantised to {0,1}, so one judgement swings balanced accuracy
by 0.5 — entering an unweighted panel mean at the same weight as a 20-item feature. Added
`MIN_ITEMS_PER_CLASS = 3`.

## MEDIUM findings

| id | finding |
|---|---|
| M-1 | A structurally thin literal control was diagnosed `judge_unparseable` **while the same dict reported `parse_failure_rate: 0.0`** — a self-contradictory verdict that sent the operator to buy a bigger model over a negative-sampling problem. New `control_unscorable` failure names the real cause. |
| M-2 | The degeneracy denominator counted unparsed batches as evidence of *non*-degeneracy — **the identical dilution the parse-rate fix three lines below had just removed**. Measured: 3 degenerate of 7 parsed (0.43, over threshold) reported as 3/10 = 0.30, which the strict `>` let through. |
| M-3 | A single-class surviving subset returned `balanced_accuracy: None` with `reason: None` — the one refusal in the module that carried no explanation. |
| M-4 | `is_degenerate`'s optional `truth` protected **zero** callers and re-armed the exact bug it was added to fix: the natural call `is_degenerate(preds)` silently rejects correct judges. Now required. |
| M-6 | The `donors` CTE had `LIMIT` with no `ORDER BY`, so PostgreSQL could return a different donor subset after a plan change — contradicting the reproducibility claim two trials depend on. |

## Test defects found in Round 1's own tests

- **C36 survived twice.** My first degeneracy test asserted the *fixture's shape*, not the rate the
  gate computes. The second attempt used a shape where both denominators cleared the threshold, so
  the mutation still survived. Only a fixture giving exactly 10 literal batches with 3 unparsed —
  where 3/7 fires and 3/10 does not — makes it bite. **Two failed attempts at one control.**
- **C14 was vacuous** (reported, fix carried to Round 3): the test builds its own SQL literals and
  never references `_EASY_NEGATIVES_SQL`, so deleting the separators leaves it green.
- `test_both_degenerate_judges_score_exactly_chance` was implicitly coupled to the fixture dividing
  into exactly one batch.

## Verified clean (attacked, no defect)

`negative_ceiling` is now produced *and* read. `score_feature` / `score_panel` / `compare_panels`
key parity holds across every branch. `assemble_items` does not alias its inputs. `fa.id` is unique
within its window partition. The 0x1F separator survives psycopg2 binding as a real byte. The
hermetic fixture's activation-scale claim was computed and holds with margin.

## Mutation controls: 37 run, 37 biting

C1–C37. Three needed more than one attempt (C1, C12, C19 in Round 1; C36 twice in Round 2).

## Carried to Round 3

- **C14 is vacuous** — the separator test does not execute the module's SQL.
- `assemble_items` guarantees a mixed batch only probabilistically: measured **1.42%** single-class
  at a 10/1/1 ratio, and because the seed is a pure function of `(panel_id, feature_id)` it fails
  **deterministically forever** for that feature.
- The round-robin tiebreak always draws the 5 highest-activating donors; 15 of 20 are never sampled.
- `keyword_judge`'s semantics reduce to the substring `"foot"`; the stop-list is 11/18 redundant.
- The hermetic DB fixture uses fixed ids in a shared database (concurrency / abnormal-exit hazard).
- Neither service module is imported by production code — Phase 4 wiring.
