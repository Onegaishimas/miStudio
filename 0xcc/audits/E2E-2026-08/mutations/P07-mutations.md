# P07 — mutation control log

**Phase:** P07 Frontend state layer · **Round:** 2 · **Date:** 2026-08-23

Frontend harness: `$SCRATCH/fmutate.sh` — same discipline (back up → one edit →
confirm it landed → run vitest → restore → verify `git diff` clean).

| # | Target | Mutation | Landed | Result |
|---|---|---|---|---|
| M22 | `frontend/src/utils/steeringStrength.ts:25` | `BASELINE_SLOPE` 2.6 → 2.4 — the IDL-27 auto-baseline formula | ✅ | **SURVIVED** (75 tests green) → MIS-E2E-127 |

## M22 — the test file exists, has 75 tests, and cannot see the slope

`steeringStrength.test.ts` is a dedicated test file for this formula, and changing
the documented coefficient left it and `steeringStore.test.ts` entirely green. The
reason is in *where* it samples:

```js
// freq 0 → intercept 2.9, below the 3.0 ceiling
expect(computeBaselineStrength(0)).toEqual({ value: 2.9, source: 'auto' });
expect(computeBaselineStrength(0.001).value).toBeLessThanOrEqual(BASELINE_MAX);
// high freq floors at 1.0 (2.9 - 2.6*1 = 0.3 → clamp)
expect(computeBaselineStrength(0.9).value).toBeGreaterThanOrEqual(BASELINE_MIN);
```

- At **freq 0** the slope is multiplied by zero, so the assertion is blind to it.
- At **freq 0.9 and 1.0** the result clamps to the floor, so the slope is irrelevant.
- The remaining assertions are **inequalities against the clamp bounds**, satisfied by
  almost any slope.

Computed, the two slopes are identical everywhere the tests look and differ only in
the mid-range nobody asserts:

```
slope 2.6 | f=0 -> 2.9 | f=0.5 -> 1.60 | f=0.9 -> 1 | f=1 -> 1
slope 2.4 | f=0 -> 2.9 | f=0.5 -> 1.70 | f=0.9 -> 1 | f=1 -> 1
```

The arithmetic that *would* distinguish them appears only in a **comment**
(`2.9 - 2.6*1 = 0.3 → clamp`), where nothing executes it.

This is the recorded J-Lens-arc trap in the other half of the codebase: *a fixture
that makes both behaviours identical*. There, `W_U = torch.eye(...)` was already
unit-norm so the unit-norm fix could be deleted with 63 tests green. Here, every
sample point is one where the slope vanishes or is clamped away.

## Equivalent mutants

None. 2.6 → 2.4 changes real output at every unclamped frequency.
