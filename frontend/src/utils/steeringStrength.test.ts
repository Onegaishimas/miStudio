import { describe, expect, it } from 'vitest';
import {
  BASELINE_MAX,
  BASELINE_MIN,
  DEFAULT_STRENGTH,
  computeBaselineStrength,
} from './steeringStrength';

describe('computeBaselineStrength', () => {
  it('matches the measured optima across the tested frequency range', () => {
    // From experiment c4a273f1 (rounded to 0.1)
    expect(computeBaselineStrength(0.037)).toEqual({ value: 2.8, source: 'auto' });
    expect(computeBaselineStrength(0.214)).toEqual({ value: 2.3, source: 'auto' });
    expect(computeBaselineStrength(0.368)).toEqual({ value: 1.9, source: 'auto' });
    expect(computeBaselineStrength(0.484)).toEqual({ value: 1.6, source: 'auto' });
  });

  it('falls back to the default strength when frequency is unavailable', () => {
    expect(computeBaselineStrength(null)).toEqual({ value: DEFAULT_STRENGTH, source: 'default' });
    expect(computeBaselineStrength(undefined)).toEqual({ value: DEFAULT_STRENGTH, source: 'default' });
    expect(computeBaselineStrength(NaN)).toEqual({ value: DEFAULT_STRENGTH, source: 'default' });
  });

  it('clamps to the [1.0, 3.0] band at the extremes', () => {
    // freq 0 → intercept 2.9, below the 3.0 ceiling
    expect(computeBaselineStrength(0)).toEqual({ value: 2.9, source: 'auto' });
    // very low freq stays under the max
    expect(computeBaselineStrength(0.001).value).toBeLessThanOrEqual(BASELINE_MAX);
    // high freq floors at 1.0 (2.9 - 2.6*1 = 0.3 → clamp)
    expect(computeBaselineStrength(1)).toEqual({ value: BASELINE_MIN, source: 'auto' });
    expect(computeBaselineStrength(0.9).value).toBeGreaterThanOrEqual(BASELINE_MIN);
  });

  it('rounds to one decimal place', () => {
    const { value } = computeBaselineStrength(0.123456);
    expect(Number.isInteger(value * 10)).toBe(true);
  });
});
