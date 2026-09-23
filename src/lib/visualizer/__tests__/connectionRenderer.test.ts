import { describe, it, expect } from 'vitest';
import { weightToStyle, maxAbsWeight } from '../connectionRenderer';

describe('connection weight encoding', () => {
  it('colours by sign', () => {
    expect(weightToStyle(0.5, 1).color).toContain('96, 165, 250');   // blue for +
    expect(weightToStyle(-0.5, 1).color).toContain('251, 113, 133'); // rose for −
  });

  it('thickness and opacity grow with |w| and saturate at the scale', () => {
    const thin = weightToStyle(0.1, 1);
    const thick = weightToStyle(0.9, 1);
    const capped = weightToStyle(5, 1);
    expect(thick.lineWidth).toBeGreaterThan(thin.lineWidth);
    expect(capped.lineWidth).toBe(weightToStyle(1, 1).lineWidth);
    expect(weightToStyle(-0.9, 1).lineWidth).toBe(thick.lineWidth);
  });

  it('never normalises by a scale below 1, so tiny initial weights stay thin', () => {
    expect(weightToStyle(0.05, 0.05).lineWidth).toBe(weightToStyle(0.05, 1).lineWidth);
  });

  it('maxAbsWeight scans the whole matrix', () => {
    expect(maxAbsWeight([[0.1, -0.7], [0.3, 0.2]])).toBe(0.7);
    expect(maxAbsWeight([])).toBe(0);
  });
});
