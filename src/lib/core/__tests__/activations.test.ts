import { describe, it, expect } from 'vitest';
import { sigmoid, dsigmoid, softmax, crossEntropyLoss } from '../activations';

describe('activations', () => {
  it('sigmoid is centred at 0.5 and bounded', () => {
    expect(sigmoid(0)).toBe(0.5);
    expect(sigmoid(50)).toBeCloseTo(1, 10);
    expect(sigmoid(-50)).toBeCloseTo(0, 10);
  });

  it('dsigmoid matches the numerical derivative of sigmoid', () => {
    const x = 0.37;
    const h = 1e-6;
    const numeric = (sigmoid(x + h) - sigmoid(x - h)) / (2 * h);
    expect(dsigmoid(sigmoid(x))).toBeCloseTo(numeric, 8);
  });

  it('softmax is a probability distribution and is shift-invariant', () => {
    const p = softmax([1, 2, 3]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(p[2]).toBeGreaterThan(p[1]);
    expect(p[1]).toBeGreaterThan(p[0]);

    const shifted = softmax([1001, 1002, 1003]); // would overflow without max-subtraction
    shifted.forEach((v, i) => expect(v).toBeCloseTo(p[i], 12));
  });

  it('cross-entropy of a one-hot target is -log(p_target)', () => {
    expect(crossEntropyLoss([0.2, 0.3, 0.5], [0, 0, 1])).toBeCloseTo(-Math.log(0.5), 12);
    expect(crossEntropyLoss([1, 0, 0], [1, 0, 0])).toBe(0);
    // Clamped so a zero probability does not produce Infinity
    expect(Number.isFinite(crossEntropyLoss([0, 1, 0], [1, 0, 0]))).toBe(true);
  });
});
