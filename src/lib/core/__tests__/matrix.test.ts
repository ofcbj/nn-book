import { describe, it, expect } from 'vitest';
import { Matrix } from '../matrix';

describe('Matrix', () => {
  it('multiplies matrices', () => {
    const a = Matrix.fromData([[1, 2], [3, 4]]);
    const b = Matrix.fromData([[5], [6]]);
    expect(Matrix.multiply(a, b).data).toEqual([[17], [39]]);
  });

  it('throws on a shape mismatch instead of returning an empty matrix', () => {
    const a = Matrix.fromData([[1, 2, 3]]);
    const b = Matrix.fromData([[1], [2]]);
    expect(() => Matrix.multiply(a, b)).toThrow(/shape mismatch/);
    expect(() => Matrix.subtract(a, b)).toThrow(/shape mismatch/);
    expect(() => a.add(b)).toThrow(/shape mismatch/);
  });

  it('transposes', () => {
    const m = Matrix.fromData([[1, 2, 3], [4, 5, 6]]);
    const t = Matrix.transpose(m);
    expect(t.rows).toBe(3);
    expect(t.cols).toBe(2);
    expect(t.data).toEqual([[1, 4], [2, 5], [3, 6]]);
  });

  it('in-place multiply supports scalars and element-wise products', () => {
    const m = Matrix.fromData([[1, 2], [3, 4]]);
    m.multiply(2);
    expect(m.data).toEqual([[2, 4], [6, 8]]);
    m.multiply(Matrix.fromData([[1, 0], [0, 1]]));
    expect(m.data).toEqual([[2, 0], [0, 8]]);
  });

  it('clone and fromData produce independent copies', () => {
    const source = [[1, 2], [3, 4]];
    const m = Matrix.fromData(source);
    const c = m.clone();
    c.data[0][0] = 99;
    source[1][1] = 77;
    expect(m.data).toEqual([[1, 2], [3, 4]]);
    expect(c.data[0][0]).toBe(99);
  });

  it('fromArray / toArray round-trip a column vector', () => {
    const m = Matrix.fromArray([0.1, 0.2, 0.3]);
    expect(m.rows).toBe(3);
    expect(m.cols).toBe(1);
    expect(m.toArray()).toEqual([0.1, 0.2, 0.3]);
  });
});
