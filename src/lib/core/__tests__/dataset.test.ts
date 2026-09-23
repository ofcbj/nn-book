import { describe, it, expect } from 'vitest';
import { NeuralNetwork } from '../network';
import { generateDataset, labelCandidate, toTrainingSample, evaluateDataset, argmax, isValidDataset, isAttitudeCapped, DATASET_SIZE } from '../dataset';
import { OUTPUT_CLASSES } from '../networkConfig';

describe('generated dataset', () => {
  const data = generateDataset();

  it('has exactly the requested size, valid balanced labels and inputs in [0, 1]', () => {
    expect(data.length).toBe(DATASET_SIZE);
    expect(isValidDataset(data)).toBe(true);
    const perClass = Array.from({ length: OUTPUT_CLASSES }, (_, c) => data.filter(d => d.label === c).length);
    expect(Math.max(...perClass) - Math.min(...perClass)).toBeLessThanOrEqual(1);
    for (const d of data) {
      for (const v of [d.grade, d.attitude, d.response]) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      }
      expect(labelCandidate(d.grade, d.attitude, d.response)).toBe(d.label);
    }
    expect(new Set(data.map(d => d.id)).size).toBe(data.length);
  });

  it('is deterministic for the same seed and interleaves classes', () => {
    expect(generateDataset()).toEqual(data);
    expect(generateDataset(60, 1)).not.toEqual(generateDataset(60, 2));
    // Not sorted by class: the first 30 items contain every class
    expect(new Set(data.slice(0, 30).map(d => d.label)).size).toBe(OUTPUT_CLASSES);
  });

  it('labelling rule: thresholds, low-attitude cap and borderline skipping', () => {
    expect(labelCandidate(0.1, 0.1, 0.1)).toBe(0);
    expect(labelCandidate(0.5, 0.5, 0.5)).toBe(1);
    expect(labelCandidate(0.9, 0.9, 0.9)).toBe(2);
    expect(labelCandidate(0.95, 0.1, 0.95)).toBe(1); // would pass, but attitude too low
    expect(labelCandidate(0.8, 0.5, 0.5)).toBeNull(); // score exactly 0.65 → borderline
    expect(labelCandidate(0.9, 0.2, 0.9)).toBeNull(); // right on the attitude cap → borderline
    expect(isAttitudeCapped({ grade: 0.95, attitude: 0.1, response: 0.95 })).toBe(true);
    expect(isAttitudeCapped({ grade: 0.5, attitude: 0.1, response: 0.5 })).toBe(false);
  });

  it('is stratified: adding the three inputs up is not enough to classify it', () => {
    // Best possible 3-class classifier that only looks at grade + attitude + response
    const sums = data.map(d => d.grade + d.attitude + d.response);
    const thresholds = Array.from({ length: 60 }, (_, i) => 0.3 + i * 0.04);
    let best = 0;
    for (const t1 of thresholds) {
      for (const t2 of thresholds) {
        if (t2 <= t1) continue;
        const correct = data.filter((d, i) => (sums[i] < t1 ? 0 : sums[i] < t2 ? 1 : 2) === d.label).length;
        best = Math.max(best, correct / data.length);
      }
    }
    expect(best).toBeLessThanOrEqual(0.86);

    // Same-sum / different-label pairs exist in quantity: count candidates that have a
    // partner with (numerically) the same input sum but another label
    const withContrastPartner = data.filter((d, i) =>
      data.some((o, j) => j !== i && o.label !== d.label && Math.abs(sums[j] - sums[i]) < 1e-6)
    ).length;
    expect(withContrastPartner).toBeGreaterThanOrEqual(data.length * 0.25);

    // Enough attitude-capped candidates for the non-linear part of the rule to be learnable
    expect(data.filter(isAttitudeCapped).length).toBeGreaterThanOrEqual(data.length * 0.1);
  });

  it('argmax picks the largest index', () => {
    expect(argmax([0.1, 0.7, 0.2])).toBe(1);
    expect(argmax([0.5, 0.3, 0.2])).toBe(0);
  });

  it('predict() does not change the stored forward pass', () => {
    const nn = new NeuralNetwork();
    nn.feedforward([0.7, 0.5, 0.8]);
    const before = nn.lastOutput!.toArray();
    nn.predict([0.1, 0.1, 0.1]);
    expect(nn.lastOutput!.toArray()).toEqual(before);
    expect(nn.lastInput!.toArray()).toEqual([0.7, 0.5, 0.8]);
  });

  it('evaluateDataset reports consistent counts', () => {
    const nn = new NeuralNetwork(0.1);
    nn.trainBatch(data.map(toTrainingSample));
    const evaluation = evaluateDataset(nn, data);
    expect(evaluation.total).toBe(data.length);
    expect(evaluation.correctCount).toBe(evaluation.results.filter(r => r.correct).length);
    expect(evaluation.accuracy).toBeCloseTo(evaluation.correctCount / evaluation.total, 12);
  });

  it('the network learns the dataset, including the attitude cap, within a few dozen passes', () => {
    const nn = new NeuralNetwork(0.25);
    const samples = data.map(toTrainingSample);
    const capped = data.filter(isAttitudeCapped);
    const capAccuracy = () => capped.filter(c => argmax(nn.predict([c.grade, c.attitude, c.response])) === 1).length / capped.length;

    let evaluation = evaluateDataset(nn, data);
    let passes = 0;
    while (passes < 40 && !(evaluation.accuracy >= 0.97 && capAccuracy() >= 0.9)) {
      nn.trainBatch(samples);
      evaluation = evaluateDataset(nn, data);
      passes++;
    }
    expect(evaluation.accuracy).toBeGreaterThanOrEqual(0.97);
    expect(capAccuracy()).toBeGreaterThanOrEqual(0.9);
    // A strong candidate with very low attitude is held at "pending", not passed
    expect(argmax(nn.predict([0.95, 0.1, 0.95]))).toBe(1);
  });
});
