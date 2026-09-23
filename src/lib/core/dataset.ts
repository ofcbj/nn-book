/**
 * Demo training data for "data training mode".
 *
 * 500 candidates: measured with the default learning rate, one pass reaches
 * ~96% accuracy and a second pass reliably reaches ~100%, so the rise is
 * visible within a single run through the list.
 *
 * Candidates are generated from a hidden rule the network never sees:
 *   score = 0.5·grade + 0.2·attitude + 0.3·response
 *   pass    if score ≥ 0.65 (but a very low attitude caps the result at "pending")
 *   fail    if score <  0.40
 *   pending otherwise
 * Borderline scores are skipped so every example is clearly one class, and the
 * three classes are balanced. The sample is stratified (see generateDataset) so
 * that adding the three inputs up is not enough to classify it. Generation is
 * seeded, so the list is the same on every load and the accuracy curve is
 * reproducible.
 */

import type { NeuralNetwork, TrainingSample } from './network';
import { toOneHot, OUTPUT_CLASSES } from './networkConfig';
import { crossEntropyLoss } from './activations';

export interface Candidate {
  id: number;
  grade: number;
  attitude: number;
  response: number;
  /** Class index: 0 = fail, 1 = pending, 2 = pass */
  label: number;
}

export const DATASET_SIZE = 500;
const DATASET_SEED = 20260922;
const PASS_THRESHOLD = 0.65;
const FAIL_THRESHOLD = 0.40;
const BORDER_MARGIN = 0.08;
const LOW_ATTITUDE = 0.2;
const ATTITUDE_MARGIN = 0.03;
/** Share of the dataset made of same-sum / different-label pairs */
const CONTRAST_SHARE = 0.3;
/** Share of the dataset made of attitude-capped "pending" candidates */
const CAP_SHARE = 0.15;

/** Small deterministic PRNG (mulberry32) */
function createRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Weighted score behind the hidden rule */
export function candidateScore(grade: number, attitude: number, response: number): number {
  return 0.5 * grade + 0.2 * attitude + 0.3 * response;
}

/** The hidden labelling rule; returns null for borderline inputs that should be skipped */
export function labelCandidate(grade: number, attitude: number, response: number): number | null {
  const score = candidateScore(grade, attitude, response);
  if (Math.abs(score - PASS_THRESHOLD) < BORDER_MARGIN || Math.abs(score - FAIL_THRESHOLD) < BORDER_MARGIN) {
    return null;
  }
  if (score < FAIL_THRESHOLD) return 0;
  if (score >= PASS_THRESHOLD) {
    // The attitude cap is a second, sharp boundary; skip candidates sitting right on it
    if (Math.abs(attitude - LOW_ATTITUDE) < ATTITUDE_MARGIN) return null;
    return attitude < LOW_ATTITUDE ? 1 : 2;
  }
  return 1;
}

/** True when the candidate would pass on score alone but is held at "pending" by the attitude cap */
export function isAttitudeCapped(c: Pick<Candidate, 'grade' | 'attitude' | 'response'>): boolean {
  return candidateScore(c.grade, c.attitude, c.response) >= PASS_THRESHOLD && c.attitude < LOW_ATTITUDE;
}

type Stats = Omit<Candidate, 'id'>;

const PERMUTATIONS: ReadonlyArray<readonly [number, number, number]> = [
  [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0],
];

/**
 * Generate a reproducible dataset of exactly `count` candidates, balanced across
 * classes (±1) and interleaved. The sample is stratified so the network cannot
 * get away with a shortcut:
 *  - CONTRAST_SHARE: pairs with the *same* input sum but different labels
 *    (a permutation of the same three numbers), so "add the three values up"
 *    is not enough and the network must learn how much each input matters;
 *  - CAP_SHARE: strong candidates held at "pending" by the low-attitude cap,
 *    the one non-linear part of the rule, so the hidden layers have a job;
 *  - the rest is uniform random, filling up each class to its quota.
 */
export function generateDataset(count: number = DATASET_SIZE, seed: number = DATASET_SEED): Candidate[] {
  const rng = createRng(seed);
  const round = (v: number) => Math.round(v * 100) / 100;
  const uniform = (lo: number, hi: number) => round(lo + rng() * (hi - lo));
  const randomStats = (): Stats | null => {
    const grade = uniform(0.05, 0.95);
    const attitude = uniform(0.05, 0.95);
    const response = uniform(0.05, 0.95);
    const label = labelCandidate(grade, attitude, response);
    return label === null ? null : { grade, attitude, response, label };
  };

  const base = Math.floor(count / OUTPUT_CLASSES);
  const remainder = count % OUTPUT_CLASSES;
  const quota = Array.from({ length: OUTPUT_CLASSES }, (_, c) => base + (c < remainder ? 1 : 0));
  const counts = Array<number>(OUTPUT_CLASSES).fill(0);
  const accepted: Stats[] = [];
  const accept = (c: Stats): boolean => {
    if (counts[c.label] >= quota[c.label]) return false;
    counts[c.label]++;
    accepted.push(c);
    return true;
  };

  const maxAttempts = count * 400;
  let attempts = 0;

  // 1. Attitude-capped candidates (all "pending")
  const capTarget = Math.round(count * CAP_SHARE);
  let caps = 0;
  while (caps < capTarget && attempts++ < maxAttempts) {
    const attitude = uniform(0.05, LOW_ATTITUDE - ATTITUDE_MARGIN);
    const grade = uniform(0.55, 0.95);
    const response = uniform(0.55, 0.95);
    const label = labelCandidate(grade, attitude, response);
    if (label !== 1 || !isAttitudeCapped({ grade, attitude, response })) continue;
    if (accept({ grade, attitude, response, label })) caps++;
  }

  // 2. Same-sum contrast pairs: a random candidate and a permutation of it with another label
  const pairTarget = Math.floor((count * CONTRAST_SHARE) / 2);
  let pairs = 0;
  while (pairs < pairTarget && attempts++ < maxAttempts) {
    const a = randomStats();
    if (!a) continue;
    const values = [a.grade, a.attitude, a.response];
    const partners: Stats[] = [];
    for (const [i, j, k] of PERMUTATIONS) {
      const label = labelCandidate(values[i], values[j], values[k]);
      if (label !== null && label !== a.label) {
        partners.push({ grade: values[i], attitude: values[j], response: values[k], label });
      }
    }
    if (partners.length === 0) continue;
    const b = partners[Math.floor(rng() * partners.length)];
    if (counts[a.label] >= quota[a.label] || counts[b.label] >= quota[b.label]) continue;
    accept(a);
    accept(b);
    pairs++;
  }

  // 3. Uniform random fill up to the class quotas
  while (accepted.length < count && attempts++ < maxAttempts) {
    const c = randomStats();
    if (c) accept(c);
  }

  // Fisher–Yates shuffle so classes and strata are interleaved
  for (let i = accepted.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [accepted[i], accepted[j]] = [accepted[j], accepted[i]];
  }

  return accepted.map((c, i) => ({ id: i + 1, ...c }));
}

export function candidateInputs(c: Candidate): number[] {
  return [c.grade, c.attitude, c.response];
}

export function toTrainingSample(c: Candidate): TrainingSample {
  return { inputs: candidateInputs(c), target: toOneHot(c.label) };
}

/** Index of the largest value (the predicted class) */
export function argmax(values: number[]): number {
  return values.reduce((best, v, i) => (v > values[best] ? i : best), 0);
}

export interface CandidateEvaluation {
  candidate: Candidate;
  probabilities: number[];
  predicted: number;
  correct: boolean;
  loss: number;
}

export interface DatasetEvaluation {
  results: CandidateEvaluation[];
  correctCount: number;
  total: number;
  /** correctCount / total, in [0, 1] */
  accuracy: number;
  meanLoss: number;
}

/** Evaluate the network on a dataset without changing its state */
export function evaluateDataset(nn: NeuralNetwork, data: readonly Candidate[]): DatasetEvaluation {
  const results = data.map(candidate => {
    const probabilities = nn.predict(candidateInputs(candidate));
    const predicted = argmax(probabilities);
    return {
      candidate,
      probabilities,
      predicted,
      correct: predicted === candidate.label,
      loss: crossEntropyLoss(probabilities, toOneHot(candidate.label)),
    };
  });
  const correctCount = results.filter(r => r.correct).length;
  const total = results.length;
  return {
    results,
    correctCount,
    total,
    accuracy: total > 0 ? correctCount / total : 0,
    meanLoss: total > 0 ? results.reduce((s, r) => s + r.loss, 0) / total : 0,
  };
}

/** Sanity check used by tests: every label is a valid class index */
export function isValidDataset(data: readonly Candidate[]): boolean {
  return data.every(c => c.label >= 0 && c.label < OUTPUT_CLASSES);
}
