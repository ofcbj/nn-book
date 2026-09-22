/**
 * Activation functions and loss.
 *
 * Kept in their own module so that `network.ts` and `backpropagation.ts`
 * can both import them without importing each other.
 */

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/** Derivative of sigmoid, expressed in terms of the output y = sigmoid(x). */
export function dsigmoid(y: number): number {
  return y * (1 - y);
}

export function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max)); // Subtract max for numerical stability
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

/** Cross-entropy loss for a one-hot (or soft) target: L = -Σ tᵢ·log(pᵢ) */
export function crossEntropyLoss(predictions: number[], target: number[]): number {
  const logLikelihood = target.reduce(
    (sum, t, i) => sum + (t > 0 ? t * Math.log(Math.max(predictions[i], 1e-7)) : 0),
    0
  );
  return 0 - logLikelihood; // `0 - x` instead of `-x` so a perfect prediction yields +0, not -0
}
