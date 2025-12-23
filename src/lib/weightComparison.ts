// Utility functions for weight comparison

import type { WeightComparisonData, LayerWeightComparison } from './types';

/**
 * Creates layer weight comparison data
 */
function createLayerComparison(
  oldWeights: number[][],
  newWeights: number[][],
  oldBiases: number[],
  newBiases: number[]
): LayerWeightComparison {
  // Calculate deltas
  const weightDeltas = oldWeights.map((neuronWeights, i) =>
    neuronWeights.map((w, j) => newWeights[i][j] - w)
  );
  
  const biasDeltas = oldBiases.map((b, i) => newBiases[i] - b);

  return {
    oldWeights,
    newWeights,
    oldBiases,
    newBiases,
    weightDeltas,
    biasDeltas,
  };
}

/**
 * Creates complete weight comparison data from before/after weights
 */
export function createWeightComparisonData(
  oldWeights: {
    layer1: number[][];
    layer2: number[][];
    output: number[][];
  },
  newWeights: {
    layer1: number[][];
    layer2: number[][];
    output: number[][];
  },
  oldBiases: {
    layer1: number[];
    layer2: number[];
    output: number[];
  },
  newBiases: {
    layer1: number[];
    layer2: number[];
    output: number[];
  },
  learningRate: number
): WeightComparisonData {
  // Create per-layer comparisons
  const layer1 = createLayerComparison(
    oldWeights.layer1,
    newWeights.layer1,
    oldBiases.layer1,
    newBiases.layer1
  );

  const layer2 = createLayerComparison(
    oldWeights.layer2,
    newWeights.layer2,
    oldBiases.layer2,
    newBiases.layer2
  );

  const output = createLayerComparison(
    oldWeights.output,
    newWeights.output,
    oldBiases.output,
    newBiases.output
  );

  // Calculate total change (sum of absolute deltas)
  const totalChange = 
    [...layer1.weightDeltas, ...layer2.weightDeltas, ...output.weightDeltas]
      .flat()
      .reduce((sum, delta) => sum + Math.abs(delta), 0) +
    [...layer1.biasDeltas, ...layer2.biasDeltas, ...output.biasDeltas]
      .reduce((sum, delta) => sum + Math.abs(delta), 0);

  // Find max weight change
  const maxWeightChange = Math.max(
    ...[...layer1.weightDeltas, ...layer2.weightDeltas, ...output.weightDeltas]
      .flat()
      .map(Math.abs)
  );

  return {
    layer1,
    layer2,
    output,
    totalChange,
    maxWeightChange,
    learningRate,
  };
}
