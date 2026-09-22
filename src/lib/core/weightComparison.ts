// Utility functions for summarizing weight changes after a training step

import type {
  WeightComparisonData,
  LayerWeightComparison,
  BackwardSteps,
  BackpropSummaryData,
} from '../types';
import { LAYER_NAMES, type LayerName } from './networkConfig';

function createLayerComparison(
  oldWeights: number[][],
  newWeights: number[][],
  oldBiases: number[],
  newBiases: number[]
): LayerWeightComparison {
  const weightDeltas = oldWeights.map((neuronWeights, i) =>
    neuronWeights.map((w, j) => newWeights[i][j] - w)
  );
  const biasDeltas = oldBiases.map((b, i) => newBiases[i] - b);

  return { oldWeights, newWeights, oldBiases, newBiases, weightDeltas, biasDeltas };
}

/** Creates complete weight comparison data from before/after weights */
export function createWeightComparisonData(
  oldWeights: Record<LayerName, number[][]>,
  newWeights: Record<LayerName, number[][]>,
  oldBiases: Record<LayerName, number[]>,
  newBiases: Record<LayerName, number[]>,
  learningRate: number
): WeightComparisonData {
  const layers = {} as Record<LayerName, LayerWeightComparison>;
  for (const name of LAYER_NAMES) {
    layers[name] = createLayerComparison(oldWeights[name], newWeights[name], oldBiases[name], newBiases[name]);
  }

  const allWeightDeltas = LAYER_NAMES.flatMap(name => layers[name].weightDeltas.flat());
  const allBiasDeltas = LAYER_NAMES.flatMap(name => layers[name].biasDeltas);

  const totalChange = [...allWeightDeltas, ...allBiasDeltas].reduce((sum, d) => sum + Math.abs(d), 0);
  const maxWeightChange = allWeightDeltas.reduce((max, d) => Math.max(max, Math.abs(d)), 0);

  return { ...layers, totalChange, maxWeightChange, learningRate };
}

/**
 * Creates a BackpropSummaryData object from BackwardSteps data.
 * Extracts weights and biases from each layer into a summary format.
 */
export function createBackpropSummaryData(
  backpropData: BackwardSteps,
  learningRate: number
): BackpropSummaryData {
  const oldWeights = {} as Record<LayerName, number[][]>;
  const newWeights = {} as Record<LayerName, number[][]>;
  const oldBiases  = {} as Record<LayerName, number[]>;
  const newBiases  = {} as Record<LayerName, number[]>;
  let totalWeightsUpdated = 0;

  for (const layer of LAYER_NAMES) {
    const neurons = backpropData[layer];
    oldWeights[layer] = neurons.map(n => [...n.oldWeights]);
    newWeights[layer] = neurons.map(n => [...n.newWeights]);
    oldBiases[layer]  = neurons.map(n => n.oldBias);
    newBiases[layer]  = neurons.map(n => n.newBias);
    totalWeightsUpdated += neurons.reduce((sum, n) => sum + n.oldWeights.length, 0);
  }

  return { oldWeights, newWeights, oldBiases, newBiases, learningRate, totalWeightsUpdated };
}
