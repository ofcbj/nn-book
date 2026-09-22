/**
 * Network Snapshot Utilities
 *
 * Deep copies of network weights/biases, used to compare the network
 * before and after a training step.
 */

import type { NeuralNetwork } from './network';
import type { WeightComparisonData } from '../types';
import { Matrix } from './matrix';
import { createWeightComparisonData } from './weightComparison';
import type { LayerName } from './networkConfig';

export interface NetworkSnapshot {
  weights: Record<LayerName, number[][]>;
  /** Column vectors, same shape as the bias matrices */
  biases: Record<LayerName, number[][]>;
}

/** Create a deep-copied snapshot of the current weights and biases */
export function createSnapshot(nn: NeuralNetwork): NetworkSnapshot {
  return {
    weights: {
      layer1: Matrix.cloneRows(nn.weightsInputHidden1.data),
      layer2: Matrix.cloneRows(nn.weightsHidden1Hidden2.data),
      output: Matrix.cloneRows(nn.weightsHidden2Output.data),
    },
    biases: {
      layer1: Matrix.cloneRows(nn.biasHidden1.data),
      layer2: Matrix.cloneRows(nn.biasHidden2.data),
      output: Matrix.cloneRows(nn.biasOutput.data),
    },
  };
}

function flattenBiases(biases: NetworkSnapshot['biases']): Record<LayerName, number[]> {
  return {
    layer1: biases.layer1.map(row => row[0]),
    layer2: biases.layer2.map(row => row[0]),
    output: biases.output.map(row => row[0]),
  };
}

/** Compare two snapshots and create weight comparison data */
export function compareSnapshots(
  oldSnapshot: NetworkSnapshot,
  newSnapshot: NetworkSnapshot,
  learningRate: number
): WeightComparisonData {
  return createWeightComparisonData(
    oldSnapshot.weights,
    newSnapshot.weights,
    flattenBiases(oldSnapshot.biases),
    flattenBiases(newSnapshot.biases),
    learningRate
  );
}
