/**
 * Network Snapshot Utilities
 *
 * Provides utilities for backing up and restoring network weights/biases.
 * Replaces repetitive JSON.parse(JSON.stringify(...)) patterns.
 */

import type { NeuralNetwork } from './network';
import type { WeightComparisonData } from '../types';
import { createWeightComparisonData } from '../visualizer/weightComparison';

export interface NetworkSnapshot {
  weights: {
    layer1: number[][];
    layer2: number[][];
    output: number[][];
  };
  biases: {
    layer1: number[][];
    layer2: number[][];
    output: number[][];
  };
}

/**
 * Create a snapshot of current network weights and biases
 */
export function createSnapshot(nn: NeuralNetwork): NetworkSnapshot {
  return {
    weights: {
      layer1: JSON.parse(JSON.stringify(nn.weightsInputHidden1.data)),
      layer2: JSON.parse(JSON.stringify(nn.weightsHidden1Hidden2.data)),
      output: JSON.parse(JSON.stringify(nn.weightsHidden2Output.data)),
    },
    biases: {
      layer1: JSON.parse(JSON.stringify(nn.biasHidden1.data)),
      layer2: JSON.parse(JSON.stringify(nn.biasHidden2.data)),
      output: JSON.parse(JSON.stringify(nn.biasOutput.data)),
    },
  };
}

/**
 * Restore network weights and biases from a snapshot
 */
export function restoreSnapshot(nn: NeuralNetwork, snapshot: NetworkSnapshot, inputs?: number[]): void {
  nn.weightsInputHidden1.data = snapshot.weights.layer1;
  nn.weightsHidden1Hidden2.data = snapshot.weights.layer2;
  nn.weightsHidden2Output.data = snapshot.weights.output;
  nn.biasHidden1.data = snapshot.biases.layer1;
  nn.biasHidden2.data = snapshot.biases.layer2;
  nn.biasOutput.data = snapshot.biases.output;

  // Optionally recalculate with restored weights
  if (inputs) {
    nn.feedforward(inputs);
  }
}

/**
 * Compare two snapshots and create weight comparison data
 */
export function compareSnapshots(
  oldSnapshot: NetworkSnapshot,
  newSnapshot: NetworkSnapshot,
  learningRate: number
): WeightComparisonData {
  // Extract flat bias arrays for comparison
  const oldBiases = {
    layer1: oldSnapshot.biases.layer1.map(row => row[0]),
    layer2: oldSnapshot.biases.layer2.map(row => row[0]),
    output: oldSnapshot.biases.output.map(row => row[0]),
  };

  const newBiases = {
    layer1: newSnapshot.biases.layer1.map(row => row[0]),
    layer2: newSnapshot.biases.layer2.map(row => row[0]),
    output: newSnapshot.biases.output.map(row => row[0]),
  };

  return createWeightComparisonData(
    oldSnapshot.weights,
    newSnapshot.weights,
    oldBiases,
    newBiases,
    learningRate
  );
}
