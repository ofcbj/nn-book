/**
 * Network Configuration
 *
 * Central configuration for the neural network architecture and the
 * neuron/stage navigation used by the animation.
 *
 * NOTE: `LAYER_SIZES` drives matrix shapes, neuron iteration and navigation,
 * but the network still has exactly three processing layers
 * (`layer1`, `layer2`, `output`) and the UI labels assume three output classes.
 */

import type { ForwardStage, BackwardStage } from '../types';

// =============================================================================
// Layer Configuration
// =============================================================================

/** Processing layers in order from input to output */
export const LAYER_NAMES = ['layer1', 'layer2', 'output'] as const;

export type LayerName = typeof LAYER_NAMES[number];

/** Number of neurons in each processing layer */
export const LAYER_SIZES: Record<LayerName, number> = {
  layer1: 5,
  layer2: 3,
  output: 3,
} as const;

/** Input layer size (number of input features) */
export const INPUT_SIZE = 3;

/** Output class count (same as output layer size) */
export const OUTPUT_CLASSES = LAYER_SIZES.output;

// =============================================================================
// Helper Functions
// =============================================================================

/** Neuron indices in forward order (0 to size-1) */
export function getForwardNeuronIndices(layer: LayerName): number[] {
  return Array.from({ length: LAYER_SIZES[layer] }, (_, i) => i);
}

/** Neuron indices in backward order (size-1 to 0) */
export function getBackwardNeuronIndices(layer: LayerName): number[] {
  const size = LAYER_SIZES[layer];
  return Array.from({ length: size }, (_, i) => size - 1 - i);
}

/** Layers in forward propagation order */
export const FORWARD_LAYER_ORDER: readonly LayerName[] = LAYER_NAMES;

/** Layers in backward propagation order */
export const BACKWARD_LAYER_ORDER: readonly LayerName[] = [...LAYER_NAMES].reverse();

/** One-hot encode a class index */
export function toOneHot(classIndex: number): number[] {
  const oneHot = Array<number>(OUTPUT_CLASSES).fill(0);
  oneHot[classIndex] = 1;
  return oneHot;
}

// =============================================================================
// Neuron Navigation Functions
// =============================================================================

export interface NeuronLocation {
  layer: LayerName;
  index: number;
}

/**
 * Next neuron in forward propagation order.
 * Returns null when at the last neuron of the output layer.
 */
export function getNextForwardNeuron(layer: LayerName, index: number): NeuronLocation | null {
  if (index < LAYER_SIZES[layer] - 1) {
    return { layer, index: index + 1 };
  }
  const nextLayer = LAYER_NAMES[LAYER_NAMES.indexOf(layer) + 1];
  return nextLayer ? { layer: nextLayer, index: 0 } : null;
}

/**
 * Next neuron in backward propagation order (reverse).
 * Returns null when at the first neuron of layer1.
 */
export function getNextBackwardNeuron(layer: LayerName, index: number): NeuronLocation | null {
  if (index > 0) {
    return { layer, index: index - 1 };
  }
  const prevLayer = LAYER_NAMES[LAYER_NAMES.indexOf(layer) - 1];
  return prevLayer ? { layer: prevLayer, index: LAYER_SIZES[prevLayer] - 1 } : null;
}

// =============================================================================
// Stage Configuration
// =============================================================================

/** Calculation stages for forward propagation animation */
export const FORWARD_STAGES: readonly ForwardStage[] = [
  'connections', 'dotProduct', 'bias', 'activation'
];

/** Calculation stages for backpropagation animation */
export const BACKPROP_STAGES: readonly BackwardStage[] = [
  'error', 'derivative', 'gradient', 'weightDelta', 'allWeightDeltas', 'update'
];

function nextStage<T>(stages: readonly T[], current: T): T | null {
  const idx = stages.indexOf(current);
  return idx >= 0 && idx < stages.length - 1 ? stages[idx + 1] : null;
}

/** Next forward stage, or null if at the last stage */
export function getNextForwardStage(current: ForwardStage): ForwardStage | null {
  return nextStage(FORWARD_STAGES, current);
}

/** Next backprop stage, or null if at the last stage */
export function getNextBackwardStage(current: BackwardStage): BackwardStage | null {
  return nextStage(BACKPROP_STAGES, current);
}
