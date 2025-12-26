/**
 * Network Configuration
 * 
 * Central configuration for neural network architecture.
 * Change layer sizes here to update the entire application.
 */

import type { CalculationStage, BackpropStage } from '../types';

// =============================================================================
// Layer Configuration
// =============================================================================

/**
 * Network layer names in order from input to output
 */
export const LAYER_NAMES = ['layer1', 'layer2', 'output'] as const;

/**
 * Type for layer names
 */
export type LayerName = typeof LAYER_NAMES[number];

/**
 * Number of neurons in each layer
 */
export const LAYER_SIZES: Record<LayerName, number> = {
  layer1: 5,
  layer2: 3,
  output: 3,
} as const;

/**
 * Input layer size (number of input features)
 */
export const INPUT_SIZE = 3;

/**
 * Output class count (same as output layer size)
 */
export const OUTPUT_CLASSES = LAYER_SIZES.output;

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Get neuron indices in forward order (0 to size-1)
 */
export function getForwardNeuronIndices(layer: LayerName): number[] {
  const size = LAYER_SIZES[layer];
  return Array.from({ length: size }, (_, i) => i);
}

/**
 * Get neuron indices in backward order (size-1 to 0)
 */
export function getBackwardNeuronIndices(layer: LayerName): number[] {
  const size = LAYER_SIZES[layer];
  return Array.from({ length: size }, (_, i) => size - 1 - i);
}

/**
 * Layers in forward propagation order
 */
export const FORWARD_LAYER_ORDER: readonly LayerName[] = LAYER_NAMES;

/**
 * Layers in backward propagation order
 */
export const BACKWARD_LAYER_ORDER: readonly LayerName[] = [...LAYER_NAMES].reverse() as LayerName[];

// =============================================================================
// Neuron Navigation Functions
// =============================================================================

/**
 * Get next neuron in forward propagation order
 * Returns null when at the last neuron of the output layer
 */
export function getNextForwardNeuron(
  layer: LayerName,
  index: number
): { layer: LayerName; index: number } | null {
  if (index < LAYER_SIZES[layer] - 1) {
    return { layer, index: index + 1 };
  }
  
  // Move to next layer
  if (layer === 'layer1') return { layer: 'layer2', index: 0 };
  if (layer === 'layer2') return { layer: 'output', index: 0 };
  
  return null; // Last neuron in output layer
}

/**
 * Get next neuron in backward propagation order (reverse)
 * Returns null when at the first neuron of layer1
 */
export function getNextBackwardNeuron(
  layer: LayerName,
  index: number
): { layer: LayerName; index: number } | null {
  if (index > 0) {
    return { layer, index: index - 1 };
  }
  
  // Move to previous layer
  if (layer === 'output') return { layer: 'layer2', index: LAYER_SIZES.layer2 - 1 };
  if (layer === 'layer2') return { layer: 'layer1', index: LAYER_SIZES.layer1 - 1 };
  
  return null; // First neuron in layer1
}

// =============================================================================
// Stage Configuration
// =============================================================================

/**
 * Calculation stages for forward propagation animation
 */
export const FORWARD_STAGES: CalculationStage[] = [
  'connections', 'dotProduct', 'bias', 'activation'
];

/**
 * Calculation stages for backpropagation animation
 */
export const BACKPROP_STAGES: BackpropStage[] = [
  'error', 'derivative', 'gradient', 'weightDelta', 'allWeightDeltas', 'update'
];

/**
 * Get next forward stage, or null if at the last stage
 */
export function getNextForwardStage(current: CalculationStage): CalculationStage | null {
  const idx = FORWARD_STAGES.indexOf(current);
  if (idx >= 0 && idx < FORWARD_STAGES.length - 1) {
    return FORWARD_STAGES[idx + 1];
  }
  return null;
}

/**
 * Get next backprop stage, or null if at the last stage
 */
export function getNextBackpropStage(current: BackpropStage): BackpropStage | null {
  const idx = BACKPROP_STAGES.indexOf(current);
  if (idx >= 0 && idx < BACKPROP_STAGES.length - 1) {
    return BACKPROP_STAGES[idx + 1];
  }
  return null;
}
