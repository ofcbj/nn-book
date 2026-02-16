/**
 * Animation Loop Abstraction
 * 
 * Common animation loop logic for forward and backward propagation.
 * Uses Strategy pattern to handle differences between modes.
 */

import type { ForwardStage, BackwardStage, ForwardCalculation, BackwardCalculation } from '../types';

// ============================================================================
// Types
// ============================================================================

import type { LayerName } from '../core';

/** Union type for all animation stages */
export type AnimationStage = ForwardStage | BackwardStage;

/** Union type for all neuron data */
export type NeuronData = ForwardCalculation | BackwardCalculation;

/** Stage durations record */
export type StageDurations = Record<string, number>;

export interface AnimationLoopConfig {
  /** Mode identifier */
  mode: 'forward' | 'backward';

  /** Layers to iterate through (in order) */
  layers: LayerName[];

  /** Get neuron indices to iterate for a layer */
  getNeuronIndices: (layer: LayerName) => number[];

  /** Stages to iterate through for each neuron */
  stages: AnimationStage[];

  /** Duration for each stage (in ms) */
  stageDurations: StageDurations;

  /** Get data for all layers */
  getData: () => Record<LayerName, NeuronData[]> | null;

  /** Called for each stage tick */
  onTick: (layer: LayerName, neuronIndex: number, stage: AnimationStage, data: NeuronData) => void;

  /** Called after visualizer update for each stage */
  onAfterVisualizer?: () => void;

  /** Called when entire animation completes */
  onComplete: () => void;

  /** Check if animation should stop */
  shouldStop: () => boolean;

  /** Sleep function (respects pause state) */
  sleep: (ms: number, speedOverride?: number) => Promise<void>;

  /** Refresh display without recalculation (preferred) */
  refreshDisplayOnly: () => void;

  /** Compute network and refresh display (legacy, kept for compatibility) */
  computeAndRefreshDisplay: () => void;

  /** Speed override for sleep (optional) */
  speedOverride?: number;

  /** Optional: Start from a specific position (for resume after pause/jump) */
  startFrom?: { layer: LayerName; neuronIndex: number };
}

// ============================================================================
// Animation Loop Runner
// ============================================================================

/**
 * Runs the animation loop with the given configuration.
 * Handles forward and backward propagation with a unified loop structure.
 * Supports starting from a specific position (for resume after pause/jump).
 *
 * @returns true if animation completed successfully, false if interrupted
 */
export async function runAnimationLoop(
  config: AnimationLoopConfig
): Promise<boolean> {
  const data = config.getData();
  if (!data) return false;

  const { startFrom, layers } = config;

  // Calculate starting position
  const startLayerIdx = startFrom ? layers.indexOf(startFrom.layer) : 0;

  for (let layerIdx = startLayerIdx; layerIdx < layers.length; layerIdx++) {
    const layer = layers[layerIdx];
    const neuronIndices = config.getNeuronIndices(layer);
    const isStartLayer = startFrom && layerIdx === startLayerIdx;

    for (const neuronIndex of neuronIndices) {
      // Skip neurons before startFrom position
      if (isStartLayer && startFrom) {
        // For forward: skip neurons <= startFrom.neuronIndex
        // For backward: skip neurons >= startFrom.neuronIndex
        const shouldSkip = config.mode === 'forward'
          ? neuronIndex <= startFrom.neuronIndex
          : neuronIndex >= startFrom.neuronIndex;
        if (shouldSkip) continue;
      }

      if (config.shouldStop()) return false;

      const neuronData = data[layer][neuronIndex];

      for (const stage of config.stages) {
        if (config.shouldStop()) return false;

        // Update state machine
        config.onTick(layer, neuronIndex, stage, neuronData);

        // Compute network and refresh display
        config.computeAndRefreshDisplay();
        config.onAfterVisualizer?.();

        // Wait for appropriate duration
        await config.sleep(config.stageDurations[stage] ?? 300, config.speedOverride);
      }
    }
  }

  // Animation complete
  config.onComplete();
  return true;
}

// ============================================================================
// Stage Durations
// ============================================================================

export const FORWARD_STAGE_DURATIONS: Record<ForwardStage, number> = {
  connections: 150,
  dotProduct: 400,
  bias: 400,
  activation: 400,
};

export const BACKWARD_STAGE_DURATIONS: Record<BackwardStage, number> = {
  error: 300,
  derivative: 350,
  gradient: 350,
  weightDelta: 350,
  allWeightDeltas: 400,
  update: 300,
};
