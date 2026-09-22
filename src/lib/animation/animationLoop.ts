/**
 * Animation Loop Abstraction
 *
 * Common loop for forward and backward propagation animations.
 * Iterates layers -> neurons -> stages, calling `onTick` for each stage.
 */

import type { ForwardStage, BackwardStage, ForwardCalculation, BackwardCalculation } from '../types';
import type { LayerName } from '../core';

// ============================================================================
// Types
// ============================================================================

export type AnimationStage = ForwardStage | BackwardStage;
export type NeuronData = ForwardCalculation | BackwardCalculation;

export interface AnimationLoopConfig<S extends AnimationStage, D extends NeuronData> {
  /** Forward iterates neurons ascending, backward descending (affects resume skipping) */
  mode: 'forward' | 'backward';
  /** Layers to iterate through, in order */
  layers: readonly LayerName[];
  /** Neuron indices to iterate for a layer */
  getNeuronIndices: (layer: LayerName) => number[];
  /** Stages to iterate through for each neuron */
  stages: readonly S[];
  /** Duration for each stage (ms, before speed scaling) */
  stageDurations: Record<S, number>;
  /** Per-layer neuron data */
  data: Record<LayerName, D[]>;
  /** Called for each stage tick (drives the state machine; the canvas redraws from state) */
  onTick: (layer: LayerName, neuronIndex: number, stage: S, data: D) => void;
  /** Called when the entire animation completes */
  onComplete: () => void;
  /** Check whether the animation should stop (pause / jump) */
  shouldStop: () => boolean;
  /** Sleep function (applies the current animation speed) */
  sleep: (ms: number) => Promise<void>;
  /** Resume from this position: neurons up to and including it are skipped */
  startFrom?: { layer: LayerName; neuronIndex: number };
}

// ============================================================================
// Animation Loop Runner
// ============================================================================

/**
 * Runs the animation loop with the given configuration.
 * @returns true if the animation completed, false if it was interrupted
 */
export async function runAnimationLoop<S extends AnimationStage, D extends NeuronData>(
  config: AnimationLoopConfig<S, D>
): Promise<boolean> {
  const { startFrom, layers } = config;
  const startLayerIdx = startFrom ? Math.max(0, layers.indexOf(startFrom.layer)) : 0;

  for (let layerIdx = startLayerIdx; layerIdx < layers.length; layerIdx++) {
    const layer = layers[layerIdx];
    const isStartLayer = startFrom !== undefined && layerIdx === startLayerIdx;

    for (const neuronIndex of config.getNeuronIndices(layer)) {
      if (isStartLayer && startFrom) {
        const alreadyShown = config.mode === 'forward'
          ? neuronIndex <= startFrom.neuronIndex
          : neuronIndex >= startFrom.neuronIndex;
        if (alreadyShown) continue;
      }

      const neuronData = config.data[layer][neuronIndex];

      for (const stage of config.stages) {
        if (config.shouldStop()) return false;
        config.onTick(layer, neuronIndex, stage, neuronData);
        await config.sleep(config.stageDurations[stage] ?? 300);
      }
    }
  }

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
