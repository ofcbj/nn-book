/**
 * Animation Loop Abstraction
 * 
 * Common animation loop logic for forward and backward propagation.
 * Uses Strategy pattern to handle differences between modes.
 */

import type { ForwardStage, BackpropStage } from '../types';

// ============================================================================
// Types
// ============================================================================

import type { LayerName } from '../core';

export interface AnimationLoopConfig<TStage extends string, TData> {
  /** Mode identifier */
  mode: 'forward' | 'backward';
  
  /** Layers to iterate through (in order) */
  layers: LayerName[];
  
  /** Get neuron indices to iterate for a layer */
  getNeuronIndices: (layer: LayerName) => number[];
  
  /** Stages to iterate through for each neuron */
  stages: TStage[];
  
  /** Duration for each stage (in ms) */
  stageDurations: Record<TStage, number>;
  
  /** Get data for all layers */
  getData: () => Record<LayerName, TData[]> | null;
  
  /** Called for each stage tick */
  onTick: (layer: LayerName, neuronIndex: number, stage: TStage, data: TData) => void;
  
  /** Called after visualizer update for each stage */
  onAfterVisualizer?: () => void;
  
  /** Called after a stage completes (e.g., for weight updates) */
  onStageComplete?: (layer: LayerName, neuronIndex: number, stage: TStage, data: TData) => void;
  
  /** Called when entire animation completes */
  onComplete: () => void;
  
  /** Check if animation should stop */
  shouldStop: () => boolean;
  
  /** Sleep function (respects pause state) */
  sleep: (ms: number, speedOverride?: number) => Promise<void>;
  
  /** Compute network and refresh display */
  computeAndRefreshDisplay: () => void;
  
  /** Speed override for sleep (optional) */
  speedOverride?: number;
}

// ============================================================================
// Animation Loop Runner
// ============================================================================

/**
 * Runs the animation loop with the given configuration.
 * Handles forward and backward propagation with a unified loop structure.
 */
export async function runAnimationLoop<TStage extends string, TData>(
  config: AnimationLoopConfig<TStage, TData>
): Promise<void> {
  const data = config.getData();
  if (!data) return;
  
  for (const layer of config.layers) {
    const neuronIndices = config.getNeuronIndices(layer);
    
    for (const neuronIndex of neuronIndices) {
      if (config.shouldStop()) return;
      
      const neuronData = data[layer][neuronIndex];
      
      for (const stage of config.stages) {
        if (config.shouldStop()) return;
        
        // Update state machine
        config.onTick(layer, neuronIndex, stage, neuronData);
        
        // Compute network and refresh display
        config.computeAndRefreshDisplay();
        config.onAfterVisualizer?.();
        
        // Wait for appropriate duration
        await config.sleep(config.stageDurations[stage], config.speedOverride);
        
        // Handle post-stage processing (e.g., weight updates for backward)
        config.onStageComplete?.(layer, neuronIndex, stage, neuronData);
      }
    }
  }
  
  // Animation complete
  config.onComplete();
}

// ============================================================================
// Configuration - imported from core/networkConfig
// ============================================================================

import { getForwardNeuronIndices, getBackwardNeuronIndices } from '../core';

export { getForwardNeuronIndices as forwardNeuronIndices, getBackwardNeuronIndices as backwardNeuronIndices };

// ============================================================================
// Stage Durations
// ============================================================================

export const FORWARD_STAGE_DURATIONS: Record<ForwardStage, number> = {
  connections: 150,
  dotProduct: 400,
  bias: 400,
  activation: 400,
};

export const BACKWARD_STAGE_DURATIONS: Record<BackpropStage, number> = {
  error: 300,
  derivative: 350,
  gradient: 350,
  weightDelta: 350,
  allWeightDeltas: 400,
  update: 300,
};
