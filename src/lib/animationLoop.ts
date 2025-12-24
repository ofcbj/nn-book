/**
 * Animation Loop Abstraction
 * 
 * Common animation loop logic for forward and backward propagation.
 * Uses Strategy pattern to handle differences between modes.
 */

import type { CalculationStage, BackpropStage } from './types';

// ============================================================================
// Types
// ============================================================================

type LayerName = 'layer1' | 'layer2' | 'output';

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
  
  /** Called after visualization update for each stage */
  onAfterVisualization?: () => void;
  
  /** Called after a stage completes (e.g., for weight updates) */
  onStageComplete?: (layer: LayerName, neuronIndex: number, stage: TStage, data: TData) => void;
  
  /** Called when entire animation completes */
  onComplete: () => void;
  
  /** Check if animation should stop */
  shouldStop: () => boolean;
  
  /** Sleep function (respects pause state) */
  sleep: (ms: number, speedOverride?: number) => Promise<void>;
  
  /** Update visualization */
  updateVisualization: () => void;
  
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
        
        // Update visualization
        config.updateVisualization();
        config.onAfterVisualization?.();
        
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
// Configuration Factories
// ============================================================================

const LAYER_SIZES: Record<LayerName, number> = {
  layer1: 5,
  layer2: 3,
  output: 3,
};

/**
 * Creates neuron indices for forward propagation (0 to max)
 */
export function forwardNeuronIndices(layer: LayerName): number[] {
  const size = LAYER_SIZES[layer];
  return Array.from({ length: size }, (_, i) => i);
}

/**
 * Creates neuron indices for backward propagation (max to 0)
 */
export function backwardNeuronIndices(layer: LayerName): number[] {
  const size = LAYER_SIZES[layer];
  return Array.from({ length: size }, (_, i) => size - 1 - i);
}

// ============================================================================
// Stage Durations
// ============================================================================

export const FORWARD_STAGE_DURATIONS: Record<CalculationStage, number> = {
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
