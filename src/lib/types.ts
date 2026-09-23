// Type definitions for Neural Network Visualizer
// Core interfaces and types used across the application

import type { LayerName } from './core/networkConfig';

// LayerType includes input layer, LayerName is for processing layers only
export type LayerType = 'input' | LayerName;

/**
 * Single neuron calculation data for visualizer
 */
export interface ForwardCalculation {
  neuronIndex: number;
  weights: number[];
  bias: number;
  inputs: number[];
  dotProduct: number;   // Σ wᵢxᵢ
  withBias: number;     // Σ wᵢxᵢ + b  (pre-activation / logit)
  activated: number;    // sigmoid(withBias) for hidden layers, softmax probability for output
}

/**
 * Complete forward propagation steps for all layers
 */
export interface ForwardSteps extends Record<LayerName, ForwardCalculation[]> {
  input: number[];
}

/**
 * Forward propagation stage during animation
 */
export type ForwardStage = 'connections' | 'dotProduct' | 'bias' | 'activation';

/**
 * Backpropagation calculation stage during animation
 */
export type BackwardStage = 'error' | 'derivative' | 'gradient' | 'weightDelta' | 'allWeightDeltas' | 'update';

/**
 * Backpropagation visualizer data for a single neuron
 */
export interface BackwardCalculation {
  neuronIndex: number;
  /** Error this neuron received: output layer = target − output, hidden = Σ δ_next · w */
  error: number;
  /** Activation value of this neuron (y) */
  activation: number;
  /** Activation derivative: σ'(y) = y(1−y) for hidden layers, 1 for softmax+cross-entropy output */
  derivative: number;
  /** δ = error × derivative. This is what is propagated to the previous layer. */
  gradient: number;
  /** ΔWⱼ = lr × δ × inputⱼ (added to the weights) */
  weightDeltas: number[];
  /** Δb = lr × δ */
  biasDelta: number;
  oldWeights: number[];
  newWeights: number[];
  oldBias: number;
  newBias: number;
  /** Inputs to this neuron (activations of the previous layer) */
  inputs: number[];
  /** Hidden layers only: δ of each neuron in the next layer */
  nextLayerDeltas?: number[];
  /** Hidden layers only: pre-update weights from this neuron to each next-layer neuron */
  nextLayerWeights?: number[];
}

/**
 * Complete backpropagation steps for visualizer
 */
export interface BackwardSteps extends Record<LayerName, BackwardCalculation[]> {
  totalLoss: number;
  targetClass: number;
  predictions: number[];
}

/**
 * Node position for canvas rendering (logical / CSS pixels)
 */
export interface NodePosition {
  x: number;
  y: number;
  width: number;
  height: number;
  centerX: number;
  centerY: number;
}

/** One recorded training step for the loss chart */
export interface LossPoint {
  epoch: number;
  loss: number;
}

/**
 * Logical drawing area of the canvas (CSS pixels, independent of devicePixelRatio)
 */
export interface Viewport {
  width: number;
  height: number;
}

export interface BackpropSummaryData {
  oldWeights: Record<LayerName, number[][]>;
  newWeights: Record<LayerName, number[][]>;
  oldBiases: Record<LayerName, number[]>;
  newBiases: Record<LayerName, number[]>;
  learningRate: number;
  totalWeightsUpdated: number;
}

export interface LayerWeightComparison {
  oldWeights: number[][];
  newWeights: number[][];
  oldBiases: number[];
  newBiases: number[];
  weightDeltas: number[][];
  biasDeltas: number[];
}

export interface WeightComparisonData extends Record<LayerName, LayerWeightComparison> {
  totalChange: number;
  maxWeightChange: number;
  learningRate: number;
}
