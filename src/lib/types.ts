// Type definitions for Neural Network Visualization
// Core interfaces and types used across the application

/**
 * Matrix class representation
 */
export interface MatrixData {
  rows: number;
  cols: number;
  data: number[][];
}

/**
 * Neural network architecture configuration
 */
export interface NetworkArchitecture {
  inputSize: number;    // 3 (성적, 태도, 응답수준)
  hidden1Size: number;  // 5 neurons (1차 면접관)
  hidden2Size: number;  // 3 neurons (2차 면접관)
  outputSize: number;   // 3 classes (불합격, 보류, 합격)
}

/**
 * Single neuron calculation data for visualization
 */
export interface NeuronCalculation {
  neuronIndex: number;
  className?: string;      // For output neurons
  weights: number[];
  bias: number;
  inputs: number[];
  dotProduct: number;
  withBias: number;
  activated: number;
  calculation?: string;
}

/**
 * Complete calculation steps for all layers
 */
export interface CalculationSteps {
  input: number[];
  layer1: NeuronCalculation[];
  layer2: NeuronCalculation[];
  output: NeuronCalculation[];
}

/**
 * Training state management
 */
export interface TrainingState {
  isTraining: boolean;
  epoch: number;
  loss: number;
  isAnimating: boolean;
  shouldStopAnimation: boolean;
  animationSpeed: number;
}

/**
 * Animation phase tracking
 */
export interface AnimationPhase {
  layer: 'input' | 'layer1' | 'layer2' | 'output';
  index: number;
}

/**
 * Calculation stage during animation
 */
export type CalculationStage = 'connections' | 'dotProduct' | 'bias' | 'activation';

/**
 * Loss display information
 */
export interface LossDisplayData {
  targetClass: number;
  targetName: string;
  predictions: number[];
  loss: number;
}

/**
 * Gradient information for backpropagation
 */
export interface GradientData {
  output: MatrixData | null;
  layer2: MatrixData | null;
  layer1: MatrixData | null;
}

/**
 * Weight delta information for visualization
 */
export interface WeightDeltas {
  output_layer2: MatrixData | null;
  layer2_layer1: MatrixData | null;
  layer1_input: MatrixData | null;
}

/**
 * Node position for canvas rendering
 */
export interface NodePosition {
  x: number;
  y: number;
  width: number;
  height: number;
  centerX: number;
  centerY: number;
}
