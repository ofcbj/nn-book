// Type definitions for Neural Network Visualizer
// Core interfaces and types used across the application

import type { LayerName } from './core/networkConfig';

// LayerType includes input layer, LayerName is for processing layers only
export type LayerType = 'input' | LayerName;

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
 * Single neuron calculation data for visualizer
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
 * Backpropagation calculation stage during animation
 */
export type BackpropStage = 'error' | 'derivative' | 'gradient' | 'weightDelta' | 'allWeightDeltas' | 'update';

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
 * Backpropagation visualizer data for a single neuron
 */
export interface BackpropNeuronData {
  neuronIndex: number;
  error: number;              // 이 뉴런이 받은 오류 크기
  gradients: number[];        // 각 가중치에 대한 그래디언트
  weightDeltas: number[];     // 실제 가중치 변화량
  biasDelta: number;          // bias 변화량
  oldWeights: number[];       // 업데이트 전 가중치
  newWeights: number[];       // 업데이트 후 가중치
  oldBias: number;
  newBias: number;
  // 추가: 계산 과정 상세 정보
  activation: number;         // 이 뉴런의 활성화 값 (y)
  derivative: number;         // sigmoid 미분값 y(1-y)
  gradient: number;           // error × derivative (최종 그래디언트)
  inputs: number[];           // 이 뉴런으로 들어온 입력값들
  nextLayerErrors?: number[]; // 다음 레이어의 오류들 (역전파 계산용)
  nextLayerWeights?: number[]; // 이 뉴런에서 다음 레이어로의 가중치들
}

/**
 * Complete backpropagation steps for visualizer
 */
export interface BackpropSteps {
  layer1: BackpropNeuronData[];
  layer2: BackpropNeuronData[];
  output: BackpropNeuronData[];
  totalLoss: number;
  targetClass: number;
  predictions: number[];
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

export interface BackpropSummaryData {
  oldWeights: {
    layer1: number[][];
    layer2: number[][];
    output: number[][];
  };
  newWeights: {
    layer1: number[][];
    layer2: number[][];
    output: number[][];
  };
  oldBiases: {
    layer1: number[];
    layer2: number[];
    output: number[];
  };
  newBiases: {
    layer1: number[];
    layer2: number[];
    output: number[];
  };
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

export interface WeightComparisonData {
  layer1: LayerWeightComparison;
  layer2: LayerWeightComparison;
  output: LayerWeightComparison;
  totalChange: number;
  maxWeightChange: number;
  learningRate: number;
}

/**
 * Creates a BackpropSummaryData object from BackpropSteps data.
 * Extracts weights and biases from each layer into a summary format.
 */
export function createBackpropSummaryData(
  backpropData: BackpropSteps,
  learningRate: number
): BackpropSummaryData {
  const layers: Array<'layer1' | 'layer2' | 'output'> = ['layer1', 'layer2', 'output'];
  
  const oldWeights: BackpropSummaryData['oldWeights'] = { layer1: [], layer2: [], output: [] };
  const newWeights: BackpropSummaryData['newWeights'] = { layer1: [], layer2: [], output: [] };
  const oldBiases: BackpropSummaryData['oldBiases'] = { layer1: [], layer2: [], output: [] };
  const newBiases: BackpropSummaryData['newBiases'] = { layer1: [], layer2: [], output: [] };
  let totalWeightsUpdated = 0;
  
  for (const layer of layers) {
    oldWeights[layer] = backpropData[layer].map(n => [...n.oldWeights]);
    newWeights[layer] = backpropData[layer].map(n => [...n.newWeights]);
    oldBiases[layer] = backpropData[layer].map(n => n.oldBias);
    newBiases[layer] = backpropData[layer].map(n => n.newBias);
    totalWeightsUpdated += backpropData[layer].reduce((sum, n) => sum + n.oldWeights.length, 0);
  }
  
  return {
    oldWeights,
    newWeights,
    oldBiases,
    newBiases,
    learningRate,
    totalWeightsUpdated,
  };
}
