// Backpropagation helper functions for neural network training
import { Matrix } from './matrix';
import type { BackpropSteps, BackpropNeuronData } from '../types';
import { LAYER_SIZES } from './networkConfig';
import type { LayerName } from './networkConfig';

// Activation function derivative
function dsigmoid(y: number): number {
  return y * (1 - y);
}

/**
 * Perform backpropagation for output layer and return deltas
 */
export function backpropOutputLayer(
  outputs: Matrix,
  targets: Matrix,
  hidden2: Matrix,
  learningRate: number
): {
  outputErrors: Matrix;
  gradientsOutput: Matrix;
  weightHoDeltas: Matrix;
  biasOutputDeltas: Matrix;
} {
  // Calculate output errors
  const outputErrors = Matrix.subtract(targets, outputs);

  // Calculate output gradient
  const gradientsOutput = Matrix.map(outputs, dsigmoid);
  gradientsOutput.multiply(outputErrors);
  gradientsOutput.multiply(learningRate);

  // Calculate hidden2 -> output deltas
  const hidden2T = Matrix.transpose(hidden2);
  const weightHoDeltas = Matrix.multiply(gradientsOutput, hidden2T);

  return {
    outputErrors,
    gradientsOutput,
    weightHoDeltas,
    biasOutputDeltas: gradientsOutput
  };
}

/**
 * Generic backpropagation for hidden layers
 * Calculates errors, gradients, and weight deltas for a layer
 */
export function backpropHiddenLayer(
  nextLayerErrors: Matrix,
  currentLayer: Matrix,
  previousLayer: Matrix,
  weightsCurrentNext: Matrix,
  learningRate: number
): {
  currentErrors: Matrix;
  currentGradients: Matrix;
  weightDeltas: Matrix;
  biasDeltas: Matrix;
} {
  // Calculate current layer errors from next layer
  const weightsTransposed = Matrix.transpose(weightsCurrentNext);
  const currentErrors = Matrix.multiply(weightsTransposed, nextLayerErrors);

  // Calculate gradients
  const currentGradients = Matrix.map(currentLayer, dsigmoid);
  currentGradients.multiply(currentErrors);
  currentGradients.multiply(learningRate);

  // Calculate weight deltas
  const previousLayerT = Matrix.transpose(previousLayer);
  const weightDeltas = Matrix.multiply(currentGradients, previousLayerT);

  return {
    currentErrors,
    currentGradients,
    weightDeltas,
    biasDeltas: currentGradients
  };
}

// =============================================================================
// Backprop Visualization Helpers
// =============================================================================

/**
 * Configuration for creating BackpropNeuronData for a single layer
 */
interface LayerBackpropConfig {
  layerName: LayerName;
  activations: Matrix;
  errors: Matrix;
  gradients: Matrix;
  weightDeltas: Matrix;
  oldWeights: number[][];
  oldBias: number[][];
  newWeights: number[][];
  newBias: number[][];
  inputs: Matrix;
  // Optional: for hidden layers, info about next layer
  nextLayerErrors?: Matrix;
  nextLayerWeights?: Matrix;
}

/**
 * Create BackpropNeuronData array for a single layer
 */
function createLayerBackpropData(config: LayerBackpropConfig): BackpropNeuronData[] {
  const {
    layerName,
    activations,
    errors,
    gradients,
    weightDeltas,
    oldWeights,
    oldBias,
    newWeights,
    newBias,
    inputs,
    nextLayerErrors,
    nextLayerWeights
  } = config;

  const neuronCount = LAYER_SIZES[layerName];
  const result: BackpropNeuronData[] = [];

  for (let i = 0; i < neuronCount; i++) {
    const activation = activations.data[i][0];
    const derivative = dsigmoid(activation);
    const error = errors.data[i][0];
    const gradient = error * derivative;

    const neuronData: BackpropNeuronData = {
      neuronIndex: i,
      error: error,
      gradients: gradients.data[i],
      weightDeltas: weightDeltas.data[i],
      biasDelta: gradients.data[i][0],
      oldWeights: oldWeights[i],
      newWeights: newWeights[i],
      oldBias: oldBias[i][0],
      newBias: newBias[i][0],
      activation: activation,
      derivative: derivative,
      gradient: gradient,
      inputs: inputs.toArray()
    };

    // Add next layer info for hidden layers
    if (nextLayerErrors && nextLayerWeights) {
      neuronData.nextLayerErrors = nextLayerErrors.toArray();
      neuronData.nextLayerWeights = nextLayerWeights.data.map(row => row[i]);
    }

    result.push(neuronData);
  }

  return result;
}

/**
 * Create BackpropSteps for visualization
 */
export function createBackpropSteps(
  inputs: Matrix,
  hidden1: Matrix,
  hidden2: Matrix,
  outputs: Matrix,
  targetArray: number[],
  outputErrors: Matrix,
  hidden2Errors: Matrix,
  hidden1Errors: Matrix,
  gradientsOutput: Matrix,
  gradientsHidden2: Matrix,
  gradientsHidden1: Matrix,
  weightHoDeltas: Matrix,
  weightH1h2Deltas: Matrix,
  weightIh1Deltas: Matrix,
  oldWeightsHo: number[][],
  oldBiasO: number[][],
  oldWeightsH1h2: number[][],
  oldBiasH2: number[][],
  oldWeightsIh1: number[][],
  oldBiasH1: number[][],
  newWeightsHo: number[][],
  newBiasO: number[][],
  newWeightsH1h2: number[][],
  newBiasH2: number[][],
  newWeightsIh1: number[][],
  newBiasH1: number[][],
  weightsHidden2Output: Matrix,
  weightsHidden1Hidden2: Matrix,
  lastLoss: number
): BackpropSteps {
  const targetClass = targetArray.indexOf(1);
  const predictions = outputs.toArray();

  return {
    output: createLayerBackpropData({
      layerName: 'output',
      activations: outputs,
      errors: outputErrors,
      gradients: gradientsOutput,
      weightDeltas: weightHoDeltas,
      oldWeights: oldWeightsHo,
      oldBias: oldBiasO,
      newWeights: newWeightsHo,
      newBias: newBiasO,
      inputs: hidden2
    }),
    layer2: createLayerBackpropData({
      layerName: 'layer2',
      activations: hidden2,
      errors: hidden2Errors,
      gradients: gradientsHidden2,
      weightDeltas: weightH1h2Deltas,
      oldWeights: oldWeightsH1h2,
      oldBias: oldBiasH2,
      newWeights: newWeightsH1h2,
      newBias: newBiasH2,
      inputs: hidden1,
      nextLayerErrors: outputErrors,
      nextLayerWeights: weightsHidden2Output
    }),
    layer1: createLayerBackpropData({
      layerName: 'layer1',
      activations: hidden1,
      errors: hidden1Errors,
      gradients: gradientsHidden1,
      weightDeltas: weightIh1Deltas,
      oldWeights: oldWeightsIh1,
      oldBias: oldBiasH1,
      newWeights: newWeightsIh1,
      newBias: newBiasH1,
      inputs: inputs,
      nextLayerErrors: hidden2Errors,
      nextLayerWeights: weightsHidden1Hidden2
    }),
    totalLoss: lastLoss,
    targetClass,
    predictions
  };
}
