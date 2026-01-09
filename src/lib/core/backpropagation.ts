// Backpropagation helper functions for neural network training
import { Matrix } from './matrix';
import { dsigmoid } from './network';
import type { BackpropSteps, BackpropNeuronData } from '../types';
import { LAYER_SIZES } from './networkConfig';
import type { LayerName } from './networkConfig';

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
// Backprop Visualizer Helpers
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
 * Create BackpropSteps for visualizer
 * 
 * Parameter interfaces for better organization
 */

// Layer-wise weight and bias data
interface LayerWeightData {
  weights: number[][];
  bias: number[][];
}

// All weights and biases before update
interface OldWeightsAndBiases {
  output: LayerWeightData;
  hidden2: LayerWeightData;
  hidden1: LayerWeightData;
}

// All weights and biases after update
interface NewWeightsAndBiases {
  output: LayerWeightData;
  hidden2: LayerWeightData;
  hidden1: LayerWeightData;
}

// Main parameter interface for createBackpropSteps
export interface BackpropData {
  // Layer activations
  activations: {
    inputs: Matrix;
    hidden1: Matrix;
    hidden2: Matrix;
    outputs: Matrix;
  };
  
  // Target values
  target: number[];
  
  // Errors for each layer
  errors: {
    output: Matrix;
    hidden2: Matrix;
    hidden1: Matrix;
  };
  
  // Gradients for each layer
  gradients: {
    output: Matrix;
    hidden2: Matrix;
    hidden1: Matrix;
  };
  
  // Weight deltas for each layer
  weightDeltas: {
    outputToHidden2: Matrix;
    hidden2ToHidden1: Matrix;
    hidden1ToInput: Matrix;
  };
  
  // Weights and biases before update
  oldWeights: OldWeightsAndBiases;
  
  // Weights and biases after update
  newWeights: NewWeightsAndBiases;
  
  // Current weight matrices (for calculating back-propagated errors)
  currentWeights: {
    hidden2ToOutput: Matrix;
    hidden1ToHidden2: Matrix;
  };
  
  // Loss value
  loss: number;
}

/**
 * Create BackpropSteps for visualizer
 */
export function createBackpropSteps(data: BackpropData): BackpropSteps {
  // Destructure data for easier access
  const { activations, target, errors, gradients, weightDeltas, oldWeights, newWeights, currentWeights, loss } = data;
  
  const targetClass = target.indexOf(1);
  const predictions = activations.outputs.toArray();

  return {
    output: createLayerBackpropData({
      layerName: 'output',
      activations: activations.outputs,
      errors: errors.output,
      gradients: gradients.output,
      weightDeltas: weightDeltas.outputToHidden2,
      oldWeights: oldWeights.output.weights,
      oldBias: oldWeights.output.bias,
      newWeights: newWeights.output.weights,
      newBias: newWeights.output.bias,
      inputs: activations.hidden2
    }),
    layer2: createLayerBackpropData({
      layerName: 'layer2',
      activations: activations.hidden2,
      errors: errors.hidden2,
      gradients: gradients.hidden2,
      weightDeltas: weightDeltas.hidden2ToHidden1,
      oldWeights: oldWeights.hidden2.weights,
      oldBias: oldWeights.hidden2.bias,
      newWeights: newWeights.hidden2.weights,
      newBias: newWeights.hidden2.bias,
      inputs: activations.hidden1,
      nextLayerErrors: errors.output,
      nextLayerWeights: currentWeights.hidden2ToOutput
    }),
    layer1: createLayerBackpropData({
      layerName: 'layer1',
      activations: activations.hidden1,
      errors: errors.hidden1,
      gradients: gradients.hidden1,
      weightDeltas: weightDeltas.hidden1ToInput,
      oldWeights: oldWeights.hidden1.weights,
      oldBias: oldWeights.hidden1.bias,
      newWeights: newWeights.hidden1.weights,
      newBias: newWeights.hidden1.bias,
      inputs: activations.inputs,
      nextLayerErrors: errors.hidden2,
      nextLayerWeights: currentWeights.hidden1ToHidden2
    }),
    totalLoss: loss,
    targetClass,
    predictions
  };
}
