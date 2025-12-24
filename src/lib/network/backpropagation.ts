// Backpropagation helper functions for neural network training
import { Matrix } from '../matrix';
import type { BackpropSteps, BackpropNeuronData } from '../types';

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
  learning_rate: number
): {
  output_errors: Matrix;
  gradients_output: Matrix;
  weight_ho_deltas: Matrix;
  bias_output_deltas: Matrix;
} {
  // Calculate output errors
  const output_errors = Matrix.subtract(targets, outputs);

  // Calculate output gradient
  const gradients_output = Matrix.map(outputs, dsigmoid);
  gradients_output.multiply(output_errors);
  gradients_output.multiply(learning_rate);

  // Calculate hidden2 -> output deltas
  const hidden2_T = Matrix.transpose(hidden2);
  const weight_ho_deltas = Matrix.multiply(gradients_output, hidden2_T);

  return {
    output_errors,
    gradients_output,
    weight_ho_deltas,
    bias_output_deltas: gradients_output
  };
}

/**
 * Generic backpropagation for hidden layers
 * Calculates errors, gradients, and weight deltas for a layer
 */
export function backpropHiddenLayer(
  next_layer_errors: Matrix,
  current_layer: Matrix,
  previous_layer: Matrix,
  weights_current_next: Matrix,
  learning_rate: number
): {
  current_errors: Matrix;
  current_gradients: Matrix;
  weight_deltas: Matrix;
  bias_deltas: Matrix;
} {
  // Calculate current layer errors from next layer
  const weights_transposed = Matrix.transpose(weights_current_next);
  const current_errors = Matrix.multiply(weights_transposed, next_layer_errors);

  // Calculate gradients
  const current_gradients = Matrix.map(current_layer, dsigmoid);
  current_gradients.multiply(current_errors);
  current_gradients.multiply(learning_rate);

  // Calculate weight deltas
  const previous_layer_T = Matrix.transpose(previous_layer);
  const weight_deltas = Matrix.multiply(current_gradients, previous_layer_T);

  return {
    current_errors,
    current_gradients,
    weight_deltas,
    bias_deltas: current_gradients
  };
}

/**
 * Create BackpropSteps for visualization
 */
export function createBackpropSteps(
  inputs: Matrix,
  hidden1: Matrix,
  hidden2: Matrix,
  outputs: Matrix,
  target_array: number[],
  output_errors: Matrix,
  hidden2_errors: Matrix,
  hidden1_errors: Matrix,
  gradients_output: Matrix,
  gradients_hidden2: Matrix,
  gradients_hidden1: Matrix,
  weight_ho_deltas: Matrix,
  weight_h1h2_deltas: Matrix,
  weight_ih1_deltas: Matrix,
  oldWeights_ho: number[][],
  oldBias_o: number[][],
  oldWeights_h1h2: number[][],
  oldBias_h2: number[][],
  oldWeights_ih1: number[][],
  oldBias_h1: number[][],
  newWeights_ho: number[][],
  newBias_o: number[][],
  newWeights_h1h2: number[][],
  newBias_h2: number[][],
  newWeights_ih1: number[][],
  newBias_h1: number[][],
  weights_hidden2_output: Matrix,
  weights_hidden1_hidden2: Matrix,
  lastLoss: number
): BackpropSteps {
  const targetClass = target_array.indexOf(1);
  const predictions = outputs.toArray();

  const backpropSteps: BackpropSteps = {
    output: [],
    layer2: [],
    layer1: [],
    totalLoss: lastLoss,
    targetClass,
    predictions
  };

  // Output layer backprop data
  for (let i = 0; i < 3; i++) {
    const activation = outputs.data[i][0];
    const derivative = dsigmoid(activation);
    const error = output_errors.data[i][0];
    const gradient = error * derivative;

    const neuronData: BackpropNeuronData = {
      neuronIndex: i,
      error: error,
      gradients: gradients_output.data[i],
      weightDeltas: weight_ho_deltas.data[i],
      biasDelta: gradients_output.data[i][0],
      oldWeights: oldWeights_ho[i],
      newWeights: newWeights_ho[i],
      oldBias: oldBias_o[i][0],
      newBias: newBias_o[i][0],
      activation: activation,
      derivative: derivative,
      gradient: gradient,
      inputs: hidden2.toArray()
    };
    backpropSteps.output.push(neuronData);
  }

  // Layer 2 backprop data
  for (let i = 0; i < 3; i++) {
    const activation = hidden2.data[i][0];
    const derivative = dsigmoid(activation);
    const error = hidden2_errors.data[i][0];
    const gradient = error * derivative;

    // Collect next layer (output) errors and weights for this neuron
    const nextErrors = output_errors.toArray();
    const nextWeights = weights_hidden2_output.data.map(row => row[i]);

    const neuronData: BackpropNeuronData = {
      neuronIndex: i,
      error: error,
      gradients: gradients_hidden2.data[i],
      weightDeltas: weight_h1h2_deltas.data[i],
      biasDelta: gradients_hidden2.data[i][0],
      oldWeights: oldWeights_h1h2[i],
      newWeights: newWeights_h1h2[i],
      oldBias: oldBias_h2[i][0],
      newBias: newBias_h2[i][0],
      activation: activation,
      derivative: derivative,
      gradient: gradient,
      inputs: hidden1.toArray(),
      nextLayerErrors: nextErrors,
      nextLayerWeights: nextWeights
    };
    backpropSteps.layer2.push(neuronData);
  }

  // Layer 1 backprop data
  for (let i = 0; i < 5; i++) {
    const activation = hidden1.data[i][0];
    const derivative = dsigmoid(activation);
    const error = hidden1_errors.data[i][0];
    const gradient = error * derivative;

    // Collect next layer (layer2) errors and weights for this neuron
    const nextErrors = hidden2_errors.toArray();
    const nextWeights = weights_hidden1_hidden2.data.map(row => row[i]);

    const neuronData: BackpropNeuronData = {
      neuronIndex: i,
      error: error,
      gradients: gradients_hidden1.data[i],
      weightDeltas: weight_ih1_deltas.data[i],
      biasDelta: gradients_hidden1.data[i][0],
      oldWeights: oldWeights_ih1[i],
      newWeights: newWeights_ih1[i],
      oldBias: oldBias_h1[i][0],
      newBias: newBias_h1[i][0],
      activation: activation,
      derivative: derivative,
      gradient: gradient,
      inputs: inputs.toArray(),
      nextLayerErrors: nextErrors,
      nextLayerWeights: nextWeights
    };
    backpropSteps.layer1.push(neuronData);
  }

  return backpropSteps;
}
