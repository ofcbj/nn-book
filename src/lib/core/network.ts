// Neural Network Implementation with TypeScript
// Architecture: 3 inputs -> 5 neurons (1차) -> 3 neurons (2차) -> 3 outputs (Softmax)

import type { CalculationSteps, BackpropSteps } from '../types';
import i18n from '../../i18n';
import { Matrix } from './matrix';
import {
  backpropOutputLayer,
  backpropHiddenLayer,
  createBackpropSteps
} from './backpropagation';


// Activation functions
export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function dsigmoid(y: number): number {
  // Derivative of sigmoid (where y is already sigmoid(x))
  return y * (1 - y);
}

export function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max)); // Subtract max for numerical stability
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

/**
 * Neural Network Class
 * Architecture: 3 -> 5 -> 3 -> 3 (3-class output with Softmax)
 */
export class NeuralNetwork {
  // Layer 1: 1차 면접관 (5 neurons)
  weights_input_hidden1: Matrix;
  bias_hidden1: Matrix;
  
  // Layer 2: 2차 면접관 (3 neurons)
  weights_hidden1_hidden2: Matrix;
  bias_hidden2: Matrix;
  
  // Output layer: 최종 결정 (3 classes: 불합격/보류/합격)
  weights_hidden2_output: Matrix;
  bias_output: Matrix;
  
  learning_rate: number = 0.1;
  
  // Store intermediate values for visualization
  lastInput: Matrix | null = null;
  lastHidden1: Matrix | null = null;
  lastHidden2: Matrix | null = null;
  lastOutput: Matrix | null = null;
  lastHidden1Raw: Matrix | null = null; // Before activation
  lastHidden2Raw: Matrix | null = null; // Before activation
  lastOutputRaw: Matrix | null = null;  // Before activation
  
  // Store gradients for backprop visualization
  lastGradients: {
    output: Matrix | null;
    layer2: Matrix | null;
    layer1: Matrix | null;
  };

  // Store weight deltas for update visualization
  lastWeightDeltas: {
    output_layer2: Matrix | null;
    layer2_layer1: Matrix | null;
    layer1_input: Matrix | null;
  };

  lastLoss: number = 0;

  // Store detailed backprop data for visualization
  lastBackpropSteps: BackpropSteps | null = null;

  constructor() {
    // Layer 1: 1차 면접관 (5 neurons, each takes 3 inputs)
    this.weights_input_hidden1 = new Matrix(5, 3);
    this.bias_hidden1 = new Matrix(5, 1);
    this.weights_input_hidden1.randomize();
    this.bias_hidden1.randomizeBias();
    
    // Layer 2: 2차 면접관 (3 neurons, each takes 5 inputs)
    this.weights_hidden1_hidden2 = new Matrix(3, 5);
    this.bias_hidden2 = new Matrix(3, 1);
    this.weights_hidden1_hidden2.randomize();
    this.bias_hidden2.randomizeBias();
    
    // Output layer: 최종 결정 (3 neurons for softmax: 불합격/보류/합격)
    this.weights_hidden2_output = new Matrix(3, 3);
    this.bias_output = new Matrix(3, 1);
    this.weights_hidden2_output.randomize();
    this.bias_output.randomizeBias();
    
    this.lastGradients = {
      output: null,
      layer2: null,
      layer1: null
    };
    
    this.lastWeightDeltas = {
      output_layer2: null,
      layer2_layer1: null,
      layer1_input: null
    };
  }

  feedforward(input_array: number[]): number[] {
    // Convert inputs to matrix
    const inputs = Matrix.fromArray(input_array);
    this.lastInput = inputs;
    
    // Layer 1: 1차 면접관
    const hidden1_raw = Matrix.multiply(this.weights_input_hidden1, inputs);
    hidden1_raw.add(this.bias_hidden1);
    this.lastHidden1Raw = hidden1_raw;
    const hidden1 = Matrix.map(hidden1_raw, sigmoid);
    this.lastHidden1 = hidden1;
    
    // Layer 2: 2차 면접관
    const hidden2_raw = Matrix.multiply(this.weights_hidden1_hidden2, hidden1);
    hidden2_raw.add(this.bias_hidden2);
    this.lastHidden2Raw = hidden2_raw;
    const hidden2 = Matrix.map(hidden2_raw, sigmoid);
    this.lastHidden2 = hidden2;
    
    // Output layer: 최종 결정 (Softmax for 3-class classification)
    const output_raw = Matrix.multiply(this.weights_hidden2_output, hidden2);
    output_raw.add(this.bias_output);
    this.lastOutputRaw = output_raw;
    
    // Apply softmax activation
    const output_logits = output_raw.toArray();
    const output_probs = softmax(output_logits);
    const outputs = Matrix.fromArray(output_probs);
    this.lastOutput = outputs;
    
    return output_probs;
  }

  train(input_array: number[], target_array: number[]): void {
    // Feedforward
    this.feedforward(input_array);

    const inputs = this.lastInput!;
    const hidden1 = this.lastHidden1!;
    const hidden2 = this.lastHidden2!;
    const outputs = this.lastOutput!;

    // Convert target to matrix
    const targets = Matrix.fromArray(target_array);

    // Store old weights before update (for visualization)
    const oldWeights_ho = JSON.parse(JSON.stringify(this.weights_hidden2_output.data));
    const oldBias_o = JSON.parse(JSON.stringify(this.bias_output.data));
    const oldWeights_h1h2 = JSON.parse(JSON.stringify(this.weights_hidden1_hidden2.data));
    const oldBias_h2 = JSON.parse(JSON.stringify(this.bias_hidden2.data));
    const oldWeights_ih1 = JSON.parse(JSON.stringify(this.weights_input_hidden1.data));
    const oldBias_h1 = JSON.parse(JSON.stringify(this.bias_hidden1.data));

    // === BACKPROPAGATION ===
    
    // Output layer
    const outputLayerResult = backpropOutputLayer(
      outputs,
      targets,
      hidden2,
      this.learning_rate
    );

    // Adjust output weights and bias
    this.weights_hidden2_output.add(outputLayerResult.weightHoDeltas);
    this.bias_output.add(outputLayerResult.biasOutputDeltas);

    // Layer 2 (Hidden2)
    const layer2Result = backpropHiddenLayer(
      outputLayerResult.outputErrors,
      hidden2,
      hidden1,
      this.weights_hidden2_output,
      this.learning_rate
    );

    // Adjust hidden2 weights and bias
    this.weights_hidden1_hidden2.add(layer2Result.weightDeltas);
    this.bias_hidden2.add(layer2Result.biasDeltas);

    // Layer 1 (Hidden1)
    const layer1Result = backpropHiddenLayer(
      layer2Result.currentErrors,
      hidden1,
      inputs,
      this.weights_hidden1_hidden2,
      this.learning_rate
    );

    // Adjust hidden1 weights and bias
    this.weights_input_hidden1.add(layer1Result.weightDeltas);
    this.bias_hidden1.add(layer1Result.biasDeltas);

    // === STORE FOR VISUALIZATION ===
    this.lastGradients.output = outputLayerResult.outputErrors;
    this.lastGradients.layer2 = layer2Result.currentErrors;
    this.lastGradients.layer1 = layer1Result.currentErrors;

    this.lastWeightDeltas.output_layer2 = outputLayerResult.weightHoDeltas;
    this.lastWeightDeltas.layer2_layer1 = layer2Result.weightDeltas;
    this.lastWeightDeltas.layer1_input = layer1Result.weightDeltas;

    // Calculate loss (cross-entropy for softmax)
    const targetOneHot = target_array;
    this.lastLoss = -targetOneHot.reduce((sum, t, i) =>
      sum + (t > 0 ? Math.log(Math.max(outputs.data[i][0], 1e-7)) : 0), 0
    );

    // Build detailed backprop steps for visualization
    this.lastBackpropSteps = createBackpropSteps(
      inputs,
      hidden1,
      hidden2,
      outputs,
      target_array,
      outputLayerResult.outputErrors,
      layer2Result.currentErrors,
      layer1Result.currentErrors,
      outputLayerResult.gradientsOutput,
      layer2Result.currentGradients,
      layer1Result.currentGradients,
      outputLayerResult.weightHoDeltas,
      layer2Result.weightDeltas,
      layer1Result.weightDeltas,
      oldWeights_ho,
      oldBias_o,
      oldWeights_h1h2,
      oldBias_h2,
      oldWeights_ih1,
      oldBias_h1,
      this.weights_hidden2_output.data,
      this.bias_output.data,
      this.weights_hidden1_hidden2.data,
      this.bias_hidden2.data,
      this.weights_input_hidden1.data,
      this.bias_hidden1.data,
      this.weights_hidden2_output,
      this.weights_hidden1_hidden2,
      this.lastLoss
    );
  }

  getCalculationSteps(): CalculationSteps | null {
    if (!this.lastInput) return null;
    
    const steps: CalculationSteps = {
      input: this.lastInput.toArray(),
      layer1: [],
      layer2: [],
      output: []
    };
    
    // Layer 1 calculations (1차 면접관)
    for (let i = 0; i < 5; i++) {
      const weights = this.weights_input_hidden1.data[i];
      const bias = this.bias_hidden1.data[i][0];
      const rawValue = this.lastHidden1Raw!.data[i][0];
      const activatedValue = this.lastHidden1!.data[i][0];
      
      steps.layer1.push({
        neuronIndex: i,
        weights: weights,
        bias: bias,
        inputs: this.lastInput.toArray(),
        dotProduct: rawValue - bias, // Subtract bias to get pure dot product
        withBias: rawValue,
        activated: activatedValue,
        calculation: `(${this.lastInput.toArray().map((v, j) => `${v.toFixed(2)}×${weights[j].toFixed(2)}`).join(' + ')}) + ${bias.toFixed(2)} = ${rawValue.toFixed(3)}`
      });
    }
    
    // Layer 2 calculations (2차 면접관)
    for (let i = 0; i < 3; i++) {
      const weights = this.weights_hidden1_hidden2.data[i];
      const bias = this.bias_hidden2.data[i][0];
      const rawValue = this.lastHidden2Raw!.data[i][0];
      const activatedValue = this.lastHidden2!.data[i][0];
      
      steps.layer2.push({
        neuronIndex: i,
        weights: weights,
        bias: bias,
        inputs: this.lastHidden1!.toArray(),
        dotProduct: rawValue - bias,
        withBias: rawValue,
        activated: activatedValue,
        calculation: `(${this.lastHidden1!.toArray().map((v, j) => `${v.toFixed(2)}×${weights[j].toFixed(2)}`).join(' + ')}) + ${bias.toFixed(2)} = ${rawValue.toFixed(3)}`
      });
    }
    
    // Output calculation (3 neurons with softmax)
    const classNames = [i18n.t('classes.fail'), i18n.t('classes.pending'), i18n.t('classes.pass')];
    for (let i = 0; i < 3; i++) {
      const weights = this.weights_hidden2_output.data[i];
      const bias = this.bias_output.data[i][0];
      const rawValue = this.lastOutputRaw!.data[i][0];
      const activatedValue = this.lastOutput!.data[i][0];
      
      steps.output.push({
        neuronIndex: i,
        className: classNames[i],
        weights: weights,
        bias: bias,
        inputs: this.lastHidden2!.toArray(),
        dotProduct: rawValue - bias,
        withBias: rawValue,
        activated: activatedValue,
        calculation: `(${this.lastHidden2!.toArray().map((v, j) => `${v.toFixed(2)}×${weights[j].toFixed(2)}`).join(' + ')}) + ${bias.toFixed(2)} = ${rawValue.toFixed(3)}`
      });
    }
    
    return steps;
  }
}
