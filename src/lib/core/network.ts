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
  weightsInputHidden1: Matrix;
  biasHidden1: Matrix;
  
  // Layer 2: 2차 면접관 (3 neurons)
  weightsHidden1Hidden2: Matrix;
  biasHidden2: Matrix;
  
  // Output layer: 최종 결정 (3 classes: 불합격/보류/합격)
  weightsHidden2Output: Matrix;
  biasOutput: Matrix;
  
  learningRate: number = 0.1;
  
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
    outputToLayer2: Matrix | null;
    layer2ToLayer1: Matrix | null;
    layer1ToInput: Matrix | null;
  };

  lastLoss: number = 0;

  // Store detailed backprop data for visualization
  lastBackpropSteps: BackpropSteps | null = null;

  constructor() {
    // Layer 1: 1차 면접관 (5 neurons, each takes 3 inputs)
    this.weightsInputHidden1 = new Matrix(5, 3);
    this.biasHidden1 = new Matrix(5, 1);
    this.weightsInputHidden1.randomize();
    this.biasHidden1.randomizeBias();
    
    // Layer 2: 2차 면접관 (3 neurons, each takes 5 inputs)
    this.weightsHidden1Hidden2 = new Matrix(3, 5);
    this.biasHidden2 = new Matrix(3, 1);
    this.weightsHidden1Hidden2.randomize();
    this.biasHidden2.randomizeBias();
    
    // Output layer: 최종 결정 (3 neurons for softmax: 불합격/보류/합격)
    this.weightsHidden2Output = new Matrix(3, 3);
    this.biasOutput = new Matrix(3, 1);
    this.weightsHidden2Output.randomize();
    this.biasOutput.randomizeBias();
    
    this.lastGradients = {
      output: null,
      layer2: null,
      layer1: null
    };
    
    this.lastWeightDeltas = {
      outputToLayer2: null,
      layer2ToLayer1: null,
      layer1ToInput: null
    };
  }

  feedforward(inputArray: number[]): number[] {
    // Convert inputs to matrix
    const inputs = Matrix.fromArray(inputArray);
    this.lastInput = inputs;
    
    // Layer 1: 1차 면접관
    const hidden1Raw = Matrix.multiply(this.weightsInputHidden1, inputs);
    hidden1Raw.add(this.biasHidden1);
    this.lastHidden1Raw = hidden1Raw;
    const hidden1 = Matrix.map(hidden1Raw, sigmoid);
    this.lastHidden1 = hidden1;
    
    // Layer 2: 2차 면접관
    const hidden2Raw = Matrix.multiply(this.weightsHidden1Hidden2, hidden1);
    hidden2Raw.add(this.biasHidden2);
    this.lastHidden2Raw = hidden2Raw;
    const hidden2 = Matrix.map(hidden2Raw, sigmoid);
    this.lastHidden2 = hidden2;
    
    // Output layer: 최종 결정 (Softmax for 3-class classification)
    const outputRaw = Matrix.multiply(this.weightsHidden2Output, hidden2);
    outputRaw.add(this.biasOutput);
    this.lastOutputRaw = outputRaw;
    
    // Apply softmax activation
    const outputLogits = outputRaw.toArray();
    const outputProbs = softmax(outputLogits);
    const outputs = Matrix.fromArray(outputProbs);
    this.lastOutput = outputs;
    
    return outputProbs;
  }

  train(inputArray: number[], targetArray: number[]): void {
    // Feedforward
    this.feedforward(inputArray);

    const inputs = this.lastInput!;
    const hidden1 = this.lastHidden1!;
    const hidden2 = this.lastHidden2!;
    const outputs = this.lastOutput!;

    // Convert target to matrix
    const targets = Matrix.fromArray(targetArray);

    // Store old weights before update (for visualization)
    const oldWeightsHo = JSON.parse(JSON.stringify(this.weightsHidden2Output.data));
    const oldBiasO = JSON.parse(JSON.stringify(this.biasOutput.data));
    const oldWeightsH1h2 = JSON.parse(JSON.stringify(this.weightsHidden1Hidden2.data));
    const oldBiasH2 = JSON.parse(JSON.stringify(this.biasHidden2.data));
    const oldWeightsIh1 = JSON.parse(JSON.stringify(this.weightsInputHidden1.data));
    const oldBiasH1 = JSON.parse(JSON.stringify(this.biasHidden1.data));

    // === BACKPROPAGATION ===
    
    // Output layer
    const outputLayerResult = backpropOutputLayer(
      outputs,
      targets,
      hidden2,
      this.learningRate
    );

    // Adjust output weights and bias
    this.weightsHidden2Output.add(outputLayerResult.weightHoDeltas);
    this.biasOutput.add(outputLayerResult.biasOutputDeltas);

    // Layer 2 (Hidden2)
    const layer2Result = backpropHiddenLayer(
      outputLayerResult.outputErrors,
      hidden2,
      hidden1,
      this.weightsHidden2Output,
      this.learningRate
    );

    // Adjust hidden2 weights and bias
    this.weightsHidden1Hidden2.add(layer2Result.weightDeltas);
    this.biasHidden2.add(layer2Result.biasDeltas);

    // Layer 1 (Hidden1)
    const layer1Result = backpropHiddenLayer(
      layer2Result.currentErrors,
      hidden1,
      inputs,
      this.weightsHidden1Hidden2,
      this.learningRate
    );

    // Adjust hidden1 weights and bias
    this.weightsInputHidden1.add(layer1Result.weightDeltas);
    this.biasHidden1.add(layer1Result.biasDeltas);

    // === STORE FOR VISUALIZATION ===
    this.lastGradients.output = outputLayerResult.outputErrors;
    this.lastGradients.layer2 = layer2Result.currentErrors;
    this.lastGradients.layer1 = layer1Result.currentErrors;

    this.lastWeightDeltas.outputToLayer2 = outputLayerResult.weightHoDeltas;
    this.lastWeightDeltas.layer2ToLayer1 = layer2Result.weightDeltas;
    this.lastWeightDeltas.layer1ToInput = layer1Result.weightDeltas;

    // Calculate loss (cross-entropy for softmax)
    const targetOneHot = targetArray;
    this.lastLoss = -targetOneHot.reduce((sum, t, i) =>
      sum + (t > 0 ? Math.log(Math.max(outputs.data[i][0], 1e-7)) : 0), 0
    );

    // Build detailed backprop steps for visualization
    this.lastBackpropSteps = createBackpropSteps(
      inputs,
      hidden1,
      hidden2,
      outputs,
      targetArray,
      outputLayerResult.outputErrors,
      layer2Result.currentErrors,
      layer1Result.currentErrors,
      outputLayerResult.gradientsOutput,
      layer2Result.currentGradients,
      layer1Result.currentGradients,
      outputLayerResult.weightHoDeltas,
      layer2Result.weightDeltas,
      layer1Result.weightDeltas,
      oldWeightsHo,
      oldBiasO,
      oldWeightsH1h2,
      oldBiasH2,
      oldWeightsIh1,
      oldBiasH1,
      this.weightsHidden2Output.data,
      this.biasOutput.data,
      this.weightsHidden1Hidden2.data,
      this.biasHidden2.data,
      this.weightsInputHidden1.data,
      this.biasHidden1.data,
      this.weightsHidden2Output,
      this.weightsHidden1Hidden2,
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
      const weights = this.weightsInputHidden1.data[i];
      const bias = this.biasHidden1.data[i][0];
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
      const weights = this.weightsHidden1Hidden2.data[i];
      const bias = this.biasHidden2.data[i][0];
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
      const weights = this.weightsHidden2Output.data[i];
      const bias = this.biasOutput.data[i][0];
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
