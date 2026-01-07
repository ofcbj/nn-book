// Neural Network Implementation with TypeScript
// Architecture: 3 inputs -> 5 neurons (1차) -> 3 neurons (2차) -> 3 outputs (Softmax)

import type { CalculationSteps, BackpropSteps, NeuronCalculation } from '../types';
import i18n from '../../i18n';
import { Matrix } from './matrix';
import { LAYER_SIZES, INPUT_SIZE } from './networkConfig';
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
  
  // Store intermediate values for visualizer
  lastInput: Matrix | null = null;
  lastHidden1: Matrix | null = null;
  lastHidden2: Matrix | null = null;
  lastOutput: Matrix | null = null;
  lastHidden1Raw: Matrix | null = null; // Before activation
  lastHidden2Raw: Matrix | null = null; // Before activation
  lastOutputRaw: Matrix | null = null;  // Before activation
  
  // Store gradients for backprop visualizer
  lastGradients: {
    output: Matrix | null;
    layer2: Matrix | null;
    layer1: Matrix | null;
  };

  // Store weight deltas for update visualizer
  lastWeightDeltas: {
    outputToLayer2: Matrix | null;
    layer2ToLayer1: Matrix | null;
    layer1ToInput: Matrix | null;
  };

  lastLoss: number = 0;

  // Store detailed backprop data for visualizer
  lastBackpropSteps: BackpropSteps | null = null;

  constructor() {
    // Layer 1: 1차 면접관 (LAYER_SIZES.layer1 neurons, each takes INPUT_SIZE inputs)
    this.weightsInputHidden1 = new Matrix(LAYER_SIZES.layer1, INPUT_SIZE);
    this.biasHidden1 = new Matrix(LAYER_SIZES.layer1, 1);
    this.weightsInputHidden1.randomize();
    this.biasHidden1.randomizeBias();
    
    // Layer 2: 2차 면접관 (LAYER_SIZES.layer2 neurons, each takes LAYER_SIZES.layer1 inputs)
    this.weightsHidden1Hidden2 = new Matrix(LAYER_SIZES.layer2, LAYER_SIZES.layer1);
    this.biasHidden2 = new Matrix(LAYER_SIZES.layer2, 1);
    this.weightsHidden1Hidden2.randomize();
    this.biasHidden2.randomizeBias();
    
    // Output layer: 최종 결정 (LAYER_SIZES.output neurons)
    this.weightsHidden2Output = new Matrix(LAYER_SIZES.output, LAYER_SIZES.layer2);
    this.biasOutput = new Matrix(LAYER_SIZES.output, 1);
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
    
    // Define layer configurations - all layers follow the same pattern:
    // 1. Weighted sum: raw = weights × input + bias
    // 2. Activation: activated = activation_function(raw)
    const layerConfigs = [
      {
        name: 'Layer 1: 1차 면접관',
        weights: this.weightsInputHidden1,
        bias: this.biasHidden1,
        activationType: 'sigmoid' as const,
        storeRaw: (m: Matrix) => { this.lastHidden1Raw = m; },
        storeActivated: (m: Matrix) => { this.lastHidden1 = m; }
      },
      {
        name: 'Layer 2: 2차 면접관',
        weights: this.weightsHidden1Hidden2,
        bias: this.biasHidden2,
        activationType: 'sigmoid' as const,
        storeRaw: (m: Matrix) => { this.lastHidden2Raw = m; },
        storeActivated: (m: Matrix) => { this.lastHidden2 = m; }
      },
      {
        name: 'Output layer: 최종 결정',
        weights: this.weightsHidden2Output,
        bias: this.biasOutput,
        activationType: 'softmax' as const,
        storeRaw: (m: Matrix) => { this.lastOutputRaw = m; },
        storeActivated: (m: Matrix) => { this.lastOutput = m; }
      }
    ];

    // Process all layers with the same pattern
    let currentInput = inputs;
    for (const config of layerConfigs) {
      // Step 1: Weighted sum (raw = weights × input + bias)
      const raw = Matrix.multiply(config.weights, currentInput);
      raw.add(config.bias);
      config.storeRaw(raw);
      
      // Step 2: Activation function
      let activated: Matrix;
      if (config.activationType === 'sigmoid') {
        activated = Matrix.map(raw, sigmoid);
      } else {
        // Softmax for output layer
        const logits = raw.toArray();
        const probs = softmax(logits);
        activated = Matrix.fromArray(probs);
      }
      
      config.storeActivated(activated);
      currentInput = activated; // Output becomes input for next layer
    }
    
    return this.lastOutput!.toArray();
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

    // Store old weights before update (for visualizer)
    const oldWeightsHo = JSON.parse(JSON.stringify(this.weightsHidden2Output.data));
    const oldBiasO = JSON.parse(JSON.stringify(this.biasOutput.data));
    const oldWeightsH1h2 = JSON.parse(JSON.stringify(this.weightsHidden1Hidden2.data));
    const oldBiasH2 = JSON.parse(JSON.stringify(this.biasHidden2.data));
    const oldWeightsIh1 = JSON.parse(JSON.stringify(this.weightsInputHidden1.data));
    const oldBiasH1 = JSON.parse(JSON.stringify(this.biasHidden1.data));

    // === BACKPROPAGATION ===
    // All layers follow the same pattern in reverse:
    // 1. Calculate error (output layer: target-output, hidden: backprop from next layer)
    // 2. Calculate gradients (error × derivative × learning_rate)
    // 3. Calculate weight deltas (gradients × previous_layer_activations^T)
    // 4. Update weights and biases
    
    // Define layer configurations for backprop (processed in reverse order)
    const backpropConfigs = [
      {
        name: 'output',
        isOutputLayer: true,
        currentLayerActivations: outputs,
        previousLayerActivations: hidden2,
        weights: this.weightsHidden2Output,
        bias: this.biasOutput,
        nextLayerWeights: null // No next layer for output
      },
      {
        name: 'layer2',
        isOutputLayer: false,
        currentLayerActivations: hidden2,
        previousLayerActivations: hidden1,
        weights: this.weightsHidden1Hidden2,
        bias: this.biasHidden2,
        nextLayerWeights: this.weightsHidden2Output
      },
      {
        name: 'layer1',
        isOutputLayer: false,
        currentLayerActivations: hidden1,
        previousLayerActivations: inputs,
        weights: this.weightsInputHidden1,
        bias: this.biasHidden1,
        nextLayerWeights: this.weightsHidden1Hidden2
      }
    ];

    // Store results for visualizer
    const layerErrors: Record<string, Matrix> = {};
    const layerGradients: Record<string, Matrix> = {};
    const layerWeightDeltas: Record<string, Matrix> = {};

    let currentError: Matrix | null = null;

    // Process each layer with unified backprop logic
    for (const config of backpropConfigs) {
      let errors: Matrix;
      let gradients: Matrix;
      let weightDeltas: Matrix;
      let biasDeltas: Matrix;

      if (config.isOutputLayer) {
        // Output layer: error = target - output
        const result = backpropOutputLayer(
          config.currentLayerActivations,
          targets,
          config.previousLayerActivations,
          this.learningRate
        );
        
        errors = result.outputErrors;
        gradients = result.gradientsOutput;
        weightDeltas = result.weightHoDeltas;
        biasDeltas = result.biasOutputDeltas;
      } else {
        // Hidden layers: error propagated from next layer
        const result = backpropHiddenLayer(
          currentError!,
          config.currentLayerActivations,
          config.previousLayerActivations,
          config.nextLayerWeights!,
          this.learningRate
        );
        
        errors = result.currentErrors;
        gradients = result.currentGradients;
        weightDeltas = result.weightDeltas;
        biasDeltas = result.biasDeltas;
      }

      // Update weights and biases
      config.weights.add(weightDeltas);
      config.bias.add(biasDeltas);

      // Store for visualizer
      layerErrors[config.name] = errors;
      layerGradients[config.name] = gradients;
      layerWeightDeltas[config.name] = weightDeltas;

      // Propagate error to previous layer
      currentError = errors;
    }

    // === STORE FOR VISUALIZATION ===
    this.lastGradients.output = layerErrors.output;
    this.lastGradients.layer2 = layerErrors.layer2;
    this.lastGradients.layer1 = layerErrors.layer1;

    this.lastWeightDeltas.outputToLayer2 = layerWeightDeltas.output;
    this.lastWeightDeltas.layer2ToLayer1 = layerWeightDeltas.layer2;
    this.lastWeightDeltas.layer1ToInput = layerWeightDeltas.layer1;

    // Calculate loss (cross-entropy for softmax)
    const targetOneHot = targetArray;
    this.lastLoss = -targetOneHot.reduce((sum, t, i) =>
      sum + (t > 0 ? Math.log(Math.max(outputs.data[i][0], 1e-7)) : 0), 0
    );

    // Build detailed backprop steps for visualizer
    this.lastBackpropSteps = createBackpropSteps(
      inputs,
      hidden1,
      hidden2,
      outputs,
      targetArray,
      layerErrors.output,
      layerErrors.layer2,
      layerErrors.layer1,
      layerGradients.output,
      layerGradients.layer2,
      layerGradients.layer1,
      layerWeightDeltas.output,
      layerWeightDeltas.layer2,
      layerWeightDeltas.layer1,
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
    
    // Define layer configurations for iteration
    const layerConfigs = [
      {
        key: 'layer1' as const,
        count: 5,
        weights: this.weightsInputHidden1,
        bias: this.biasHidden1,
        rawValues: this.lastHidden1Raw!,
        activatedValues: this.lastHidden1!,
        inputs: this.lastInput,
      },
      {
        key: 'layer2' as const,
        count: 3,
        weights: this.weightsHidden1Hidden2,
        bias: this.biasHidden2,
        rawValues: this.lastHidden2Raw!,
        activatedValues: this.lastHidden2!,
        inputs: this.lastHidden1!,
      },
      {
        key: 'output' as const,
        count: 3,
        weights: this.weightsHidden2Output,
        bias: this.biasOutput,
        rawValues: this.lastOutputRaw!,
        activatedValues: this.lastOutput!,
        inputs: this.lastHidden2!,
      },
    ];
    
    const classNames = [i18n.t('classes.fail'), i18n.t('classes.pending'), i18n.t('classes.pass')];
    
    for (const config of layerConfigs) {
      for (let i = 0; i < config.count; i++) {
        const weights = config.weights.data[i];
        const bias = config.bias.data[i][0];
        const rawValue = config.rawValues.data[i][0];
        const activatedValue = config.activatedValues.data[i][0];
        const inputArray = config.inputs.toArray();
        
        const neuronData: NeuronCalculation = {
          neuronIndex: i,
          weights,
          bias,
          inputs: inputArray,
          dotProduct: rawValue - bias,
          withBias: rawValue,
          activated: activatedValue,
          calculation: `(${inputArray.map((v, j) => `${v.toFixed(2)}×${weights[j].toFixed(2)}`).join(' + ')}) + ${bias.toFixed(2)} = ${rawValue.toFixed(3)}`
        };
        
        // Add className only for output layer
        if (config.key === 'output') {
          neuronData.className = classNames[i];
        }
        
        steps[config.key].push(neuronData);
      }
    }
    
    return steps;
  }

  /**
   * Update weights and bias for a specific neuron in a layer.
   * This provides a flexible way to update neuron parameters without hardcoding layer names.
   */
  updateNeuronWeights(
    layer: 'layer1' | 'layer2' | 'output',
    neuronIndex: number,
    newWeights: number[],
    newBias: number
  ): void {
    switch (layer) {
      case 'output':
        this.weightsHidden2Output.data[neuronIndex] = newWeights;
        this.biasOutput.data[neuronIndex][0] = newBias;
        break;
      case 'layer2':
        this.weightsHidden1Hidden2.data[neuronIndex] = newWeights;
        this.biasHidden2.data[neuronIndex][0] = newBias;
        break;
      case 'layer1':
        this.weightsInputHidden1.data[neuronIndex] = newWeights;
        this.biasHidden1.data[neuronIndex][0] = newBias;
        break;
    }
  }
}
