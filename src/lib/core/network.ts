// Neural Network Implementation with TypeScript
// Architecture: 3 inputs -> 5 neurons (1차) -> 3 neurons (2차) -> 3 outputs (Softmax)

import type { ForwardSteps, BackwardSteps, ForwardCalculation, BackwardCalculation } from '../types';
import { Matrix } from './matrix';
import { sigmoid, softmax, crossEntropyLoss } from './activations';
import { LAYER_SIZES, INPUT_SIZE } from './networkConfig';
import type { LayerName } from './networkConfig';
import {
  backpropOutputLayer,
  backpropHiddenLayer,
  createBackwardSteps
} from './backpropagation';

export const DEFAULT_LEARNING_RATE = 0.25;

/**
 * Neural Network Class
 * Architecture: 3 -> 5 -> 3 -> 3 (3-class output with Softmax)
 *
 * Hidden layers use sigmoid; the output layer uses softmax with cross-entropy loss.
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

  learningRate: number;

  // Intermediate values from the last feedforward (for the visualizer)
  lastInput      : Matrix | null = null;
  lastHidden1    : Matrix | null = null;
  lastHidden2    : Matrix | null = null;
  lastOutput     : Matrix | null = null;
  lastHidden1Raw : Matrix | null = null; // Before activation
  lastHidden2Raw : Matrix | null = null; // Before activation
  lastOutputRaw  : Matrix | null = null; // Before activation (logits)

  lastLoss: number = 0;

  // Detailed backprop data from the last train() call (for the visualizer)
  lastBackwardSteps: BackwardSteps | null = null;

  constructor(learningRate: number = DEFAULT_LEARNING_RATE) {
    this.learningRate = learningRate;

    this.weightsInputHidden1 = new Matrix(LAYER_SIZES.layer1, INPUT_SIZE);
    this.biasHidden1 = new Matrix(LAYER_SIZES.layer1, 1);
    this.weightsInputHidden1.randomize();
    this.biasHidden1.randomizeBias();

    this.weightsHidden1Hidden2 = new Matrix(LAYER_SIZES.layer2, LAYER_SIZES.layer1);
    this.biasHidden2 = new Matrix(LAYER_SIZES.layer2, 1);
    this.weightsHidden1Hidden2.randomize();
    this.biasHidden2.randomizeBias();

    this.weightsHidden2Output = new Matrix(LAYER_SIZES.output, LAYER_SIZES.layer2);
    this.biasOutput = new Matrix(LAYER_SIZES.output, 1);
    this.weightsHidden2Output.randomize();
    this.biasOutput.randomizeBias();
  }

  feedforward(inputArray: number[]): number[] {
    const inputs = Matrix.fromArray(inputArray);
    this.lastInput = inputs;

    // Every layer follows the same pattern:
    // 1. raw = weights × input + bias
    // 2. activated = activation(raw)
    const layerConfigs = [
      {
        weights: this.weightsInputHidden1,
        bias: this.biasHidden1,
        activationType: 'sigmoid' as const,
        storeRaw: (m: Matrix) => { this.lastHidden1Raw = m; },
        storeActivated: (m: Matrix) => { this.lastHidden1 = m; }
      },
      {
        weights: this.weightsHidden1Hidden2,
        bias: this.biasHidden2,
        activationType: 'sigmoid' as const,
        storeRaw: (m: Matrix) => { this.lastHidden2Raw = m; },
        storeActivated: (m: Matrix) => { this.lastHidden2 = m; }
      },
      {
        weights: this.weightsHidden2Output,
        bias: this.biasOutput,
        activationType: 'softmax' as const,
        storeRaw: (m: Matrix) => { this.lastOutputRaw = m; },
        storeActivated: (m: Matrix) => { this.lastOutput = m; }
      }
    ];

    let currentInput = inputs;
    for (const config of layerConfigs) {
      const raw = Matrix.multiply(config.weights, currentInput);
      raw.add(config.bias);
      config.storeRaw(raw);

      const activated = config.activationType === 'sigmoid'
        ? Matrix.map(raw, sigmoid)
        : Matrix.fromArray(softmax(raw.toArray()));

      config.storeActivated(activated);
      currentInput = activated;
    }

    return this.lastOutput!.toArray();
  }

  /**
   * One training step: feedforward, backpropagation, weight update.
   * Detailed per-neuron data is stored in `lastBackwardSteps`.
   */
  train(inputArray: number[], targetArray: number[]): void {
    this.feedforward(inputArray);
    this.backpropagate(targetArray);
  }

  private backpropagate(targetArray: number[]): void {
    const inputs  = this.lastInput!;
    const hidden1 = this.lastHidden1!;
    const hidden2 = this.lastHidden2!;
    const outputs = this.lastOutput!;
    const targets = Matrix.fromArray(targetArray);

    // Snapshot weights BEFORE any update. Every gradient (including the
    // errors propagated to hidden layers) must be computed against these.
    const old = {
      output: { weights: this.weightsHidden2Output.clone(),  bias: this.biasOutput.clone() },
      layer2: { weights: this.weightsHidden1Hidden2.clone(), bias: this.biasHidden2.clone() },
      layer1: { weights: this.weightsInputHidden1.clone(),   bias: this.biasHidden1.clone() },
    };

    // === BACKPROPAGATION (output -> layer2 -> layer1) ===
    const outputResult = backpropOutputLayer(outputs, targets, hidden2, this.learningRate);
    const layer2Result = backpropHiddenLayer(outputResult.deltas, hidden2, hidden1, old.output.weights, this.learningRate);
    const layer1Result = backpropHiddenLayer(layer2Result.deltas, hidden1, inputs,  old.layer2.weights, this.learningRate);

    // === UPDATE (only after all gradients are computed) ===
    this.weightsHidden2Output.add(outputResult.weightDeltas);
    this.biasOutput.add(outputResult.biasDeltas);
    this.weightsHidden1Hidden2.add(layer2Result.weightDeltas);
    this.biasHidden2.add(layer2Result.biasDeltas);
    this.weightsInputHidden1.add(layer1Result.weightDeltas);
    this.biasHidden1.add(layer1Result.biasDeltas);

    const predictions = outputs.toArray();
    this.lastLoss = crossEntropyLoss(predictions, targetArray);

    // === STORE FOR VISUALIZATION ===
    this.lastBackwardSteps = createBackwardSteps({
      layers: {
        output: {
          activations: outputs,
          inputs: hidden2,
          result: outputResult,
          oldWeights: old.output.weights, oldBias: old.output.bias,
          newWeights: this.weightsHidden2Output, newBias: this.biasOutput,
        },
        layer2: {
          activations: hidden2,
          inputs: hidden1,
          result: layer2Result,
          oldWeights: old.layer2.weights, oldBias: old.layer2.bias,
          newWeights: this.weightsHidden1Hidden2, newBias: this.biasHidden2,
          nextLayerDeltas: outputResult.deltas,
          nextLayerWeights: old.output.weights,
        },
        layer1: {
          activations: hidden1,
          inputs: inputs,
          result: layer1Result,
          oldWeights: old.layer1.weights, oldBias: old.layer1.bias,
          newWeights: this.weightsInputHidden1, newBias: this.biasHidden1,
          nextLayerDeltas: layer2Result.deltas,
          nextLayerWeights: old.layer2.weights,
        },
      },
      target: targetArray,
      predictions,
      loss: this.lastLoss,
    });
  }

  getForwardSteps(): ForwardSteps | null {
    if (!this.lastInput || !this.lastHidden1 || !this.lastHidden2 || !this.lastOutput) return null;

    const layerConfigs = [
      { key: 'layer1' as const, weights: this.weightsInputHidden1,   bias: this.biasHidden1, raw: this.lastHidden1Raw!, activated: this.lastHidden1, inputs: this.lastInput },
      { key: 'layer2' as const, weights: this.weightsHidden1Hidden2, bias: this.biasHidden2, raw: this.lastHidden2Raw!, activated: this.lastHidden2, inputs: this.lastHidden1 },
      { key: 'output' as const, weights: this.weightsHidden2Output,  bias: this.biasOutput,  raw: this.lastOutputRaw!,  activated: this.lastOutput,  inputs: this.lastHidden2 },
    ];

    const steps: ForwardSteps = {
      input: this.lastInput.toArray(),
      layer1: [],
      layer2: [],
      output: []
    };

    for (const config of layerConfigs) {
      const inputArray = config.inputs.toArray();
      const count = LAYER_SIZES[config.key];
      for (let i = 0; i < count; i++) {
        const weights  = [...config.weights.data[i]];
        const bias     = config.bias.data[i][0];
        const rawValue = config.raw.data[i][0];

        const neuronData: ForwardCalculation = {
          neuronIndex: i,
          weights,
          bias,
          inputs: [...inputArray],
          dotProduct: rawValue - bias,
          withBias: rawValue,
          activated: config.activated.data[i][0],
        };
        steps[config.key].push(neuronData);
      }
    }
    return steps;
  }

  /** Forward propagation data for a specific neuron (from the last feedforward). */
  getForwardNeuronData(layer: LayerName, index: number): ForwardCalculation | null {
    const steps = this.getForwardSteps();
    return steps?.[layer]?.[index] ?? null;
  }

  /** Backward propagation data for a specific neuron (from the last train()). */
  getBackwardNeuronData(layer: LayerName, index: number): BackwardCalculation | null {
    return this.lastBackwardSteps?.[layer]?.[index] ?? null;
  }
}
