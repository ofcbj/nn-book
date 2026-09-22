// Backpropagation helper functions for neural network training
import { Matrix } from './matrix';
import { dsigmoid } from './activations';
import type { BackwardSteps, BackwardCalculation } from '../types';
import { LAYER_SIZES, LAYER_NAMES } from './networkConfig';
import type { LayerName } from './networkConfig';

/**
 * Result of backpropagation through one layer.
 *
 * - errors:       the error each neuron "received"
 *                 (output: target − output, hidden: Σ δ_next · w)
 * - deltas:       δ = errors ⊙ f'(y); this is what gets propagated to the previous layer
 * - weightDeltas: ΔW = lr · δ · xᵀ   (added to the weights)
 * - biasDeltas:   Δb = lr · δ        (added to the biases)
 */
export interface LayerBackpropResult {
  errors: Matrix;
  deltas: Matrix;
  weightDeltas: Matrix;
  biasDeltas: Matrix;
}

/**
 * Output layer: softmax activation + cross-entropy loss.
 *
 * For this pairing ∂L/∂z = output − target, so the activation derivative
 * is already folded in and δ = target − output directly (no σ' factor).
 */
export function backpropOutputLayer(
  outputs: Matrix,
  targets: Matrix,
  previousLayer: Matrix,
  learningRate: number
): LayerBackpropResult {
  const errors = Matrix.subtract(targets, outputs);
  const deltas = errors.clone();

  const weightDeltas = Matrix.multiply(deltas, Matrix.transpose(previousLayer));
  weightDeltas.multiply(learningRate);
  const biasDeltas = Matrix.map(deltas, d => d * learningRate);

  return { errors, deltas, weightDeltas, biasDeltas };
}

/**
 * Hidden layer with sigmoid activation.
 *
 * error_i = Σ_j δ_next[j] · W[j][i]   (W = weights from this layer to the next, BEFORE update)
 * δ_i     = error_i · σ'(y_i)
 */
export function backpropHiddenLayer(
  nextLayerDeltas: Matrix,
  currentLayer: Matrix,
  previousLayer: Matrix,
  weightsCurrentToNext: Matrix,
  learningRate: number
): LayerBackpropResult {
  const errors = Matrix.multiply(Matrix.transpose(weightsCurrentToNext), nextLayerDeltas);

  const deltas = Matrix.map(currentLayer, dsigmoid);
  deltas.multiply(errors);

  const weightDeltas = Matrix.multiply(deltas, Matrix.transpose(previousLayer));
  weightDeltas.multiply(learningRate);
  const biasDeltas = Matrix.map(deltas, d => d * learningRate);

  return { errors, deltas, weightDeltas, biasDeltas };
}

// =============================================================================
// Backprop Visualizer Helpers
// =============================================================================

/** Everything the visualizer needs to know about one layer's backward pass. */
export interface LayerBackpropInput {
  activations: Matrix;
  inputs: Matrix;
  result: LayerBackpropResult;
  oldWeights: Matrix;
  oldBias: Matrix;
  newWeights: Matrix;
  newBias: Matrix;
  /** Hidden layers only: δ of the next layer and the (pre-update) weights to it */
  nextLayerDeltas?: Matrix;
  nextLayerWeights?: Matrix;
}

export interface BackpropData {
  layers: Record<LayerName, LayerBackpropInput>;
  target: number[];
  predictions: number[];
  loss: number;
}

function createLayerBackpropData(layerName: LayerName, layer: LayerBackpropInput): BackwardCalculation[] {
  const { activations, inputs, result, oldWeights, oldBias, newWeights, newBias, nextLayerDeltas, nextLayerWeights } = layer;
  const isOutputLayer = layerName === 'output';
  const inputArray = inputs.toArray();
  const neuronCount = LAYER_SIZES[layerName];
  const data: BackwardCalculation[] = [];

  for (let i = 0; i < neuronCount; i++) {
    const activation = activations.data[i][0];
    // Softmax + cross-entropy: derivative already folded into the error, so it is 1.
    const derivative = isOutputLayer ? 1 : dsigmoid(activation);

    const neuronData: BackwardCalculation = {
      neuronIndex : i,
      error       : result.errors.data[i][0],
      activation,
      derivative,
      gradient    : result.deltas.data[i][0],
      weightDeltas: [...result.weightDeltas.data[i]],
      biasDelta   : result.biasDeltas.data[i][0],
      oldWeights  : [...oldWeights.data[i]],
      newWeights  : [...newWeights.data[i]],
      oldBias     : oldBias.data[i][0],
      newBias     : newBias.data[i][0],
      inputs      : [...inputArray],
    };

    if (nextLayerDeltas && nextLayerWeights) {
      neuronData.nextLayerDeltas = nextLayerDeltas.toArray();
      // Column i of the next layer's weight matrix = weights leaving this neuron
      neuronData.nextLayerWeights = nextLayerWeights.data.map(row => row[i]);
    }

    data.push(neuronData);
  }

  return data;
}

/**
 * Create BackwardSteps for the visualizer.
 * All arrays are copies, so later training steps do not mutate stored data.
 */
export function createBackwardSteps(data: BackpropData): BackwardSteps {
  const steps = {} as Record<LayerName, BackwardCalculation[]>;
  for (const name of LAYER_NAMES) {
    steps[name] = createLayerBackpropData(name, data.layers[name]);
  }

  return {
    ...steps,
    totalLoss  : data.loss,
    targetClass: data.target.indexOf(1),
    predictions: [...data.predictions],
  };
}
