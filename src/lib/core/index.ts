/**
 * Core Neural Network Module
 *
 * Contains fundamental neural network components:
 * - Matrix operations
 * - Activation functions and loss
 * - Network architecture and forward pass
 * - Backpropagation algorithms
 * - Snapshot / comparison helpers
 */

export { Matrix } from './matrix';
export { sigmoid, dsigmoid, softmax, crossEntropyLoss } from './activations';
export { NeuralNetwork, DEFAULT_LEARNING_RATE } from './network';
export {
  backpropOutputLayer,
  backpropHiddenLayer,
  createBackwardSteps,
} from './backpropagation';
export type { LayerBackpropResult } from './backpropagation';
export {
  LAYER_NAMES,
  LAYER_SIZES,
  INPUT_SIZE,
  OUTPUT_CLASSES,
  toOneHot,
  getForwardNeuronIndices,
  getBackwardNeuronIndices,
  FORWARD_LAYER_ORDER,
  BACKWARD_LAYER_ORDER,
  getNextForwardNeuron,
  getNextBackwardNeuron,
  FORWARD_STAGES,
  BACKPROP_STAGES,
  getNextForwardStage,
  getNextBackwardStage,
} from './networkConfig';
export type { LayerName, NeuronLocation } from './networkConfig';
export { createWeightComparisonData, createBackpropSummaryData } from './weightComparison';
export { createSnapshot, compareSnapshots } from './networkSnapshot';
export type { NetworkSnapshot } from './networkSnapshot';
