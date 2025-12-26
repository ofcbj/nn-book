/**
 * Core Neural Network Module
 * 
 * Contains fundamental neural network components:
 * - Matrix operations
 * - Network architecture and forward pass
 * - Backpropagation algorithms
 */

export { Matrix } from './matrix';
export { NeuralNetwork, sigmoid, dsigmoid, softmax } from './network';
export {
  backpropOutputLayer,
  backpropHiddenLayer,
  createBackpropSteps,
} from './backpropagation';
export {
  LAYER_NAMES,
  LAYER_SIZES,
  INPUT_SIZE,
  OUTPUT_CLASSES,
  getForwardNeuronIndices,
  getBackwardNeuronIndices,
  FORWARD_LAYER_ORDER,
  BACKWARD_LAYER_ORDER,
  getNextForwardNeuron,
  getNextBackwardNeuron,
  FORWARD_STAGES,
  BACKPROP_STAGES,
  getNextForwardStage,
  getNextBackpropStage,
} from './networkConfig';
export type { LayerName } from './networkConfig';
