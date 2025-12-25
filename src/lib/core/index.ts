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
