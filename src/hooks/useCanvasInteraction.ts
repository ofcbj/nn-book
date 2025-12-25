/**
 * Canvas Interaction Hook
 * 
 * Handles canvas click events for neural network visualization.
 * Extracted from useNeuralNetwork for better separation of concerns.
 */

import { useCallback, RefObject } from 'react';
import type { NeuralNetwork } from '../lib/core';
import type { Visualizer } from '../lib/visualizer';
import type { UseNetworkStateReturn } from './useNetworkState';
import type { AnimationStateMachine } from './useAnimationStateMachine';
import type { UseNetworkAnimationReturn } from './useNetworkAnimation';
import {
  getNextForwardStage,
  getNextBackpropStage,
  getNextForwardNeuron,
  getNextBackwardNeuron,
} from './useAnimationStateMachine';

export interface UseCanvasInteractionReturn {
  handleCanvasClick: (x?: number, y?: number) => void;
}

export function useCanvasInteraction(
  nnRef: RefObject<NeuralNetwork>,
  visualizerRef: RefObject<Visualizer | null>,
  state: UseNetworkStateReturn,
  animationMachine: AnimationStateMachine,
  animation: UseNetworkAnimationReturn
): UseCanvasInteractionReturn {

  const handleCanvasClick = useCallback((x?: number, y?: number) => {
    if (!animationMachine.isAnimating) return;

    const visualizer = visualizerRef.current;
    const nn = nnRef.current;
    const machineState = animationMachine.state;

    // If coordinates provided, check if clicking on a neuron
    if (x !== undefined && y !== undefined && visualizer) {
      const neuron = visualizer.findNeuronAtPosition(x, y);

      if (neuron && (neuron.layer === 'layer1' || neuron.layer === 'layer2' || neuron.layer === 'output')) {
        // Check if clicking on same neuron currently being animated
        const isSameNeuronForward = machineState.type === 'forward_animating' &&
          machineState.layer === neuron.layer &&
          machineState.neuronIndex === neuron.index;

        const isSameNeuronBackward = machineState.type === 'backward_animating' &&
          machineState.layer === neuron.layer &&
          machineState.neuronIndex === neuron.index;

        if (isSameNeuronForward) {
          // Same neuron clicked in forward mode - advance to next stage
          const nextStage = getNextForwardStage(machineState.stage);
          if (nextStage) {
            animationMachine.forwardTick(neuron.layer, neuron.index, nextStage, machineState.neuronData);
            animation.updateVisualization();
          } else {
            // All stages done for this neuron - move to next neuron
            const nextNeuron = getNextForwardNeuron(neuron.layer, neuron.index);
            if (nextNeuron) {
              const calcSteps = nn.getCalculationSteps();
              if (calcSteps) {
                const layerData = { layer1: calcSteps.layer1, layer2: calcSteps.layer2, output: calcSteps.output };
                const nextNeuronData = layerData[nextNeuron.layer as 'layer1' | 'layer2' | 'output'][nextNeuron.index];
                animationMachine.jumpToNeuron(nextNeuron.layer, nextNeuron.index);
                animationMachine.forwardTick(nextNeuron.layer, nextNeuron.index, 'dotProduct', nextNeuronData);
                animation.updateVisualization();
              }
            } else {
              // No next neuron - forward pass complete
              completeForwardPass(nn, state, animationMachine);
            }
          }
          return;
        }

        if (isSameNeuronBackward) {
          // Same neuron clicked in backward mode - advance to next backprop stage
          const nextStage = getNextBackpropStage(machineState.stage);
          if (nextStage) {
            animationMachine.backwardTick(neuron.layer, neuron.index, nextStage, machineState.neuronData);
            animation.updateVisualization();
          } else {
            // All stages done for this neuron - move to next neuron in backward order
            const nextNeuron = getNextBackwardNeuron(neuron.layer, neuron.index);
            if (nextNeuron) {
              const backpropData = nn.lastBackpropSteps;
              if (backpropData) {
                const layerData = { layer1: backpropData.layer1, layer2: backpropData.layer2, output: backpropData.output };
                const nextNeuronData = layerData[nextNeuron.layer as 'layer1' | 'layer2' | 'output'][nextNeuron.index];
                animationMachine.jumpToNeuron(nextNeuron.layer, nextNeuron.index);
                animationMachine.backwardTick(nextNeuron.layer, nextNeuron.index, 'error', nextNeuronData);
                animation.updateVisualization();
              }
            } else {
              // No next neuron - backward pass complete
              animationMachine.backwardComplete();
            }
          }
          return;
        }

        // Different neuron clicked - jump to it
        animation.shouldStopRef.current = true;
        animationMachine.jumpToNeuron(neuron.layer, neuron.index);

        // Set up neuron data based on current mode
        if (machineState.type === 'forward_animating') {
          const calcSteps = nn.getCalculationSteps();
          if (calcSteps) {
            const layerData = { layer1: calcSteps.layer1, layer2: calcSteps.layer2, output: calcSteps.output };
            const neuronData = layerData[neuron.layer][neuron.index];
            animationMachine.forwardTick(neuron.layer, neuron.index, 'dotProduct', neuronData);
          }
        } else if (machineState.type === 'backward_animating') {
          const backpropData = nn.lastBackpropSteps;
          if (backpropData) {
            const layerData = { layer1: backpropData.layer1, layer2: backpropData.layer2, output: backpropData.output };
            const neuronData = layerData[neuron.layer][neuron.index];
            animationMachine.backwardTick(neuron.layer, neuron.index, 'error', neuronData);
          }
        }

        animation.updateVisualization();
        return;
      }
    }

    // Click on empty space - pause/resume or next step
    if (animationMachine.state.speed > 0) {
      animationMachine.pause();
    } else {
      // We're paused - check if jumped to a neuron
      if (machineState.isJumped) {
        handleJumpedStateClick(nn, state, animationMachine, animation, machineState);
      } else {
        animationMachine.nextStep();
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animationMachine, animation]);

  return {
    handleCanvasClick,
  };
}

// Helper: Complete forward pass and show loss modal
function completeForwardPass(
  nn: NeuralNetwork,
  state: UseNetworkStateReturn,
  animationMachine: AnimationStateMachine
): void {
  const inputs = [state.grade, state.attitude, state.response];
  const targetOneHot = [0, 0, 0];
  targetOneHot[state.targetValue] = 1;

  // Backup old weights
  const oldWeights = {
    layer1: JSON.parse(JSON.stringify(nn.weights_input_hidden1.data)),
    layer2: JSON.parse(JSON.stringify(nn.weights_hidden1_hidden2.data)),
    output: JSON.parse(JSON.stringify(nn.weights_hidden2_output.data))
  };
  const oldBiases = {
    layer1: JSON.parse(JSON.stringify(nn.bias_hidden1.data)),
    layer2: JSON.parse(JSON.stringify(nn.bias_hidden2.data)),
    output: JSON.parse(JSON.stringify(nn.bias_output.data))
  };

  // Train to prepare backprop data
  nn.train(inputs, targetOneHot);
  const predictions = nn.lastOutput?.toArray() || [0, 0, 0];
  const currentLoss = nn.lastLoss;

  // Restore old weights for backprop visualization
  nn.weights_input_hidden1.data = oldWeights.layer1;
  nn.weights_hidden1_hidden2.data = oldWeights.layer2;
  nn.weights_hidden2_output.data = oldWeights.output;
  nn.bias_hidden1.data = oldBiases.layer1;
  nn.bias_hidden2.data = oldBiases.layer2;
  nn.bias_output.data = oldBiases.output;
  nn.feedforward(inputs);

  // Show loss modal
  animationMachine.forwardComplete();
  state.setLossModalData({ targetClass: state.targetValue, predictions, loss: currentLoss });
}

// Helper: Handle click when in jumped (paused) state
function handleJumpedStateClick(
  nn: NeuralNetwork,
  state: UseNetworkStateReturn,
  animationMachine: AnimationStateMachine,
  animation: UseNetworkAnimationReturn,
  machineState: ReturnType<typeof animationMachine.state extends infer T ? () => T : never>
): void {
  if (machineState.type === 'forward_animating') {
    const nextStage = getNextForwardStage(machineState.stage);
    if (nextStage) {
      animationMachine.forwardTick(machineState.layer, machineState.neuronIndex, nextStage, machineState.neuronData);
      animation.updateVisualization();
    } else {
      // All stages done - move to next neuron while staying paused
      const nextNeuron = getNextForwardNeuron(machineState.layer, machineState.neuronIndex);
      if (nextNeuron) {
        const calcSteps = nn.getCalculationSteps();
        if (calcSteps) {
          const layerData = { layer1: calcSteps.layer1, layer2: calcSteps.layer2, output: calcSteps.output };
          const nextNeuronData = layerData[nextNeuron.layer as 'layer1' | 'layer2' | 'output'][nextNeuron.index];
          animationMachine.jumpToNeuron(nextNeuron.layer, nextNeuron.index);
          animationMachine.forwardTick(nextNeuron.layer, nextNeuron.index, 'dotProduct', nextNeuronData);
          animation.updateVisualization();
        }
      } else {
        // No more neurons - forward pass complete
        completeForwardPass(nn, state, animationMachine);
      }
    }
  } else if (machineState.type === 'backward_animating') {
    const nextStage = getNextBackpropStage(machineState.stage);
    if (nextStage) {
      // Apply update if completing 'update' stage
      if (machineState.stage === 'update' && machineState.neuronData) {
        const neuronData = machineState.neuronData;
        if (machineState.layer === 'output') {
          nn.weights_hidden2_output.data[machineState.neuronIndex] = neuronData.newWeights;
          nn.bias_output.data[machineState.neuronIndex][0] = neuronData.newBias;
        } else if (machineState.layer === 'layer2') {
          nn.weights_hidden1_hidden2.data[machineState.neuronIndex] = neuronData.newWeights;
          nn.bias_hidden2.data[machineState.neuronIndex][0] = neuronData.newBias;
        } else if (machineState.layer === 'layer1') {
          nn.weights_input_hidden1.data[machineState.neuronIndex] = neuronData.newWeights;
          nn.bias_hidden1.data[machineState.neuronIndex][0] = neuronData.newBias;
        }
        nn.feedforward(nn.lastInput!.toArray());
      }
      animationMachine.backwardTick(machineState.layer, machineState.neuronIndex, nextStage, machineState.neuronData);
      animation.updateVisualization();
    } else {
      // All stages done - move to next neuron while staying paused
      const nextNeuron = getNextBackwardNeuron(machineState.layer, machineState.neuronIndex);
      if (nextNeuron) {
        const backpropData = nn.lastBackpropSteps;
        if (backpropData) {
          const layerData = { layer1: backpropData.layer1, layer2: backpropData.layer2, output: backpropData.output };
          const nextNeuronData = layerData[nextNeuron.layer as 'layer1' | 'layer2' | 'output'][nextNeuron.index];
          animationMachine.jumpToNeuron(nextNeuron.layer, nextNeuron.index);
          animationMachine.backwardTick(nextNeuron.layer, nextNeuron.index, 'error', nextNeuronData);
          animation.updateVisualization();
        }
      } else {
        // No more neurons - backward pass complete
        animationMachine.backwardComplete();
      }
    }
  }
}
