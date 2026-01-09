/**
 * Canvas Interaction Hook
 * 
 * Handles canvas click events for neural network visualizer.
 * Extracted from useNeuralNetwork for better separation of concerns.
 */

import { useCallback, RefObject } from 'react';
import type { NeuralNetwork, LayerName } from '../lib/core';
import type { Visualizer } from '../lib/visualizer';
import type { UseNetworkStateReturn } from './useNetworkState';
import type { AnimationStateMachine } from './useAnimationStateMachine';
import type { UseNetworkAnimationReturn } from './useNetworkAnimation';
import type { 
  ForwardAnimatingState, 
  BackwardAnimatingState,
  AnimationState,
} from '../lib/animation';
import {
  getNextForwardStage,
  getNextBackpropStage,
  getNextForwardNeuron,
  getNextBackwardNeuron,
} from './useAnimationStateMachine';

export interface UseCanvasInteractionReturn {
  handleCanvasClick: (x?: number, y?: number) => void;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper Types
// ─────────────────────────────────────────────────────────────────────────────

type NeuronLocation = { layer: LayerName; index: number };

interface AnimationContext {
  nn: NeuralNetwork;
  state: UseNetworkStateReturn;
  animationMachine: AnimationStateMachine;
  animation: UseNetworkAnimationReturn;
}

// ─────────────────────────────────────────────────────────────────────────────
// Data Access Helpers
// ─────────────────────────────────────────────────────────────────────────────

function getForwardNeuronData(nn: NeuralNetwork, layer: LayerName, index: number) {
  const calcSteps = nn.getCalculationSteps();
  if (!calcSteps) return null;
  return calcSteps[layer]?.[index] ?? null;
}

function getBackwardNeuronData(nn: NeuralNetwork, layer: LayerName, index: number) {
  const backpropData = nn.lastBackpropSteps;
  if (!backpropData) return null;
  return backpropData[layer]?.[index] ?? null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Neuron Transition Helpers
// ─────────────────────────────────────────────────────────────────────────────

function transitionToNextForwardNeuron(
  ctx: AnimationContext,
  nextNeuron: NeuronLocation
): boolean {
  const neuronData = getForwardNeuronData(ctx.nn, nextNeuron.layer, nextNeuron.index);
  if (!neuronData) return false;
  
  ctx.animationMachine.jumpToNeuron(nextNeuron.layer, nextNeuron.index);
  ctx.animationMachine.forwardTick(nextNeuron.layer, nextNeuron.index, 'dotProduct', neuronData);
  ctx.animation.refreshDisplayOnly();  // No recalculation needed - uses pre-calculated data
  return true;
}

function transitionToNextBackwardNeuron(
  ctx: AnimationContext,
  nextNeuron: NeuronLocation
): boolean {
  const neuronData = getBackwardNeuronData(ctx.nn, nextNeuron.layer, nextNeuron.index);
  if (!neuronData) return false;
  
  ctx.animationMachine.jumpToNeuron(nextNeuron.layer, nextNeuron.index);
  ctx.animationMachine.backwardTick(nextNeuron.layer, nextNeuron.index, 'error', neuronData);
  ctx.animation.refreshDisplayOnly();  // No recalculation needed - uses pre-calculated data
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward/Backward Stage Handlers
// ─────────────────────────────────────────────────────────────────────────────

function handleForwardSameNeuronClick(
  ctx: AnimationContext,
  neuron: NeuronLocation,
  machineState: ForwardAnimatingState
): void {
  const nextStage = getNextForwardStage(machineState.stage);
  
  if (nextStage) {
    ctx.animationMachine.forwardTick(neuron.layer, neuron.index, nextStage, machineState.neuronData);
    ctx.animation.refreshDisplayOnly();  // No recalculation needed - just update visualization
    return;
  }
  
  // All stages done - move to next neuron
  const nextNeuron = getNextForwardNeuron(neuron.layer, neuron.index);
  if (nextNeuron) {
    transitionToNextForwardNeuron(ctx, nextNeuron);
  } else {
    completeForwardPass(ctx);
  }
}

function handleBackwardSameNeuronClick(
  ctx: AnimationContext,
  neuron: NeuronLocation,
  machineState: BackwardAnimatingState
): void {
  const nextStage = getNextBackpropStage(machineState.stage);
  
  if (nextStage) {
    ctx.animationMachine.backwardTick(neuron.layer, neuron.index, nextStage, machineState.neuronData);
    ctx.animation.refreshDisplayOnly();  // No recalculation needed - just update visualization
    return;
  }
  
  // All stages done - move to next neuron in backward order
  const nextNeuron = getNextBackwardNeuron(neuron.layer, neuron.index);
  if (nextNeuron) {
    transitionToNextBackwardNeuron(ctx, nextNeuron);
  } else {
    ctx.animationMachine.backwardComplete();
  }
}

function handleJumpToDifferentNeuron(
  ctx: AnimationContext,
  neuron: NeuronLocation,
  machineState: AnimationState
): void {
  ctx.animation.shouldStopRef.current = true;
  ctx.animationMachine.jumpToNeuron(neuron.layer, neuron.index);

  if (machineState.type === 'forward_animating') {
    const neuronData = getForwardNeuronData(ctx.nn, neuron.layer, neuron.index);
    if (neuronData) {
      ctx.animationMachine.forwardTick(neuron.layer, neuron.index, 'dotProduct', neuronData);
    }
  } else if (machineState.type === 'backward_animating') {
    const neuronData = getBackwardNeuronData(ctx.nn, neuron.layer, neuron.index);
    if (neuronData) {
      ctx.animationMachine.backwardTick(neuron.layer, neuron.index, 'error', neuronData);
    }
  }

  ctx.animation.refreshDisplayOnly();  // No recalculation needed - just visualizing existing data
}

// ─────────────────────────────────────────────────────────────────────────────
// Paused State Click Handlers
// ─────────────────────────────────────────────────────────────────────────────

function handleForwardJumpedClick(
  ctx: AnimationContext,
  machineState: ForwardAnimatingState
): void {
  const nextStage = getNextForwardStage(machineState.stage);
  
  if (nextStage) {
    ctx.animationMachine.forwardTick(machineState.layer, machineState.neuronIndex, nextStage, machineState.neuronData);
    ctx.animation.refreshDisplayOnly();  // No recalculation needed - just update visualization
    return;
  }
  
  // All stages done - move to next neuron while staying paused
  const nextNeuron = getNextForwardNeuron(machineState.layer, machineState.neuronIndex);
  if (nextNeuron) {
    transitionToNextForwardNeuron(ctx, nextNeuron);
  } else {
    completeForwardPass(ctx);
  }
}

function handleBackwardJumpedClick(
  ctx: AnimationContext,
  machineState: BackwardAnimatingState
): void {
  const nextStage = getNextBackpropStage(machineState.stage);
  
  if (nextStage) {
    // Apply update if completing 'update' stage
    if (machineState.stage === 'update' && machineState.neuronData) {
      const neuronData = machineState.neuronData;
      ctx.nn.updateNeuronWeights(machineState.layer, machineState.neuronIndex, neuronData.newWeights, neuronData.newBias);
      ctx.nn.feedforward(ctx.nn.lastInput!.toArray());
    }
    ctx.animationMachine.backwardTick(machineState.layer, machineState.neuronIndex, nextStage, machineState.neuronData);
    ctx.animation.refreshDisplayOnly();  // No recalculation needed - just update visualization
    return;
  }
  
  // All stages done - move to next neuron while staying paused
  const nextNeuron = getNextBackwardNeuron(machineState.layer, machineState.neuronIndex);
  if (nextNeuron) {
    transitionToNextBackwardNeuron(ctx, nextNeuron);
  } else {
    ctx.animationMachine.backwardComplete();
  }
}

function handleJumpedStateClick(ctx: AnimationContext, machineState: AnimationState): void {
  if (machineState.type === 'forward_animating') {
    handleForwardJumpedClick(ctx, machineState);
  } else if (machineState.type === 'backward_animating') {
    handleBackwardJumpedClick(ctx, machineState);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward Pass Completion
// ─────────────────────────────────────────────────────────────────────────────

function completeForwardPass(ctx: AnimationContext): void {
  const { nn, state, animationMachine } = ctx;
  const inputs = [state.inputs.grade, state.inputs.attitude, state.inputs.response];
  const targetOneHot = [0, 0, 0];
  targetOneHot[state.inputs.targetValue] = 1;

  // Backup old weights
  const oldWeights = {
    layer1: JSON.parse(JSON.stringify(nn.weightsInputHidden1.data)),
    layer2: JSON.parse(JSON.stringify(nn.weightsHidden1Hidden2.data)),
    output: JSON.parse(JSON.stringify(nn.weightsHidden2Output.data))
  };
  const oldBiases = {
    layer1: JSON.parse(JSON.stringify(nn.biasHidden1.data)),
    layer2: JSON.parse(JSON.stringify(nn.biasHidden2.data)),
    output: JSON.parse(JSON.stringify(nn.biasOutput.data))
  };

  // Train to prepare backprop data
  nn.train(inputs, targetOneHot);
  const predictions = nn.lastOutput?.toArray() || [0, 0, 0];
  const currentLoss = nn.lastLoss;

  // Restore old weights for backprop visualizer
  nn.weightsInputHidden1.data = oldWeights.layer1;
  nn.weightsHidden1Hidden2.data = oldWeights.layer2;
  nn.weightsHidden2Output.data = oldWeights.output;
  nn.biasHidden1.data = oldBiases.layer1;
  nn.biasHidden2.data = oldBiases.layer2;
  nn.biasOutput.data = oldBiases.output;
  nn.feedforward(inputs);

  // Show loss modal
  animationMachine.forwardComplete();
  state.modalSetters.setLossModalData({ targetClass: state.inputs.targetValue, predictions, loss: currentLoss });
}

// ─────────────────────────────────────────────────────────────────────────────
// Neuron Click Detection
// ─────────────────────────────────────────────────────────────────────────────

function isValidNeuronLayer(layer: string): layer is LayerName {
  return layer === 'layer1' || layer === 'layer2' || layer === 'output';
}

function isSameNeuron(
  machineState: AnimationState, 
  expectedType: 'forward_animating' | 'backward_animating',
  neuron: NeuronLocation
): boolean {
  if (machineState.type !== expectedType) return false;
  // Type narrowing happens after the check above
  const animatingState = machineState as ForwardAnimatingState | BackwardAnimatingState;
  return animatingState.layer === neuron.layer && animatingState.neuronIndex === neuron.index;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Hook
// ─────────────────────────────────────────────────────────────────────────────

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
    const ctx: AnimationContext = { nn, state, animationMachine, animation };

    // Handle neuron click if coordinates provided
    if (x !== undefined && y !== undefined && visualizer) {
      const neuron = visualizer.findNeuronAtPosition(x, y);

      if (neuron && isValidNeuronLayer(neuron.layer)) {
        const neuronLoc: NeuronLocation = { layer: neuron.layer, index: neuron.index };

        if (isSameNeuron(machineState, 'forward_animating', neuronLoc)) {
          handleForwardSameNeuronClick(ctx, neuronLoc, machineState as ForwardAnimatingState);
          return;
        }

        if (isSameNeuron(machineState, 'backward_animating', neuronLoc)) {
          handleBackwardSameNeuronClick(ctx, neuronLoc, machineState as BackwardAnimatingState);
          return;
        }

        // Different neuron clicked - jump to it
        handleJumpToDifferentNeuron(ctx, neuronLoc, machineState);
        return;
      }
    }

    // Click on empty space - pause/resume or next step
    if (animationMachine.state.speed > 0) {
      animationMachine.pause();
    } else if (machineState.isJumped) {
      handleJumpedStateClick(ctx, machineState);
    } else {
      animationMachine.nextStep();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animationMachine, animation]);

  return {
    handleCanvasClick,
  };
}
