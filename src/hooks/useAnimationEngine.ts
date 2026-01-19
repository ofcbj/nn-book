/**
 * Animation Engine Hook
 *
 * Core animation and training orchestrator that combines:
 * - Animation State Machine (FSM for animation states)
 * - Animation Loops (forward/backward propagation animation)
 * - Training Controls (train step/epoch, reset)
 * - Modal Controls (loss modal, backprop modal)
 * - Canvas Interaction (neuron click handling)
 *
 * State management is handled by useNetworkState hook.
 *
 * Architecture: 5 hooks → 2 hooks (useNetworkState + useAnimationEngine)
 */

import { useReducer, useCallback, useRef, RefObject, useEffect, useMemo } from 'react';
import { NeuralNetwork } from '../lib/core';
import type { LayerName } from '../lib/core';
import { getForwardNeuronIndices, getBackwardNeuronIndices } from '../lib/core';
import type { Visualizer } from '../lib/visualizer';
import type { UseNetworkStateReturn } from './useNetworkState';
import type { ForwardStage, BackwardStage, ForwardCalculation, BackwardCalculation } from '../lib/types';
import { createBackpropSummaryData } from '../lib/types';
import { createSnapshot, compareSnapshots } from '../lib/core/networkSnapshot';
import { AnimationState, AnimationAction, animationReducer, 
  initialAnimationState, checkAnimating, checkPaused, 
  getNextForwardStage, getNextBackwardStage, getForwardStage, getBackwardStage,
  getNextForwardNeuron, getNextBackwardNeuron, getHighlightedNeuron,
  getCurrentForwardNeuronData, getCurrentBackwardNeuronData,
  isForwardAnimatingAtNeuron, isBackwardAnimatingAtNeuron,
  FORWARD_STAGES, BACKPROP_STAGES, InterruptReason, runAnimationLoop
} from '../lib/animation';
import {
  FORWARD_STAGE_DURATIONS,
  BACKWARD_STAGE_DURATIONS,
} from '../lib/animation/animationLoop';

export interface UseAnimationEngineReturn {
  // === Animation State ===
  state               : AnimationState;
  dispatch            : React.Dispatch<AnimationAction>;
  isAnimating         : boolean;
  isPaused            : boolean;
  // === Visualization Data ===
  highlightedNeuron   : { layer: LayerName; index: number } | null;
  forwardStage        : ForwardStage | null;
  backpropStage       : BackwardStage | null;
  currentForwardData  : ForwardCalculation | null;
  currentBackwardData  : BackwardCalculation | null;
  // === Training Controls ===
  trainOneStepWithAnimation      : () => Promise<void>;
  trainOneEpochWithoutAnimation  : () => void;
  toggleTraining                 : () => void;
  reset                          : () => void;
  computeAndRefreshDisplay       : () => void;
  // === Modal Controls ===
  closeLossModal                 : () => Promise<void>;
  closeBackpropModal             : () => void;
  // === Canvas Interaction ===
  handleCanvasClick              : (x?: number, y?: number) => void;
  // === Visualizer ===
  setVisualizer                  : (v: Visualizer) => void;
  // === Utilities ===
  handleLearningRateChange       : (v: number) => void;
  trainingIntervalRef            : RefObject<number | undefined>;
}

type NeuronLocation = { layer: LayerName; index: number };

export function useAnimationEngine(
  displayNNRef  : RefObject<NeuralNetwork>,  // For visualization (weights unchanged during forward)
  trainNNRef    : RefObject<NeuralNetwork>,  // For training (loss calculation, backprop data)
  visualizerRef : RefObject<Visualizer | null>,
  state         : UseNetworkStateReturn
): UseAnimationEngineReturn {
  // 1. ANIMATION STATE MACHINE
  const [animationState, dispatch]= useReducer(animationReducer, initialAnimationState);
  const animationStateRef         = useRef(animationState);  // Always-current ref for async access
  const interruptReasonRef        = useRef<InterruptReason>('none'); // Synced with FSM
  const trainingIntervalRef       = useRef<number | undefined>(undefined);

  // Sync refs with FSM state (refs are needed for async loop access)
  useEffect(() => {
    animationStateRef.current     = animationState;
    interruptReasonRef.current    = animationState.interruptReason;
  }, [animationState]);
  // FSM action creators (grouped for cleaner code)
  const fsmActions = useMemo(() => ({
    startTraining     : () => dispatch({ type: 'START_TRAINING' }),
    pause             : () => dispatch({ type: 'PAUSE' }),
    resume            : () => dispatch({ type: 'RESUME' }),
    reset             : () => dispatch({ type: 'RESET' }),
    jumpToNeuron      : (layer: LayerName, neuronIndex: number) =>
      dispatch({ type: 'JUMP_TO_NEURON', layer, neuronIndex }),    
    
    forwardTick       : (layer: LayerName, neuronIndex: number, stage: ForwardStage, neuronData: ForwardCalculation | null) =>
      dispatch({ type: 'FORWARD_TICK', layer, neuronIndex, stage, neuronData }),
    forwardComplete   : () => dispatch({ type: 'FORWARD_COMPLETE' }),
    closeLossModal    : () => dispatch({ type: 'CLOSE_LOSS_MODAL' }),
    
    backwardTick      : (layer: LayerName, neuronIndex: number, stage: BackwardStage, neuronData: BackwardCalculation | null) =>
      dispatch({ type: 'BACKWARD_TICK', layer, neuronIndex, stage, neuronData }),
    backwardComplete  : () => dispatch({ type: 'BACKWARD_COMPLETE' }),
    closeBackpropModal: () => dispatch({ type: 'CLOSE_BACKPROP_MODAL' }),
  }), []);

  // Derived state
  const isAnimating = checkAnimating(animationState);
  const isPaused    = checkPaused(animationState);

  // Animation Control Helpers
  const shouldPauseAnimation = useCallback(() => {
    return interruptReasonRef.current !== 'none';
  }, []);

  const clearTrainingInterval = useCallback(() => {
    if (trainingIntervalRef.current) {
      clearInterval(trainingIntervalRef.current);
      trainingIntervalRef.current = undefined;
    }
  }, []);

  // 2. ANIMATION LOOPS & VISUALIZATION
  // Helper: Get current input array from state
  const getCurrentInputs = useCallback(() => {
    return [state.inputs.grade, state.inputs.attitude, state.inputs.response];
  }, [state.inputs.grade, state.inputs.attitude, state.inputs.response]);
  
  // Helper: Get target one-hot array from state
  const getTargetOneHot = useCallback(() => {
    const oneHot = [0, 0, 0];
    oneHot[state.inputs.targetValue] = 1;
    return oneHot;
  }, [state.inputs.targetValue]);
  
  // Helper: Create weight comparison data after training
  const createWeightComparisonAfterTraining = useCallback((
    oldSnapshot: ReturnType<typeof createSnapshot>,
    newSnapshot: ReturnType<typeof createSnapshot>
  ) => {
    const comparisonData = compareSnapshots(oldSnapshot, newSnapshot, state.stats.learningRate);
    state.modalSetters.setWeightComparisonData(comparisonData);
  }, [state.stats.learningRate, state.modalSetters]);
  
  // Helper: Show loss modal with current predictions (uses trainNN for loss data)
  const showLossModal = useCallback(() => {
    const trainNN       = trainNNRef.current;
    const predictions   = trainNN.lastOutput?.toArray() || [0, 0, 0];
    const loss          = trainNN.lastLoss;
    state.modalSetters.setLossModalData({
      targetClass: state.inputs.targetValue,
      predictions,
      loss
    });
  }, [trainNNRef, state.inputs.targetValue, state.modalSetters]);

  // Refresh display without recalculation (uses displayNN for visualization)
  const refreshDisplayOnly = useCallback(() => {
    if (visualizerRef.current) {
      visualizerRef.current.update(displayNNRef.current, animationStateRef.current);
    }
  }, [displayNNRef, visualizerRef]);

  // Compute and refresh display (full recalculation using displayNN)
  const computeAndRefreshDisplay = useCallback(() => {
    const nn = displayNNRef.current;
    nn.feedforward(getCurrentInputs());

    if (nn.lastOutput) {
      state.statsSetters.setOutput(nn.lastOutput.toArray());
    }
    state.statsSetters.setSteps(nn.getForwardSteps());
    // Update activations for heatmap
    if (nn.lastInput && nn.lastHidden1 && nn.lastHidden2 && nn.lastOutput) {
      state.visualizerSetters.setActivations({
        input : nn.lastInput.toArray(),
        layer1: nn.lastHidden1.toArray(),
        layer2: nn.lastHidden2.toArray(),
        output: nn.lastOutput.toArray(),
      });
    }

    refreshDisplayOnly();
  }, [getCurrentInputs, refreshDisplayOnly, displayNNRef, state.statsSetters, state.visualizerSetters]);
  
  // Sleep utility for animation timing
  const sleep = useCallback(async (ms: number, overrideSpeed?: number): Promise<void> => {
    const effectiveSpeed = overrideSpeed ?? state.training.animationSpeed;
    await new Promise(resolve => setTimeout(resolve, ms / effectiveSpeed));
  }, [state.training.animationSpeed]);

  // Helper: Complete forward pass (used by canvas click handler and animation loop)
  // Uses trainNN for loss calculation - displayNN stays unchanged
  const completeForwardPass = useCallback((options?: { skipForwardComplete?: boolean }) => {
    const trainNN = trainNNRef.current;
    const inputs  = getCurrentInputs();

    // Sync trainNN with displayNN before training (to get same forward pass results)
    trainNN.copyWeightsFrom(displayNNRef.current);
    // Train on trainNN to get loss and backprop data (displayNN unchanged)
    trainNN.train(inputs, getTargetOneHot());

    if (!options?.skipForwardComplete) {
      fsmActions.forwardComplete();
    }
    showLossModal();
  }, [trainNNRef, displayNNRef, getCurrentInputs, getTargetOneHot, showLossModal, fsmActions]);

  // Forward propagation animation (unified: can start from beginning or from a specific position)
  // Uses displayNN for visualization
  const animateForwardPropagation = useCallback(async (
    startFrom?: { layer: LayerName; neuronIndex: number }
  ): Promise<boolean> => {
    const displayNN   = displayNNRef.current;
    const forwardSteps= displayNN.getForwardSteps();
    if (!forwardSteps)
      return false;

    const layerData = {
      layer1: forwardSteps.layer1,
      layer2: forwardSteps.layer2,
      output: forwardSteps.output
    };

    const completed = await runAnimationLoop({
      mode              : 'forward',
      layers            : ['layer1', 'layer2', 'output'],
      getNeuronIndices  : getForwardNeuronIndices,
      stages            : FORWARD_STAGES,
      stageDurations    : FORWARD_STAGE_DURATIONS,
      getData           : () => layerData,
      onTick            : fsmActions.forwardTick,
      onComplete        : fsmActions.forwardComplete,
      refreshDisplayOnly: refreshDisplayOnly,
      shouldStop        : shouldPauseAnimation,
      sleep             : sleep,
      computeAndRefreshDisplay: computeAndRefreshDisplay,
      startFrom         : startFrom,
    });

    // Show loss modal if animation completed
    if (completed) {
      completeForwardPass({ skipForwardComplete: true });
    }

    return completed;
  }, [fsmActions, refreshDisplayOnly, sleep, computeAndRefreshDisplay, displayNNRef, shouldPauseAnimation, completeForwardPass]);
  // Backward propagation animation (unified: can start from beginning or from a specific position)
  // Uses trainNN for backprop data, updates displayNN weights during animation
  const animateBackwardPropagation = useCallback(async (
    startFrom?: { layer: LayerName; neuronIndex: number }
  ): Promise<boolean> => {
    const displayNN   = displayNNRef.current;
    const trainNN     = trainNNRef.current;
    const backpropData= trainNN.lastBackwardSteps;
    if (!backpropData) return false;

    const layerData = {
      layer1: backpropData.layer1,
      layer2: backpropData.layer2,
      output: backpropData.output
    };

    // Helper for weight updates on 'update' stage - updates displayNN
    const handleStageComplete = (layer: LayerName, neuronIndex: number, stage: BackwardStage, data: BackwardCalculation) => {
      if (stage === 'update') {
        displayNN.updateNeuronWeights(layer, neuronIndex, data.newWeights, data.newBias);
        displayNN.feedforward(displayNN.lastInput!.toArray());
        refreshDisplayOnly();
      }
    };

    const completed = await runAnimationLoop({
      mode              : 'backward',
      layers            : ['output', 'layer2', 'layer1'],
      getNeuronIndices  : getBackwardNeuronIndices,
      stages            : BACKPROP_STAGES,
      stageDurations    : BACKWARD_STAGE_DURATIONS,
      getData           : () => layerData,
      onTick            : fsmActions.backwardTick,
      onStageComplete   : handleStageComplete,
      onComplete        : fsmActions.backwardComplete,
      shouldStop        : shouldPauseAnimation,
      sleep             : sleep,
      refreshDisplayOnly: refreshDisplayOnly,
      computeAndRefreshDisplay: computeAndRefreshDisplay,
      speedOverride     : state.training.animationSpeed,
      startFrom         : startFrom,
    });

    // Collect summary data only if animation completed (not stopped)
    if (completed) {
      const summaryData = createBackpropSummaryData(backpropData, state.stats.learningRate);
      state.modalSetters.setBackpropSummaryData(summaryData);
    }
    return completed;
  }, [fsmActions, sleep, refreshDisplayOnly, computeAndRefreshDisplay, displayNNRef, trainNNRef, state.stats.learningRate, state.modalSetters, shouldPauseAnimation, state.training.animationSpeed]);
  
  // Continue from jumped position (dispatcher)
  const continueFromJumpedPosition = useCallback(async () => {
    if (animationState.type === 'forward_animating') {
      await animateForwardPropagation({
        layer: animationState.layer,
        neuronIndex: animationState.neuronIndex
      });
    } else if (animationState.type === 'backward_animating') {
      await animateBackwardPropagation({
        layer: animationState.layer,
        neuronIndex: animationState.neuronIndex
      });
    }
  }, [animationState, animateForwardPropagation, animateBackwardPropagation]);

  // 3. TRAINING CONTROLS
  const setVisualizer = useCallback((v: Visualizer) => {
    visualizerRef.current = v;
  }, [visualizerRef]);

  // Train one epoch without animation (directly updates displayNN)
  const trainOneEpochWithoutAnimation = useCallback(() => {
    const displayNN = displayNNRef.current;
    const oldSnapshot = createSnapshot(displayNN);
    displayNN.train(getCurrentInputs(), getTargetOneHot());

    const newSnapshot = createSnapshot(displayNN);
    createWeightComparisonAfterTraining(oldSnapshot, newSnapshot);

    state.statsSetters.setLoss(displayNN.lastLoss);
    state.statsSetters.setEpoch(prev => prev + 1);
    computeAndRefreshDisplay();
  }, [getCurrentInputs, getTargetOneHot, createWeightComparisonAfterTraining, state.statsSetters, computeAndRefreshDisplay, displayNNRef]);

  // Train one step with animation
  const trainOneStepWithAnimation = useCallback(async () => {
    if (isAnimating) {
      if (isPaused) {
        // Resume from paused position
        interruptReasonRef.current = 'none';
        fsmActions.resume();
        await continueFromJumpedPosition();
      } else {
        // Pause running animation
        fsmActions.pause();
      }
    } else {
      // Start new animation
      fsmActions.startTraining();
      const displayNN = displayNNRef.current;
      displayNN.feedforward(getCurrentInputs());
      await animateForwardPropagation();
    }
  }, [isAnimating, isPaused, state.training.animationSpeed, getCurrentInputs, fsmActions, continueFromJumpedPosition, animateForwardPropagation, displayNNRef]);

  // Toggle auto training
  const toggleTraining = useCallback(() => {
    if (state.training.isTraining) {
      state.trainingSetters.setIsTraining(false);
      clearTrainingInterval();
    } else {
      state.trainingSetters.setIsTraining(true);
      trainingIntervalRef.current = window.setInterval(() => {
        trainOneEpochWithoutAnimation();
        if (displayNNRef.current.lastLoss < 0.001) {
          state.trainingSetters.setIsTraining(false);
          clearTrainingInterval();
        }
      }, 50);
    }
  }, [state.training.isTraining, state.trainingSetters, trainOneEpochWithoutAnimation, displayNNRef, clearTrainingInterval]);
  // Reset (resets both displayNN and trainNN)
  const reset = useCallback(() => {
    // Stop auto training if running
    if (state.training.isTraining) {
      state.trainingSetters.setIsTraining(false);
      clearTrainingInterval();
    }
    // Reset both networks
    displayNNRef.current = new NeuralNetwork();
    trainNNRef.current = new NeuralNetwork();
    trainNNRef.current.copyWeightsFrom(displayNNRef.current);
    // Reset all state (stats, modals, inputs)
    state.resetAllState();
    // Reset FSM and refresh
    fsmActions.reset();
    computeAndRefreshDisplay();
  }, [state.training.isTraining, state.trainingSetters, state.resetAllState, fsmActions, computeAndRefreshDisplay, displayNNRef, trainNNRef, clearTrainingInterval]);

  // Learning rate change (updates both networks)
  const handleLearningRateChange = useCallback((v: number) => {
    state.statsSetters.setLearningRate(v);
    displayNNRef.current.learningRate = v;
    trainNNRef.current.learningRate = v;
  }, [state.statsSetters, displayNNRef, trainNNRef]);

  // 4. MODAL CONTROLS
  // Close loss modal - start backward propagation
  // Backward animation updates displayNN weights incrementally
  const closeLossModal = useCallback(async () => {
    state.modalSetters.setLossModalData(null);
    fsmActions.closeLossModal();

    const displayNN = displayNNRef.current;
    const oldSnapshot = createSnapshot(displayNN);

    const completed = await animateBackwardPropagation();
    await sleep(500, state.training.animationSpeed);
    // Update stats and weight comparison data if animation completed
    if (completed) {
      const newSnapshot = createSnapshot(displayNN);
      const comparisonData = compareSnapshots(oldSnapshot, newSnapshot, state.stats.learningRate);
      state.modalSetters.setWeightComparisonData(comparisonData);
      state.statsSetters.setEpoch(prev => prev + 1);
      state.statsSetters.setLoss(displayNN.lastLoss);
    }

    computeAndRefreshDisplay();
  }, [
    state.modalSetters,
    state.statsSetters,
    state.training.animationSpeed,
    state.stats.learningRate,
    fsmActions,
    animateBackwardPropagation,
    sleep,
    computeAndRefreshDisplay,
    displayNNRef,
  ]);

  // Close backprop modal
  const closeBackpropModal = useCallback(() => {
    state.modalSetters.setBackpropSummaryData(null);
    fsmActions.closeBackpropModal();
    refreshDisplayOnly();
  }, [state.modalSetters, fsmActions, refreshDisplayOnly]);

  // 5. CANVAS INTERACTION

  // Helper: Advance forward animation to next stage
  const advanceForwardStage = useCallback((neuronLoc: NeuronLocation) => {
    if (animationState.type !== 'forward_animating') return;

    const nextStage = getNextForwardStage(animationState.stage);
    if (nextStage) {
      fsmActions.forwardTick(neuronLoc.layer, neuronLoc.index, nextStage, animationState.neuronData);
      refreshDisplayOnly();
      return;
    }

    // No more stages - move to next neuron or complete
    const nextNeuron = getNextForwardNeuron(neuronLoc.layer, neuronLoc.index);
    if (nextNeuron) {
      const neuronData = displayNNRef.current.getForwardNeuronData(nextNeuron.layer, nextNeuron.index);
      if (neuronData) {
        fsmActions.jumpToNeuron(nextNeuron.layer, nextNeuron.index);
        fsmActions.forwardTick(nextNeuron.layer, nextNeuron.index, 'dotProduct', neuronData);
        refreshDisplayOnly();
      }
    } else {
      completeForwardPass();
    }
  }, [animationState, fsmActions, refreshDisplayOnly, displayNNRef, completeForwardPass]);

  // Helper: Advance backward animation to next stage
  const advanceBackwardStage = useCallback((neuronLoc: NeuronLocation) => {
    if (animationState.type !== 'backward_animating') return;

    const nextStage = getNextBackwardStage(animationState.stage);
    if (nextStage) {
      fsmActions.backwardTick(neuronLoc.layer, neuronLoc.index, nextStage, animationState.neuronData);
      refreshDisplayOnly();
      return;
    }

    // No more stages - move to next neuron or complete
    const nextNeuron = getNextBackwardNeuron(neuronLoc.layer, neuronLoc.index);
    if (nextNeuron) {
      const neuronData = trainNNRef.current.getBackwardNeuronData(nextNeuron.layer, nextNeuron.index);
      if (neuronData) {
        fsmActions.jumpToNeuron(nextNeuron.layer, nextNeuron.index);
        fsmActions.backwardTick(nextNeuron.layer, nextNeuron.index, 'error', neuronData);
        refreshDisplayOnly();
      }
    } else {
      // Backward complete - show summary
      const backpropData = trainNNRef.current.lastBackwardSteps;
      if (backpropData) {
        const summaryData = createBackpropSummaryData(backpropData, state.stats.learningRate);
        state.modalSetters.setBackpropSummaryData(summaryData);
      }
      fsmActions.backwardComplete();
      refreshDisplayOnly();
    }
  }, [animationState, fsmActions, refreshDisplayOnly, trainNNRef, state.stats.learningRate, state.modalSetters]);

  // Helper: Jump to a different neuron
  const jumpToNeuron = useCallback((neuronLoc: NeuronLocation) => {
    fsmActions.jumpToNeuron(neuronLoc.layer, neuronLoc.index);

    if (animationState.type === 'forward_animating') {
      const neuronData = displayNNRef.current.getForwardNeuronData(neuronLoc.layer, neuronLoc.index);
      if (neuronData) {
        fsmActions.forwardTick(neuronLoc.layer, neuronLoc.index, 'dotProduct', neuronData);
      }
    } else if (animationState.type === 'backward_animating') {
      const neuronData = trainNNRef.current.getBackwardNeuronData(neuronLoc.layer, neuronLoc.index);
      if (neuronData) {
        fsmActions.backwardTick(neuronLoc.layer, neuronLoc.index, 'error', neuronData);
      }
    }
    refreshDisplayOnly();
  }, [animationState, fsmActions, displayNNRef, trainNNRef, refreshDisplayOnly]);

  // Canvas click handler
  const handleCanvasClick = useCallback((x?: number, y?: number) => {
    if (!isAnimating) return;
    if (x === undefined || y === undefined) return;

    const visualizer = visualizerRef.current;
    if (!visualizer) return;

    const neuron = visualizer.findNeuronAtPosition(x, y);
    if (!neuron) return;
    if (neuron.layer !== 'layer1' && neuron.layer !== 'layer2' && neuron.layer !== 'output') return;

    const neuronLoc: NeuronLocation = { layer: neuron.layer, index: neuron.index };

    // Case 1: Same neuron clicked during forward - advance stage
    if (isForwardAnimatingAtNeuron(animationState, neuronLoc.layer, neuronLoc.index)) {
      advanceForwardStage(neuronLoc);
    }
    // Case 2: Same neuron clicked during backward - advance stage
    else if (isBackwardAnimatingAtNeuron(animationState, neuronLoc.layer, neuronLoc.index)) {
      advanceBackwardStage(neuronLoc);
    }
    // Case 3: Different neuron clicked - jump to it
    else {
      jumpToNeuron(neuronLoc);
    }
  }, [isAnimating, animationState, visualizerRef, advanceForwardStage, advanceBackwardStage, jumpToNeuron]);

  // RETURN
  return {
    // Animation state
    state: animationState,
    dispatch,
    isAnimating,
    isPaused,
    // Visualization data
    highlightedNeuron   : getHighlightedNeuron(animationState),
    forwardStage        : getForwardStage(animationState),
    backpropStage       : getBackwardStage(animationState),
    currentForwardData  : getCurrentForwardNeuronData(animationState),
    currentBackwardData : getCurrentBackwardNeuronData(animationState),
    // Training controls
    trainOneStepWithAnimation,
    trainOneEpochWithoutAnimation,
    toggleTraining,
    reset,
    computeAndRefreshDisplay,
    // Modal controls
    closeLossModal,
    closeBackpropModal,
    // Canvas interaction
    handleCanvasClick,
    // Visualizer
    setVisualizer,
    // Utilities
    handleLearningRateChange,
    trainingIntervalRef,
  };
}
