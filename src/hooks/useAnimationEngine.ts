/**
 * Animation Engine Hook
 *
 * Core animation and training orchestrator that combines:
 * - Animation State Machine (FSM for animation states)
 * - Animation Loops (forward/backward propagation animation)
 * - Training Controls (train step/epoch, reset)
 * - Canvas Interaction (neuron click handling)
 *
 * Modal lifecycle is handled by useModalActions hook.
 * State management is handled by useNetworkState hook.
 *
 * Architecture: 5 hooks → 3 hooks (useNetworkState + useAnimationEngine + useModalActions)
 */

import { useReducer, useCallback, useRef, RefObject, useEffect } from 'react';
import { NeuralNetwork } from '../lib/core';
import type { LayerName } from '../lib/core';
import { getForwardNeuronIndices, getBackwardNeuronIndices } from '../lib/core';
import type { Visualizer } from '../lib/visualizer';
import type { UseNetworkStateReturn } from './useNetworkState';
import type { ForwardStage, BackwardStage, ForwardCalculation, BackwardCalculation } from '../lib/types';
import { createBackpropSummaryData } from '../lib/types';
import { createSnapshot, restoreSnapshot, compareSnapshots } from '../lib/core/networkSnapshot';
import {
  AnimationState,
  AnimationAction,
  animationReducer,
  initialAnimationState,
  isAnimating,
  isPaused,
  getNextForwardStage,
  getNextBackwardStage,
  getNextForwardNeuron,
  getNextBackwardNeuron,
  FORWARD_STAGES,
  BACKPROP_STAGES,
  getHighlightedNeuron,
  getForwardStage,
  getBackwardStage,
  getCurrentNeuronData,
  getCurrentBackpropData,
  InterruptReason,
  runAnimationLoop,
} from '../lib/animation';
import {
  FORWARD_STAGE_DURATIONS,
  BACKWARD_STAGE_DURATIONS,
} from '../lib/animation/animationLoop';
import { useModalActions } from './useModalActions';

export interface UseAnimationEngineReturn {
  // === Animation State ===
  state: AnimationState;
  dispatch: React.Dispatch<AnimationAction>;
  isAnimating: boolean;
  isPaused: boolean;
  // === Visualization Data ===
  highlightedNeuron: { layer: LayerName; index: number } | null;
  forwardStage: ForwardStage | null;
  backpropStage: BackwardStage | null;
  currentNeuronData: ForwardCalculation | null;
  currentBackpropData: BackwardCalculation | null;
  // === Training Controls ===
  trainOneStepWithAnimation: () => Promise<void>;
  trainOneEpochWithoutAnimation: () => void;
  toggleTraining: () => void;
  reset: () => void;
  computeAndRefreshDisplay: () => void;
  // === Modal Controls ===
  closeLossModal: () => Promise<void>;
  closeBackpropModal: () => void;
  // === Canvas Interaction ===
  handleCanvasClick: (x?: number, y?: number) => void;
  // === Visualizer ===
  setVisualizer: (v: Visualizer) => void;
  // === Utilities ===
  handleLearningRateChange: (v: number) => void;
  trainingIntervalRef: RefObject<number | undefined>;
}

type NeuronLocation = { layer: LayerName; index: number };

export function useAnimationEngine(
  nnRef: RefObject<NeuralNetwork>,
  visualizerRef: RefObject<Visualizer | null>,
  state: UseNetworkStateReturn
): UseAnimationEngineReturn {
  // 1. ANIMATION STATE MACHINE
  const [animationState, dispatch]= useReducer(animationReducer, initialAnimationState);
  const stepResolverRef           = useRef<(() => void) | null>(null);
  const prevSpeedRef              = useRef(animationState.speed);
  const interruptReasonRef        = useRef<InterruptReason>('none'); // Synced with FSM
  const trainingIntervalRef       = useRef<number | undefined>(undefined);

  // Sync interruptReasonRef with FSM state (ref is needed for async loop access)
  useEffect(() => {
    interruptReasonRef.current = animationState.interruptReason;
  }, [animationState.interruptReason]);
  // Auto-resolve step when speed changes from 0 to > 0
  useEffect(() => {
    if (prevSpeedRef.current === 0 && animationState.speed > 0) {
      if (stepResolverRef.current) {
        stepResolverRef.current();
        stepResolverRef.current = null;
      }
    }
    prevSpeedRef.current = animationState.speed;
  }, [animationState.speed]);
  // FSM action creators
  const startTraining = useCallback(() => {
    dispatch({ type: 'START_TRAINING' });
  }, []);

  const pause = useCallback(() => {
    dispatch({ type: 'PAUSE' });
  }, []);

  const resume = useCallback((speed?: number) => {
    dispatch({ type: 'RESUME', speed: speed ?? 1.0 });
  }, []);

  const fsmReset = useCallback(() => {
    dispatch({ type: 'RESET' });
    stepResolverRef.current = null;
  }, []);

  const forwardTick = useCallback((
    layer: LayerName,
    neuronIndex: number,
    stage: ForwardStage,
    neuronData: ForwardCalculation | null
  ) => {
    dispatch({ type: 'FORWARD_TICK', layer, neuronIndex, stage, neuronData });
  }, []);

  const forwardComplete = useCallback(() => {
    dispatch({ type: 'FORWARD_COMPLETE' });
  }, []);

  const closeLossModalAction = useCallback(() => {
    dispatch({ type: 'CLOSE_LOSS_MODAL' });
  }, []);

  const backwardTick = useCallback((
    layer       : LayerName,
    neuronIndex : number,
    stage       : BackwardStage,
    neuronData  : BackwardCalculation | null
  ) => {
    dispatch({ type: 'BACKWARD_TICK', layer, neuronIndex, stage, neuronData });
  }, []);

  const backwardComplete = useCallback(() => {
    dispatch({ type: 'BACKWARD_COMPLETE' });
  }, []);

  const closeBackpropModalAction = useCallback(() => {
    dispatch({ type: 'CLOSE_BACKPROP_MODAL' });
  }, []);

  const jumpToNeuron = useCallback((layer: LayerName, neuronIndex: number) => {
    dispatch({ type: 'JUMP_TO_NEURON', layer, neuronIndex });
  }, []);

  const waitForNextStep = useCallback((): Promise<void> => {
    if (animationState.speed === 0) {
      return new Promise<void>(resolve => {
        stepResolverRef.current = resolve;
      });
    }
    return Promise.resolve();
  }, [animationState.speed]);
  // Derived state
  const animating = isAnimating(animationState);
  const paused = isPaused(animationState);
  // Animation Control Helpers
  const shouldPauseAnimation = useCallback(() => {
    return interruptReasonRef.current !== 'none';
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
  
  // Helper: Show loss modal with current predictions
  const showLossModal = useCallback(() => {
    const nn = nnRef.current;
    const predictions = nn.lastOutput?.toArray() || [0, 0, 0];
    const loss = nn.lastLoss;
    state.modalSetters.setLossModalData({
      targetClass: state.inputs.targetValue,
      predictions,
      loss
    });
  }, [nnRef, state.inputs.targetValue, state.modalSetters]);

  // Sync visualizer state with animation machine
  const syncVisualizerState = useCallback(() => {
    if (visualizerRef.current) {
      visualizerRef.current.update(nnRef.current, animationState);
    }
  }, [animationState, nnRef, visualizerRef]);
  
  // Refresh display without recalculation
  const refreshDisplayOnly = useCallback(() => {
    syncVisualizerState();
  }, [syncVisualizerState]);
  
  // Compute and refresh display (full recalculation)
  const computeAndRefreshDisplay = useCallback(() => {
    const nn = nnRef.current;
    nn.feedforward(getCurrentInputs());

    if (nn.lastOutput) {
      state.statsSetters.setOutput(nn.lastOutput.toArray());
    }
    state.statsSetters.setSteps(nn.getCalculationSteps());
    // Update activations for heatmap
    if (nn.lastInput && nn.lastHidden1 && nn.lastHidden2 && nn.lastOutput) {
      state.visualizerSetters.setActivations({
        input : nn.lastInput.toArray(),
        layer1: nn.lastHidden1.toArray(),
        layer2: nn.lastHidden2.toArray(),
        output: nn.lastOutput.toArray(),
      });
    }

    syncVisualizerState();
  }, [getCurrentInputs, syncVisualizerState, nnRef, state.statsSetters, state.visualizerSetters]);
  
  // Sleep utility with pause support
  const sleep = useCallback(async (ms: number, overrideSpeed?: number): Promise<void> => {
    const effectiveSpeed = overrideSpeed ?? animationState.speed;
    if (effectiveSpeed === 0) {
      await waitForNextStep();
    } else {
      await new Promise(resolve => setTimeout(resolve, ms / effectiveSpeed));
    }
  }, [animationState.speed, waitForNextStep]);

  // Forward propagation animation (unified: can start from beginning or from a specific position)
  const animateForwardPropagation = useCallback(async (
    startFrom?: { layer: LayerName; neuronIndex: number }
  ): Promise<boolean> => {
    const nn = nnRef.current;
    const inputs = getCurrentInputs();
    const calcSteps = nn.getCalculationSteps();
    if (!calcSteps) return false;

    const layerData = {
      layer1: calcSteps.layer1,
      layer2: calcSteps.layer2,
      output: calcSteps.output
    };

    const completed = await runAnimationLoop({
      mode: 'forward',
      layers: ['layer1', 'layer2', 'output'],
      getNeuronIndices: getForwardNeuronIndices,
      stages: FORWARD_STAGES,
      stageDurations: FORWARD_STAGE_DURATIONS,
      getData: () => layerData,
      onTick: forwardTick,
      onComplete: forwardComplete,
      refreshDisplayOnly,
      shouldStop: shouldPauseAnimation,
      sleep,
      computeAndRefreshDisplay,
      startFrom,
    });

    // Show loss modal if animation completed
    if (completed) {
      const snapshot = createSnapshot(nn);
      nn.train(inputs, getTargetOneHot());
      restoreSnapshot(nn, snapshot, inputs);
      showLossModal();
    }

    return completed;
  }, [forwardTick, forwardComplete, refreshDisplayOnly, sleep, computeAndRefreshDisplay, nnRef, shouldPauseAnimation, getCurrentInputs, getTargetOneHot, showLossModal]);
  
  // Backward propagation animation (unified: can start from beginning or from a specific position)
  const animateBackwardPropagation = useCallback(async (
    speedOverride: number = 1.0,
    startFrom?: { layer: LayerName; neuronIndex: number }
  ): Promise<boolean> => {
    const nn = nnRef.current;
    const backpropData = nn.lastBackwardSteps;
    if (!backpropData) return false;

    const layerData = {
      layer1: backpropData.layer1,
      layer2: backpropData.layer2,
      output: backpropData.output
    };

    // Helper for weight updates on 'update' stage
    const handleStageComplete = (layer: LayerName, neuronIndex: number, stage: BackwardStage, data: BackwardCalculation) => {
      if (stage === 'update') {
        nn.updateNeuronWeights(layer, neuronIndex, data.newWeights, data.newBias);
        nn.feedforward(nn.lastInput!.toArray());
        refreshDisplayOnly();
      }
    };

    const completed = await runAnimationLoop({
      mode: 'backward',
      layers: ['output', 'layer2', 'layer1'],
      getNeuronIndices: getBackwardNeuronIndices,
      stages: BACKPROP_STAGES,
      stageDurations: BACKWARD_STAGE_DURATIONS,
      getData: () => layerData,
      onTick: backwardTick,
      onStageComplete: handleStageComplete,
      onComplete: backwardComplete,
      shouldStop: shouldPauseAnimation,
      sleep,
      refreshDisplayOnly,
      computeAndRefreshDisplay,
      speedOverride,
      startFrom,
    });

    // Collect summary data only if animation completed (not stopped)
    if (completed) {
      const summaryData = createBackpropSummaryData(backpropData, state.stats.learningRate);
      state.modalSetters.setBackpropSummaryData(summaryData);
    }
    return completed;
  }, [backwardTick, backwardComplete, sleep, refreshDisplayOnly, computeAndRefreshDisplay, nnRef, state.stats.learningRate, state.modalSetters, shouldPauseAnimation]);
  
  // Continue from jumped position (dispatcher)
  const continueFromJumpedPosition = useCallback(async () => {
    if (animationState.type === 'forward_animating') {
      await animateForwardPropagation({
        layer: animationState.layer,
        neuronIndex: animationState.neuronIndex
      });
    } else if (animationState.type === 'backward_animating') {
      await animateBackwardPropagation(state.training.animationSpeed, {
        layer: animationState.layer,
        neuronIndex: animationState.neuronIndex
      });
    }
  }, [animationState, animateForwardPropagation, animateBackwardPropagation, state.training.animationSpeed]);

  // 3. TRAINING CONTROLS
  const setVisualizer = useCallback((v: Visualizer) => {
    visualizerRef.current = v;
  }, [visualizerRef]);

  // Train one epoch without animation
  const trainOneEpochWithoutAnimation = useCallback(() => {
    const nn = nnRef.current;
    const oldSnapshot = createSnapshot(nn);
    nn.train(getCurrentInputs(), getTargetOneHot());
    const newSnapshot = createSnapshot(nn);

    createWeightComparisonAfterTraining(oldSnapshot, newSnapshot);
    state.statsSetters.setLoss(nn.lastLoss);
    state.statsSetters.setEpoch(prev => prev + 1);
    computeAndRefreshDisplay();
  }, [getCurrentInputs, getTargetOneHot, createWeightComparisonAfterTraining, state.statsSetters, computeAndRefreshDisplay, nnRef]);

  // Train one step with animation
  const trainOneStepWithAnimation = useCallback(async () => {
    // Case 1: Resume from paused position (continue from next neuron)
    if (animating && paused) {
      // Manually clear ref before dispatch to prevent race condition
      interruptReasonRef.current = 'none';
      resume(state.training.animationSpeed);
      await continueFromJumpedPosition();
      return;
    }
    // Case 2: Pause running animation
    if (animating) {
      pause();
      return;
    }
    // Case 3: Start new animation
    startTraining();

    const nn = nnRef.current;
    nn.feedforward(getCurrentInputs());
    await animateForwardPropagation();
    // Loss modal is shown inside animateForwardPropagation if completed
  }, [animating, paused, state.training.animationSpeed, getCurrentInputs, resume, continueFromJumpedPosition, pause, startTraining, animateForwardPropagation, nnRef]);

  // Toggle auto training
  const toggleTraining = useCallback(() => {
    if (state.training.isTraining) {
      state.trainingSetters.setIsTraining(false);
      if (trainingIntervalRef.current) {
        clearInterval(trainingIntervalRef.current);
      }
    } else {
      state.trainingSetters.setIsTraining(true);
      trainingIntervalRef.current = window.setInterval(() => {
        trainOneEpochWithoutAnimation();
        if (nnRef.current.lastLoss < 0.001) {
          state.trainingSetters.setIsTraining(false);
          if (trainingIntervalRef.current) {
            clearInterval(trainingIntervalRef.current);
          }
        }
      }, 50);
    }
  }, [state.training.isTraining, state.trainingSetters, trainOneEpochWithoutAnimation, nnRef]);
  // Reset
  const reset = useCallback(() => {
    if (state.training.isTraining) {
      state.trainingSetters.setIsTraining(false);
      if (trainingIntervalRef.current) {
        clearInterval(trainingIntervalRef.current);
      }
    }
    nnRef.current = new NeuralNetwork();
    state.statsSetters.setEpoch(0);
    state.statsSetters.setLoss(0);
    state.statsSetters.setOutput(null);
    state.modalSetters.setLossModalData(null);
    state.modalSetters.setBackpropSummaryData(null);
    state.modalSetters.setWeightComparisonData(null);

    state.inputSetters.setGrade(Math.random());
    state.inputSetters.setAttitude(Math.random());
    state.inputSetters.setResponse(Math.random());
    state.inputSetters.setTargetValue(Math.floor(Math.random() * 3));

    fsmReset();
    computeAndRefreshDisplay();
  }, [state.training.isTraining, state.trainingSetters, state.statsSetters, state.modalSetters, state.inputSetters, fsmReset, computeAndRefreshDisplay, nnRef]);

  // Learning rate change
  const handleLearningRateChange = useCallback((v: number) => {
    state.statsSetters.setLearningRate(v);
    nnRef.current.learningRate = v;
  }, [state.statsSetters, nnRef]);

  // 4. MODAL CONTROLS
  const modalControls = useModalActions({
    nnRef,
    state,
    animateBackwardPropagation,
    sleep,
    computeAndRefreshDisplay,
    closeLossModalAction,
    closeBackpropModalAction,
    refreshDisplayOnly,
    animationSpeed: state.training.animationSpeed,
  });

  // 5. CANVAS INTERACTION
  // Helper: Get neuron data
  const getForwardNeuronData = useCallback((layer: LayerName, index: number) => {
    const calcSteps = nnRef.current.getCalculationSteps();
    if (!calcSteps) return null;
    return calcSteps[layer]?.[index] ?? null;
  }, [nnRef]);

  const getBackwardNeuronData = useCallback((layer: LayerName, index: number) => {
    const backpropData = nnRef.current.lastBackwardSteps;
    if (!backpropData) return null;
    return backpropData[layer]?.[index] ?? null;
  }, [nnRef]);

  // Helper: Complete forward pass (used by canvas click handler)
  const completeForwardPass = useCallback(() => {
    const nn = nnRef.current;
    const inputs = getCurrentInputs();
    const snapshot = createSnapshot(nn);
    nn.train(inputs, getTargetOneHot());
    restoreSnapshot(nn, snapshot, inputs);
    forwardComplete();
    showLossModal();
  }, [nnRef, getCurrentInputs, getTargetOneHot, showLossModal, forwardComplete]);

  // Canvas click handler
  const handleCanvasClick = useCallback((x?: number, y?: number) => {
    if (!animating) 
      return;
    
    const visualizer = visualizerRef.current;
    if (x !== undefined && y !== undefined && visualizer) {
      const neuron = visualizer.findNeuronAtPosition(x, y);

      if (neuron && (neuron.layer === 'layer1' || neuron.layer === 'layer2' || neuron.layer === 'output')) {
        const neuronLoc: NeuronLocation = { layer: neuron.layer, index: neuron.index };

        // Case 1: Same neuron clicked during forward animation - advance to next stage/neuron
        if (animationState.type === 'forward_animating' &&
            animationState.layer === neuronLoc.layer &&
            animationState.neuronIndex === neuronLoc.index) {
          const nextStage = getNextForwardStage(animationState.stage);

          if (nextStage) {
            forwardTick(neuronLoc.layer, neuronLoc.index, nextStage, animationState.neuronData);
            refreshDisplayOnly();
          } else {
            const nextNeuron = getNextForwardNeuron(neuronLoc.layer, neuronLoc.index);
            if (nextNeuron) {
              const neuronData = getForwardNeuronData(nextNeuron.layer, nextNeuron.index);
              if (neuronData) {
                jumpToNeuron(nextNeuron.layer, nextNeuron.index);
                forwardTick(nextNeuron.layer, nextNeuron.index, 'dotProduct', neuronData);
                refreshDisplayOnly();
              }
            } else {
              completeForwardPass();
            }
          }
        }
        // Case 2: Same neuron clicked during backward animation - advance to next stage/neuron
        else if (animationState.type === 'backward_animating' &&
                 animationState.layer === neuronLoc.layer &&
                 animationState.neuronIndex === neuronLoc.index) {
          const nextStage = getNextBackwardStage(animationState.stage);

          if (nextStage) {
            backwardTick(neuronLoc.layer, neuronLoc.index, nextStage, animationState.neuronData);
            refreshDisplayOnly();
          } else {
            const nextNeuron = getNextBackwardNeuron(neuronLoc.layer, neuronLoc.index);
            if (nextNeuron) {
              const neuronData = getBackwardNeuronData(nextNeuron.layer, nextNeuron.index);
              if (neuronData) {
                jumpToNeuron(nextNeuron.layer, nextNeuron.index);
                backwardTick(nextNeuron.layer, nextNeuron.index, 'error', neuronData);
                refreshDisplayOnly();
              }
            } else {
              // Backward propagation completed via click-through - show summary
              const nn = nnRef.current;
              const backpropData = nn.lastBackwardSteps;
              if (backpropData) {
                const summaryData = createBackpropSummaryData(backpropData, state.stats.learningRate);
                state.modalSetters.setBackpropSummaryData(summaryData);
              }
              backwardComplete();
              refreshDisplayOnly();
            }
          }
        }
        // Case 3: Different neuron clicked - jump to that neuron
        else {
          jumpToNeuron(neuronLoc.layer, neuronLoc.index);

          if (animationState.type === 'forward_animating') {
            const neuronData = getForwardNeuronData(neuronLoc.layer, neuronLoc.index);
            if (neuronData) {
              forwardTick(neuronLoc.layer, neuronLoc.index, 'dotProduct', neuronData);
            }
          } else if (animationState.type === 'backward_animating') {
            const neuronData = getBackwardNeuronData(neuronLoc.layer, neuronLoc.index);
            if (neuronData) {
              backwardTick(neuronLoc.layer, neuronLoc.index, 'error', neuronData);
            }
          }

          refreshDisplayOnly();
        }
      }
    }
  }, [animating, animationState, visualizerRef, nnRef, forwardTick, backwardTick, jumpToNeuron, refreshDisplayOnly, getForwardNeuronData, getBackwardNeuronData, completeForwardPass, backwardComplete]);

  // RETURN
  return {
    // Animation state
    state: animationState,
    dispatch,
    isAnimating: animating,
    isPaused: paused,
    // Visualization data
    highlightedNeuron: getHighlightedNeuron(animationState),
    forwardStage: getForwardStage(animationState),
    backpropStage: getBackwardStage(animationState),
    currentNeuronData: getCurrentNeuronData(animationState),
    currentBackpropData: getCurrentBackpropData(animationState),
    // Training controls
    trainOneStepWithAnimation,
    trainOneEpochWithoutAnimation,
    toggleTraining,
    reset,
    computeAndRefreshDisplay,
    // Modal controls
    closeLossModal: modalControls.closeLossModal,
    closeBackpropModal: modalControls.closeBackpropModal,
    // Canvas interaction
    handleCanvasClick,
    // Visualizer
    setVisualizer,
    // Utilities
    handleLearningRateChange,
    trainingIntervalRef,
  };
}
