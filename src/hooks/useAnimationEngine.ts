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

import { useReducer, useCallback, useRef, RefObject, useEffect, useMemo } from 'react';
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
  checkAnimating,
  checkPaused,
  getNextForwardStage,
  getNextBackwardStage,
  getNextForwardNeuron,
  getNextBackwardNeuron,
  FORWARD_STAGES,
  BACKPROP_STAGES,
  getHighlightedNeuron,
  getForwardStage,
  getBackwardStage,
  getCurrentForwardData,
  getCurrentBackwardData,
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
  nnRef         : RefObject<NeuralNetwork>,
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
    animationStateRef.current = animationState;
    interruptReasonRef.current = animationState.interruptReason;
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
  
  // Helper: Show loss modal with current predictions
  const showLossModal = useCallback(() => {
    const nn            = nnRef.current;
    const predictions   = nn.lastOutput?.toArray() || [0, 0, 0];
    const loss          = nn.lastLoss;
    state.modalSetters.setLossModalData({
      targetClass: state.inputs.targetValue,
      predictions,
      loss
    });
  }, [nnRef, state.inputs.targetValue, state.modalSetters]);

  // Refresh display without recalculation (uses ref to avoid stale closure)
  const refreshDisplayOnly = useCallback(() => {
    if (visualizerRef.current) {
      visualizerRef.current.update(nnRef.current, animationStateRef.current);
    }
  }, [nnRef, visualizerRef]);
  
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

    refreshDisplayOnly();
  }, [getCurrentInputs, refreshDisplayOnly, nnRef, state.statsSetters, state.visualizerSetters]);
  
  // Sleep utility for animation timing
  const sleep = useCallback(async (ms: number, overrideSpeed?: number): Promise<void> => {
    const effectiveSpeed = overrideSpeed ?? state.training.animationSpeed;
    await new Promise(resolve => setTimeout(resolve, ms / effectiveSpeed));
  }, [state.training.animationSpeed]);

  // Forward propagation animation (unified: can start from beginning or from a specific position)
  const animateForwardPropagation = useCallback(async (
    startFrom?: { layer: LayerName; neuronIndex: number }
  ): Promise<boolean> => {
    const nn        = nnRef.current;
  const inputs = getCurrentInputs();
    const calcSteps = nn.getCalculationSteps();
    if (!calcSteps) return false;

    const layerData = {
      layer1: calcSteps.layer1,
      layer2: calcSteps.layer2,
      output: calcSteps.output
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
      const snapshot = createSnapshot(nn);
      nn.train(inputs, getTargetOneHot());
      restoreSnapshot(nn, snapshot, inputs);
      showLossModal();
    }

    return completed;
  }, [fsmActions, refreshDisplayOnly, sleep, computeAndRefreshDisplay, nnRef, shouldPauseAnimation, getCurrentInputs, getTargetOneHot, showLossModal]);
  
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
      speedOverride     : speedOverride,
      startFrom         : startFrom,
    });

    // Collect summary data only if animation completed (not stopped)
    if (completed) {
      const summaryData = createBackpropSummaryData(backpropData, state.stats.learningRate);
      state.modalSetters.setBackpropSummaryData(summaryData);
    }
    return completed;
  }, [fsmActions, sleep, refreshDisplayOnly, computeAndRefreshDisplay, nnRef, state.stats.learningRate, state.modalSetters, shouldPauseAnimation]);
  
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
      const nn = nnRef.current;
      nn.feedforward(getCurrentInputs());
      await animateForwardPropagation();
    }
  }, [isAnimating, isPaused, state.training.animationSpeed, getCurrentInputs, fsmActions, continueFromJumpedPosition, animateForwardPropagation, nnRef]);

  // Toggle auto training
  const toggleTraining = useCallback(() => {
    if (state.training.isTraining) {
      state.trainingSetters.setIsTraining(false);
      clearTrainingInterval();
    } else {
      state.trainingSetters.setIsTraining(true);
      trainingIntervalRef.current = window.setInterval(() => {
        trainOneEpochWithoutAnimation();
        if (nnRef.current.lastLoss < 0.001) {
          state.trainingSetters.setIsTraining(false);
          clearTrainingInterval();
        }
      }, 50);
    }
  }, [state.training.isTraining, state.trainingSetters, trainOneEpochWithoutAnimation, nnRef, clearTrainingInterval]);
  // Reset
  const reset = useCallback(() => {
    // Stop auto training if running
    if (state.training.isTraining) {
      state.trainingSetters.setIsTraining(false);
      clearTrainingInterval();
    }
    // Reset network
    nnRef.current = new NeuralNetwork();
    // Reset all state (stats, modals, inputs)
    state.resetAllState();
    // Reset FSM and refresh
    fsmActions.reset();
    computeAndRefreshDisplay();
  }, [state.training.isTraining, state.trainingSetters, state.resetAllState, fsmActions, computeAndRefreshDisplay, nnRef, clearTrainingInterval]);

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
    closeLossModalAction: fsmActions.closeLossModal,
    closeBackpropModalAction: fsmActions.closeBackpropModal,
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
    fsmActions.forwardComplete();
    showLossModal();
  }, [nnRef, getCurrentInputs, getTargetOneHot, showLossModal, fsmActions]);

  // Canvas click handler
  const handleCanvasClick = useCallback((x?: number, y?: number) => {
    if (!isAnimating) 
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
            fsmActions.forwardTick(neuronLoc.layer, neuronLoc.index, nextStage, animationState.neuronData);
            refreshDisplayOnly();
          } else {
            const nextNeuron = getNextForwardNeuron(neuronLoc.layer, neuronLoc.index);
            if (nextNeuron) {
              const neuronData = getForwardNeuronData(nextNeuron.layer, nextNeuron.index);
              if (neuronData) {
                fsmActions.jumpToNeuron(nextNeuron.layer, nextNeuron.index);
                fsmActions.forwardTick(nextNeuron.layer, nextNeuron.index, 'dotProduct', neuronData);
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
            fsmActions.backwardTick(neuronLoc.layer, neuronLoc.index, nextStage, animationState.neuronData);
            refreshDisplayOnly();
          } else {
            const nextNeuron = getNextBackwardNeuron(neuronLoc.layer, neuronLoc.index);
            if (nextNeuron) {
              const neuronData = getBackwardNeuronData(nextNeuron.layer, nextNeuron.index);
              if (neuronData) {
                fsmActions.jumpToNeuron(nextNeuron.layer, nextNeuron.index);
                fsmActions.backwardTick(nextNeuron.layer, nextNeuron.index, 'error', neuronData);
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
              fsmActions.backwardComplete();
              refreshDisplayOnly();
            }
          }
        }
        // Case 3: Different neuron clicked - jump to that neuron
        else {
          fsmActions.jumpToNeuron(neuronLoc.layer, neuronLoc.index);

          if (animationState.type === 'forward_animating') {
            const neuronData = getForwardNeuronData(neuronLoc.layer, neuronLoc.index);
            if (neuronData) {
              fsmActions.forwardTick(neuronLoc.layer, neuronLoc.index, 'dotProduct', neuronData);
            }
          } else if (animationState.type === 'backward_animating') {
            const neuronData = getBackwardNeuronData(neuronLoc.layer, neuronLoc.index);
            if (neuronData) {
              fsmActions.backwardTick(neuronLoc.layer, neuronLoc.index, 'error', neuronData);
            }
          }
          refreshDisplayOnly();
        }
      }
    }
  }, [isAnimating, animationState, visualizerRef, nnRef, fsmActions, refreshDisplayOnly, getForwardNeuronData, getBackwardNeuronData, completeForwardPass, state.stats.learningRate, state.modalSetters]);

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
    currentForwardData  : getCurrentForwardData(animationState),
    currentBackwardData : getCurrentBackwardData(animationState),
    // Training controls
    trainOneStepWithAnimation,
    trainOneEpochWithoutAnimation,
    toggleTraining,
    reset,
    computeAndRefreshDisplay,
    // Modal controls
    closeLossModal    : modalControls.closeLossModal,
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
