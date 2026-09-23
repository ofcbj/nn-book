/**
 * Animation Engine Hook
 *
 * Core animation and training orchestrator:
 * - Animation state machine (FSM) and the async animation loops
 * - Training controls (animated step, plain epoch, auto-train, reset)
 * - Modal controls (loss modal -> backprop animation -> summary modal)
 * - Canvas interaction (click a neuron to advance / jump)
 *
 * The canvas is redrawn from the FSM state: every state change triggers a redraw
 * (see the sync effect below), so callers never redraw manually after dispatching.
 * React state (inputs, stats, modals) lives in useNetworkState.
 */

import { useReducer, useCallback, useRef, RefObject, useEffect, useMemo } from 'react';
import {
  NeuralNetwork,
  toOneHot,
  OUTPUT_CLASSES,
  FORWARD_LAYER_ORDER,
  BACKWARD_LAYER_ORDER,
  getForwardNeuronIndices,
  getBackwardNeuronIndices,
  createSnapshot,
  compareSnapshots,
  createBackpropSummaryData,
} from '../lib/core';
import type { LayerName, NeuronLocation, TrainingSample } from '../lib/core';
import type { Visualizer } from '../lib/visualizer';
import type { UseNetworkStateReturn } from './useNetworkState';
import type {
  ForwardStage, BackwardStage, ForwardCalculation, BackwardCalculation, WeightComparisonData,
} from '../lib/types';
import {
  animationReducer, initialAnimationState, checkAnimating, checkPaused,
  getNextForwardStage, getNextBackwardStage, getNextForwardNeuron, getNextBackwardNeuron,
  isAnimatingAtNeuron, FORWARD_STAGES, BACKPROP_STAGES, runAnimationLoop,
  FORWARD_STAGE_DURATIONS, BACKWARD_STAGE_DURATIONS,
} from '../lib/animation';
import type { AnimationState, InterruptReason } from '../lib/animation';

export interface UseAnimationEngineReturn {
  // === Animation State ===
  state       : AnimationState;
  isAnimating : boolean;
  isPaused    : boolean;
  // === Training Controls ===
  /** Animated training step. Without an argument it trains the slider candidate;
   *  pass a sample to walk through that candidate instead (data mode inspection).
   *  While an animation runs, calling it toggles pause / resume. */
  trainOneStepWithAnimation    : (sample?: AnimatedSample) => Promise<void>;
  /** Abort a running animation (and its modals) without touching the network */
  stopAnimation                : () => void;
  /** One training step on the slider candidate without animation; returns its loss */
  trainOneEpochWithoutAnimation: () => number;
  toggleTraining               : () => void;
  reset                        : () => void;
  computeAndRefreshDisplay     : () => void;
  /** Like computeAndRefreshDisplay but for an arbitrary input vector (data training mode) */
  displayInputs                : (inputs: number[]) => void;
  refreshDisplayOnly           : () => void;
  // === Modal Controls ===
  closeLossModal    : () => Promise<void>;
  closeBackpropModal: () => void;
  // === Canvas Interaction ===
  handleCanvasClick: (x: number, y: number) => void;
  // === Visualizer ===
  setVisualizer: (v: Visualizer) => void;
  // === Utilities ===
  handleLearningRateChange: (v: number) => void;
}

/** What the animated step is training on */
export interface AnimatedSample extends TrainingSample {
  targetClass: number;
}

const AUTO_TRAIN_INTERVAL_MS = 50;
const AUTO_TRAIN_STOP_LOSS = 0.001;
const BACKWARD_COMPLETE_PAUSE_MS = 500;

export function useAnimationEngine(
  nnRef         : RefObject<NeuralNetwork>,
  visualizerRef : RefObject<Visualizer | null>,
  state         : UseNetworkStateReturn
): UseAnimationEngineReturn {
  // =========================================================================
  // 1. ANIMATION STATE MACHINE
  // =========================================================================
  const [animationState, dispatch] = useReducer(animationReducer, initialAnimationState);

  // Refs read by the async animation loops (always current, unlike closures)
  const animationStateRef   = useRef(animationState);
  const interruptReasonRef  = useRef<InterruptReason>('none');
  const animationSpeedRef   = useRef(state.training.animationSpeed);
  // Auto-training
  const trainingIntervalRef = useRef<number | undefined>(undefined);
  const trainStepRef        = useRef<() => number>(() => Infinity);
  // Weight comparison of the most recent training step
  const pendingComparisonRef = useRef<WeightComparisonData | null>(null);
  // Sample the running animation is about (slider candidate by default)
  const activeSampleRef = useRef<AnimatedSample | null>(null);
  // Set by stopAnimation(); unlike interruptReasonRef it is not re-synced from the FSM,
  // so a loop that is asleep when the user stops cannot resume and "complete" afterwards
  const cancelledRef = useRef(false);

  useEffect(() => {
    animationSpeedRef.current = state.training.animationSpeed;
  }, [state.training.animationSpeed]);

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

  const isAnimating = checkAnimating(animationState);
  const isPaused    = checkPaused(animationState);

  const shouldPauseAnimation = useCallback(() => cancelledRef.current || interruptReasonRef.current !== 'none', []);

  const clearTrainingInterval = useCallback(() => {
    if (trainingIntervalRef.current !== undefined) {
      clearInterval(trainingIntervalRef.current);
      trainingIntervalRef.current = undefined;
    }
  }, []);

  // Stop auto-training when the component unmounts
  useEffect(() => clearTrainingInterval, [clearTrainingInterval]);

  // =========================================================================
  // 2. DISPLAY
  // =========================================================================
  const getCurrentInputs = useCallback(() => {
    return [state.inputs.grade, state.inputs.attitude, state.inputs.response];
  }, [state.inputs.grade, state.inputs.attitude, state.inputs.response]);

  const getTargetOneHot = useCallback(() => toOneHot(state.inputs.targetValue), [state.inputs.targetValue]);

  /** Redraw the canvas from the current network and animation state (no recalculation) */
  const refreshDisplayOnly = useCallback(() => {
    visualizerRef.current?.update(nnRef.current, animationStateRef.current, state.stats.learningRate);
  }, [nnRef, visualizerRef, state.stats.learningRate]);

  // Sync refs with the FSM state and redraw whenever it changes
  useEffect(() => {
    animationStateRef.current  = animationState;
    interruptReasonRef.current = animationState.interruptReason;
    refreshDisplayOnly();
  }, [animationState, refreshDisplayOnly]);

  /** Run feedforward on the given inputs, update React state and redraw */
  const displayInputs = useCallback((inputs: number[]) => {
    const nn = nnRef.current;
    nn.feedforward(inputs);

    if (nn.lastOutput) {
      state.statsSetters.setOutput(nn.lastOutput.toArray());
    }
    state.statsSetters.setSteps(nn.getForwardSteps());
    if (nn.lastInput && nn.lastHidden1 && nn.lastHidden2 && nn.lastOutput) {
      state.visualizerSetters.setActivations({
        input : nn.lastInput.toArray(),
        layer1: nn.lastHidden1.toArray(),
        layer2: nn.lastHidden2.toArray(),
        output: nn.lastOutput.toArray(),
      });
    }

    refreshDisplayOnly();
  }, [refreshDisplayOnly, nnRef, state.statsSetters, state.visualizerSetters]);

  /** Run feedforward on the slider candidate, update React state and redraw */
  const computeAndRefreshDisplay = useCallback(() => {
    displayInputs(getCurrentInputs());
  }, [displayInputs, getCurrentInputs]);

  /** Sleep scaled by the *current* animation speed (read from a ref so slider changes apply immediately) */
  const sleep = useCallback((ms: number): Promise<void> => {
    return new Promise(resolve => setTimeout(resolve, ms / animationSpeedRef.current));
  }, []);

  // =========================================================================
  // 3. TRAINING PRIMITIVES
  // =========================================================================

  /** Run `train` on the network and remember the before/after weight comparison */
  const withComparison = useCallback(<T,>(train: (nn: NeuralNetwork) => T): T => {
    const nn = nnRef.current;
    const before = createSnapshot(nn);
    const result = train(nn);
    pendingComparisonRef.current = compareSnapshots(before, createSnapshot(nn), nn.learningRate);
    return result;
  }, [nnRef]);

  /** The slider candidate as a training sample */
  const sliderSample = useCallback((): AnimatedSample => ({
    inputs: getCurrentInputs(),
    target: getTargetOneHot(),
    targetClass: state.inputs.targetValue,
  }), [getCurrentInputs, getTargetOneHot, state.inputs.targetValue]);

  /** One training step on the slider candidate; returns its loss */
  const trainSliderCandidate = useCallback((): number => {
    return withComparison(nn => {
      nn.train(getCurrentInputs(), getTargetOneHot());
      return nn.lastLoss;
    });
  }, [withComparison, getCurrentInputs, getTargetOneHot]);

  /** One training step on whatever the running animation is about */
  const trainActiveSample = useCallback((): number => {
    const sample = activeSampleRef.current ?? sliderSample();
    return withComparison(nn => {
      nn.train(sample.inputs, sample.target);
      return nn.lastLoss;
    });
  }, [withComparison, sliderSample]);

  /** Publish the pending comparison and epoch/loss stats after a completed training step */
  const commitTrainingStats = useCallback((loss: number) => {
    if (pendingComparisonRef.current) {
      state.modalSetters.setWeightComparisonData(pendingComparisonRef.current);
    }
    state.statsSetters.setLoss(loss);
    state.statsSetters.recordLoss(loss);
    state.statsSetters.setEpoch(prev => prev + 1);
  }, [state.modalSetters, state.statsSetters]);

  const showLossModal = useCallback(() => {
    const nn = nnRef.current;
    state.modalSetters.setLossModalData({
      targetClass: activeSampleRef.current?.targetClass ?? state.inputs.targetValue,
      predictions: nn.lastOutput?.toArray() ?? Array<number>(OUTPUT_CLASSES).fill(0),
      loss: nn.lastLoss,
    });
  }, [nnRef, state.inputs.targetValue, state.modalSetters]);

  /** End of the forward pass: train (weights update, backprop data is stored) and show the loss modal */
  const completeForwardPass = useCallback((options?: { skipForwardComplete?: boolean }) => {
    trainActiveSample();
    if (!options?.skipForwardComplete) {
      fsmActions.forwardComplete();
    }
    showLossModal();
  }, [trainActiveSample, showLossModal, fsmActions]);

  // =========================================================================
  // 4. ANIMATION LOOPS
  // =========================================================================
  const animateForwardPropagation = useCallback(async (startFrom?: NeuronLocation): Promise<boolean> => {
    const forwardSteps = nnRef.current.getForwardSteps();
    if (!forwardSteps) return false;

    const completed = await runAnimationLoop<ForwardStage, ForwardCalculation>({
      mode            : 'forward',
      layers          : FORWARD_LAYER_ORDER,
      getNeuronIndices: getForwardNeuronIndices,
      stages          : FORWARD_STAGES,
      stageDurations  : FORWARD_STAGE_DURATIONS,
      data            : forwardSteps,
      onTick          : fsmActions.forwardTick,
      onComplete      : fsmActions.forwardComplete,
      shouldStop      : shouldPauseAnimation,
      sleep,
      startFrom       : startFrom && { layer: startFrom.layer, neuronIndex: startFrom.index },
    });

    if (completed) {
      completeForwardPass({ skipForwardComplete: true });
    }
    return completed;
  }, [nnRef, fsmActions, shouldPauseAnimation, sleep, completeForwardPass]);

  const animateBackwardPropagation = useCallback(async (startFrom?: NeuronLocation): Promise<boolean> => {
    const backpropData = nnRef.current.lastBackwardSteps;
    if (!backpropData) return false;

    const completed = await runAnimationLoop<BackwardStage, BackwardCalculation>({
      mode            : 'backward',
      layers          : BACKWARD_LAYER_ORDER,
      getNeuronIndices: getBackwardNeuronIndices,
      stages          : BACKPROP_STAGES,
      stageDurations  : BACKWARD_STAGE_DURATIONS,
      data            : backpropData,
      onTick          : fsmActions.backwardTick,
      onComplete      : fsmActions.backwardComplete,
      shouldStop      : shouldPauseAnimation,
      sleep,
      startFrom       : startFrom && { layer: startFrom.layer, neuronIndex: startFrom.index },
    });

    if (completed) {
      state.modalSetters.setBackpropSummaryData(createBackpropSummaryData(backpropData, nnRef.current.learningRate));
    }
    return completed;
  }, [nnRef, fsmActions, shouldPauseAnimation, sleep, state.modalSetters]);

  /** Resume the running animation from the neuron the FSM currently points at */
  const continueFromCurrentPosition = useCallback(async () => {
    if (animationState.type === 'forward_animating') {
      await animateForwardPropagation({ layer: animationState.layer, index: animationState.neuronIndex });
    } else if (animationState.type === 'backward_animating') {
      await animateBackwardPropagation({ layer: animationState.layer, index: animationState.neuronIndex });
    }
  }, [animationState, animateForwardPropagation, animateBackwardPropagation]);

  // =========================================================================
  // 5. TRAINING CONTROLS
  // =========================================================================
  const setVisualizer = useCallback((v: Visualizer) => {
    visualizerRef.current = v;
  }, [visualizerRef]);

  const trainOneEpochWithoutAnimation = useCallback((): number => {
    const loss = trainSliderCandidate();
    commitTrainingStats(loss);
    computeAndRefreshDisplay();
    return loss;
  }, [trainSliderCandidate, commitTrainingStats, computeAndRefreshDisplay]);

  // The auto-train interval calls through this ref so it always uses the latest inputs
  useEffect(() => {
    trainStepRef.current = trainOneEpochWithoutAnimation;
  }, [trainOneEpochWithoutAnimation]);

  const trainOneStepWithAnimation = useCallback(async (sample?: AnimatedSample) => {
    if (isAnimating) {
      if (isPaused) {
        interruptReasonRef.current = 'none'; // the loop checks this synchronously
        fsmActions.resume();
        await continueFromCurrentPosition();
      } else {
        fsmActions.pause();
      }
      return;
    }

    activeSampleRef.current = sample ?? sliderSample();
    cancelledRef.current = false;
    fsmActions.startTraining();
    displayInputs(activeSampleRef.current.inputs);
    await animateForwardPropagation();
  }, [isAnimating, isPaused, fsmActions, continueFromCurrentPosition, sliderSample, displayInputs, animateForwardPropagation]);

  /** Abort the running animation: stop the loop, close its modals, back to idle. Weights are untouched. */
  const stopAnimation = useCallback(() => {
    cancelledRef.current = true; // any running loop returns at its next check and skips completion
    state.modalSetters.setLossModalData(null);
    state.modalSetters.setBackpropSummaryData(null);
    activeSampleRef.current = null;
    fsmActions.reset();
    computeAndRefreshDisplay();
  }, [state.modalSetters, fsmActions, computeAndRefreshDisplay]);

  const toggleTraining = useCallback(() => {
    if (state.training.isTraining) {
      state.trainingSetters.setIsTraining(false);
      clearTrainingInterval();
      return;
    }

    state.trainingSetters.setIsTraining(true);
    trainingIntervalRef.current = window.setInterval(() => {
      const loss = trainStepRef.current();
      if (loss < AUTO_TRAIN_STOP_LOSS) {
        state.trainingSetters.setIsTraining(false);
        clearTrainingInterval();
      }
    }, AUTO_TRAIN_INTERVAL_MS);
  }, [state.training.isTraining, state.trainingSetters, clearTrainingInterval]);

  const reset = useCallback(() => {
    if (state.training.isTraining) {
      state.trainingSetters.setIsTraining(false);
      clearTrainingInterval();
    }
    // New network keeps the learning rate the user selected
    nnRef.current = new NeuralNetwork(state.stats.learningRate);
    pendingComparisonRef.current = null;
    state.resetAllState();
    fsmActions.reset();
    computeAndRefreshDisplay();
  }, [state.training.isTraining, state.trainingSetters, state.stats.learningRate, state.resetAllState, fsmActions, computeAndRefreshDisplay, nnRef, clearTrainingInterval]);

  const handleLearningRateChange = useCallback((v: number) => {
    state.statsSetters.setLearningRate(v);
    nnRef.current.learningRate = v;
  }, [state.statsSetters, nnRef]);

  // =========================================================================
  // 6. MODAL CONTROLS
  // =========================================================================

  /** Closing the loss modal starts the backward animation */
  const closeLossModal = useCallback(async () => {
    state.modalSetters.setLossModalData(null);
    fsmActions.closeLossModal();

    // Weights were already updated by train(); lastBackwardSteps holds old/new values
    const completed = await animateBackwardPropagation();
    await sleep(BACKWARD_COMPLETE_PAUSE_MS);

    if (completed) {
      commitTrainingStats(nnRef.current.lastLoss);
    }
    computeAndRefreshDisplay();
  }, [state.modalSetters, fsmActions, animateBackwardPropagation, sleep, commitTrainingStats, computeAndRefreshDisplay, nnRef]);

  const closeBackpropModal = useCallback(() => {
    state.modalSetters.setBackpropSummaryData(null);
    fsmActions.closeBackpropModal();
  }, [state.modalSetters, fsmActions]);

  // =========================================================================
  // 7. CANVAS INTERACTION
  // =========================================================================

  /** Same neuron clicked during forward animation: advance to the next stage / neuron */
  const advanceForwardStage = useCallback((neuronLoc: NeuronLocation) => {
    if (animationState.type !== 'forward_animating') return;

    const nextStage = getNextForwardStage(animationState.stage);
    if (nextStage) {
      fsmActions.forwardTick(neuronLoc.layer, neuronLoc.index, nextStage, animationState.neuronData);
      return;
    }

    const nextNeuron = getNextForwardNeuron(neuronLoc.layer, neuronLoc.index);
    if (nextNeuron) {
      const neuronData = nnRef.current.getForwardNeuronData(nextNeuron.layer, nextNeuron.index);
      if (neuronData) {
        fsmActions.jumpToNeuron(nextNeuron.layer, nextNeuron.index);
        fsmActions.forwardTick(nextNeuron.layer, nextNeuron.index, 'dotProduct', neuronData);
      }
    } else {
      completeForwardPass();
    }
  }, [animationState, fsmActions, nnRef, completeForwardPass]);

  /** Same neuron clicked during backward animation: advance to the next stage / neuron */
  const advanceBackwardStage = useCallback((neuronLoc: NeuronLocation) => {
    if (animationState.type !== 'backward_animating') return;

    const nextStage = getNextBackwardStage(animationState.stage);
    if (nextStage) {
      fsmActions.backwardTick(neuronLoc.layer, neuronLoc.index, nextStage, animationState.neuronData);
      return;
    }

    const nextNeuron = getNextBackwardNeuron(neuronLoc.layer, neuronLoc.index);
    if (nextNeuron) {
      const neuronData = nnRef.current.getBackwardNeuronData(nextNeuron.layer, nextNeuron.index);
      if (neuronData) {
        fsmActions.jumpToNeuron(nextNeuron.layer, nextNeuron.index);
        fsmActions.backwardTick(nextNeuron.layer, nextNeuron.index, 'error', neuronData);
      }
    } else {
      const backpropData = nnRef.current.lastBackwardSteps;
      if (backpropData) {
        state.modalSetters.setBackpropSummaryData(createBackpropSummaryData(backpropData, nnRef.current.learningRate));
      }
      fsmActions.backwardComplete();
    }
  }, [animationState, fsmActions, nnRef, state.modalSetters]);

  /** Different neuron clicked: jump there (pauses the running loop) */
  const jumpToNeuron = useCallback((neuronLoc: NeuronLocation) => {
    fsmActions.jumpToNeuron(neuronLoc.layer, neuronLoc.index);

    if (animationState.type === 'forward_animating') {
      const neuronData = nnRef.current.getForwardNeuronData(neuronLoc.layer, neuronLoc.index);
      if (neuronData) {
        fsmActions.forwardTick(neuronLoc.layer, neuronLoc.index, 'dotProduct', neuronData);
      }
    } else if (animationState.type === 'backward_animating') {
      const neuronData = nnRef.current.getBackwardNeuronData(neuronLoc.layer, neuronLoc.index);
      if (neuronData) {
        fsmActions.backwardTick(neuronLoc.layer, neuronLoc.index, 'error', neuronData);
      }
    }
  }, [animationState, fsmActions, nnRef]);

  const handleCanvasClick = useCallback((x: number, y: number) => {
    if (!isAnimating) return;

    const neuron = visualizerRef.current?.findNeuronAtPosition(x, y);
    if (!neuron) return;

    if (isAnimatingAtNeuron(animationState, neuron.layer, neuron.index)) {
      if (animationState.type === 'forward_animating') {
        advanceForwardStage(neuron);
      } else if (animationState.type === 'backward_animating') {
        advanceBackwardStage(neuron);
      }
    } else {
      jumpToNeuron(neuron);
    }
  }, [isAnimating, animationState, visualizerRef, advanceForwardStage, advanceBackwardStage, jumpToNeuron]);

  return {
    state: animationState,
    isAnimating,
    isPaused,
    trainOneStepWithAnimation,
    stopAnimation,
    trainOneEpochWithoutAnimation,
    toggleTraining,
    reset,
    computeAndRefreshDisplay,
    displayInputs,
    refreshDisplayOnly,
    closeLossModal,
    closeBackpropModal,
    handleCanvasClick,
    setVisualizer,
    handleLearningRateChange,
  };
}
