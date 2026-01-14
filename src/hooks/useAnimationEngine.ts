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
import { LAYER_SIZES, FORWARD_LAYER_ORDER, BACKWARD_LAYER_ORDER } from '../lib/core';
import type { Visualizer } from '../lib/visualizer';
import type { UseNetworkStateReturn } from './useNetworkState';
import type { ForwardStage, BackpropStage, NeuronCalculation, BackpropNeuronData } from '../lib/types';
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
  getNextBackpropStage,
  getNextForwardNeuron,
  getNextBackwardNeuron,
  FORWARD_STAGES,
  BACKPROP_STAGES,
  getHighlightedNeuron,
  getForwardStage,
  getBackpropStage,
  getCurrentNeuronData,
  getCurrentBackpropData,
  runAnimationLoop,
} from '../lib/animation';
import {
  forwardNeuronIndices,
  backwardNeuronIndices,
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
  isForwardMode: boolean;
  isBackwardMode: boolean;
  // === Visualization Data ===
  highlightedNeuron: { layer: LayerName; index: number } | null;
  forwardStage: ForwardStage | null;
  backpropStage: BackpropStage | null;
  currentNeuronData: NeuronCalculation | null;
  currentBackpropData: BackpropNeuronData | null;
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
  const shouldStopRef             = useRef(false);
  const trainingIntervalRef       = useRef<number | undefined>(undefined);
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
    neuronData: NeuronCalculation | null
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
    stage       : BackpropStage,
    neuronData  : BackpropNeuronData | null
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
  const isForwardMode = animationState.type === 'forward_animating' || animationState.type === 'showing_loss_modal';
  const isBackwardMode = animationState.type === 'backward_animating' || animationState.type === 'showing_backprop_modal';
  // Animation Control Helpers - Meaningful abstractions for shouldStopRef
  /**
   * Stops the animation by setting the stop flag.
   * Use when pausing or interrupting an animation.
   */
  const stopAnimation = useCallback(() => {
    shouldStopRef.current = true;
  }, []);

  const startAnimation = useCallback(() => {
    shouldStopRef.current = false;
  }, []);

  const isAnimationStopped = useCallback(() => {
    return shouldStopRef.current;
  }, []);

  const shouldContinueAnimation = useCallback(() => {
    return !shouldStopRef.current;
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
  
  // Helper: Compute backprop without applying weights (for preview)
  const computeBackpropPreview = useCallback(() => {
    const nn = nnRef.current;
    const inputs = getCurrentInputs();
    const snapshot = createSnapshot(nn);
    nn.computeBackpropagation(inputs, getTargetOneHot());
    restoreSnapshot(nn, snapshot, inputs);
  }, [nnRef, getCurrentInputs, getTargetOneHot]);
  
  // Sync visualizer state with animation machine
  const syncVisualizerState = useCallback(() => {
    if (visualizerRef.current) {
      const nn = nnRef.current;

      if (animationState.type === 'forward_animating') {
        visualizerRef.current.setForwardAnimationState(
          animationState.layer, animationState.neuronIndex, animationState.stage, 
          animationState.neuronData
        );
      } else if (animationState.type === 'backward_animating') {
        visualizerRef.current.setBackwardAnimationState(
          animationState.layer, animationState.neuronIndex, animationState.stage,
          animationState.neuronData, nn.lastBackpropSteps
        );
      } else {
        visualizerRef.current.clearAnimationState();
      }
      visualizerRef.current.update(nn);
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
  
  // Forward propagation animation
  const animateForwardPropagation = useCallback(async () => {
    const nn = nnRef.current;
    
    await runAnimationLoop({
      mode: 'forward',
      layers: ['layer1', 'layer2', 'output'],
      getNeuronIndices: forwardNeuronIndices,
      stages: FORWARD_STAGES,
      stageDurations: FORWARD_STAGE_DURATIONS,
      getData: () => {
        const calcSteps = nn.getCalculationSteps();
        if (!calcSteps) return null;
        return { layer1: calcSteps.layer1, layer2: calcSteps.layer2, output: calcSteps.output };
      },
      onTick: forwardTick,
      onComplete: forwardComplete,
      refreshDisplayOnly,
      shouldStop: isAnimationStopped,
      sleep,
      computeAndRefreshDisplay,
    });
  }, [forwardTick, forwardComplete, refreshDisplayOnly, sleep, computeAndRefreshDisplay, nnRef, isAnimationStopped]);
  
  // Backward propagation animation
  const animateBackwardPropagation = useCallback(async (speedOverride: number = 1.0) => {
    const nn = nnRef.current;
    const backpropData = nn.lastBackpropSteps;
    if (!backpropData) return;

    await runAnimationLoop({
      mode: 'backward',
      layers: ['output', 'layer2', 'layer1'],
      getNeuronIndices: backwardNeuronIndices,
      stages: BACKPROP_STAGES,
      stageDurations: BACKWARD_STAGE_DURATIONS,
      getData: () => {
        return { layer1: backpropData.layer1, layer2: backpropData.layer2, output: backpropData.output };
      },
      onTick: backwardTick,
      onStageComplete: (layer, neuronIndex, stage, data) => {
        if (stage === 'update') {
          nn.updateNeuronWeights(layer, neuronIndex, data.newWeights, data.newBias);
          nn.feedforward(nn.lastInput!.toArray());
        }
      },
      onComplete: backwardComplete,
      shouldStop: isAnimationStopped,
      sleep,
      refreshDisplayOnly,
      computeAndRefreshDisplay,
      speedOverride,
    });
    // Collect summary data only if animation completed (not stopped)
    if (shouldContinueAnimation()) {
      const summaryData = createBackpropSummaryData(backpropData, state.stats.learningRate);
      state.modalSetters.setBackpropSummaryData(summaryData);
    }
  }, [backwardTick, backwardComplete, sleep, refreshDisplayOnly, computeAndRefreshDisplay, nnRef, state.stats.learningRate, state.modalSetters, isAnimationStopped, shouldContinueAnimation]);
  
  // Helper: Generic animation continuation from jumped position
  /**
   * Animates remaining neurons from the current jumped position.
   * This is a unified function that handles both forward and backward animations.
   */
  const continueAnimationFromJumpedPosition = useCallback(async <
    TData extends NeuronCalculation | BackpropNeuronData,
    TStage extends string
  >(config: {
    layers: readonly LayerName[];
    currentLayerIndex: number;
    startNeuronIndex: number;
    getDefaultNeuronIndex: (layer: LayerName) => number;
    shouldIncrement: boolean; // true for forward (++), false for backward (--)
    layerData: Record<LayerName, TData[]>;
    stages: readonly TStage[];
    stageDurations: Record<TStage, number>;
    onTick: (layer: LayerName, neuronIndex: number, stage: TStage, data: TData) => void;
    onComplete: () => void;
    onStageComplete?: (layer: LayerName, neuronIndex: number, stage: TStage, data: TData) => void;
    onAnimationComplete?: () => void;
  }) => {
    const {
      layers, currentLayerIndex, startNeuronIndex, getDefaultNeuronIndex,
      shouldIncrement, layerData, stages, stageDurations,
      onTick, onComplete, onStageComplete, onAnimationComplete
    } = config;
    // Animate remaining layers
    for (let layerIdx = currentLayerIndex; layerIdx < layers.length; layerIdx++) {
      const layer = layers[layerIdx];
      const isCurrentLayer = layerIdx === currentLayerIndex;
      const initialNeuronIndex = isCurrentLayer ? startNeuronIndex : getDefaultNeuronIndex(layer);
      // Determine iteration direction and bounds
      const shouldContinue = shouldIncrement
        ? (idx: number) => idx < LAYER_SIZES[layer]
        : (idx: number) => idx >= 0;
      const nextIndex = shouldIncrement ? (idx: number) => idx + 1 : (idx: number) => idx - 1;
      // Animate each neuron in this layer
      for (let neuronIndex = initialNeuronIndex; shouldContinue(neuronIndex); neuronIndex = nextIndex(neuronIndex)) {
        if (isAnimationStopped()) return;

        const neuronData = layerData[layer][neuronIndex];
        // Animate each stage of this neuron
        for (const stage of stages) {
          if (isAnimationStopped()) return;

          onTick(layer, neuronIndex, stage, neuronData);
          refreshDisplayOnly();
          await sleep(stageDurations[stage]);
          // Optional callback after each stage completes
          if (onStageComplete) {
            onStageComplete(layer, neuronIndex, stage, neuronData);
          }
        }
      }
    }

    onComplete();
    // Optional callback after entire animation completes
    if (shouldContinueAnimation() && onAnimationComplete) {
      onAnimationComplete();
    }
  }, [refreshDisplayOnly, sleep, isAnimationStopped, shouldContinueAnimation]);

  // Continue forward animation from jumped position
  const continueForwardFromJumpedPosition = useCallback(async () => {
    if (animationState.type !== 'forward_animating') return;

    const nn = nnRef.current;
    const calcSteps = nn.getCalculationSteps();
    if (!calcSteps) return;

    const layerData = {
      layer1: calcSteps.layer1,
      layer2: calcSteps.layer2,
      output: calcSteps.output
    };

    await continueAnimationFromJumpedPosition<NeuronCalculation, ForwardStage>({
      layers: FORWARD_LAYER_ORDER,
      currentLayerIndex: FORWARD_LAYER_ORDER.indexOf(animationState.layer),
      startNeuronIndex: animationState.neuronIndex + 1,
      getDefaultNeuronIndex: () => 0,
      shouldIncrement: true,
      layerData,
      stages: FORWARD_STAGES,
      stageDurations: FORWARD_STAGE_DURATIONS,
      onTick: (layer, neuronIndex, stage, data) => {
        forwardTick(layer, neuronIndex, stage, data);
      },
      onComplete: forwardComplete,
    });
  }, [animationState, nnRef, forwardTick, forwardComplete, continueAnimationFromJumpedPosition]);

  // Continue backward animation from jumped position
  const continueBackwardFromJumpedPosition = useCallback(async () => {
    if (animationState.type !== 'backward_animating') return;

    const nn = nnRef.current;
    const backpropData = nn.lastBackpropSteps;
    if (!backpropData) return;

    const layerData = {
      layer1: backpropData.layer1,
      layer2: backpropData.layer2,
      output: backpropData.output
    };

    await continueAnimationFromJumpedPosition<BackpropNeuronData, BackpropStage>({
      layers: BACKWARD_LAYER_ORDER,
      currentLayerIndex: BACKWARD_LAYER_ORDER.indexOf(animationState.layer),
      startNeuronIndex: animationState.neuronIndex - 1,
      getDefaultNeuronIndex: (layer) => LAYER_SIZES[layer] - 1,
      shouldIncrement: false,
      layerData,
      stages: BACKPROP_STAGES,
      stageDurations: BACKWARD_STAGE_DURATIONS,
      onTick: (layer, neuronIndex, stage, data) => {
        backwardTick(layer, neuronIndex, stage, data);
      },
      onComplete: backwardComplete,
      onStageComplete: (layer, neuronIndex, stage, data) => {
        // Apply weight updates immediately after 'update' stage
        if (stage === 'update') {
          nn.updateNeuronWeights(layer, neuronIndex, data.newWeights, data.newBias);
          nn.feedforward(nn.lastInput!.toArray());
          refreshDisplayOnly();
        }
      },
      onAnimationComplete: () => {
        // Show backprop summary only if animation completed without interruption
        const summaryData = createBackpropSummaryData(backpropData, state.stats.learningRate);
        state.modalSetters.setBackpropSummaryData(summaryData);
      },
    });
  }, [animationState, nnRef, backwardTick, backwardComplete, continueAnimationFromJumpedPosition, refreshDisplayOnly, state.stats.learningRate, state.modalSetters]);
  
  // Continue from jumped position (dispatcher)
  const continueFromJumpedPosition = useCallback(async () => {
    if (animationState.type === 'forward_animating') {
      await continueForwardFromJumpedPosition();
    } else if (animationState.type === 'backward_animating') {
      await continueBackwardFromJumpedPosition();
    }
  }, [animationState.type, continueForwardFromJumpedPosition, continueBackwardFromJumpedPosition]);

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
    // Case 1: Resume from paused position
    if (animating && animationState.isJumped) {
      startAnimation();
      resume(state.training.animationSpeed);
      await continueFromJumpedPosition();

      if (animationState.type === 'forward_animating' && shouldContinueAnimation()) {
        computeBackpropPreview();
        showLossModal();
      }
      return;
    }
    // Case 2: Pause running animation
    if (animating) {
      pause();
      stopAnimation();
      return;
    }
    // Case 3: Start new animation
    startAnimation();
    startTraining();

    const nn = nnRef.current;
    const inputs = getCurrentInputs();
    nn.feedforward(inputs);
    await animateForwardPropagation();

    if (isAnimationStopped()) return;

    const snapshot = createSnapshot(nn);
    nn.train(inputs, getTargetOneHot());
    restoreSnapshot(nn, snapshot, inputs);
    showLossModal();
  }, [animating, animationState, state.training.animationSpeed, getCurrentInputs, getTargetOneHot, showLossModal, resume, continueFromJumpedPosition, pause, startTraining, animateForwardPropagation, nnRef, startAnimation, stopAnimation, shouldContinueAnimation, isAnimationStopped]);

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
    stopAnimation();
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
  }, [state.training.isTraining, state.trainingSetters, state.statsSetters, state.modalSetters, state.inputSetters, fsmReset, computeAndRefreshDisplay, nnRef, stopAnimation]);

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
    startAnimation,
    shouldContinueAnimation,
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
    const backpropData = nnRef.current.lastBackpropSteps;
    if (!backpropData) return null;
    return backpropData[layer]?.[index] ?? null;
  }, [nnRef]);

  // Helper: Complete forward pass
  const completeForwardPass = useCallback(() => {
    computeBackpropPreview();
    forwardComplete();
    showLossModal();
  }, [computeBackpropPreview, showLossModal, forwardComplete]);

  // Canvas click handler
  const handleCanvasClick = useCallback((x?: number, y?: number) => {
    if (!animating) return;
    const visualizer = visualizerRef.current;
    if (x !== undefined && y !== undefined && visualizer) {
      const neuron = visualizer.findNeuronAtPosition(x, y);

      if (neuron && (neuron.layer === 'layer1' || neuron.layer === 'layer2' || neuron.layer === 'output')) {
        const neuronLoc: NeuronLocation = { layer: neuron.layer, index: neuron.index };

        // Same neuron click - advance stage
        if (animationState.type === 'forward_animating' &&
            animationState.layer === neuronLoc.layer &&
            animationState.neuronIndex === neuronLoc.index) {
          const nextStage = getNextForwardStage(animationState.stage);

          if (nextStage) {
            forwardTick(neuronLoc.layer, neuronLoc.index, nextStage, animationState.neuronData);
            refreshDisplayOnly();
            return;
          }

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
          return;
        }

        if (animationState.type === 'backward_animating' &&
            animationState.layer === neuronLoc.layer &&
            animationState.neuronIndex === neuronLoc.index) {
          const nextStage = getNextBackpropStage(animationState.stage);

          if (nextStage) {
            backwardTick(neuronLoc.layer, neuronLoc.index, nextStage, animationState.neuronData);
            refreshDisplayOnly();
            return;
          }

          const nextNeuron = getNextBackwardNeuron(neuronLoc.layer, neuronLoc.index);
          if (nextNeuron) {
            const neuronData = getBackwardNeuronData(nextNeuron.layer, nextNeuron.index);
            if (neuronData) {
              jumpToNeuron(nextNeuron.layer, nextNeuron.index);
              backwardTick(nextNeuron.layer, nextNeuron.index, 'error', neuronData);
              refreshDisplayOnly();
            }
          } else {
            // Backward propagation completed via click-through
            // Don't set weight comparison data here - it should only be set
            // when backward animation naturally completes in closeLossModal
            backwardComplete();
            refreshDisplayOnly();
          }
          return;
        }
        // Different neuron clicked - jump
        stopAnimation();
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
  }, [animating, animationState, visualizerRef, nnRef, forwardTick, backwardTick, jumpToNeuron, refreshDisplayOnly, getForwardNeuronData, getBackwardNeuronData, completeForwardPass, backwardComplete, stopAnimation]);

  // RETURN
  return {
    // Animation state
    state: animationState,
    dispatch,
    isAnimating: animating,
    isPaused: paused,
    isForwardMode,
    isBackwardMode,
    // Visualization data
    highlightedNeuron: getHighlightedNeuron(animationState),
    forwardStage: getForwardStage(animationState),
    backpropStage: getBackpropStage(animationState),
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
