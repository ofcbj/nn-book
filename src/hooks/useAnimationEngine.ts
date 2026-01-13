/**
 * Animation Engine Hook
 *
 * Unified hook that combines:
 * - Animation State Machine (FSM for animation states)
 * - Animation Loops (forward/backward propagation animation)
 * - Training Controls (train step/epoch, reset)
 * - Canvas Interaction (neuron click handling)
 *
 * This consolidation reduces the number of hooks from 5 to 2,
 * simplifying data flow and reducing cognitive overhead.
 */

import { useReducer, useCallback, useRef, RefObject, useEffect } from 'react';
import { NeuralNetwork } from '../lib/core';
import type { LayerName } from '../lib/core';
import { LAYER_SIZES, FORWARD_LAYER_ORDER, BACKWARD_LAYER_ORDER } from '../lib/core';
import type { Visualizer } from '../lib/visualizer';
import type { UseNetworkStateReturn } from './useNetworkState';
import type { ForwardStage, BackpropStage, NeuronCalculation, BackpropNeuronData } from '../lib/types';
import { createBackpropSummaryData } from '../lib/types';
import { createWeightComparisonData } from '../lib/visualizer/weightComparison';
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

// =============================================================================
// Return Type
// =============================================================================

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

// =============================================================================
// Helper Types
// =============================================================================

type NeuronLocation = { layer: LayerName; index: number };

// =============================================================================
// Main Hook
// =============================================================================

export function useAnimationEngine(
  nnRef: RefObject<NeuralNetwork>,
  visualizerRef: RefObject<Visualizer | null>,
  state: UseNetworkStateReturn
): UseAnimationEngineReturn {

  // ===========================================================================
  // 1. ANIMATION STATE MACHINE
  // ===========================================================================

  const [animationState, dispatch] = useReducer(animationReducer, initialAnimationState);
  const stepResolverRef = useRef<(() => void) | null>(null);
  const prevSpeedRef = useRef(animationState.speed);
  const shouldStopRef = useRef(false);
  const trainingIntervalRef = useRef<number | undefined>(undefined);

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
    layer: LayerName,
    neuronIndex: number,
    stage: BackpropStage,
    neuronData: BackpropNeuronData | null
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

  // ===========================================================================
  // 2. ANIMATION LOOPS & VISUALIZATION
  // ===========================================================================

  // Sync visualizer state with animation machine
  const syncVisualizerState = useCallback(() => {
    if (visualizerRef.current) {
      const nn = nnRef.current;

      if (animationState.type === 'forward_animating') {
        visualizerRef.current.setForwardAnimationState(
          animationState.layer, animationState.neuronIndex, animationState.stage, animationState.neuronData
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
    const inputs = [state.inputs.grade, state.inputs.attitude, state.inputs.response];
    nn.feedforward(inputs);

    if (nn.lastOutput) {
      state.statsSetters.setOutput(nn.lastOutput.toArray());
    }
    state.statsSetters.setSteps(nn.getCalculationSteps());

    // Update activations for heatmap
    if (nn.lastInput && nn.lastHidden1 && nn.lastHidden2 && nn.lastOutput) {
      state.visualizerSetters.setActivations({
        input: nn.lastInput.toArray(),
        layer1: nn.lastHidden1.toArray(),
        layer2: nn.lastHidden2.toArray(),
        output: nn.lastOutput.toArray(),
      });
    }

    syncVisualizerState();
  }, [state.inputs.grade, state.inputs.attitude, state.inputs.response, syncVisualizerState, nnRef, state.statsSetters, state.visualizerSetters]);

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
      shouldStop: () => shouldStopRef.current,
      sleep,
      computeAndRefreshDisplay,
    });
  }, [forwardTick, forwardComplete, refreshDisplayOnly, sleep, computeAndRefreshDisplay, nnRef]);

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
      shouldStop: () => shouldStopRef.current,
      sleep,
      refreshDisplayOnly,
      computeAndRefreshDisplay,
      speedOverride,
    });

    // Collect summary data only if animation completed (not stopped)
    if (!shouldStopRef.current) {
      const summaryData = createBackpropSummaryData(backpropData, state.stats.learningRate);
      state.modalSetters.setBackpropSummaryData(summaryData);
    }
  }, [backwardTick, backwardComplete, sleep, refreshDisplayOnly, computeAndRefreshDisplay, nnRef, state.stats.learningRate, state.modalSetters]);

  // Continue from jumped position
  const continueFromJumpedPosition = useCallback(async () => {
    if (animationState.type !== 'forward_animating' && animationState.type !== 'backward_animating') return;

    const nn = nnRef.current;

    if (animationState.type === 'forward_animating') {
      const calcSteps = nn.getCalculationSteps();
      if (!calcSteps) return;

      const layers = FORWARD_LAYER_ORDER;
      const layerData = { layer1: calcSteps.layer1, layer2: calcSteps.layer2, output: calcSteps.output };
      const stageDurations: Record<ForwardStage, number> = {
        connections: 150,
        dotProduct: 400,
        bias: 400,
        activation: 400,
      };

      const startLayerIdx = layers.indexOf(animationState.layer);
      let startNeuronIdx = animationState.neuronIndex + 1;

      for (let layerIdx = startLayerIdx; layerIdx < layers.length; layerIdx++) {
        const layer = layers[layerIdx];
        const startIdx = layerIdx === startLayerIdx ? startNeuronIdx : 0;

        for (let neuronIndex = startIdx; neuronIndex < LAYER_SIZES[layer]; neuronIndex++) {
          if (shouldStopRef.current) return;

          const neuronData = layerData[layer][neuronIndex];

          for (const stage of FORWARD_STAGES) {
            if (shouldStopRef.current) return;

            forwardTick(layer, neuronIndex, stage, neuronData);
            refreshDisplayOnly();
            await sleep(stageDurations[stage]);
          }
        }
      }

      forwardComplete();

    } else if (animationState.type === 'backward_animating') {
      const backpropData = nn.lastBackpropSteps;
      if (!backpropData) return;

      const layers = BACKWARD_LAYER_ORDER;
      const layerStartIndices = { output: LAYER_SIZES.output - 1, layer2: LAYER_SIZES.layer2 - 1, layer1: LAYER_SIZES.layer1 - 1 };
      const layerData = { layer1: backpropData.layer1, layer2: backpropData.layer2, output: backpropData.output };
      const stageDurations: Record<BackpropStage, number> = {
        error: 300,
        derivative: 350,
        gradient: 350,
        weightDelta: 350,
        allWeightDeltas: 400,
        update: 300,
      };

      const startLayerIdx = layers.indexOf(animationState.layer);
      let startNeuronIdx = animationState.neuronIndex - 1;

      for (let layerIdx = startLayerIdx; layerIdx < layers.length; layerIdx++) {
        const layer = layers[layerIdx];
        const startIdx = layerIdx === startLayerIdx ? startNeuronIdx : layerStartIndices[layer];

        for (let neuronIndex = startIdx; neuronIndex >= 0; neuronIndex--) {
          if (shouldStopRef.current) return;

          const neuronData = layerData[layer][neuronIndex];

          for (const stage of BACKPROP_STAGES) {
            if (shouldStopRef.current) return;

            backwardTick(layer, neuronIndex, stage, neuronData);
            refreshDisplayOnly();
            await sleep(stageDurations[stage]);

            if (stage === 'update') {
              nn.updateNeuronWeights(layer, neuronIndex, neuronData.newWeights, neuronData.newBias);
              nn.feedforward(nn.lastInput!.toArray());
              refreshDisplayOnly();
            }
          }
        }
      }

      backwardComplete();

      if (!shouldStopRef.current) {
        const summaryData = createBackpropSummaryData(backpropData, state.stats.learningRate);
        state.modalSetters.setBackpropSummaryData(summaryData);
      }
    }
  }, [animationState, nnRef, forwardTick, backwardTick, forwardComplete, backwardComplete, refreshDisplayOnly, sleep, state.stats.learningRate, state.modalSetters]);

  // ===========================================================================
  // 3. TRAINING CONTROLS
  // ===========================================================================

  const setVisualizer = useCallback((v: Visualizer) => {
    visualizerRef.current = v;
  }, [visualizerRef]);

  // Train one epoch without animation
  const trainOneEpochWithoutAnimation = useCallback(() => {
    const nn = nnRef.current;
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
      layer1: nn.biasHidden1.data.map(row => row[0]),
      layer2: nn.biasHidden2.data.map(row => row[0]),
      output: nn.biasOutput.data.map(row => row[0])
    };

    nn.train(inputs, targetOneHot);

    const newWeights = {
      layer1: JSON.parse(JSON.stringify(nn.weightsInputHidden1.data)),
      layer2: JSON.parse(JSON.stringify(nn.weightsHidden1Hidden2.data)),
      output: JSON.parse(JSON.stringify(nn.weightsHidden2Output.data))
    };
    const newBiases = {
      layer1: nn.biasHidden1.data.map(row => row[0]),
      layer2: nn.biasHidden2.data.map(row => row[0]),
      output: nn.biasOutput.data.map(row => row[0])
    };

    const comparisonData = createWeightComparisonData(oldWeights, newWeights, oldBiases, newBiases, state.stats.learningRate);
    state.modalSetters.setWeightComparisonData(comparisonData);

    state.statsSetters.setLoss(nn.lastLoss);
    state.statsSetters.setEpoch(prev => prev + 1);
    computeAndRefreshDisplay();
  }, [state.inputs, state.stats.learningRate, state.modalSetters, state.statsSetters, computeAndRefreshDisplay, nnRef]);

  // Train one step with animation
  const trainOneStepWithAnimation = useCallback(async () => {
    // Case 1: Resume from paused position
    if (animating && animationState.isJumped) {
      shouldStopRef.current = false;
      resume(state.training.animationSpeed);
      await continueFromJumpedPosition();

      if (animationState.type === 'forward_animating' && !shouldStopRef.current) {
        const nn = nnRef.current;
        const inputs = [state.inputs.grade, state.inputs.attitude, state.inputs.response];
        const targetOneHot = [0, 0, 0];
        targetOneHot[state.inputs.targetValue] = 1;

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

        nn.computeBackpropagation(inputs, targetOneHot);

        // Restore
        nn.weightsInputHidden1.data = oldWeights.layer1;
        nn.weightsHidden1Hidden2.data = oldWeights.layer2;
        nn.weightsHidden2Output.data = oldWeights.output;
        nn.biasHidden1.data = oldBiases.layer1;
        nn.biasHidden2.data = oldBiases.layer2;
        nn.biasOutput.data = oldBiases.output;
        nn.feedforward(inputs);

        const predictions = nn.lastOutput?.toArray() || [0, 0, 0];
        const loss = nn.lastLoss;
        state.modalSetters.setLossModalData({ targetClass: state.inputs.targetValue, predictions, loss });
      }
      return;
    }

    // Case 2: Pause running animation
    if (animating) {
      pause();
      shouldStopRef.current = true;
      return;
    }

    // Case 3: Start new animation
    shouldStopRef.current = false;
    startTraining();

    const nn = nnRef.current;
    const inputs = [state.inputs.grade, state.inputs.attitude, state.inputs.response];
    const targetOneHot = [0, 0, 0];
    targetOneHot[state.inputs.targetValue] = 1;

    nn.feedforward(inputs);
    await animateForwardPropagation();

    if (shouldStopRef.current) return;

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

    nn.train(inputs, targetOneHot);

    nn.weightsInputHidden1.data = oldWeights.layer1;
    nn.weightsHidden1Hidden2.data = oldWeights.layer2;
    nn.weightsHidden2Output.data = oldWeights.output;
    nn.biasHidden1.data = oldBiases.layer1;
    nn.biasHidden2.data = oldBiases.layer2;
    nn.biasOutput.data = oldBiases.output;
    nn.feedforward(inputs);

    const predictions = nn.lastOutput?.toArray() || [0, 0, 0];
    const loss = nn.lastLoss;
    state.modalSetters.setLossModalData({ targetClass: state.inputs.targetValue, predictions, loss });
  }, [animating, animationState, state.training.animationSpeed, state.inputs, state.modalSetters, resume, continueFromJumpedPosition, pause, startTraining, animateForwardPropagation, nnRef]);

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

    shouldStopRef.current = true;
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
  }, [state, fsmReset, computeAndRefreshDisplay, nnRef]);

  // Learning rate change
  const handleLearningRateChange = useCallback((v: number) => {
    state.statsSetters.setLearningRate(v);
    nnRef.current.learningRate = v;
  }, [state.statsSetters, nnRef]);

  // ===========================================================================
  // 4. MODAL CONTROLS
  // ===========================================================================

  // Close loss modal - start backward propagation
  const closeLossModal = useCallback(async () => {
    state.modalSetters.setLossModalData(null);
    closeLossModalAction();

    const nn = nnRef.current;
    shouldStopRef.current = false;

    const oldWeights = {
      layer1: JSON.parse(JSON.stringify(nn.weightsInputHidden1.data)),
      layer2: JSON.parse(JSON.stringify(nn.weightsHidden1Hidden2.data)),
      output: JSON.parse(JSON.stringify(nn.weightsHidden2Output.data))
    };
    const oldBiases = {
      layer1: nn.biasHidden1.data.map(row => row[0]),
      layer2: nn.biasHidden2.data.map(row => row[0]),
      output: nn.biasOutput.data.map(row => row[0])
    };

    await animateBackwardPropagation(state.training.animationSpeed);
    await sleep(500, state.training.animationSpeed);

    const newWeights = {
      layer1: JSON.parse(JSON.stringify(nn.weightsInputHidden1.data)),
      layer2: JSON.parse(JSON.stringify(nn.weightsHidden1Hidden2.data)),
      output: JSON.parse(JSON.stringify(nn.weightsHidden2Output.data))
    };
    const newBiases = {
      layer1: nn.biasHidden1.data.map(row => row[0]),
      layer2: nn.biasHidden2.data.map(row => row[0]),
      output: nn.biasOutput.data.map(row => row[0])
    };

    const comparisonData = createWeightComparisonData(oldWeights, newWeights, oldBiases, newBiases, state.stats.learningRate);
    state.modalSetters.setWeightComparisonData(comparisonData);

    state.statsSetters.setEpoch(prev => prev + 1);
    state.statsSetters.setLoss(nn.lastLoss);
    computeAndRefreshDisplay();
  }, [state, closeLossModalAction, animateBackwardPropagation, sleep, computeAndRefreshDisplay, nnRef]);

  // Close backprop modal
  const closeBackpropModal = useCallback(() => {
    state.modalSetters.setBackpropSummaryData(null);
    closeBackpropModalAction();
    refreshDisplayOnly();
  }, [state.modalSetters, closeBackpropModalAction, refreshDisplayOnly]);

  // ===========================================================================
  // 5. CANVAS INTERACTION
  // ===========================================================================

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
    const nn = nnRef.current;
    const inputs = [state.inputs.grade, state.inputs.attitude, state.inputs.response];
    const targetOneHot = [0, 0, 0];
    targetOneHot[state.inputs.targetValue] = 1;

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

    nn.computeBackpropagation(inputs, targetOneHot);
    const predictions = nn.lastOutput?.toArray() || [0, 0, 0];
    const currentLoss = nn.lastLoss;

    nn.weightsInputHidden1.data = oldWeights.layer1;
    nn.weightsHidden1Hidden2.data = oldWeights.layer2;
    nn.weightsHidden2Output.data = oldWeights.output;
    nn.biasHidden1.data = oldBiases.layer1;
    nn.biasHidden2.data = oldBiases.layer2;
    nn.biasOutput.data = oldBiases.output;
    nn.feedforward(inputs);

    forwardComplete();
    state.modalSetters.setLossModalData({ targetClass: state.inputs.targetValue, predictions, loss: currentLoss });
  }, [nnRef, state.inputs, state.modalSetters, forwardComplete]);

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
            backwardComplete();
          }
          return;
        }

        // Different neuron clicked - jump
        shouldStopRef.current = true;
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
  }, [animating, animationState, visualizerRef, nnRef, forwardTick, backwardTick, jumpToNeuron, refreshDisplayOnly, getForwardNeuronData, getBackwardNeuronData, completeForwardPass, backwardComplete]);

  // ===========================================================================
  // RETURN
  // ===========================================================================

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
