/**
 * Network Animation Hook
 * 
 * Manages animation logic for forward and backward propagation.
 * Extracted from useNeuralNetwork for better separation of concerns.
 */

import { useCallback, useRef, useEffect, RefObject } from 'react';
import type { NeuralNetwork } from '../lib/core';
import { LAYER_SIZES, FORWARD_LAYER_ORDER, BACKWARD_LAYER_ORDER } from '../lib/core';
import type { Visualizer } from '../lib/visualizer';
import type { ForwardStage, BackpropStage } from '../lib/types';
import { createBackpropSummaryData } from '../lib/types';
import type { UseNetworkStateReturn } from './useNetworkState';
import type { AnimationStateMachine } from './useAnimationStateMachine';
import {
  FORWARD_STAGES,
  BACKPROP_STAGES,
  runAnimationLoop,
} from '../lib/animation';
import {
  forwardNeuronIndices,
  backwardNeuronIndices,
  FORWARD_STAGE_DURATIONS,
  BACKWARD_STAGE_DURATIONS,
} from '../lib/animation/animationLoop';

export interface UseNetworkAnimationReturn {
  animateForwardPropagation: () => Promise<void>;
  animateBackwardPropagation: (speedOverride?: number) => Promise<void>;
  continueFromJumpedPosition: () => Promise<void>;
  sleep: (ms: number, overrideSpeed?: number) => Promise<void>;
  shouldStopRef: RefObject<boolean>;
  computeAndRefreshDisplay: () => void;
}

export function useNetworkAnimation(
  nnRef: RefObject<NeuralNetwork>,
  visualizerRef: RefObject<Visualizer | null>,
  state: UseNetworkStateReturn,
  animationMachine: AnimationStateMachine
): UseNetworkAnimationReturn {
  const shouldStopRef = useRef(false);
  const continueFromJumpedPositionRef = useRef<(() => Promise<void>) | null>(null);
  const prevSpeedRef = useRef(animationMachine.state.speed);

  // =========================================================================
  // Compute Network and Refresh Display
  // Recalculates the network with current inputs and updates both UI state and canvas
  // =========================================================================
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

    if (visualizerRef.current) {
      // Sync visualizer state from state machine
      const machineState = animationMachine.state;

      if (machineState.type === 'forward_animating') {
        visualizerRef.current.setForwardAnimationState(
          machineState.layer, machineState.neuronIndex, machineState.stage, machineState.neuronData
        );
      } else if (machineState.type === 'backward_animating') {
        visualizerRef.current.setBackwardAnimationState(
          machineState.layer, machineState.neuronIndex, machineState.stage, 
          machineState.neuronData, nn.lastBackpropSteps
        );
      } else {
        visualizerRef.current.clearAnimationState();
      }

      visualizerRef.current.update(nn);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.inputs.grade, state.inputs.attitude, state.inputs.response, animationMachine.state]);

  // =========================================================================
  // Sleep utility
  // =========================================================================
  const sleep = useCallback(async (ms: number, overrideSpeed?: number): Promise<void> => {
    const effectiveSpeed = overrideSpeed ?? state.training.animationSpeed;

    if (state.training.isManualMode || effectiveSpeed === 0) {
      await animationMachine.waitForNextStep();
    } else {
      await new Promise(resolve => setTimeout(resolve, ms / effectiveSpeed));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animationMachine, state.training.isManualMode, state.training.animationSpeed]);

  // =========================================================================
  // Forward Propagation Animation
  // =========================================================================
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
      onTick: (layer, neuronIndex, stage, data) => {
        animationMachine.forwardTick(layer, neuronIndex, stage, data);
      },
      onComplete: () => {
        animationMachine.forwardComplete();
      },
      shouldStop: () => shouldStopRef.current,
      sleep,
      computeAndRefreshDisplay,
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animationMachine, sleep, computeAndRefreshDisplay]);

  // =========================================================================
  // Backward Propagation Animation
  // =========================================================================
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
      onTick: (layer, neuronIndex, stage, data) => {
        animationMachine.backwardTick(layer, neuronIndex, stage, data);
      },
      onStageComplete: (layer, neuronIndex, stage, data) => {
        if (stage === 'update') {
          nn.updateNeuronWeights(layer, neuronIndex, data.newWeights, data.newBias);
          nn.feedforward(nn.lastInput!.toArray());
          computeAndRefreshDisplay();
        }
      },
      onComplete: () => {
        animationMachine.backwardComplete();
      },
      shouldStop: () => shouldStopRef.current,
      sleep,
      computeAndRefreshDisplay,
      speedOverride,
    });

    // Collect summary data using helper function
    const summaryData = createBackpropSummaryData(backpropData, state.stats.learningRate);
    state.modalSetters.setBackpropSummaryData(summaryData);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animationMachine, sleep, computeAndRefreshDisplay, state.stats.learningRate]);

  // =========================================================================
  // Continue Animation from Jumped Position
  // =========================================================================
  const continueFromJumpedPosition = useCallback(async () => {
    const machineState = animationMachine.state;
    if (machineState.type !== 'forward_animating' && machineState.type !== 'backward_animating') return;

    const nn = nnRef.current;

    if (machineState.type === 'forward_animating') {
      const calcSteps = nn.getCalculationSteps();
      if (!calcSteps) return;

      const layers = FORWARD_LAYER_ORDER;
      const layerData = { layer1: calcSteps.layer1, layer2: calcSteps.layer2, output: calcSteps.output };

      const baseDelay = 400;
      const connectionDelay = 150;
      const stageDurations: Record<ForwardStage, number> = {
        connections: connectionDelay,
        dotProduct: baseDelay,
        bias: baseDelay,
        activation: baseDelay,
      };

      const startLayerIdx = layers.indexOf(machineState.layer);
      let startNeuronIdx = machineState.neuronIndex + 1;

      for (let layerIdx = startLayerIdx; layerIdx < layers.length; layerIdx++) {
        const layer = layers[layerIdx];
        const startIdx = layerIdx === startLayerIdx ? startNeuronIdx : 0;

        for (let neuronIndex = startIdx; neuronIndex < LAYER_SIZES[layer]; neuronIndex++) {
          if (shouldStopRef.current) return;

          const neuronData = layerData[layer][neuronIndex];

          for (const stage of FORWARD_STAGES) {
            if (shouldStopRef.current) return;

            animationMachine.forwardTick(layer, neuronIndex, stage, neuronData);
            computeAndRefreshDisplay();
            await sleep(stageDurations[stage]);
          }
        }
      }

      animationMachine.forwardComplete();

    } else if (machineState.type === 'backward_animating') {
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

      const startLayerIdx = layers.indexOf(machineState.layer);
      let startNeuronIdx = machineState.neuronIndex - 1;

      for (let layerIdx = startLayerIdx; layerIdx < layers.length; layerIdx++) {
        const layer = layers[layerIdx];
        const startIdx = layerIdx === startLayerIdx ? startNeuronIdx : layerStartIndices[layer];

        for (let neuronIndex = startIdx; neuronIndex >= 0; neuronIndex--) {
          if (shouldStopRef.current) return;

          const neuronData = layerData[layer][neuronIndex];

          for (const stage of BACKPROP_STAGES) {
            if (shouldStopRef.current) return;

            animationMachine.backwardTick(layer, neuronIndex, stage, neuronData);
            computeAndRefreshDisplay();
            await sleep(stageDurations[stage]);

            if (stage === 'update') {
              nn.updateNeuronWeights(layer, neuronIndex, neuronData.newWeights, neuronData.newBias);
              nn.feedforward(nn.lastInput!.toArray());
              computeAndRefreshDisplay();
            }
          }
        }
      }

      animationMachine.backwardComplete();

      // Summary data using helper function
      const summaryData = createBackpropSummaryData(backpropData, state.stats.learningRate);
      state.modalSetters.setBackpropSummaryData(summaryData);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animationMachine, sleep, computeAndRefreshDisplay, state.stats.learningRate]);

  // Store reference for useEffect
  continueFromJumpedPositionRef.current = continueFromJumpedPosition;

  // =========================================================================
  // Effect: Resume animation when speed changes from 0 to > 0 while jumped
  // =========================================================================
  useEffect(() => {
    const currentSpeed = animationMachine.state.speed;
    const wasZero = prevSpeedRef.current === 0;
    const isNowPositive = currentSpeed > 0;
    const isJumped = animationMachine.state.isJumped;

    if (wasZero && isNowPositive && isJumped) {
      shouldStopRef.current = false;
      if (continueFromJumpedPositionRef.current) {
        continueFromJumpedPositionRef.current();
      }
    }

    prevSpeedRef.current = currentSpeed;
  }, [animationMachine.state.speed, animationMachine.state.isJumped]);

  return {
    animateForwardPropagation,
    animateBackwardPropagation,
    continueFromJumpedPosition,
    sleep,
    shouldStopRef,
    computeAndRefreshDisplay,
  };
}
