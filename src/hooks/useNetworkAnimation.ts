/**
 * Network Animation Hook
 * 
 * Manages animation logic for forward and backward propagation.
 * Extracted from useNeuralNetwork for better separation of concerns.
 */

import { useCallback, useRef, useEffect, MutableRefObject } from 'react';
import type { NeuralNetwork } from '../lib/core';
import { LAYER_SIZES, FORWARD_LAYER_ORDER, BACKWARD_LAYER_ORDER } from '../lib/core';
import type { Visualizer } from '../lib/visualizer';
import type { BackpropSummaryData, CalculationStage, BackpropStage } from '../lib/types';
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
  shouldStopRef: MutableRefObject<boolean>;
  updateVisualization: () => void;
}

export function useNetworkAnimation(
  nnRef: MutableRefObject<NeuralNetwork>,
  visualizerRef: MutableRefObject<Visualizer | null>,
  state: UseNetworkStateReturn,
  animationMachine: AnimationStateMachine
): UseNetworkAnimationReturn {
  const shouldStopRef = useRef(false);
  const continueFromJumpedPositionRef = useRef<(() => Promise<void>) | null>(null);
  const prevSpeedRef = useRef(animationMachine.state.speed);

  // =========================================================================
  // Update Visualization
  // =========================================================================
  const updateVisualization = useCallback(() => {
    const nn = nnRef.current;
    const inputs = [state.grade, state.attitude, state.response];
    nn.feedforward(inputs);

    if (nn.lastOutput) {
      state.setOutput(nn.lastOutput.toArray());
    }
    state.setSteps(nn.getCalculationSteps());

    // Update activations for heatmap
    if (nn.lastInput && nn.lastHidden1 && nn.lastHidden2 && nn.lastOutput) {
      state.setActivations({
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
        visualizerRef.current.highlightedNeuron = { layer: machineState.layer, index: machineState.neuronIndex };
        visualizerRef.current.calculationStage = machineState.stage;
        visualizerRef.current.currentNeuronData = machineState.neuronData;
        visualizerRef.current.backpropPhase = null;
        visualizerRef.current.currentBackpropData = null;
        visualizerRef.current.backpropStage = null;
        visualizerRef.current.allBackpropData = null;
      } else if (machineState.type === 'backward_animating') {
        visualizerRef.current.highlightedNeuron = null;
        visualizerRef.current.calculationStage = null;
        visualizerRef.current.currentNeuronData = null;
        visualizerRef.current.backpropPhase = { layer: machineState.layer, index: machineState.neuronIndex };
        visualizerRef.current.currentBackpropData = machineState.neuronData;
        visualizerRef.current.backpropStage = machineState.stage;
        visualizerRef.current.allBackpropData = nn.lastBackpropSteps;
      } else {
        visualizerRef.current.highlightedNeuron = null;
        visualizerRef.current.calculationStage = null;
        visualizerRef.current.currentNeuronData = null;
        visualizerRef.current.backpropPhase = null;
        visualizerRef.current.currentBackpropData = null;
        visualizerRef.current.backpropStage = null;
        visualizerRef.current.allBackpropData = null;
      }

      visualizerRef.current.update(nn);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.grade, state.attitude, state.response, animationMachine.state]);

  // =========================================================================
  // Sleep utility
  // =========================================================================
  const sleep = useCallback(async (ms: number, overrideSpeed?: number): Promise<void> => {
    const effectiveSpeed = overrideSpeed ?? state.animationSpeed;

    if (state.isManualMode || effectiveSpeed === 0) {
      await animationMachine.waitForNextStep();
    } else {
      await new Promise(resolve => setTimeout(resolve, ms / effectiveSpeed));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animationMachine, state.isManualMode, state.animationSpeed]);

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
      updateVisualization,
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animationMachine, sleep, updateVisualization]);

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
          if (layer === 'output') {
            nn.weights_hidden2_output.data[neuronIndex] = data.newWeights;
            nn.bias_output.data[neuronIndex][0] = data.newBias;
          } else if (layer === 'layer2') {
            nn.weights_hidden1_hidden2.data[neuronIndex] = data.newWeights;
            nn.bias_hidden2.data[neuronIndex][0] = data.newBias;
          } else if (layer === 'layer1') {
            nn.weights_input_hidden1.data[neuronIndex] = data.newWeights;
            nn.bias_hidden1.data[neuronIndex][0] = data.newBias;
          }
          nn.feedforward(nn.lastInput!.toArray());
          updateVisualization();
        }
      },
      onComplete: () => {
        animationMachine.backwardComplete();
      },
      shouldStop: () => shouldStopRef.current,
      sleep,
      updateVisualization,
      speedOverride,
    });

    // Collect summary data
    const summaryData: BackpropSummaryData = {
      oldWeights: {
        layer1: backpropData.layer1.map(n => [...n.oldWeights]),
        layer2: backpropData.layer2.map(n => [...n.oldWeights]),
        output: backpropData.output.map(n => [...n.oldWeights]),
      },
      newWeights: {
        layer1: backpropData.layer1.map(n => [...n.newWeights]),
        layer2: backpropData.layer2.map(n => [...n.newWeights]),
        output: backpropData.output.map(n => [...n.newWeights]),
      },
      oldBiases: {
        layer1: backpropData.layer1.map(n => n.oldBias),
        layer2: backpropData.layer2.map(n => n.oldBias),
        output: backpropData.output.map(n => n.oldBias),
      },
      newBiases: {
        layer1: backpropData.layer1.map(n => n.newBias),
        layer2: backpropData.layer2.map(n => n.newBias),
        output: backpropData.output.map(n => n.newBias),
      },
      learningRate: state.learningRate,
      totalWeightsUpdated:
        backpropData.layer1.reduce((sum, n) => sum + n.oldWeights.length, 0) +
        backpropData.layer2.reduce((sum, n) => sum + n.oldWeights.length, 0) +
        backpropData.output.reduce((sum, n) => sum + n.oldWeights.length, 0),
    };

    state.setBackpropSummaryData(summaryData);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animationMachine, sleep, updateVisualization, state.learningRate]);

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
      const stageDurations: Record<CalculationStage, number> = {
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
            updateVisualization();
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
            updateVisualization();
            await sleep(stageDurations[stage]);

            if (stage === 'update') {
              if (layer === 'output') {
                nn.weights_hidden2_output.data[neuronIndex] = neuronData.newWeights;
                nn.bias_output.data[neuronIndex][0] = neuronData.newBias;
              } else if (layer === 'layer2') {
                nn.weights_hidden1_hidden2.data[neuronIndex] = neuronData.newWeights;
                nn.bias_hidden2.data[neuronIndex][0] = neuronData.newBias;
              } else if (layer === 'layer1') {
                nn.weights_input_hidden1.data[neuronIndex] = neuronData.newWeights;
                nn.bias_hidden1.data[neuronIndex][0] = neuronData.newBias;
              }
              nn.feedforward(nn.lastInput!.toArray());
              updateVisualization();
            }
          }
        }
      }

      animationMachine.backwardComplete();

      // Summary data
      const summaryData: BackpropSummaryData = {
        oldWeights: {
          layer1: backpropData.layer1.map(n => [...n.oldWeights]),
          layer2: backpropData.layer2.map(n => [...n.oldWeights]),
          output: backpropData.output.map(n => [...n.oldWeights]),
        },
        newWeights: {
          layer1: backpropData.layer1.map(n => [...n.newWeights]),
          layer2: backpropData.layer2.map(n => [...n.newWeights]),
          output: backpropData.output.map(n => [...n.newWeights]),
        },
        oldBiases: {
          layer1: backpropData.layer1.map(n => n.oldBias),
          layer2: backpropData.layer2.map(n => n.oldBias),
          output: backpropData.output.map(n => n.oldBias),
        },
        newBiases: {
          layer1: backpropData.layer1.map(n => n.newBias),
          layer2: backpropData.layer2.map(n => n.newBias),
          output: backpropData.output.map(n => n.newBias),
        },
        learningRate: state.learningRate,
        totalWeightsUpdated:
          backpropData.layer1.reduce((sum, n) => sum + n.oldWeights.length, 0) +
          backpropData.layer2.reduce((sum, n) => sum + n.oldWeights.length, 0) +
          backpropData.output.reduce((sum, n) => sum + n.oldWeights.length, 0),
      };
      state.setBackpropSummaryData(summaryData);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animationMachine, sleep, updateVisualization, state.learningRate]);

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
    updateVisualization,
  };
}
