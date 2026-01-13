/**
 * Network Animation Hook
 * 
 * Manages animation logic for forward and backward propagation.
 * Extracted from useNeuralNetwork for better separation of concerns.
 */

import { useCallback, useRef, RefObject } from 'react';
import type { NeuralNetwork } from '../lib/core';
import { LAYER_SIZES, FORWARD_LAYER_ORDER, BACKWARD_LAYER_ORDER, type LayerName } from '../lib/core';
import type { Visualizer } from '../lib/visualizer';
import { createBackpropSummaryData, type ForwardStage, type BackpropStage } from '../lib/types';
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
  shouldStopRef: RefObject<boolean>;  // Kept for backward compatibility
  // Semantic animation control functions
  stopAnimation: () => void;
  resumeAnimation: () => void;
  isStopped: () => boolean;
  computeAndRefreshDisplay: () => void;
  refreshDisplayOnly: () => void;
}

export function useNetworkAnimation(
  nnRef: RefObject<NeuralNetwork>,
  visualizerRef: RefObject<Visualizer | null>,
  state: UseNetworkStateReturn,
  animationMachine: AnimationStateMachine
): UseNetworkAnimationReturn {
  const shouldStopRef = useRef(false);
  const continueFromJumpedPositionRef = useRef<(() => Promise<void>) | null>(null);
  const animationMachineRef = useRef(animationMachine);

  // Always keep the ref updated
  animationMachineRef.current = animationMachine;

  // =========================================================================
  // Common visualizer sync logic
  // =========================================================================
  const syncVisualizerState = useCallback(() => {
    if (visualizerRef.current) {
      const nn = nnRef.current;
      // Use ref to get latest state, not closure!
      const machineState = animationMachineRef.current.state;


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
  }, []);  // No dependencies - always use refs for latest values!

  // =========================================================================
  // Refresh Display Only (no recalculation)
  // =========================================================================
  const refreshDisplayOnly = useCallback(() => {
    // Just sync visualizer and update canvas - no feedforward
    syncVisualizerState();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animationMachine.state]);  // Only depend on state, not the function

  // =========================================================================
  // Compute and Refresh Display (full recalculation + refresh)
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

    // Sync visualizer and update canvas
    syncVisualizerState();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.inputs.grade, state.inputs.attitude, state.inputs.response]);  // Removed syncVisualizerState - causes infinite loop

  // =========================================================================
  // Sleep utility
  // =========================================================================
  const sleep = useCallback(async (ms: number, overrideSpeed?: number): Promise<void> => {
    // Always read the latest speed from the ref to get real-time updates
    const effectiveSpeed = overrideSpeed ?? animationMachineRef.current.state.speed;
    if (effectiveSpeed === 0) {
      // Paused state - wait indefinitely until resumed
      await animationMachineRef.current.waitForNextStep();
    } else {
      await new Promise(resolve => setTimeout(resolve, ms / effectiveSpeed));
    }
  }, []); // No dependencies - always uses ref for latest value

  // =========================================================================
  // Forward Propagation Animation
  // =========================================================================
  const animateForwardPropagation = useCallback(async () => {
    const nn = nnRef.current;

    // Manually trigger first tick to ensure state changes from idle to forward_animating
    const calcSteps = nn.getCalculationSteps();
    if (calcSteps) {
      const firstData = calcSteps.layer1[0];

      animationMachine.forwardTick('layer1', 0, 'connections', firstData);
      await sleep(FORWARD_STAGE_DURATIONS.connections);
    }

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
      refreshDisplayOnly,
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
          // Removed duplicate computeAndRefreshDisplay() - already called in loop
        }
      },
      onComplete: () => {
        animationMachine.backwardComplete();
      },
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
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animationMachine, sleep, computeAndRefreshDisplay, state.stats.learningRate]);

  // =========================================================================
  // Continue Animation from Jumped Position
  // =========================================================================
  const continueFromJumpedPosition = useCallback(async () => {
    const machineState = animationMachine.state;
    if (machineState.type !== 'forward_animating' && machineState.type !== 'backward_animating')
      return;

    const nn = nnRef.current;

    // Common animation configuration
    const commonConfig = {
      shouldStop: () => shouldStopRef.current,
      sleep,
      refreshDisplayOnly,
      computeAndRefreshDisplay,
    };

    // Mode-specific configurations
    const animationConfigs = {
      forward: () => {
        const calcSteps = nn.getCalculationSteps();
        if (!calcSteps) return null;

        return {
          mode: 'forward' as const,
          layers: FORWARD_LAYER_ORDER,
          getNeuronIndices: forwardNeuronIndices,
          stages: FORWARD_STAGES,
          stageDurations: FORWARD_STAGE_DURATIONS,
          getData: () => {
            const calcSteps = nn.getCalculationSteps();
            if (!calcSteps) return null;
            return { layer1: calcSteps.layer1, layer2: calcSteps.layer2, output: calcSteps.output };
          },
          onTick: (layer: LayerName, neuronIndex: number, stage: ForwardStage, data: any) => {
            animationMachine.forwardTick(layer, neuronIndex, stage, data);
          },
          onComplete: () => {
            animationMachine.forwardComplete();
          },
          resumeFrom: {
            layerIndex: FORWARD_LAYER_ORDER.indexOf(machineState.layer),
            neuronIndex: machineState.neuronIndex + 1
          }
        };
      },

      backward: () => {
        const backpropData = nn.lastBackpropSteps;
        if (!backpropData) return null;

        return {
          mode: 'backward' as const,
          layers: BACKWARD_LAYER_ORDER,
          getNeuronIndices: backwardNeuronIndices,
          stages: BACKPROP_STAGES,
          stageDurations: BACKWARD_STAGE_DURATIONS,
          getData: () => {
            return { layer1: backpropData.layer1, layer2: backpropData.layer2, output: backpropData.output };
          },
          onTick: (layer: LayerName, neuronIndex: number, stage: BackpropStage, data: any) => {
            animationMachine.backwardTick(layer, neuronIndex, stage, data);
          },
          onStageComplete: (layer: LayerName, neuronIndex: number, stage: BackpropStage, data: any) => {
            if (stage === 'update') {
              nn.updateNeuronWeights(layer, neuronIndex, data.newWeights, data.newBias);
              nn.feedforward(nn.lastInput!.toArray());
            }
          },
          onComplete: () => {
            animationMachine.backwardComplete();

            // Summary data only if animation completed (not stopped)
            if (!shouldStopRef.current) {
              const summaryData = createBackpropSummaryData(backpropData, state.stats.learningRate);
              state.modalSetters.setBackpropSummaryData(summaryData);
            }
          },
          resumeFrom: {
            layerIndex: BACKWARD_LAYER_ORDER.indexOf(machineState.layer),
            // For backward animation, getNeuronIndices returns reversed array
            neuronIndex: LAYER_SIZES[machineState.layer] - machineState.neuronIndex
          }
        };
      }
    };

    // Select and execute appropriate animation configuration
    const mode = machineState.type === 'forward_animating' ? 'forward' : 'backward';
    const modeConfig = animationConfigs[mode]();
    
    if (!modeConfig) return;

    await runAnimationLoop({
      ...commonConfig,
      ...modeConfig
    } as any);  // Type assertion needed due to union type complexity
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animationMachine, sleep, computeAndRefreshDisplay, state.stats.learningRate]);

  // Store reference for useEffect
  continueFromJumpedPositionRef.current = continueFromJumpedPosition;

  // =========================================================================
  // Semantic Animation Control Functions
  // =========================================================================
  const stopAnimation = useCallback(() => {
    shouldStopRef.current = true;
  }, []);

  const resumeAnimation = useCallback(() => {
    shouldStopRef.current = false;
  }, []);

  const isStopped = useCallback(() => {
    return shouldStopRef.current;
  }, []);

  return {
    animateForwardPropagation,
    animateBackwardPropagation,
    continueFromJumpedPosition,
    sleep,
    shouldStopRef,  // Kept for backward compatibility
    stopAnimation,
    resumeAnimation,
    isStopped,
    computeAndRefreshDisplay,
    refreshDisplayOnly,
  };
}
