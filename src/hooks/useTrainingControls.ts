/**
 * Training Controls Hook
 * 
 * Manages training-related actions: animated training, auto-training, reset.
 * Extracted from useNeuralNetwork for better separation of concerns.
 */

import { useCallback, useRef, RefObject } from 'react';
import { NeuralNetwork } from '../lib/core';
import type { Visualizer } from '../lib/visualizer';
import type { UseNetworkStateReturn } from './useNetworkState';
import type { AnimationStateMachine } from './useAnimationStateMachine';
import type { UseNetworkAnimationReturn } from './useNetworkAnimation';
import { createWeightComparisonData } from '../lib/visualizer/weightComparison';

export interface UseTrainingControlsReturn {
  trainOneStepWithAnimation: () => Promise<void>;
  closeLossModal: () => Promise<void>;
  closeBackpropModal: () => void;
  toggleTraining: () => void;
  reset: () => void;
  nextStep: () => void;
  handleLearningRateChange: (v: number) => void;
  toggleCanvasHeatmap: () => void;
  setVisualizer: (v: Visualizer) => void;
  trainingIntervalRef: RefObject<number | undefined>;
}

export function useTrainingControls(
  nnRef: RefObject<NeuralNetwork>,
  visualizerRef: RefObject<Visualizer | null>,
  state: UseNetworkStateReturn,
  animationMachine: AnimationStateMachine,
  animation: UseNetworkAnimationReturn
): UseTrainingControlsReturn {
  const trainingIntervalRef = useRef<number | undefined>(undefined);

  // =========================================================================
  // Set Visualizer
  // =========================================================================
  const setVisualizer = useCallback((v: Visualizer) => {
    visualizerRef.current = v;
  }, [visualizerRef]);

  // =========================================================================
  // Train One Step (non-animated)
  // =========================================================================
  const trainOneStep = useCallback(() => {
    const nn = nnRef.current;
    const inputs = [state.inputs.grade, state.inputs.attitude, state.inputs.response];
    const targetOneHot = [0, 0, 0];
    targetOneHot[state.inputs.targetValue] = 1;

    nn.train(inputs, targetOneHot);
    state.statsSetters.setLoss(nn.lastLoss);
    state.statsSetters.setEpoch(prev => prev + 1);
    animation.computeAndRefreshDisplay();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.inputs.grade, state.inputs.attitude, state.inputs.response, state.inputs.targetValue, animation]);

  // =========================================================================
  // Train One Step With Animation
  // =========================================================================
  const trainOneStepWithAnimation = useCallback(async () => {
    if (animationMachine.isAnimating) {
      animation.shouldStopRef.current = true;
      return;
    }

    animation.shouldStopRef.current = false;
    animationMachine.startTraining();

    const nn = nnRef.current;
    const inputs = [state.inputs.grade, state.inputs.attitude, state.inputs.response];
    const targetOneHot = [0, 0, 0];
    targetOneHot[state.inputs.targetValue] = 1;

    // Forward pass
    nn.feedforward(inputs);
    await animation.animateForwardPropagation();

    if (animation.shouldStopRef.current) return;

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

    // Train
    nn.train(inputs, targetOneHot);
    const predictions = nn.lastOutput?.toArray() || [0, 0, 0];
    const currentLoss = nn.lastLoss;

    // Restore old weights for backprop visualization
    nn.weightsInputHidden1.data = oldWeights.layer1;
    nn.weightsHidden1Hidden2.data = oldWeights.layer2;
    nn.weightsHidden2Output.data = oldWeights.output;
    nn.biasHidden1.data = oldBiases.layer1;
    nn.biasHidden2.data = oldBiases.layer2;
    nn.biasOutput.data = oldBiases.output;
    nn.feedforward(inputs);

    state.modalSetters.setLossModalData({ targetClass: state.inputs.targetValue, predictions, loss: currentLoss });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.inputs.grade, state.inputs.attitude, state.inputs.response, state.inputs.targetValue, animation, animationMachine]);

  // =========================================================================
  // Close Loss Modal - Start Backward Propagation
  // =========================================================================
  const closeLossModal = useCallback(async () => {
    state.modalSetters.setLossModalData(null);
    animationMachine.closeLossModal();

    // Ensure animation speed is at least 1.0 for backprop
    if (state.training.animationSpeed === 0) {
      state.trainingSetters.setAnimationSpeed(1.0);
    }

    const nn = nnRef.current;
    animation.shouldStopRef.current = false;

    // Store old weights for comparison
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

    // Use speed of 1.0 if current speed is 0
    const backpropSpeed = state.training.animationSpeed > 0 ? state.training.animationSpeed : 1.0;
    await animation.animateBackwardPropagation(backpropSpeed);
    await animation.sleep(500, backpropSpeed);

    // Collect new weights
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
    animation.computeAndRefreshDisplay();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animationMachine, animation, state.stats.learningRate, state.training.animationSpeed]);

  // =========================================================================
  // Close Backprop Modal
  // =========================================================================
  const closeBackpropModal = useCallback(() => {
    state.modalSetters.setBackpropSummaryData(null);
    animationMachine.closeBackpropModal();
    animation.computeAndRefreshDisplay();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animationMachine, animation]);

  // =========================================================================
  // Toggle Training (auto)
  // =========================================================================
  const toggleTraining = useCallback(() => {
    if (state.training.isTraining) {
      state.trainingSetters.setIsTraining(false);
      if (trainingIntervalRef.current) {
        clearInterval(trainingIntervalRef.current);
      }
    } else {
      state.trainingSetters.setIsTraining(true);
      trainingIntervalRef.current = window.setInterval(() => {
        trainOneStep();
        if (nnRef.current.lastLoss < 0.001) {
          state.trainingSetters.setIsTraining(false);
          if (trainingIntervalRef.current) {
            clearInterval(trainingIntervalRef.current);
          }
        }
      }, 50);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.training.isTraining, trainOneStep]);

  // =========================================================================
  // Reset
  // =========================================================================
  const reset = useCallback(() => {
    if (state.training.isTraining) {
      state.trainingSetters.setIsTraining(false);
      if (trainingIntervalRef.current) {
        clearInterval(trainingIntervalRef.current);
      }
    }

    animation.shouldStopRef.current = true;
    nnRef.current = new NeuralNetwork();
    state.statsSetters.setEpoch(0);
    state.statsSetters.setLoss(0);
    state.statsSetters.setOutput(null);
    state.modalSetters.setLossModalData(null);
    state.modalSetters.setBackpropSummaryData(null);

    animationMachine.reset();
    animation.computeAndRefreshDisplay();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.training.isTraining, animation, animationMachine]);

  // =========================================================================
  // Next Step (manual mode)
  // =========================================================================
  const nextStep = useCallback(() => {
    animationMachine.nextStep();
  }, [animationMachine]);

  // =========================================================================
  // Learning Rate Handler
  // =========================================================================
  const handleLearningRateChange = useCallback((v: number) => {
    state.statsSetters.setLearningRate(v);
    nnRef.current.learningRate = v;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // =========================================================================
  // Canvas Heatmap Toggle (needs visualizer access)
  // =========================================================================
  const toggleCanvasHeatmap = useCallback(() => {
    const newValue = !state.visualizer.showCanvasHeatmap;
    state.visualizationSetters.setShowCanvasHeatmap(newValue);
    if (visualizerRef.current) {
      visualizerRef.current.setHeatmapMode(newValue);
      visualizerRef.current.update(nnRef.current);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.visualizer.showCanvasHeatmap]);

  return {
    trainOneStepWithAnimation,
    closeLossModal,
    closeBackpropModal,
    toggleTraining,
    reset,
    nextStep,
    handleLearningRateChange,
    toggleCanvasHeatmap,
    setVisualizer,
    trainingIntervalRef,
  };
}
