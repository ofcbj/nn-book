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
  trainOneStepWithAnimation     : () => Promise<void>;
  trainOneEpochWithoutAnimation : () => void;
  closeLossModal                : () => Promise<void>;
  closeBackpropModal            : () => void;
  toggleTraining                : () => void;
  reset                         : () => void;
  handleLearningRateChange      : (v: number) => void;
  setVisualizer                 : (v: Visualizer) => void;
  trainingIntervalRef           : RefObject<number | undefined>;
}

export function useTrainingControls(
  nnRef                         : RefObject<NeuralNetwork>,
  visualizerRef                 : RefObject<Visualizer | null>,
  state                         : UseNetworkStateReturn,
  animationMachine              : AnimationStateMachine,
  animation                     : UseNetworkAnimationReturn
): UseTrainingControlsReturn {
  const trainingIntervalRef = useRef<number | undefined>(undefined);

  // =========================================================================
  // Set Visualizer
  // =========================================================================
  const setVisualizer = useCallback((v: Visualizer) => {
    visualizerRef.current = v;
  }, [visualizerRef]);

  // =========================================================================
  // Train One Epoch Without Animation
  // =========================================================================
  const trainOneEpochWithoutAnimation = useCallback(() => {
    const nn = nnRef.current;
    const inputs = [state.inputs.grade, state.inputs.attitude, state.inputs.response];
    const targetOneHot = [0, 0, 0];
    targetOneHot[state.inputs.targetValue] = 1;
    // Backup old weights and biases before training
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
    // Train
    nn.train(inputs, targetOneHot);
    // Collect new weights and biases after training
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
    // Create and save weight comparison data
    const comparisonData = createWeightComparisonData(oldWeights, newWeights, oldBiases, newBiases, state.stats.learningRate);
    state.modalSetters.setWeightComparisonData(comparisonData);
    // Update stats
    state.statsSetters.setLoss(nn.lastLoss);
    state.statsSetters.setEpoch(prev => prev + 1);
    animation.computeAndRefreshDisplay();
  }, [state.inputs.grade, state.inputs.attitude, state.inputs.response, state.inputs.targetValue, animation, state.stats.learningRate]);
  // =========================================================================
  // Train One Step With Animation (Stop/Resume Toggle)
  // =========================================================================
  // =========================================================================
  // Helper Functions
  // =========================================================================
  
  const prepareTrainingInputs = useCallback(() => {
    const inputs = [state.inputs.grade, state.inputs.attitude, state.inputs.response];
    const targetOneHot = [0, 0, 0];
    targetOneHot[state.inputs.targetValue] = 1;
    return { inputs, targetOneHot };
  }, [state.inputs]);

  const backupWeightsAndBiases = useCallback((nn: NeuralNetwork) => {
    return {
      weights: {
        layer1: JSON.parse(JSON.stringify(nn.weightsInputHidden1.data)),
        layer2: JSON.parse(JSON.stringify(nn.weightsHidden1Hidden2.data)),
        output: JSON.parse(JSON.stringify(nn.weightsHidden2Output.data))
      },
      biases: {
        layer1: JSON.parse(JSON.stringify(nn.biasHidden1.data)),
        layer2: JSON.parse(JSON.stringify(nn.biasHidden2.data)),
        output: JSON.parse(JSON.stringify(nn.biasOutput.data))
      }
    };
  }, []);

  const restoreWeightsAndBiases = useCallback((
    nn: NeuralNetwork, 
    backup: ReturnType<typeof backupWeightsAndBiases>,
    inputs: number[]
  ) => {
    nn.weightsInputHidden1.data   = backup.weights.layer1;
    nn.weightsHidden1Hidden2.data = backup.weights.layer2;
    nn.weightsHidden2Output.data  = backup.weights.output;
    nn.biasHidden1.data           = backup.biases.layer1;
    nn.biasHidden2.data           = backup.biases.layer2;
    nn.biasOutput.data            = backup.biases.output;
    nn.feedforward(inputs);
  }, []);

  const showLossModal = useCallback((nn: NeuralNetwork) => {
    const predictions = nn.lastOutput?.toArray() || [0, 0, 0];
    const loss = nn.lastLoss;
    state.modalSetters.setLossModalData({ 
      targetClass: state.inputs.targetValue, 
      predictions, 
      loss 
    });
  }, [state.inputs.targetValue, state.modalSetters]);

  // =========================================================================
  // Train One Step With Animation (Stop/Resume Toggle)
  // =========================================================================
  const trainOneStepWithAnimation = useCallback(async () => {
    const machineState = animationMachine.state;
    
    // Case 1: Resume from paused position
    if (animationMachine.isAnimating && machineState.isJumped) {
      animation.shouldStopRef.current = false;
      animationMachine.resume(state.training.animationSpeed);
      await animation.continueFromJumpedPosition();
      // If completed forward propagation, prepare for backprop
      if (machineState.type === 'forward_animating' && !animation.shouldStopRef.current) {
        const nn = nnRef.current;
        const { inputs, targetOneHot } = prepareTrainingInputs();
        const backup = backupWeightsAndBiases(nn);
        
        nn.computeBackpropagation(inputs, targetOneHot);
        restoreWeightsAndBiases(nn, backup, inputs);
        showLossModal(nn);
      }
      return;
    }
    // Case 2: Pause running animation
    if (animationMachine.isAnimating) {
      animationMachine.pause();
      animation.shouldStopRef.current = true;
      return;
    }
    // Case 3: Start new animation
    animation.shouldStopRef.current = false;
    animationMachine.startTraining();

    const nn = nnRef.current;
    const { inputs, targetOneHot } = prepareTrainingInputs();

    nn.feedforward(inputs);
    await animation.animateForwardPropagation();

    if (animation.shouldStopRef.current) return;

    const backup = backupWeightsAndBiases(nn);
    nn.train(inputs, targetOneHot);
    restoreWeightsAndBiases(nn, backup, inputs);
    showLossModal(nn);
  }, [
    state.inputs, 
    state.training.animationSpeed, 
    animation, 
    animationMachine, 
    prepareTrainingInputs,
    backupWeightsAndBiases,
    restoreWeightsAndBiases,
    showLossModal
  ]);

  // =========================================================================
  // Close Loss Modal - Start Backward Propagation
  // =========================================================================
  const closeLossModal = useCallback(async () => {
    state.modalSetters.setLossModalData(null);
    animationMachine.closeLossModal();


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

    await animation.animateBackwardPropagation(state.training.animationSpeed);
    await animation.sleep(500, state.training.animationSpeed);

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
    animation.refreshDisplayOnly();  // No recalculation needed - just closing modal
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
        trainOneEpochWithoutAnimation();
        if (nnRef.current.lastLoss < 0.001) {
          state.trainingSetters.setIsTraining(false);
          if (trainingIntervalRef.current) {
            clearInterval(trainingIntervalRef.current);
          }
        }
      }, 50);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.training.isTraining, trainOneEpochWithoutAnimation]);

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
    state.modalSetters.setWeightComparisonData(null);

    // Randomize input values
    state.inputSetters.setGrade(Math.random());
    state.inputSetters.setAttitude(Math.random());
    state.inputSetters.setResponse(Math.random());
    state.inputSetters.setTargetValue(Math.floor(Math.random() * 3)); // 0, 1, or 2

    animationMachine.reset();
    animation.computeAndRefreshDisplay();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.training.isTraining, animation, animationMachine]);


  // =========================================================================
  // Learning Rate Handler
  // =========================================================================
  const handleLearningRateChange = useCallback((v: number) => {
    state.statsSetters.setLearningRate(v);
    nnRef.current.learningRate = v;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    trainOneStepWithAnimation,
    trainOneEpochWithoutAnimation,
    closeLossModal,
    closeBackpropModal,
    toggleTraining,
    reset,
    handleLearningRateChange,
    setVisualizer,
    trainingIntervalRef,
  };
}
