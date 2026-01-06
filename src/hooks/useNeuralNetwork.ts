/**
 * Neural Network Hook - Refactored with Animation State Machine
 * 
 * This hook is the main orchestrator that combines all specialized hooks:
 * - useNetworkState: State management
 * - useAnimationStateMachine: Animation state transitions
 * - useNetworkAnimation: Animation logic (forward/backward propagation)
 * - useCanvasInteraction: Canvas click handling
 * - useTrainingControls: Training controls and modal management
 */

import { useRef, useEffect } from 'react';
import { NeuralNetwork } from '../lib/core';
import { Visualizer } from '../lib/visualizer';
import type { CalculationSteps, BackpropSummaryData, WeightComparisonData } from '../lib/types';
import type { ActivationData } from '../components/ActivationHeatmap';
import { useNetworkState } from './useNetworkState';
import { useAnimationStateMachine } from './useAnimationStateMachine';
import { useNetworkAnimation } from './useNetworkAnimation';
import { useCanvasInteraction } from './useCanvasInteraction';
import { useTrainingControls } from './useTrainingControls';

// ============================================================================
// Grouped Return Types
// ============================================================================

export interface NetworkCore {
  nn: NeuralNetwork;
  visualizer: Visualizer | null;
  setVisualizer: (v: Visualizer) => void;
}

export interface InputValues {
  grade: number;
  attitude: number;
  response: number;
  targetValue: number;
  learningRate: number;
  animationSpeed: number;
  isManualMode: boolean;
}

export interface InputControls {
  setGrade: (v: number) => void;
  setAttitude: (v: number) => void;
  setResponse: (v: number) => void;
  setTargetValue: (v: number) => void;
  setLearningRate: (v: number) => void;
  setAnimationSpeed: (v: number) => void;
  setIsManualMode: (v: boolean) => void;
  nextStep: () => void;
}

export interface NetworkStats {
  epoch: number;
  loss: number;
  output: number[] | null;
  steps: CalculationSteps | null;
}

export interface TrainingState {
  isTraining: boolean;
  isAnimating: boolean;
}

export interface ModalState {
  loss: {
    show: boolean;
    data: { targetClass: number; predictions: number[]; loss: number } | null;
    close: () => Promise<void>;
  };
  backprop: {
    show: boolean;
    data: BackpropSummaryData | null;
    close: () => void;
  };
  comparison: {
    show: boolean;
    data: WeightComparisonData | null;
    open: () => void;
    close: () => void;
  };
}

export interface VisualizationState {
  showCanvasHeatmap: boolean;
  showGridHeatmap: boolean;
  activations: ActivationData | null;
  toggleCanvasHeatmap: () => void;
  toggleGridHeatmap: () => void;
}

export interface TrainingActions {
  trainOneStep: () => Promise<void>;
  toggleTraining: () => void;
  reset: () => void;
  computeAndRefreshDisplay: () => void;
  handleCanvasClick: (x?: number, y?: number) => void;
}

export interface UseNeuralNetworkReturn {
  network: NetworkCore;
  inputs: InputValues;
  controls: InputControls;
  stats: NetworkStats;
  training: TrainingState;
  modals: ModalState;
  visualization: VisualizationState;
  actions: TrainingActions;
}

export function useNeuralNetwork(): UseNeuralNetworkReturn {
  // =========================================================================
  // Core Refs
  // =========================================================================
  const nnRef = useRef(new NeuralNetwork());
  const visualizerRef = useRef<Visualizer | null>(null);

  // =========================================================================
  // Hooks
  // =========================================================================
  const state = useNetworkState();
  const animationMachine = useAnimationStateMachine();
  
  const animation = useNetworkAnimation(
    nnRef,
    visualizerRef,
    state,
    animationMachine
  );
  
  const canvasInteraction = useCanvasInteraction(
    nnRef,
    visualizerRef,
    state,
    animationMachine,
    animation
  );
  
  const trainingControls = useTrainingControls(
    nnRef,
    visualizerRef,
    state,
    animationMachine,
    animation
  );

  // =========================================================================
  // Sync animation speed with state machine
  // =========================================================================
  useEffect(() => {
    animationMachine.setSpeed(state.animationSpeed);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.animationSpeed]);

  // =========================================================================
  // Handle learning rate changes
  // =========================================================================
  const handleLearningRateChange = (v: number) => {
    state.setLearningRate(v);
    nnRef.current.learningRate = v;
  };

  // =========================================================================
  // Return combined interface
  // =========================================================================
  return {
    network: {
      nn: nnRef.current,
      visualizer: visualizerRef.current,
      setVisualizer: trainingControls.setVisualizer,
    },

    inputs: {
      grade: state.grade,
      attitude: state.attitude,
      response: state.response,
      targetValue: state.targetValue,
      learningRate: state.learningRate,
      animationSpeed: state.animationSpeed,
      isManualMode: state.isManualMode,
    },

    controls: {
      setGrade: state.setGrade,
      setAttitude: state.setAttitude,
      setResponse: state.setResponse,
      setTargetValue: state.setTargetValue,
      setLearningRate: handleLearningRateChange,
      setAnimationSpeed: state.setAnimationSpeed,
      setIsManualMode: state.setIsManualMode,
      nextStep: trainingControls.nextStep,
    },

    stats: {
      epoch: state.epoch,
      loss: state.loss,
      output: state.output,
      steps: state.steps,
    },

    training: {
      isTraining: state.isTraining,
      isAnimating: animationMachine.isAnimating,
    },

    modals: {
      loss: {
        show: animationMachine.state.type === 'showing_loss_modal',
        data: state.lossModalData,
        close: trainingControls.closeLossModal,
      },
      backprop: {
        show: animationMachine.state.type === 'showing_backprop_modal',
        data: state.backpropSummaryData,
        close: trainingControls.closeBackpropModal,
      },
      comparison: {
        show: state.showComparisonModal,
        data: state.weightComparisonData,
        open: state.openComparisonModal,
        close: state.closeComparisonModal,
      },
    },

    visualization: {
      showCanvasHeatmap: state.showCanvasHeatmap,
      showGridHeatmap: state.showGridHeatmap,
      activations: state.activations,
      toggleCanvasHeatmap: trainingControls.toggleCanvasHeatmap,
      toggleGridHeatmap: state.toggleGridHeatmap,
    },

    actions: {
      trainOneStep: trainingControls.trainOneStepWithAnimation,
      toggleTraining: trainingControls.toggleTraining,
      reset: trainingControls.reset,
      computeAndRefreshDisplay: animation.computeAndRefreshDisplay,
      handleCanvasClick: canvasInteraction.handleCanvasClick,
    },
  };
}
