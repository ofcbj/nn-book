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
import type { BackpropSummaryData, WeightComparisonData } from '../lib/types';
import { useNetworkState, NetworkStats, VisualizerState, InputState } from './useNetworkState';
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


export interface TrainingActions {
  trainOneStep: () => Promise<void>;
  toggleTraining: () => void;
  reset: () => void;
  computeAndRefreshDisplay: () => void;
  handleCanvasClick: (x?: number, y?: number) => void;
}

export interface UseNeuralNetworkReturn {
  network: NetworkCore;
  inputs: InputState & {
    learningRate: number;
    animationSpeed: number;
    isManualMode: boolean;
  };
  controls: InputControls;
  stats: NetworkStats;
  training: {
    isTraining: boolean;
    isAnimating: boolean;
  };
  modals: ModalState;
  visualizer: VisualizerState & {
    toggleCanvasHeatmap: () => void;
    toggleGridHeatmap: () => void;
  };
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
    animationMachine.setSpeed(state.training.animationSpeed);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.training.animationSpeed]);

  // =========================================================================
  // Handle learning rate changes
  // =========================================================================
  const handleLearningRateChange = (v: number) => {
    state.statsSetters.setLearningRate(v);
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
      ...state.inputs,
      learningRate: state.stats.learningRate,
      animationSpeed: state.training.animationSpeed,
      isManualMode: state.training.isManualMode,
    },

    controls: {
      ...state.inputSetters,
      setLearningRate: handleLearningRateChange,
      setAnimationSpeed: state.trainingSetters.setAnimationSpeed,
      setIsManualMode: state.trainingSetters.setIsManualMode,
      nextStep: trainingControls.nextStep,
    },

    stats: {
      epoch: state.stats.epoch,
      loss: state.stats.loss,
      learningRate: state.stats.learningRate,
      output: state.stats.output,
      steps: state.stats.steps,
    },

    training: {
      isTraining: state.training.isTraining,
      isAnimating: animationMachine.isAnimating,
    },

    modals: {
      loss: {
        ...state.modals.loss,
        close: trainingControls.closeLossModal,
      },
      backprop: {
        ...state.modals.backprop,
        close: trainingControls.closeBackpropModal,
      },
      comparison: {
        ...state.modals.comparison,
        open: state.modalActions.openComparisonModal,
        close: state.modalActions.closeComparisonModal,
      },
    },

    visualizer: {
      ...state.visualizer,
      toggleCanvasHeatmap: trainingControls.toggleCanvasHeatmap,
      toggleGridHeatmap: state.visualizerActions.toggleGridHeatmap,
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
