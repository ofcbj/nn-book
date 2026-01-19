/**
 * Neural Network Hook - Simplified Architecture
 *
 * This hook is the main orchestrator that combines:
 * - useNetworkState: State management
 * - useAnimationEngine: Unified animation, training, and interaction logic
 *
 * Previous 5-hook architecture has been consolidated to 2 hooks for better maintainability.
 */

import { useRef, useMemo } from 'react';
import { NeuralNetwork } from '../lib/core';
import { Visualizer } from '../lib/visualizer';
import type { BackpropSummaryData, WeightComparisonData } from '../lib/types';
import { useNetworkState, NetworkStats, VisualizerState, InputState } from './useNetworkState';
import { useAnimationEngine } from './useAnimationEngine';

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
  trainOneEpoch: () => void;
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
  };
  controls: InputControls;
  stats: NetworkStats;
  training: {
    isTraining: boolean;
    isAnimating: boolean;
    isPaused: boolean;
  };
  modals: ModalState;
  visualizer: VisualizerState;
  actions: TrainingActions;
}

export function useNeuralNetwork(): UseNeuralNetworkReturn {
  // =========================================================================
  // Core Refs - Dual Buffer Pattern
  // displayNN: for visualization (weights unchanged during forward animation)
  // trainNN: for training calculations (loss, backprop data)
  // =========================================================================
  const displayNNRef = useRef(new NeuralNetwork());
  const trainNNRef = useRef(new NeuralNetwork());
  const visualizerRef = useRef<Visualizer | null>(null);

  // Sync trainNN weights with displayNN on mount
  trainNNRef.current.copyWeightsFrom(displayNNRef.current);

  // =========================================================================
  // Hooks - Simplified from 5 to 2
  // =========================================================================
  const state = useNetworkState();
  const engine = useAnimationEngine(displayNNRef, trainNNRef, visualizerRef, state);

  // =========================================================================
  // Return combined interface - Memoized to prevent recreating objects
  // =========================================================================
  const network = useMemo(() => ({
    nn: displayNNRef.current,
    visualizer: visualizerRef.current,
    setVisualizer: engine.setVisualizer,
  }), [engine.setVisualizer]);

  const inputs = useMemo(() => ({
    ...state.inputs,
    learningRate: state.stats.learningRate,
    animationSpeed: state.training.animationSpeed,
  }), [state.inputs, state.stats.learningRate, state.training.animationSpeed]);

  const controls = useMemo(() => ({
    ...state.inputSetters,
    setLearningRate: engine.handleLearningRateChange,
    setAnimationSpeed: state.trainingSetters.setAnimationSpeed,
  }), [state.inputSetters, engine.handleLearningRateChange, state.trainingSetters.setAnimationSpeed]);

  const stats = useMemo(() => ({
    epoch: state.stats.epoch,
    loss: state.stats.loss,
    learningRate: state.stats.learningRate,
    output: state.stats.output,
    steps: state.stats.steps,
  }), [state.stats.epoch, state.stats.loss, state.stats.learningRate, state.stats.output, state.stats.steps]);

  const training = useMemo(() => ({
    isTraining: state.training.isTraining,
    isAnimating: engine.isAnimating,
    isPaused: engine.state.interruptReason !== 'none',
  }), [state.training.isTraining, engine.isAnimating, engine.state.interruptReason]);

  const modals = useMemo(() => ({
    loss: {
      ...state.modals.loss,
      close: engine.closeLossModal,
    },
    backprop: {
      ...state.modals.backprop,
      close: engine.closeBackpropModal,
    },
    comparison: {
      ...state.modals.comparison,
      // Wrapper for open to make it no-argument (data already set by training)
      open: () => {
        if (state.modals.comparison.data) {
          state.modals.comparison.open(state.modals.comparison.data);
        }
      },
    },
  }), [state.modals.loss, state.modals.backprop, state.modals.comparison, engine.closeLossModal, engine.closeBackpropModal]);

  const visualizer = useMemo(() => ({
    ...state.visualizer,
  }), [state.visualizer]);

  const actions = useMemo(() => ({
    trainOneStep            : engine.trainOneStepWithAnimation,
    trainOneEpoch           : engine.trainOneEpochWithoutAnimation,
    toggleTraining          : engine.toggleTraining,
    reset                   : engine.reset,
    computeAndRefreshDisplay: engine.computeAndRefreshDisplay,
    handleCanvasClick       : engine.handleCanvasClick,
  }), [engine.trainOneStepWithAnimation, engine.trainOneEpochWithoutAnimation, 
    engine.toggleTraining, engine.reset, engine.computeAndRefreshDisplay, engine.handleCanvasClick]);

  return {
    network,
    inputs,
    controls,
    stats,
    training,
    modals,
    visualizer,
    actions,
  };
}
