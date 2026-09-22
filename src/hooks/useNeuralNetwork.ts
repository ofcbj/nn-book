/**
 * Neural Network Hook
 *
 * Main orchestrator that combines:
 * - useNetworkState: React state
 * - useAnimationEngine: animation, training and interaction logic
 *
 * and exposes a grouped, memoized API to the App component.
 */

import { useRef, useMemo } from 'react';
import { NeuralNetwork } from '../lib/core';
import { Visualizer } from '../lib/visualizer';
import type { BackpropSummaryData, WeightComparisonData } from '../lib/types';
import { useNetworkState, NetworkStats, VisualizerState, InputState, LossModalData } from './useNetworkState';
import { useAnimationEngine } from './useAnimationEngine';

// ============================================================================
// Grouped Return Types
// ============================================================================

export interface NetworkCore {
  /** Register the canvas visualizer once it is created */
  setVisualizer: (v: Visualizer) => void;
  /** Redraw the canvas from the current network and animation state */
  redraw: () => void;
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
    data: LossModalData | null;
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
  handleCanvasClick: (x: number, y: number) => void;
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
  // Single network / visualizer instance (the network is replaced on reset)
  const nnRef = useRef(new NeuralNetwork());
  const visualizerRef = useRef<Visualizer | null>(null);

  const state = useNetworkState();
  const engine = useAnimationEngine(nnRef, visualizerRef, state);

  const network = useMemo<NetworkCore>(() => ({
    setVisualizer: engine.setVisualizer,
    redraw: engine.refreshDisplayOnly,
  }), [engine.setVisualizer, engine.refreshDisplayOnly]);

  const inputs = useMemo(() => ({
    ...state.inputs,
    learningRate: state.stats.learningRate,
    animationSpeed: state.training.animationSpeed,
  }), [state.inputs, state.stats.learningRate, state.training.animationSpeed]);

  const controls = useMemo<InputControls>(() => ({
    ...state.inputSetters,
    setLearningRate: engine.handleLearningRateChange,
    setAnimationSpeed: state.trainingSetters.setAnimationSpeed,
  }), [state.inputSetters, engine.handleLearningRateChange, state.trainingSetters.setAnimationSpeed]);

  const training = useMemo(() => ({
    isTraining: state.training.isTraining,
    isAnimating: engine.isAnimating,
    isPaused: engine.isPaused,
  }), [state.training.isTraining, engine.isAnimating, engine.isPaused]);

  const modals = useMemo<ModalState>(() => ({
    loss: {
      show: state.modals.loss.show,
      data: state.modals.loss.data,
      close: engine.closeLossModal,
    },
    backprop: {
      show: state.modals.backprop.show,
      data: state.modals.backprop.data,
      close: engine.closeBackpropModal,
    },
    comparison: {
      show: state.modals.comparison.show,
      data: state.modals.comparison.data,
      // Data is stored by training; the "View" button just re-opens it
      open: () => {
        if (state.modals.comparison.data) {
          state.modals.comparison.open(state.modals.comparison.data);
        }
      },
      close: state.modals.comparison.close,
    },
  }), [state.modals.loss, state.modals.backprop, state.modals.comparison, engine.closeLossModal, engine.closeBackpropModal]);

  const actions = useMemo<TrainingActions>(() => ({
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
    stats: state.stats,
    training,
    modals,
    visualizer: state.visualizer,
    actions,
  };
}
