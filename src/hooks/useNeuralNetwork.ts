/**
 * Neural Network Hook
 *
 * Main orchestrator that combines:
 * - useNetworkState: React state
 * - useAnimationEngine: single-candidate animation, training and interaction
 * - useDatasetTraining: data training mode (stream of generated candidates)
 *
 * and exposes a grouped, memoized API to the App component.
 */

import { useRef, useMemo, useCallback, useEffect } from 'react';
import { NeuralNetwork, candidateInputs, toOneHot } from '../lib/core';
import type { Candidate } from '../lib/core';
import { Visualizer } from '../lib/visualizer';
import type { BackpropSummaryData, WeightComparisonData } from '../lib/types';
import { useNetworkState, NetworkStats, VisualizerState, InputState, LossModalData, TrainingMode } from './useNetworkState';
import { useAnimationEngine } from './useAnimationEngine';
import { useDatasetTraining, type UseDatasetTrainingReturn } from './useDatasetTraining';

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

/** A data-mode candidate whose training step is being shown on the network */
export interface Inspection {
  candidate: Candidate;
  isPaused: boolean;
}

export interface TrainingActions {
  trainOneStep: () => Promise<void>;
  /** Data mode: pause the stream and walk through one training step on this candidate */
  inspectCandidate: (candidate: Candidate) => Promise<void>;
  /** Pause / resume the inspection animation */
  toggleInspection: () => void;
  /** Abort the inspection animation */
  stopInspection: () => void;
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
    mode: TrainingMode;
    /** Switch modes; ignored while an animation or auto-training is running */
    setMode: (mode: TrainingMode) => void;
    /** True while something is running that must finish before the mode can change */
    busy: boolean;
    isTraining: boolean;
    isAnimating: boolean;
    isPaused: boolean;
  };
  /** Data training mode: the candidate stream */
  stream: UseDatasetTrainingReturn;
  /** Data training mode: candidate currently being walked through, if any */
  inspection: Inspection | null;
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
  const stream = useDatasetTraining(nnRef, {
    displayInputs: engine.displayInputs,
    statsSetters: state.statsSetters,
    enabled: state.training.mode === 'dataset',
  });

  const busy = engine.isAnimating || state.training.isTraining || stream.running;

  // Data mode: click a record → stop the stream, load it into the sliders, animate one training step on it
  const inspectedRef = useRef<Candidate | null>(null);
  const inspectCandidate = useCallback(async (candidate: Candidate) => {
    if (engine.isAnimating) return;
    stream.pause();
    state.inputSetters.setGrade(candidate.grade);
    state.inputSetters.setAttitude(candidate.attitude);
    state.inputSetters.setResponse(candidate.response);
    state.inputSetters.setTargetValue(candidate.label);
    inspectedRef.current = candidate;
    await engine.trainOneStepWithAnimation({
      inputs: candidateInputs(candidate),
      target: toOneHot(candidate.label),
      targetClass: candidate.label,
    });
  }, [engine.isAnimating, engine.trainOneStepWithAnimation, stream.pause, state.inputSetters]);

  const stopInspection = useCallback(() => {
    inspectedRef.current = null;
    engine.stopAnimation();
  }, [engine.stopAnimation]);

  const animationIdle = engine.state.type === 'idle';
  const inspection = useMemo<Inspection | null>(() => (
    !animationIdle && inspectedRef.current
      ? { candidate: inspectedRef.current, isPaused: engine.isPaused }
      : null
  ), [animationIdle, engine.isPaused]);

  // When an inspection finishes, the network has changed: refresh the dataset accuracy
  useEffect(() => {
    if (animationIdle && inspectedRef.current) {
      inspectedRef.current = null;
      if (state.training.mode === 'dataset') stream.refresh();
    }
  }, [animationIdle, state.training.mode, stream.refresh]);

  const setMode = useCallback((mode: TrainingMode) => {
    if (mode === state.training.mode || engine.isAnimating || state.training.isTraining) return;
    stream.pause();
    state.trainingSetters.setMode(mode);
    // Each mode gets a fresh loss curve and epoch counter (the weights are kept)
    state.statsSetters.clearLossHistory();
    state.statsSetters.setEpoch(0);
    state.statsSetters.setLoss(0);
    engine.computeAndRefreshDisplay();
  }, [state.training.mode, state.training.isTraining, state.trainingSetters, state.statsSetters, engine.isAnimating, engine.computeAndRefreshDisplay, stream.pause]);

  const reset = useCallback(() => {
    engine.reset();      // replaces nnRef.current
    stream.restart();    // rewinds the stream and evaluates the new network
  }, [engine.reset, stream.restart]);

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
    mode: state.training.mode,
    setMode,
    busy,
    isTraining: state.training.isTraining,
    isAnimating: engine.isAnimating,
    isPaused: engine.isPaused,
  }), [state.training.mode, setMode, busy, state.training.isTraining, engine.isAnimating, engine.isPaused]);

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
    trainOneStep            : () => engine.trainOneStepWithAnimation(),
    inspectCandidate,
    toggleInspection        : () => { void engine.trainOneStepWithAnimation(); },
    stopInspection,
    trainOneEpoch           : engine.trainOneEpochWithoutAnimation,
    toggleTraining          : engine.toggleTraining,
    reset,
    computeAndRefreshDisplay: engine.computeAndRefreshDisplay,
    handleCanvasClick       : engine.handleCanvasClick,
  }), [engine.trainOneStepWithAnimation, inspectCandidate, stopInspection, engine.trainOneEpochWithoutAnimation,
    engine.toggleTraining, reset, engine.computeAndRefreshDisplay, engine.handleCanvasClick]);

  return {
    network,
    inputs,
    controls,
    stats: state.stats,
    training,
    stream,
    inspection,
    modals,
    visualizer: state.visualizer,
    actions,
  };
}
