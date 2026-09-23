/**
 * Network State Hook
 *
 * Manages all React state for the neural network visualizer,
 * organized by topic (inputs, stats, training, visualizer, modals).
 */

import { useState, useMemo, useCallback } from 'react';
import type { ForwardSteps, BackpropSummaryData, WeightComparisonData, LossPoint } from '../lib/types';
import type { ActivationData } from '../components/ActivationHeatmap';
import { DEFAULT_LEARNING_RATE, OUTPUT_CLASSES } from '../lib/core';
import { useModal, type UseModalReturn } from './useModalState';

// =============================================================================
// State Interfaces
// =============================================================================

export interface InputState {
  grade: number;
  attitude: number;
  response: number;
  targetValue: number;
}

export interface NetworkStats {
  epoch: number;
  loss: number;
  /** Loss of every committed training step, oldest first (capped at MAX_LOSS_HISTORY) */
  lossHistory: LossPoint[];
  learningRate: number;
  output: number[] | null;
  steps: ForwardSteps | null;
}

/** Upper bound on stored loss points; auto-training adds 20 per second */
export const MAX_LOSS_HISTORY = 2000;

/** 'single': the slider candidate is trained (step-by-step animation, train once, auto train).
 *  'dataset': a stream of generated candidates is trained one by one (data training mode). */
export type TrainingMode = 'single' | 'dataset';

export interface TrainingConfig {
  isTraining: boolean;
  animationSpeed: number;
  mode: TrainingMode;
}

export interface VisualizerState {
  activations: ActivationData | null;
}

export type LossModalData = { targetClass: number; predictions: number[]; loss: number };

export interface ModalState {
  loss: UseModalReturn<LossModalData>;
  backprop: UseModalReturn<BackpropSummaryData>;
  comparison: UseModalReturn<WeightComparisonData>;
}

// =============================================================================
// Setters
// =============================================================================

export interface InputSetters {
  setGrade: (v: number) => void;
  setAttitude: (v: number) => void;
  setResponse: (v: number) => void;
  setTargetValue: (v: number) => void;
}

export interface StatsSetters {
  setEpoch: (v: number | ((prev: number) => number)) => void;
  setLoss: (v: number) => void;
  /** Append the loss of a completed training step (epoch = previous point's epoch + 1) */
  recordLoss: (loss: number) => void;
  clearLossHistory: () => void;
  setLearningRate: (v: number) => void;
  setOutput: (v: number[] | null) => void;
  setSteps: (v: ForwardSteps | null) => void;
}

export interface TrainingSetters {
  setIsTraining: (v: boolean) => void;
  setAnimationSpeed: (v: number) => void;
  setMode: (v: TrainingMode) => void;
}

export interface VisualizerSetters {
  setActivations: (v: ActivationData | null) => void;
}

export interface ModalSetters {
  /** Open the loss modal with data, or close it with null */
  setLossModalData: (v: LossModalData | null) => void;
  /** Open the backprop summary modal with data, or close it with null */
  setBackpropSummaryData: (v: BackpropSummaryData | null) => void;
  /** Store comparison data without opening the modal (the "View" button opens it) */
  setWeightComparisonData: (v: WeightComparisonData | null) => void;
}

// =============================================================================
// Return Type
// =============================================================================

export interface UseNetworkStateReturn {
  inputs            : InputState;
  stats             : NetworkStats;
  training          : TrainingConfig;
  visualizer        : VisualizerState;
  modals            : ModalState;

  inputSetters      : InputSetters;
  statsSetters      : StatsSetters;
  trainingSetters   : TrainingSetters;
  visualizerSetters : VisualizerSetters;
  modalSetters      : ModalSetters;

  /** Reset stats and modals, and randomize the inputs (learning rate is kept) */
  resetAllState     : () => void;
}

// =============================================================================
// Hook Implementation
// =============================================================================

export function useNetworkState(): UseNetworkStateReturn {
  // Input state
  const [grade, setGrade] = useState(0.7);
  const [attitude, setAttitude] = useState(0.5);
  const [response, setResponse] = useState(0.8);
  const [targetValue, setTargetValue] = useState(2);

  // Network stats
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(0);
  const [lossHistory, setLossHistory] = useState<LossPoint[]>([]);
  const [learningRate, setLearningRate] = useState(DEFAULT_LEARNING_RATE);
  const [output, setOutput] = useState<number[] | null>(null);
  const [steps, setSteps] = useState<ForwardSteps | null>(null);

  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [animationSpeed, setAnimationSpeed] = useState(1.0);
  const [mode, setMode] = useState<TrainingMode>('single');

  // Visualizer state
  const [activations, setActivations] = useState<ActivationData | null>(null);

  // Modal state
  const lossModal = useModal<LossModalData>();
  const backpropModal = useModal<BackpropSummaryData>();
  const comparisonModal = useModal<WeightComparisonData>();

  // useState setters are stable, so these groups never need to be recreated
  const inputSetters = useMemo<InputSetters>(() => ({ setGrade, setAttitude, setResponse, setTargetValue }), []);
  const recordLoss = useCallback((loss: number) => {
    setLossHistory(prev => {
      const epoch = (prev[prev.length - 1]?.epoch ?? 0) + 1;
      const kept = prev.length >= MAX_LOSS_HISTORY ? prev.slice(prev.length - MAX_LOSS_HISTORY + 1) : prev;
      return [...kept, { epoch, loss }];
    });
  }, []);
  const clearLossHistory = useCallback(() => setLossHistory([]), []);
  const statsSetters = useMemo<StatsSetters>(() => ({ setEpoch, setLoss, recordLoss, clearLossHistory, setLearningRate, setOutput, setSteps }), [recordLoss, clearLossHistory]);
  const trainingSetters = useMemo<TrainingSetters>(() => ({ setIsTraining, setAnimationSpeed, setMode }), []);
  const visualizerSetters = useMemo<VisualizerSetters>(() => ({ setActivations }), []);

  const modalSetters = useMemo<ModalSetters>(() => ({
    setLossModalData: (data) => data ? lossModal.open(data) : lossModal.close(),
    setBackpropSummaryData: (data) => data ? backpropModal.open(data) : backpropModal.close(),
    setWeightComparisonData: (data) => comparisonModal.setData(data),
  }), [lossModal.open, lossModal.close, backpropModal.open, backpropModal.close, comparisonModal.setData]);

  const resetAllState = useCallback(() => {
    setEpoch(0);
    setLoss(0);
    setLossHistory([]);
    setOutput(null);

    lossModal.close();
    backpropModal.close();
    comparisonModal.close();
    comparisonModal.setData(null);

    setGrade(Math.random());
    setAttitude(Math.random());
    setResponse(Math.random());
    setTargetValue(Math.floor(Math.random() * OUTPUT_CLASSES));
  }, [lossModal.close, backpropModal.close, comparisonModal.close, comparisonModal.setData]);

  const inputs = useMemo<InputState>(() => ({ grade, attitude, response, targetValue }), [grade, attitude, response, targetValue]);
  const stats = useMemo<NetworkStats>(() => ({ epoch, loss, lossHistory, learningRate, output, steps }), [epoch, loss, lossHistory, learningRate, output, steps]);
  const training = useMemo<TrainingConfig>(() => ({ isTraining, animationSpeed, mode }), [isTraining, animationSpeed, mode]);
  const visualizer = useMemo<VisualizerState>(() => ({ activations }), [activations]);
  const modals = useMemo<ModalState>(() => ({ loss: lossModal, backprop: backpropModal, comparison: comparisonModal }), [lossModal, backpropModal, comparisonModal]);

  return {
    inputs,
    stats,
    training,
    visualizer,
    modals,
    inputSetters,
    statsSetters,
    trainingSetters,
    visualizerSetters,
    modalSetters,
    resetAllState,
  };
}
