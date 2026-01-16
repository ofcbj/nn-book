/**
 * Network State Hook
 * 
 * Manages all state variables for the neural network visualizer.
 * State is organized by topic for better clarity and maintainability.
 */

import { useState, useMemo, useCallback } from 'react';
import type { ForwardSteps, BackpropSummaryData, WeightComparisonData } from '../lib/types';
import type { ActivationData } from '../components/ActivationHeatmap';
import { useModal, type UseModalReturn } from './useModalState';

// =============================================================================
// State Interfaces - Organized by Topic
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
  learningRate: number;
  output: number[] | null;
  steps: ForwardSteps | null;
}

export interface TrainingConfig {
  isTraining: boolean;
  animationSpeed: number;
}

export interface VisualizerState {
  activations: ActivationData | null;
}

// Modal data types
export type LossModalData = { targetClass: number; predictions: number[]; loss: number };

export interface ModalState {
  loss: UseModalReturn<LossModalData>;
  backprop: UseModalReturn<BackpropSummaryData>;
  comparison: UseModalReturn<WeightComparisonData>;
}

// =============================================================================
// Setters and Actions
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
  setLearningRate: (v: number) => void;
  setOutput: (v: number[] | null) => void;
  setSteps: (v: ForwardSteps | null) => void;
}

export interface TrainingSetters {
  setIsTraining: (v: boolean) => void;
  setAnimationSpeed: (v: number) => void;
}

export interface VisualizerSetters {
  setActivations: (v: ActivationData | null) => void;
}

export interface VisualizerActions {
}

// Modal setters and actions are now part of the modal state itself
// Keeping these interfaces for backward compatibility during transition
export interface ModalSetters {
  setLossModalData: (v: LossModalData | null) => void;
  setBackpropSummaryData: (v: BackpropSummaryData | null) => void;
  setWeightComparisonData: (v: WeightComparisonData | null) => void;
}

export interface ModalActions {
  // No longer needed - actions are part of modal state
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
  
  // Setters
  inputSetters      : InputSetters;
  statsSetters      : StatsSetters;
  trainingSetters   : TrainingSetters;
  visualizerSetters : VisualizerSetters;
  modalSetters      : ModalSetters;
  
  // Actions
  visualizerActions : VisualizerActions;
  modalActions      : ModalActions;
  
  // Reset
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
  const [learningRate, setLearningRate] = useState(0.1);
  const [output, setOutput] = useState<number[] | null>(null);
  const [steps, setSteps] = useState<ForwardSteps | null>(null);

  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [animationSpeed, setAnimationSpeed] = useState(1.0);

  // Visualizer state
  const [activations, setActivations] = useState<ActivationData | null>(null);

  // Modal state - now using generic useModal hook
  const lossModal = useModal<LossModalData>();
  const backpropModal = useModal<BackpropSummaryData>();
  const comparisonModal = useModal<WeightComparisonData>();

  // Memoize setters to prevent recreating objects every render
  const inputSetters = useMemo(() => ({
    setGrade,
    setAttitude,
    setResponse,
    setTargetValue,
  }), []); // eslint-disable-line react-hooks/exhaustive-deps

  const statsSetters = useMemo(() => ({
    setEpoch,
    setLoss,
    setLearningRate,
    setOutput,
    setSteps,
  }), []); // eslint-disable-line react-hooks/exhaustive-deps

  const trainingSetters = useMemo(() => ({
    setIsTraining,
    setAnimationSpeed,
  }), []); // eslint-disable-line react-hooks/exhaustive-deps

  const visualizerSetters = useMemo(() => ({
    setActivations,
  }), []); // eslint-disable-line react-hooks/exhaustive-deps

  const modalSetters = useMemo(() => ({
    setLossModalData: (data: LossModalData | null) => data ? lossModal.open(data) : lossModal.close(),
    setBackpropSummaryData: (data: BackpropSummaryData | null) => data ? backpropModal.open(data) : backpropModal.close(),
    setWeightComparisonData: (data: WeightComparisonData | null) => comparisonModal.setData(data),
  }), [lossModal.open, lossModal.close, backpropModal.open, backpropModal.close, comparisonModal.setData]);

  // Reset all state to initial values
  const resetAllState = useCallback(() => {
    // Stats
    setEpoch(0);
    setLoss(0);
    setOutput(null);
    
    // Modals
    lossModal.close();
    backpropModal.close();
    comparisonModal.close();
    
    // Inputs - randomize
    setGrade(Math.random());
    setAttitude(Math.random());
    setResponse(Math.random());
    setTargetValue(Math.floor(Math.random() * 3));
  }, [lossModal.close, backpropModal.close, comparisonModal.close]);

  return {
    // Grouped state
    inputs: {
      grade,
      attitude,
      response,
      targetValue,
    },

    stats: {
      epoch,
      loss,
      learningRate,
      output,
      steps,
    },

    training: {
      isTraining,
      animationSpeed,
    },

    visualizer: {
      activations,
    },

    modals: {
      loss: lossModal,
      backprop: backpropModal,
      comparison: comparisonModal,
    },

    // Setters - now memoized
    inputSetters,
    statsSetters,
    trainingSetters,
    visualizerSetters,
    modalSetters,

    // Actions
    visualizerActions: {
    },

    modalActions: {
      // No longer needed - use modals directly
    },
    
    // Reset
    resetAllState,
  };
}
