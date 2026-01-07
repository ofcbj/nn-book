/**
 * Network State Hook
 * 
 * Manages all state variables for the neural network visualizer.
 * State is organized by topic for better clarity and maintainability.
 */

import { useState, useCallback } from 'react';
import type { ForwardSteps, BackpropSummaryData, WeightComparisonData } from '../lib/types';
import type { ActivationData } from '../components/ActivationHeatmap';

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
  isManualMode: boolean;
}

export interface VisualizerState {
  showCanvasHeatmap: boolean;
  showGridHeatmap: boolean;
  activations: ActivationData | null;
}

export interface ModalState {
  loss: {
    show: boolean;
    data: { targetClass: number; predictions: number[]; loss: number } | null;
  };
  backprop: {
    show: boolean;
    data: BackpropSummaryData | null;
  };
  comparison: {
    show: boolean;
    data: WeightComparisonData | null;
  };
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
  setIsManualMode: (v: boolean) => void;
}

export interface VisualizerSetters {
  setShowCanvasHeatmap: (v: boolean) => void;
  setShowGridHeatmap: (v: boolean) => void;
  setActivations: (v: ActivationData | null) => void;
}

export interface VisualizerActions {
  toggleCanvasHeatmap: () => void;
  toggleGridHeatmap: () => void;
}

export interface ModalSetters {
  setLossModalData: (v: { targetClass: number; predictions: number[]; loss: number } | null) => void;
  setBackpropSummaryData: (v: BackpropSummaryData | null) => void;
  setShowComparisonModal: (v: boolean) => void;
  setWeightComparisonData: (v: WeightComparisonData | null) => void;
}

export interface ModalActions {
  openComparisonModal: () => void;
  closeComparisonModal: () => void;
}

// =============================================================================
// Return Type
// =============================================================================

export interface UseNetworkStateReturn {
  inputs: InputState;
  stats: NetworkStats;
  training: TrainingConfig;
  visualizer: VisualizerState;
  modals: ModalState;
  
  // Setters
  inputSetters: InputSetters;
  statsSetters: StatsSetters;
  trainingSetters: TrainingSetters;
  visualizerSetters: VisualizerSetters;
  modalSetters: ModalSetters;
  
  // Actions
  visualizerActions: VisualizerActions;
  modalActions: ModalActions;
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
  const [isManualMode, setIsManualMode] = useState(false);

  // Visualizer state
  const [showCanvasHeatmap, setShowCanvasHeatmap] = useState(false);
  const [showGridHeatmap, setShowGridHeatmap] = useState(true);
  const [activations, setActivations] = useState<ActivationData | null>(null);

  // Modal state
  const [lossModalData, setLossModalData] = useState<{ targetClass: number; predictions: number[]; loss: number } | null>(null);
  const [backpropSummaryData, setBackpropSummaryData] = useState<BackpropSummaryData | null>(null);
  const [showComparisonModal, setShowComparisonModal] = useState(false);
  const [weightComparisonData, setWeightComparisonData] = useState<WeightComparisonData | null>(null);

  // Visualizer actions
  const toggleCanvasHeatmap = useCallback(() => {
    setShowCanvasHeatmap(prev => !prev);
  }, []);

  const toggleGridHeatmap = useCallback(() => {
    setShowGridHeatmap(prev => !prev);
  }, []);

  // Modal actions
  const openComparisonModal = useCallback(() => {
    setShowComparisonModal(true);
  }, []);

  const closeComparisonModal = useCallback(() => {
    setShowComparisonModal(false);
  }, []);

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
      isManualMode,
    },
    
    visualizer: {
      showCanvasHeatmap,
      showGridHeatmap,
      activations,
    },
    
    modals: {
      loss: {
        show: lossModalData !== null,
        data: lossModalData,
      },
      backprop: {
        show: backpropSummaryData !== null,
        data: backpropSummaryData,
      },
      comparison: {
        show: showComparisonModal,
        data: weightComparisonData,
      },
    },

    // Setters
    inputSetters: {
      setGrade,
      setAttitude,
      setResponse,
      setTargetValue,
    },
    
    statsSetters: {
      setEpoch,
      setLoss,
      setLearningRate,
      setOutput,
      setSteps,
    },
    
    trainingSetters: {
      setIsTraining,
      setAnimationSpeed,
      setIsManualMode,
    },
    
    visualizerSetters: {
      setShowCanvasHeatmap,
      setShowGridHeatmap,
      setActivations,
    },
    
    modalSetters: {
      setLossModalData,
      setBackpropSummaryData,
      setShowComparisonModal,
      setWeightComparisonData,
    },

    // Actions
    visualizerActions: {
      toggleCanvasHeatmap,
      toggleGridHeatmap,
    },
    
    modalActions: {
      openComparisonModal,
      closeComparisonModal,
    },
  };
}
