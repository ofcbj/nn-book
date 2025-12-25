/**
 * Network State Hook
 * 
 * Manages all state variables for the neural network visualization.
 * Extracted from useNeuralNetwork for better separation of concerns.
 */

import { useState, useCallback } from 'react';
import type { CalculationSteps, BackpropSummaryData, WeightComparisonData } from '../lib/types';
import type { ActivationData } from '../components/ActivationHeatmap';

export interface NetworkState {
  // Input values
  grade: number;
  attitude: number;
  response: number;
  targetValue: number;
  learningRate: number;
  animationSpeed: number;
  isManualMode: boolean;

  // Stats
  epoch: number;
  loss: number;
  output: number[] | null;
  steps: CalculationSteps | null;

  // Training state
  isTraining: boolean;

  // Loss modal
  lossModalData: { targetClass: number; predictions: number[]; loss: number } | null;

  // Backprop summary
  backpropSummaryData: BackpropSummaryData | null;

  // Heatmap
  showCanvasHeatmap: boolean;
  showGridHeatmap: boolean;
  activations: ActivationData | null;

  // Weight comparison
  showComparisonModal: boolean;
  weightComparisonData: WeightComparisonData | null;
}

export interface NetworkStateSetters {
  setGrade: (v: number) => void;
  setAttitude: (v: number) => void;
  setResponse: (v: number) => void;
  setTargetValue: (v: number) => void;
  setLearningRate: (v: number) => void;
  setAnimationSpeed: (v: number) => void;
  setIsManualMode: (v: boolean) => void;
  setEpoch: (v: number | ((prev: number) => number)) => void;
  setLoss: (v: number) => void;
  setOutput: (v: number[] | null) => void;
  setSteps: (v: CalculationSteps | null) => void;
  setIsTraining: (v: boolean) => void;
  setLossModalData: (v: { targetClass: number; predictions: number[]; loss: number } | null) => void;
  setBackpropSummaryData: (v: BackpropSummaryData | null) => void;
  setShowCanvasHeatmap: (v: boolean) => void;
  setShowGridHeatmap: (v: boolean) => void;
  setActivations: (v: ActivationData | null) => void;
  setShowComparisonModal: (v: boolean) => void;
  setWeightComparisonData: (v: WeightComparisonData | null) => void;
}

export interface NetworkStateActions {
  toggleCanvasHeatmap: () => void;
  toggleGridHeatmap: () => void;
  openComparisonModal: () => void;
  closeComparisonModal: () => void;
}

export interface UseNetworkStateReturn extends NetworkState, NetworkStateSetters, NetworkStateActions {}

export function useNetworkState(): UseNetworkStateReturn {
  // Input values
  const [grade, setGrade] = useState(0.7);
  const [attitude, setAttitude] = useState(0.5);
  const [response, setResponse] = useState(0.8);
  const [targetValue, setTargetValue] = useState(2);
  const [learningRate, setLearningRate] = useState(0.1);
  const [animationSpeed, setAnimationSpeed] = useState(1.0);
  const [isManualMode, setIsManualMode] = useState(false);

  // Stats
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(0);
  const [output, setOutput] = useState<number[] | null>(null);
  const [steps, setSteps] = useState<CalculationSteps | null>(null);

  // Training state
  const [isTraining, setIsTraining] = useState(false);

  // Loss modal
  const [lossModalData, setLossModalData] = useState<{ targetClass: number; predictions: number[]; loss: number } | null>(null);

  // Backprop summary
  const [backpropSummaryData, setBackpropSummaryData] = useState<BackpropSummaryData | null>(null);

  // Heatmap
  const [showCanvasHeatmap, setShowCanvasHeatmap] = useState(false);
  const [showGridHeatmap, setShowGridHeatmap] = useState(true);
  const [activations, setActivations] = useState<ActivationData | null>(null);

  // Weight comparison
  const [showComparisonModal, setShowComparisonModal] = useState(false);
  const [weightComparisonData, setWeightComparisonData] = useState<WeightComparisonData | null>(null);

  // Actions
  const toggleCanvasHeatmap = useCallback(() => {
    setShowCanvasHeatmap(prev => !prev);
  }, []);

  const toggleGridHeatmap = useCallback(() => {
    setShowGridHeatmap(prev => !prev);
  }, []);

  const openComparisonModal = useCallback(() => {
    setShowComparisonModal(true);
  }, []);

  const closeComparisonModal = useCallback(() => {
    setShowComparisonModal(false);
  }, []);

  return {
    // State
    grade,
    attitude,
    response,
    targetValue,
    learningRate,
    animationSpeed,
    isManualMode,
    epoch,
    loss,
    output,
    steps,
    isTraining,
    lossModalData,
    backpropSummaryData,
    showCanvasHeatmap,
    showGridHeatmap,
    activations,
    showComparisonModal,
    weightComparisonData,

    // Setters
    setGrade,
    setAttitude,
    setResponse,
    setTargetValue,
    setLearningRate,
    setAnimationSpeed,
    setIsManualMode,
    setEpoch,
    setLoss,
    setOutput,
    setSteps,
    setIsTraining,
    setLossModalData,
    setBackpropSummaryData,
    setShowCanvasHeatmap,
    setShowGridHeatmap,
    setActivations,
    setShowComparisonModal,
    setWeightComparisonData,

    // Actions
    toggleCanvasHeatmap,
    toggleGridHeatmap,
    openComparisonModal,
    closeComparisonModal,
  };
}
