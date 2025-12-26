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

export interface UseNeuralNetworkReturn {
  // Neural network
  nn: NeuralNetwork;
  visualizer: Visualizer | null;
  setVisualizer: (v: Visualizer) => void;

  // Input values
  grade: number;
  attitude: number;
  response: number;
  targetValue: number;
  learningRate: number;
  animationSpeed: number;
  isManualMode: boolean;

  // Setters
  setGrade: (v: number) => void;
  setAttitude: (v: number) => void;
  setResponse: (v: number) => void;
  setTargetValue: (v: number) => void;
  setLearningRate: (v: number) => void;
  setAnimationSpeed: (v: number) => void;
  setIsManualMode: (v: boolean) => void;
  nextStep: () => void;

  // Stats
  epoch: number;
  loss: number;
  output: number[] | null;
  steps: CalculationSteps | null;

  // Training state
  isTraining: boolean;
  isAnimating: boolean;

  // Loss modal
  showLossModal: boolean;
  lossModalData: { targetClass: number; predictions: number[]; loss: number } | null;

  // Backprop summary modal
  showBackpropModal: boolean;
  backpropSummaryData: BackpropSummaryData | null;

  // Heatmap visualization
  showCanvasHeatmap: boolean;
  showGridHeatmap: boolean;
  activations: ActivationData | null;
  toggleCanvasHeatmap: () => void;
  toggleGridHeatmap: () => void;

  // Weight comparison
  showComparisonModal: boolean;
  weightComparisonData: WeightComparisonData | null;
  openComparisonModal: () => void;
  closeComparisonModal: () => void;

  // Actions
  trainOneStepWithAnimation: () => Promise<void>;
  toggleTraining: () => void;
  reset: () => void;
  closeLossModal: () => Promise<void>;
  closeBackpropModal: () => void;
  updateVisualization: () => void;
  handleCanvasClick: (x?: number, y?: number) => void;
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
    // Neural network
    nn: nnRef.current,
    visualizer: visualizerRef.current,
    setVisualizer: trainingControls.setVisualizer,

    // Input values
    grade: state.grade,
    attitude: state.attitude,
    response: state.response,
    targetValue: state.targetValue,
    learningRate: state.learningRate,
    animationSpeed: state.animationSpeed,
    isManualMode: state.isManualMode,

    // Setters
    setGrade: state.setGrade,
    setAttitude: state.setAttitude,
    setResponse: state.setResponse,
    setTargetValue: state.setTargetValue,
    setLearningRate: handleLearningRateChange,
    setAnimationSpeed: state.setAnimationSpeed,
    setIsManualMode: state.setIsManualMode,
    nextStep: trainingControls.nextStep,

    // Stats
    epoch: state.epoch,
    loss: state.loss,
    output: state.output,
    steps: state.steps,

    // Training state
    isTraining: state.isTraining,
    isAnimating: animationMachine.isAnimating,

    // Loss modal
    showLossModal: animationMachine.state.type === 'showing_loss_modal',
    lossModalData: state.lossModalData,

    // Backprop summary modal
    showBackpropModal: animationMachine.state.type === 'showing_backprop_modal',
    backpropSummaryData: state.backpropSummaryData,

    // Heatmap visualization
    showCanvasHeatmap: state.showCanvasHeatmap,
    showGridHeatmap: state.showGridHeatmap,
    activations: state.activations,
    toggleCanvasHeatmap: trainingControls.toggleCanvasHeatmap,
    toggleGridHeatmap: state.toggleGridHeatmap,

    // Weight comparison
    showComparisonModal: state.showComparisonModal,
    weightComparisonData: state.weightComparisonData,
    openComparisonModal: state.openComparisonModal,
    closeComparisonModal: state.closeComparisonModal,

    // Actions
    trainOneStepWithAnimation: trainingControls.trainOneStepWithAnimation,
    toggleTraining: trainingControls.toggleTraining,
    reset: trainingControls.reset,
    closeLossModal: trainingControls.closeLossModal,
    closeBackpropModal: trainingControls.closeBackpropModal,
    updateVisualization: animation.updateVisualization,
    handleCanvasClick: canvasInteraction.handleCanvasClick,
  };
}
