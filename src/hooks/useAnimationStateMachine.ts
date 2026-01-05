/**
 * Animation State Machine Hook
 * 
 * Provides a React hook wrapper around the animation state machine,
 * handling the animation loop and timing.
 */

import { useReducer, useCallback, useRef, useEffect } from 'react';
import {
  AnimationState,
  AnimationAction,
  animationReducer,
  initialAnimationState,
  isAnimating,
  isPaused,
  getNextForwardStage,
  getNextBackpropStage,
  getNextForwardNeuron,
  getNextBackwardNeuron,
  FORWARD_STAGES,
  BACKPROP_STAGES,
  getHighlightedNeuron,
  getForwardStage,
  getBackpropStage,
  getCurrentNeuronData,
  getCurrentBackpropData,
} from '../lib/animation';
import type { LayerName } from '../lib/core';
import type { NeuronCalculation, BackpropNeuronData, CalculationStage, BackpropStage } from '../lib/types';

export interface UseAnimationStateMachineReturn {
  // State
  state: AnimationState;
  
  // Derived state helpers
  isAnimating: boolean;
  isPaused: boolean;
  isForwardMode: boolean;
  isBackwardMode: boolean;
  highlightedNeuron: { layer: LayerName; index: number } | null;
  forwardStage: CalculationStage | null;
  backpropStage: BackpropStage | null;
  currentNeuronData: NeuronCalculation | null;
  currentBackpropData: BackpropNeuronData | null;
  
  // Actions
  dispatch: React.Dispatch<AnimationAction>;
  startTraining: () => void;
  setSpeed: (speed: number) => void;
  pause: () => void;
  resume: (speed?: number) => void;
  nextStep: () => void;
  jumpToNeuron: (layer: LayerName, neuronIndex: number) => void;
  reset: () => void;
  
  // Animation tick callbacks (called by animation loop)
  forwardTick: (
    layer: LayerName,
    neuronIndex: number,
    stage: CalculationStage,
    neuronData: NeuronCalculation | null
  ) => void;
  forwardComplete: () => void;
  closeLossModal: () => void;
  backwardTick: (
    layer: LayerName,
    neuronIndex: number,
    stage: BackpropStage,
    neuronData: BackpropNeuronData | null
  ) => void;
  backwardComplete: () => void;
  closeBackpropModal: () => void;
  
  // Manual step resolver for paused mode
  waitForNextStep: () => Promise<void>;
  resolveStep: () => void;
}

// Type alias for external use
export type AnimationStateMachine = UseAnimationStateMachineReturn;

export function useAnimationStateMachine(): UseAnimationStateMachineReturn {
  const [state, dispatch] = useReducer(animationReducer, initialAnimationState);
  
  // Resolver for manual stepping (when paused)
  const stepResolverRef = useRef<(() => void) | null>(null);
  
  // Track previous speed to detect resume from pause
  const prevSpeedRef = useRef(state.speed);
  
  // Effect to auto-resolve step when speed changes from 0 to > 0
  useEffect(() => {
    if (prevSpeedRef.current === 0 && state.speed > 0) {
      // Speed was increased from paused state, resolve pending step
      if (stepResolverRef.current) {
        stepResolverRef.current();
        stepResolverRef.current = null;
      }
    }
    prevSpeedRef.current = state.speed;
  }, [state.speed]);
  
  // Wait for next step (used when paused)
  const waitForNextStep = useCallback((): Promise<void> => {
    if (state.speed === 0) {
      return new Promise<void>(resolve => {
        stepResolverRef.current = resolve;
      });
    }
    return Promise.resolve();
  }, [state.speed]);
  
  // Resolve pending step (called on click or button)
  const resolveStep = useCallback(() => {
    if (stepResolverRef.current) {
      stepResolverRef.current();
      stepResolverRef.current = null;
    }
  }, []);
  
  // Action creators
  const startTraining = useCallback(() => {
    dispatch({ type: 'START_TRAINING' });
  }, []);
  
  const setSpeed = useCallback((speed: number) => {
    dispatch({ type: 'SET_SPEED', speed });
  }, []);
  
  const pause = useCallback(() => {
    dispatch({ type: 'PAUSE' });
  }, []);
  
  const resume = useCallback((speed?: number) => {
    dispatch({ type: 'RESUME', speed: speed ?? 1.0 });
  }, []);
  
  const nextStep = useCallback(() => {
    dispatch({ type: 'NEXT_STEP' });
    resolveStep();
  }, [resolveStep]);
  
  const jumpToNeuron = useCallback((
    layer: LayerName,
    neuronIndex: number
  ) => {
    dispatch({ type: 'JUMP_TO_NEURON', layer, neuronIndex });
  }, []);
  
  const reset = useCallback(() => {
    dispatch({ type: 'RESET' });
    stepResolverRef.current = null;
  }, []);
  
  // Forward animation tick
  const forwardTick = useCallback((
    layer: LayerName,
    neuronIndex: number,
    stage: CalculationStage,
    neuronData: NeuronCalculation | null
  ) => {
    dispatch({ type: 'FORWARD_TICK', layer, neuronIndex, stage, neuronData });
  }, []);
  
  const forwardComplete = useCallback(() => {
    dispatch({ type: 'FORWARD_COMPLETE' });
  }, []);
  
  const closeLossModal = useCallback(() => {
    dispatch({ type: 'CLOSE_LOSS_MODAL' });
  }, []);
  
  // Backward animation tick
  const backwardTick = useCallback((
    layer: LayerName,
    neuronIndex: number,
    stage: BackpropStage,
    neuronData: BackpropNeuronData | null
  ) => {
    dispatch({ type: 'BACKWARD_TICK', layer, neuronIndex, stage, neuronData });
  }, []);
  
  const backwardComplete = useCallback(() => {
    dispatch({ type: 'BACKWARD_COMPLETE' });
  }, []);
  
  const closeBackpropModal = useCallback(() => {
    dispatch({ type: 'CLOSE_BACKPROP_MODAL' });
  }, []);
  
  // Derived state
  const animatingState = isAnimating(state);
  const pausedState = isPaused(state);
  const isForwardMode = state.type === 'forward_animating' || state.type === 'showing_loss_modal';
  const isBackwardMode = state.type === 'backward_animating' || state.type === 'showing_backprop_modal';
  
  return {
    state,
    isAnimating: animatingState,
    isPaused: pausedState,
    isForwardMode,
    isBackwardMode,
    highlightedNeuron: getHighlightedNeuron(state),
    forwardStage: getForwardStage(state),
    backpropStage: getBackpropStage(state),
    currentNeuronData: getCurrentNeuronData(state),
    currentBackpropData: getCurrentBackpropData(state),
    dispatch,
    startTraining,
    setSpeed,
    pause,
    resume,
    nextStep,
    jumpToNeuron,
    reset,
    forwardTick,
    forwardComplete,
    closeLossModal,
    backwardTick,
    backwardComplete,
    closeBackpropModal,
    waitForNextStep,
    resolveStep,
  };
}

// Export helper functions and types for external use
export {
  getNextForwardStage,
  getNextBackpropStage,
  getNextForwardNeuron,
  getNextBackwardNeuron,
  FORWARD_STAGES,
  BACKPROP_STAGES,
};
export type { AnimationState };
