/**
 * Animation State Machine for Neural Network Visualizer
 * 
 * This module defines a finite state machine (FSM) to manage animation states
 * in a predictable and bug-free manner. All state transitions are explicit.
 */

import type { ForwardStage, BackpropStage, NeuronCalculation, BackpropNeuronData } from '../types';
import { type LayerName } from '../core';

// ============================================================================
// State Definitions
// ============================================================================

export type AnimationMode = 'forward' | 'backward';

/** Base state shared by all animation states */
interface BaseAnimationState {
  /** Speed multiplier (0 = paused) */
  speed: number;
  /** Whether user jumped to a specific neuron */
  isJumped: boolean;
}

/** Idle - No animation running */
export interface IdleState extends BaseAnimationState {
  type: 'idle';
}

/** Forward propagation animation */
export interface ForwardAnimatingState extends BaseAnimationState {
  type: 'forward_animating';
  layer: LayerName;
  neuronIndex: number;
  stage: ForwardStage;
  neuronData: NeuronCalculation | null;
}

/** Showing loss modal after forward propagation */
export interface ShowingLossModalState extends BaseAnimationState {
  type: 'showing_loss_modal';
}

/** Backward propagation animation */
export interface BackwardAnimatingState extends BaseAnimationState {
  type: 'backward_animating';
  layer: LayerName;
  neuronIndex: number;
  stage: BackpropStage;
  neuronData: BackpropNeuronData | null;
}

/** Showing backprop summary modal */
export interface ShowingBackpropModalState extends BaseAnimationState {
  type: 'showing_backprop_modal';
}

export type AnimationState =
  | IdleState
  | ForwardAnimatingState
  | ShowingLossModalState
  | BackwardAnimatingState
  | ShowingBackpropModalState;

// ============================================================================
// Action Definitions
// ============================================================================

export type AnimationAction =
  | { type: 'START_TRAINING' }
  | { type: 'SET_SPEED'; speed: number }
  | { type: 'PAUSE' }
  | { type: 'RESUME'; speed: number }
  | { type: 'NEXT_STEP' }
  | { 
      type: 'JUMP_TO_NEURON'; 
      layer: LayerName; 
      neuronIndex: number;
    }
  | { 
      type: 'FORWARD_TICK'; 
      layer: LayerName; 
      neuronIndex: number; 
      stage: ForwardStage;
      neuronData: NeuronCalculation | null;
    }
  | { type: 'FORWARD_COMPLETE' }
  | { type: 'CLOSE_LOSS_MODAL' }
  | { 
      type: 'BACKWARD_TICK'; 
      layer: LayerName; 
      neuronIndex: number; 
      stage: BackpropStage;
      neuronData: BackpropNeuronData | null;
    }
  | { type: 'BACKWARD_COMPLETE' }
  | { type: 'CLOSE_BACKPROP_MODAL' }
  | { type: 'RESET' };

// ============================================================================
// Initial State
// ============================================================================

export const initialAnimationState: AnimationState = {
  type: 'idle',
  speed: 1.0,
  isJumped: false,
};

// ============================================================================
// State Reducer
// ============================================================================

export function animationReducer(
  state: AnimationState,
  action: AnimationAction
): AnimationState {
  switch (action.type) {
    // -------------------------------------------------------------------------
    // Global Actions (work in any state)
    // -------------------------------------------------------------------------
    case 'SET_SPEED':
      return { ...state, speed: action.speed };

    case 'PAUSE':
      return { ...state, speed: 0 };

    case 'RESUME':
      return { ...state, speed: action.speed > 0 ? action.speed : 1.0 };

    case 'RESET':
      return initialAnimationState;

    // -------------------------------------------------------------------------
    // Idle State Transitions
    // -------------------------------------------------------------------------
    case 'START_TRAINING':
      if (state.type !== 'idle') return state;
      return {
        type: 'forward_animating',
        layer: 'layer1',
        neuronIndex: 0,
        stage: 'connections',
        neuronData: null,
        speed: state.speed,
        isJumped: false,
      };

    // -------------------------------------------------------------------------
    // Forward Animation Transitions
    // -------------------------------------------------------------------------
    case 'FORWARD_TICK':
      if (state.type !== 'forward_animating') return state;
      return {
        ...state,
        layer: action.layer,
        neuronIndex: action.neuronIndex,
        stage: action.stage,
        neuronData: action.neuronData,
      };

    case 'JUMP_TO_NEURON':
      // Can jump during forward or backward animation
      if (state.type === 'forward_animating') {
        return {
          ...state,
          layer: action.layer,
          neuronIndex: action.neuronIndex,
          stage: 'connections', // Reset to first stage
          speed: 0, // Pause when jumping
          isJumped: true,
        };
      }
      if (state.type === 'backward_animating') {
        return {
          ...state,
          layer: action.layer,
          neuronIndex: action.neuronIndex,
          stage: 'error', // Reset to first backprop stage
          speed: 0,
          isJumped: true,
        };
      }
      return state;

    case 'FORWARD_COMPLETE':
      if (state.type !== 'forward_animating') return state;
      return {
        type: 'showing_loss_modal',
        speed: state.speed,
        isJumped: false,
      };

    // -------------------------------------------------------------------------
    // Loss Modal Transitions
    // -------------------------------------------------------------------------
    case 'CLOSE_LOSS_MODAL':
      if (state.type !== 'showing_loss_modal') return state;
      return {
        type: 'backward_animating',
        layer: 'output',
        neuronIndex: 2, // Start from last output neuron
        stage: 'error',
        neuronData: null,
        speed: state.speed > 0 ? state.speed : 1.0, // Ensure non-zero speed
        isJumped: false,
      };

    // -------------------------------------------------------------------------
    // Backward Animation Transitions
    // -------------------------------------------------------------------------
    case 'BACKWARD_TICK':
      if (state.type !== 'backward_animating') return state;
      return {
        ...state,
        layer: action.layer,
        neuronIndex: action.neuronIndex,
        stage: action.stage,
        neuronData: action.neuronData,
      };

    case 'BACKWARD_COMPLETE':
      if (state.type !== 'backward_animating') return state;
      return {
        type: 'showing_backprop_modal',
        speed: state.speed,
        isJumped: false,
      };

    // -------------------------------------------------------------------------
    // Backprop Modal Transitions
    // -------------------------------------------------------------------------
    case 'CLOSE_BACKPROP_MODAL':
      if (state.type !== 'showing_backprop_modal') return state;
      return {
        type: 'idle',
        speed: state.speed,
        isJumped: false,
      };

    case 'NEXT_STEP':
      // This is handled by the animation loop, not the reducer
      // It's a signal to advance to the next stage/neuron
      return state;

    default:
      return state;
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

/** Check if animation is currently running (not idle and not showing modal) */
export function isAnimating(state: AnimationState): boolean {
  return state.type === 'forward_animating' || state.type === 'backward_animating';
}

/** Check if animation is paused (speed is 0 and animating) */
export function isPaused(state: AnimationState): boolean {
  return isAnimating(state) && state.speed === 0;
}

/** Check if we're in forward propagation mode */
export function isForwardMode(state: AnimationState): boolean {
  return state.type === 'forward_animating' || state.type === 'showing_loss_modal';
}

/** Check if we're in backward propagation mode */
export function isBackwardMode(state: AnimationState): boolean {
  return state.type === 'backward_animating' || state.type === 'showing_backprop_modal';
}

/** Get current highlighted neuron info for visualizer */
export function getHighlightedNeuron(state: AnimationState): {
  layer: LayerName;
  index: number;
} | null {
  if (state.type === 'forward_animating' || state.type === 'backward_animating') {
    return { layer: state.layer, index: state.neuronIndex };
  }
  return null;
}

/** Get forward propagation stage for visualizer */
export function getForwardStage(state: AnimationState): ForwardStage | null {
  if (state.type === 'forward_animating') {
    return state.stage;
  }
  return null;
}

/** Get backward propagation stage for visualizer */
export function getBackpropStage(state: AnimationState): BackpropStage | null {
  if (state.type === 'backward_animating') {
    return state.stage;
  }
  return null;
}

/** Get current neuron data for calculation overlay */
export function getCurrentNeuronData(state: AnimationState): NeuronCalculation | null {
  if (state.type === 'forward_animating') {
    return state.neuronData;
  }
  return null;
}

/** Get current backprop data for backprop overlay */
export function getCurrentBackpropData(state: AnimationState): BackpropNeuronData | null {
  if (state.type === 'backward_animating') {
    return state.neuronData;
  }
  return null;
}

// ============================================================================
// Re-exports from core/networkConfig
// (Centralized configuration for layers, stages, and navigation)
// ============================================================================

export {
  FORWARD_STAGES,
  BACKPROP_STAGES,
  getNextForwardStage,
  getNextBackpropStage,
  getNextForwardNeuron,
  getNextBackwardNeuron,
} from '../core';
