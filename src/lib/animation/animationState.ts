/**
 * Animation State Machine for Neural Network Visualizer
 *
 * This module defines a finite state machine (FSM) to manage animation states
 * in a predictable manner. All state transitions are explicit.
 */

import type { ForwardStage, BackwardStage, ForwardCalculation, BackwardCalculation } from '../types';
import { LAYER_SIZES, FORWARD_LAYER_ORDER, BACKWARD_LAYER_ORDER, FORWARD_STAGES, BACKPROP_STAGES, type LayerName } from '../core';

// ============================================================================
// State Definitions
// ============================================================================

export type AnimationMode = 'forward' | 'backward';

/** Interrupt reason: none = running, paused = user paused, jumped = user clicked another neuron */
export type InterruptReason = 'none' | 'paused' | 'jumped';

interface BaseAnimationState {
  interruptReason: InterruptReason;
}

/** Idle - No animation running */
interface IdleState extends BaseAnimationState {
  type: 'idle';
}

/** Forward propagation animation */
export interface ForwardAnimatingState extends BaseAnimationState {
  type        : 'forward_animating';
  layer       : LayerName;
  neuronIndex : number;
  stage       : ForwardStage;
  neuronData  : ForwardCalculation | null;
}

/** Showing loss modal after forward propagation */
interface ShowingLossModalState extends BaseAnimationState {
  type: 'showing_loss_modal';
}

/** Backward propagation animation */
export interface BackwardAnimatingState extends BaseAnimationState {
  type        : 'backward_animating';
  layer       : LayerName;
  neuronIndex : number;
  stage       : BackwardStage;
  neuronData  : BackwardCalculation | null;
}

/** Showing backprop summary modal */
interface ShowingBackpropModalState extends BaseAnimationState {
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
  | { type: 'PAUSE' }
  | { type: 'RESUME' }
  | { type: 'JUMP_TO_NEURON'; layer: LayerName; neuronIndex: number }
  | { type: 'FORWARD_TICK'; layer: LayerName; neuronIndex: number; stage: ForwardStage; neuronData: ForwardCalculation | null }
  | { type: 'FORWARD_COMPLETE' }
  | { type: 'CLOSE_LOSS_MODAL' }
  | { type: 'BACKWARD_TICK'; layer: LayerName; neuronIndex: number; stage: BackwardStage; neuronData: BackwardCalculation | null }
  | { type: 'BACKWARD_COMPLETE' }
  | { type: 'CLOSE_BACKPROP_MODAL' }
  | { type: 'RESET' };

// ============================================================================
// Initial State
// ============================================================================

export const initialAnimationState: AnimationState = {
  type: 'idle',
  interruptReason: 'none',
};

const FIRST_FORWARD_LAYER  = FORWARD_LAYER_ORDER[0];
const FIRST_BACKWARD_LAYER = BACKWARD_LAYER_ORDER[0];

// ============================================================================
// State Reducer
// ============================================================================

export function animationReducer(state: AnimationState, action: AnimationAction): AnimationState {
  switch (action.type) {
    // --- Global actions (work in any state) ---
    case 'PAUSE':
      return { ...state, interruptReason: 'paused' };

    case 'RESUME':
      return { ...state, interruptReason: 'none' };

    case 'RESET':
      return initialAnimationState;

    // --- Idle -> Forward ---
    case 'START_TRAINING':
      if (state.type !== 'idle') return state;
      return {
        type            : 'forward_animating',
        layer           : FIRST_FORWARD_LAYER,
        neuronIndex     : 0,
        stage           : FORWARD_STAGES[0],
        neuronData      : null,
        interruptReason : 'none',
      };

    // --- Forward animation ---
    case 'FORWARD_TICK':
      if (state.type !== 'forward_animating') return state;
      return {
        ...state,
        layer       : action.layer,
        neuronIndex : action.neuronIndex,
        stage       : action.stage,
        neuronData  : action.neuronData,
      };

    case 'JUMP_TO_NEURON':
      if (state.type === 'forward_animating') {
        return {
          ...state,
          layer           : action.layer,
          neuronIndex     : action.neuronIndex,
          stage           : FORWARD_STAGES[0],
          interruptReason : 'jumped',
        };
      }
      if (state.type === 'backward_animating') {
        return {
          ...state,
          layer           : action.layer,
          neuronIndex     : action.neuronIndex,
          stage           : BACKPROP_STAGES[0],
          interruptReason : 'jumped',
        };
      }
      return state;

    case 'FORWARD_COMPLETE':
      if (state.type !== 'forward_animating') return state;
      return { type: 'showing_loss_modal', interruptReason: 'none' };

    // --- Loss modal -> Backward ---
    case 'CLOSE_LOSS_MODAL':
      if (state.type !== 'showing_loss_modal') return state;
      return {
        type            : 'backward_animating',
        layer           : FIRST_BACKWARD_LAYER,
        neuronIndex     : LAYER_SIZES[FIRST_BACKWARD_LAYER] - 1, // Start from the last output neuron
        stage           : BACKPROP_STAGES[0],
        neuronData      : null,
        interruptReason : 'none',
      };

    // --- Backward animation ---
    case 'BACKWARD_TICK':
      if (state.type !== 'backward_animating') return state;
      return {
        ...state,
        layer       : action.layer,
        neuronIndex : action.neuronIndex,
        stage       : action.stage,
        neuronData  : action.neuronData,
      };

    case 'BACKWARD_COMPLETE':
      if (state.type !== 'backward_animating') return state;
      return { type: 'showing_backprop_modal', interruptReason: 'none' };

    // --- Backprop modal -> Idle ---
    case 'CLOSE_BACKPROP_MODAL':
      if (state.type !== 'showing_backprop_modal') return state;
      return { type: 'idle', interruptReason: 'none' };

    default:
      return state;
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

/** True while a forward or backward animation is running (not idle, not showing a modal) */
export function checkAnimating(state: AnimationState): state is ForwardAnimatingState | BackwardAnimatingState {
  return state.type === 'forward_animating' || state.type === 'backward_animating';
}

/** True when an animation is interrupted (user pause or neuron jump) */
export function checkPaused(state: AnimationState): boolean {
  return checkAnimating(state) && state.interruptReason !== 'none';
}

/** True when the state belongs to the given propagation phase (animation or its modal) */
export function checkMode(state: AnimationState, mode: AnimationMode): boolean {
  if (mode === 'forward') {
    return state.type === 'forward_animating' || state.type === 'showing_loss_modal';
  }
  return state.type === 'backward_animating' || state.type === 'showing_backprop_modal';
}

/** Neuron currently being animated, if any */
export function getAnimatingNeuron(state: AnimationState): { layer: LayerName; index: number } | null {
  if (checkAnimating(state)) {
    return { layer: state.layer, index: state.neuronIndex };
  }
  return null;
}

/** True when the animation is currently at the given neuron */
export function isAnimatingAtNeuron(state: AnimationState, layer: LayerName, neuronIndex: number): boolean {
  return checkAnimating(state) && state.layer === layer && state.neuronIndex === neuronIndex;
}

// Re-exports from core/networkConfig for convenience
export {
  FORWARD_STAGES,
  BACKPROP_STAGES,
  getNextForwardStage,
  getNextBackwardStage,
  getNextForwardNeuron,
  getNextBackwardNeuron,
} from '../core';
