/**
 * Animation Module
 *
 * - Animation state machine and reducer
 * - Animation loop runner and stage durations
 */

export {
  animationReducer,
  initialAnimationState,
  checkAnimating,
  checkPaused,
  checkMode,
  getAnimatingNeuron,
  isAnimatingAtNeuron,
  getNextForwardStage,
  getNextBackwardStage,
  getNextForwardNeuron,
  getNextBackwardNeuron,
  FORWARD_STAGES,
  BACKPROP_STAGES,
} from './animationState';
export type {
  AnimationState,
  AnimationAction,
  AnimationMode,
  ForwardAnimatingState,
  BackwardAnimatingState,
  InterruptReason,
} from './animationState';

export {
  runAnimationLoop,
  FORWARD_STAGE_DURATIONS,
  BACKWARD_STAGE_DURATIONS,
} from './animationLoop';
export type { AnimationLoopConfig, AnimationStage, NeuronData } from './animationLoop';
