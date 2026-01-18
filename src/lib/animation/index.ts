/**
 * Animation Module
 * 
 * Contains animation state management and loop utilities:
 * - Animation state machine and reducer
 * - Animation loop runner
 */

export {
  animationReducer,
  initialAnimationState,
  checkAnimating,
  checkPaused,
  checkForwardMode,
  checkBackwardMode,
  getNextForwardStage,
  getNextBackwardStage,
  getNextForwardNeuron,
  getNextBackwardNeuron,
  FORWARD_STAGES,
  BACKPROP_STAGES,
  getHighlightedNeuron,
  getForwardStage,
  getBackwardStage,
  getCurrentForwardNeuronData,
  getCurrentBackwardNeuronData,
  isForwardAnimatingAtNeuron,
  isBackwardAnimatingAtNeuron,
  isAnimatingAtNeuron,
} from './animationState';
export type { 
  AnimationState, 
  AnimationAction,
  ForwardAnimatingState,
  BackwardAnimatingState,
  InterruptReason,
} from './animationState';

export {
  runAnimationLoop,
} from './animationLoop';
export type { AnimationLoopConfig } from './animationLoop';
