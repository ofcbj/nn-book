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
  isAnimating,
  isPaused,
  getNextForwardStage,
  getNextBackwardStage,
  getNextForwardNeuron,
  getNextBackwardNeuron,
  FORWARD_STAGES,
  BACKPROP_STAGES,
  getHighlightedNeuron,
  getForwardStage,
  getBackwardStage,
  getCurrentNeuronData,
  getCurrentBackpropData,
} from './animationState';
export type { 
  AnimationState, 
  AnimationAction,
  ForwardAnimatingState,
  BackwardAnimatingState,
} from './animationState';

export {
  runAnimationLoop,
} from './animationLoop';
export type { AnimationLoopConfig } from './animationLoop';
