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
