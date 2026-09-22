import { describe, it, expect } from 'vitest';
import { animationReducer, initialAnimationState, checkAnimating, checkPaused, checkMode } from '../animationState';
import type { AnimationState } from '../animationState';
import { LAYER_SIZES } from '../../core';

function run(...actions: Parameters<typeof animationReducer>[1][]): AnimationState {
  return actions.reduce(animationReducer, initialAnimationState);
}

describe('animationReducer', () => {
  it('follows idle -> forward -> loss modal -> backward -> summary -> idle', () => {
    const forward = run({ type: 'START_TRAINING' });
    expect(forward.type).toBe('forward_animating');
    expect(checkAnimating(forward)).toBe(true);

    const lossModal = animationReducer(forward, { type: 'FORWARD_COMPLETE' });
    expect(lossModal.type).toBe('showing_loss_modal');
    expect(checkMode(lossModal, 'forward')).toBe(true);
    expect(checkAnimating(lossModal)).toBe(false);

    const backward = animationReducer(lossModal, { type: 'CLOSE_LOSS_MODAL' });
    expect(backward).toMatchObject({ type: 'backward_animating', layer: 'output', neuronIndex: LAYER_SIZES.output - 1, stage: 'error' });

    const summary = animationReducer(backward, { type: 'BACKWARD_COMPLETE' });
    expect(summary.type).toBe('showing_backprop_modal');
    expect(checkMode(summary, 'backward')).toBe(true);

    expect(animationReducer(summary, { type: 'CLOSE_BACKPROP_MODAL' })).toEqual(initialAnimationState);
  });

  it('ignores transitions that do not apply to the current state', () => {
    expect(run({ type: 'FORWARD_COMPLETE' })).toEqual(initialAnimationState);
    expect(run({ type: 'CLOSE_LOSS_MODAL' })).toEqual(initialAnimationState);
    const forward = run({ type: 'START_TRAINING' });
    expect(animationReducer(forward, { type: 'START_TRAINING' })).toBe(forward);
  });

  it('pause / jump set the interrupt reason and resume clears it', () => {
    const paused = run({ type: 'START_TRAINING' }, { type: 'PAUSE' });
    expect(checkPaused(paused)).toBe(true);

    const jumped = run({ type: 'START_TRAINING' }, { type: 'JUMP_TO_NEURON', layer: 'layer2', neuronIndex: 1 });
    expect(jumped).toMatchObject({ layer: 'layer2', neuronIndex: 1, stage: 'connections', interruptReason: 'jumped' });

    const resumed = animationReducer(jumped, { type: 'RESUME' });
    expect(checkPaused(resumed)).toBe(false);
  });

  it('RESET returns to idle from any state', () => {
    expect(run({ type: 'START_TRAINING' }, { type: 'FORWARD_COMPLETE' }, { type: 'RESET' })).toEqual(initialAnimationState);
  });
});
