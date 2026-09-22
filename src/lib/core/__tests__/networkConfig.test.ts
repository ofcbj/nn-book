import { describe, it, expect } from 'vitest';
import {
  LAYER_SIZES, OUTPUT_CLASSES, toOneHot,
  getNextForwardNeuron, getNextBackwardNeuron,
  getForwardNeuronIndices, getBackwardNeuronIndices,
  getNextForwardStage, getNextBackwardStage,
  FORWARD_LAYER_ORDER, BACKWARD_LAYER_ORDER,
} from '../networkConfig';

describe('networkConfig', () => {
  it('layer orders are reverses of each other', () => {
    expect([...BACKWARD_LAYER_ORDER].reverse()).toEqual([...FORWARD_LAYER_ORDER]);
  });

  it('toOneHot encodes the class index', () => {
    expect(toOneHot(0)).toEqual([1, 0, 0]);
    expect(toOneHot(OUTPUT_CLASSES - 1)).toHaveLength(OUTPUT_CLASSES);
  });

  it('neuron indices cover each layer in the right direction', () => {
    expect(getForwardNeuronIndices('layer1')).toEqual([0, 1, 2, 3, 4]);
    expect(getBackwardNeuronIndices('output')).toEqual([2, 1, 0]);
  });

  it('forward navigation walks layer1 -> layer2 -> output -> null', () => {
    expect(getNextForwardNeuron('layer1', 0)).toEqual({ layer: 'layer1', index: 1 });
    expect(getNextForwardNeuron('layer1', LAYER_SIZES.layer1 - 1)).toEqual({ layer: 'layer2', index: 0 });
    expect(getNextForwardNeuron('layer2', LAYER_SIZES.layer2 - 1)).toEqual({ layer: 'output', index: 0 });
    expect(getNextForwardNeuron('output', LAYER_SIZES.output - 1)).toBeNull();
  });

  it('backward navigation walks output -> layer2 -> layer1 -> null', () => {
    expect(getNextBackwardNeuron('output', 1)).toEqual({ layer: 'output', index: 0 });
    expect(getNextBackwardNeuron('output', 0)).toEqual({ layer: 'layer2', index: LAYER_SIZES.layer2 - 1 });
    expect(getNextBackwardNeuron('layer2', 0)).toEqual({ layer: 'layer1', index: LAYER_SIZES.layer1 - 1 });
    expect(getNextBackwardNeuron('layer1', 0)).toBeNull();
  });

  it('stage navigation ends with null', () => {
    expect(getNextForwardStage('connections')).toBe('dotProduct');
    expect(getNextForwardStage('activation')).toBeNull();
    expect(getNextBackwardStage('error')).toBe('derivative');
    expect(getNextBackwardStage('update')).toBeNull();
  });
});
