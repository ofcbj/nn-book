import { describe, it, expect } from 'vitest';
import { NeuralNetwork } from '../network';
import { Matrix } from '../matrix';
import { crossEntropyLoss } from '../activations';
import { createSnapshot, type NetworkSnapshot } from '../networkSnapshot';
import { LAYER_NAMES, LAYER_SIZES, INPUT_SIZE, toOneHot, type LayerName } from '../networkConfig';

const INPUTS = [0.7, 0.5, 0.8];
const TARGET = toOneHot(2);

function layerMatrices(nn: NeuralNetwork): Record<LayerName, { weights: Matrix; bias: Matrix }> {
  return {
    layer1: { weights: nn.weightsInputHidden1,   bias: nn.biasHidden1 },
    layer2: { weights: nn.weightsHidden1Hidden2, bias: nn.biasHidden2 },
    output: { weights: nn.weightsHidden2Output,  bias: nn.biasOutput },
  };
}

function loadSnapshot(nn: NeuralNetwork, snapshot: NetworkSnapshot): void {
  const m = layerMatrices(nn);
  for (const layer of LAYER_NAMES) {
    m[layer].weights.data = Matrix.cloneRows(snapshot.weights[layer]);
    m[layer].bias.data = Matrix.cloneRows(snapshot.biases[layer]);
  }
}

function lossOf(nn: NeuralNetwork, inputs: number[], target: number[]): number {
  return crossEntropyLoss(nn.feedforward(inputs), target);
}

describe('NeuralNetwork feedforward', () => {
  it('has the configured shapes', () => {
    const nn = new NeuralNetwork();
    expect(nn.weightsInputHidden1.rows).toBe(LAYER_SIZES.layer1);
    expect(nn.weightsInputHidden1.cols).toBe(INPUT_SIZE);
    expect(nn.weightsHidden2Output.rows).toBe(LAYER_SIZES.output);
    expect(nn.weightsHidden2Output.cols).toBe(LAYER_SIZES.layer2);
  });

  it('returns a probability distribution over the output classes', () => {
    const out = new NeuralNetwork().feedforward(INPUTS);
    expect(out).toHaveLength(LAYER_SIZES.output);
    expect(out.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    out.forEach(p => {
      expect(p).toBeGreaterThan(0);
      expect(p).toBeLessThan(1);
    });
  });

  it('getForwardSteps returns copies, not live weight rows', () => {
    const nn = new NeuralNetwork();
    nn.feedforward(INPUTS);
    const steps = nn.getForwardSteps()!;
    const before = steps.layer1[0].weights[0];
    nn.weightsInputHidden1.data[0][0] = before + 1;
    expect(steps.layer1[0].weights[0]).toBe(before);
  });
});

describe('NeuralNetwork training', () => {
  it('records the cross-entropy loss of the pre-update prediction', () => {
    const nn = new NeuralNetwork();
    const expected = lossOf(nn, INPUTS, TARGET);
    nn.train(INPUTS, TARGET);
    expect(nn.lastLoss).toBeCloseTo(expected, 12);
  });

  it('applies ΔW = −lr · ∂L/∂W for every weight and bias (numerical gradient check)', () => {
    const lr = 0.1;
    const nn = new NeuralNetwork(lr);
    const before = createSnapshot(nn);

    nn.train(INPUTS, TARGET);
    const after = createSnapshot(nn);

    // Independent copy of the pre-update network for finite differences
    const probe = new NeuralNetwork(lr);
    loadSnapshot(probe, before);
    const probeMatrices = layerMatrices(probe);
    const h = 1e-6;

    const numericalGradient = (m: Matrix, i: number, j: number): number => {
      const original = m.data[i][j];
      m.data[i][j] = original + h;
      const plus = lossOf(probe, INPUTS, TARGET);
      m.data[i][j] = original - h;
      const minus = lossOf(probe, INPUTS, TARGET);
      m.data[i][j] = original;
      return (plus - minus) / (2 * h);
    };

    for (const layer of LAYER_NAMES) {
      const { weights, bias } = probeMatrices[layer];
      for (let i = 0; i < weights.rows; i++) {
        for (let j = 0; j < weights.cols; j++) {
          const delta = after.weights[layer][i][j] - before.weights[layer][i][j];
          expect(delta).toBeCloseTo(-lr * numericalGradient(weights, i, j), 7);
        }
        const biasDelta = after.biases[layer][i][0] - before.biases[layer][i][0];
        expect(biasDelta).toBeCloseTo(-lr * numericalGradient(bias, i, 0), 7);
      }
    }
  });

  it('reduces the loss on the training sample', () => {
    const nn = new NeuralNetwork(0.25);
    const initial = lossOf(nn, INPUTS, TARGET);
    for (let i = 0; i < 20; i++) nn.train(INPUTS, TARGET);
    expect(lossOf(nn, INPUTS, TARGET)).toBeLessThan(initial);
  });
});

describe('NeuralNetwork backward steps (visualizer data)', () => {
  const lr = 0.2;
  const nn = new NeuralNetwork(lr);
  const before = createSnapshot(nn);
  nn.train(INPUTS, TARGET);
  const steps = nn.lastBackwardSteps!;

  it('output layer uses softmax + cross-entropy: derivative 1, δ = target − output', () => {
    steps.output.forEach((n, i) => {
      expect(n.error).toBeCloseTo(TARGET[i] - steps.predictions[i], 12);
      expect(n.derivative).toBe(1);
      expect(n.gradient).toBeCloseTo(n.error, 12);
    });
    expect(steps.targetClass).toBe(2);
    expect(steps.totalLoss).toBeCloseTo(nn.lastLoss, 12);
  });

  it('hidden layers use δ = error × σ\'(y) and error = Σ δ_next × w (pre-update weights)', () => {
    for (const layer of ['layer1', 'layer2'] as const) {
      const nextLayer: LayerName = layer === 'layer1' ? 'layer2' : 'output';
      steps[layer].forEach((n, i) => {
        expect(n.derivative).toBeCloseTo(n.activation * (1 - n.activation), 12);
        expect(n.gradient).toBeCloseTo(n.error * n.derivative, 12);

        const nextDeltas = n.nextLayerDeltas!;
        const nextWeights = n.nextLayerWeights!;
        expect(nextDeltas).toEqual(steps[nextLayer].map(m => m.gradient));
        nextWeights.forEach((w, j) => expect(w).toBe(before.weights[nextLayer][j][i]));
        const expectedError = nextDeltas.reduce((sum, d, j) => sum + d * nextWeights[j], 0);
        expect(n.error).toBeCloseTo(expectedError, 12);
      });
    }
  });

  it('weight deltas are lr × δ × input and connect old to new weights', () => {
    for (const layer of LAYER_NAMES) {
      steps[layer].forEach((n, i) => {
        n.weightDeltas.forEach((d, j) => {
          expect(d).toBeCloseTo(lr * n.gradient * n.inputs[j], 12);
          expect(n.oldWeights[j]).toBe(before.weights[layer][i][j]);
          expect(n.newWeights[j]).toBeCloseTo(n.oldWeights[j] + d, 12);
        });
        expect(n.biasDelta).toBeCloseTo(lr * n.gradient, 12);
        expect(n.newBias).toBeCloseTo(n.oldBias + n.biasDelta, 12);
      });
    }
  });

  it('stored arrays are copies that survive later training steps', () => {
    const storedNew = [...steps.output[0].newWeights];
    nn.train(INPUTS, TARGET);
    expect(steps.output[0].newWeights).toEqual(storedNew);
  });
});
