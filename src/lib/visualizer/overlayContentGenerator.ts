/**
 * Overlay Content Generator
 *
 * Generates text content for calculation popups.
 * Separates content generation from rendering for better maintainability.
 */

import type { ForwardStage, ForwardCalculation, BackwardStage, BackwardCalculation } from '../types';
import type { LayerName } from '../core';
import { DEFAULT_LEARNING_RATE } from '../core';
import i18n from '../../i18n';

// ============================================================================
// Common Types
// ============================================================================

export interface OverlayContent {
  title: string;
  color: string;
  lines: string[];
}

// ============================================================================
// Forward Propagation Content
// ============================================================================

export function generateForwardContent(
  stage: ForwardStage,
  data: ForwardCalculation,
  layer: LayerName
): OverlayContent {
  switch (stage) {
    case 'connections':
      return {
        title: i18n.t('calculation.connections'),
        color: '#60a5fa',
        lines: [i18n.t('calculation.connectionsCalc')]
      };

    case 'dotProduct': {
      const terms = data.inputs.map((input, i) =>
        `${input.toFixed(2)}×${data.weights[i].toFixed(2)}`
      );
      const result = ` = ${data.dotProduct.toFixed(3)}`;
      // Split into two lines if too many terms
      const lines = terms.length > 3
        ? (() => {
            const mid = Math.ceil(terms.length / 2);
            return [terms.slice(0, mid).join(' + ') + ' +', terms.slice(mid).join(' + ') + result];
          })()
        : [terms.join(' + ') + result];
      return {
        title: i18n.t('calculation.dotProduct'),
        color: '#a5b4fc',
        lines
      };
    }

    case 'bias':
      return {
        title: i18n.t('calculation.bias'),
        color: '#fbbf24',
        lines: [`${data.dotProduct.toFixed(3)} + ${data.bias.toFixed(2)} = ${data.withBias.toFixed(3)}`]
      };

    case 'activation':
      if (layer === 'output') {
        // Softmax depends on all logits of the layer, not just this neuron's
        return {
          title: i18n.t('calculation.activation'),
          color: '#34d399',
          lines: [
            `${i18n.t('calculation.activationSoftmax')}(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)`,
            `zᵢ = ${data.withBias.toFixed(3)} → ${data.activated.toFixed(3)} (${(data.activated * 100).toFixed(1)}%)`
          ]
        };
      }
      return {
        title: i18n.t('calculation.activation'),
        color: '#34d399',
        lines: [`${i18n.t('calculation.activationSigmoid')}(${data.withBias.toFixed(3)}) = ${data.activated.toFixed(3)}`]
      };

    default:
      return { title: '', color: '', lines: [] };
  }
}

// ============================================================================
// Backward Propagation Content
// ============================================================================

/** Error content for output layer neurons */
function generateOutputErrorContent(data: BackwardCalculation): string[] {
  const prediction = data.activation;
  const target = data.error + prediction;
  return [
    i18n.t('backprop.outputError'),
    '',
    `error = target - prediction`,
    `      = ${target.toFixed(4)} - ${prediction.toFixed(4)}`,
    `      = ${data.error.toFixed(4)}`,
    '',
    target > prediction ? i18n.t('backprop.needIncreaseWeight') : i18n.t('backprop.needDecreaseWeight')
  ];
}

/** Neuron label based on layer */
function getNeuronLabel(layer: LayerName, index: number): string {
  if (layer === 'layer2') return `${i18n.t('layers.layer2Prefix')}#${index + 1}`;
  if (layer === 'layer1') return `${i18n.t('layers.layer1Prefix')}#${index + 1}`;
  return `neuron[${index}]`;
}

/** Labels of the neurons in the layer after `currentLayer` */
function getNextLayerLabels(currentLayer: LayerName, count: number): string[] {
  if (currentLayer === 'layer2') {
    return [i18n.t('classes.fail'), i18n.t('classes.pending'), i18n.t('classes.pass')];
  }
  if (currentLayer === 'layer1') {
    return Array.from({ length: count }, (_, i) => `${i18n.t('layers.layer2Prefix')}#${i + 1}`);
  }
  return Array.from({ length: count }, (_, i) => `next[${i}]`);
}

/** Error content for hidden layer neurons: error = Σ δ_next × w */
function generateHiddenErrorContent(data: BackwardCalculation, currentLayer: LayerName): string[] {
  const { nextLayerDeltas, nextLayerWeights } = data;
  if (!nextLayerDeltas || !nextLayerWeights) {
    return [
      i18n.t('backprop.hiddenError'),
      '',
      i18n.t('backprop.nextLayerPropagation'),
      i18n.t('backprop.thisNeuron'),
      '',
      `error = Σ(δ_next × w)`,
      `      = ${data.error.toFixed(4)}`,
      '',
      i18n.t('backprop.neuronResponsibility')
    ];
  }

  const currentNeuronLabel = getNeuronLabel(currentLayer, data.neuronIndex);
  const nextLayerLabels = getNextLayerLabels(currentLayer, nextLayerDeltas.length);

  const content = [
    i18n.t('backprop.hiddenError'),
    '',
    `${currentNeuronLabel}: ${i18n.t('backprop.nextLayerErrors')}`,
    ''
  ];

  const contributions: string[] = [];
  nextLayerDeltas.forEach((nextDelta, idx) => {
    const weight = nextLayerWeights[idx];
    const term = nextDelta * weight;
    const nextLabel = nextLayerLabels[idx];

    content.push(`${nextLabel}: δ=${nextDelta.toFixed(4)}`);
    content.push(`  × W[${currentNeuronLabel}→${nextLabel}]=${weight.toFixed(4)}`);
    content.push(`  = ${term.toFixed(4)}`);
    content.push('');
    contributions.push(term.toFixed(4));
  });

  content.push(i18n.t('backprop.sumAll'));
  content.push(`error = ${contributions.join(' + ')}`);
  content.push(`      = ${data.error.toFixed(4)}`);

  return content;
}

function generateErrorContent(data: BackwardCalculation, currentLayer: LayerName): string[] {
  return currentLayer === 'output'
    ? generateOutputErrorContent(data)
    : generateHiddenErrorContent(data, currentLayer);
}

/** δ = error × f'(y). Softmax + cross-entropy folds the derivative into the error. */
function generateDerivativeContent(data: BackwardCalculation, currentLayer: LayerName): string[] {
  if (currentLayer === 'output') {
    return [
      'Softmax + Cross-Entropy',
      i18n.t('backprop.softmaxDerivativeNote'),
      `∂L/∂z = prediction - target = -error`,
      i18n.t('backprop.softmaxDerivativeResult'),
      `δ = error × 1 = ${data.gradient.toFixed(4)}`
    ];
  }
  const y = data.activation;
  return [
    `σ'(y) = y × (1 - y)`,
    `σ'(${y.toFixed(3)}) = ${y.toFixed(3)} × (1 - ${y.toFixed(3)})`,
    `= ${data.derivative.toFixed(4)}`
  ];
}

export function generateBackpropContent(
  stage: BackwardStage,
  data: BackwardCalculation,
  currentLayer: LayerName,
  learningRate: number = DEFAULT_LEARNING_RATE
): OverlayContent {
  const deriv = data.derivative;
  const mostChangedIdx = data.weightDeltas.reduce((max, d, i) =>
    Math.abs(d) > Math.abs(data.weightDeltas[max]) ? i : max, 0
  );
  const inputVal = data.inputs[mostChangedIdx];
  const weightDelta = data.weightDeltas[mostChangedIdx];

  switch (stage) {
    case 'error':
      return {
        title: i18n.t('backprop.error'),
        color: '#fca5a5',
        lines: generateErrorContent(data, currentLayer)
      };

    case 'derivative':
      return {
        title: i18n.t('backprop.delta'),
        color: '#a5b4fc',
        lines: generateDerivativeContent(data, currentLayer)
      };

    case 'gradient':
      return {
        title: i18n.t('backprop.gradient'),
        color: '#60a5fa',
        lines: [
          `δ = error × f'(y)`,
          `= ${data.error.toFixed(4)} × ${deriv.toFixed(4)}`,
          `= ${data.gradient.toFixed(4)}`
        ]
      };

    case 'weightDelta':
      return {
        title: i18n.t('backprop.weightDelta'),
        color: '#fbbf24',
        lines: [
          i18n.t('backprop.weightDeltaCalc'),
          `ΔW = δ × input × ${i18n.t('controls.learningRate')}(${learningRate})`,
          ``,
          `${i18n.t('backprop.example')} input[${mostChangedIdx}]${i18n.t('backprop.connectedWeight')}:`,
          `ΔW[${mostChangedIdx}] = ${data.gradient.toFixed(4)} × ${inputVal.toFixed(3)} × ${learningRate} = ${weightDelta.toFixed(5)}`
        ]
      };

    case 'allWeightDeltas': {
      const lines = [
        i18n.t('backprop.allWeightChanges'),
        ''
      ];

      data.inputs.forEach((input, i) => {
        lines.push(`W[${i}] = ${data.oldWeights[i].toFixed(4)}  → ΔW[${i}] = η × δ × x[${i}]`);
        lines.push(`     = ${learningRate} × ${data.gradient.toFixed(4)} × ${input.toFixed(3)} = ${data.weightDeltas[i].toFixed(5)}`);
      });

      lines.push('');
      lines.push(`b = ${data.oldBias.toFixed(4)}  → Δb = ${learningRate} × ${data.gradient.toFixed(4)} = ${data.biasDelta.toFixed(5)}`);

      return {
        title: i18n.t('backprop.allWeightDeltas'),
        color: '#fcd34d',
        lines
      };
    }

    case 'update': {
      const biasChange = data.newBias - data.oldBias;
      const lines = [
        i18n.t('backprop.allWeightUpdate'),
        ''
      ];

      data.oldWeights.forEach((oldW, i) => {
        const delta = data.weightDeltas[i];
        lines.push(`W[${i}]: ${oldW.toFixed(4)} ${delta > 0 ? '↑' : '↓'} ${data.newWeights[i].toFixed(4)} (Δ${delta.toFixed(5)})`);
      });

      lines.push('');
      lines.push(`Bias: ${data.oldBias.toFixed(4)} ${biasChange > 0 ? '↑' : '↓'} ${data.newBias.toFixed(4)} (Δ${biasChange.toFixed(5)})`);

      return {
        title: i18n.t('backprop.update'),
        color: '#34d399',
        lines
      };
    }

    default:
      return { title: '', color: '', lines: [] };
  }
}
