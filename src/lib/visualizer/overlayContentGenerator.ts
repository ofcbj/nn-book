/**
 * Overlay Content Generator
 * 
 * Generates text content for calculation popups.
 * Separates content generation from rendering for better maintainability.
 */

import type { CalculationStage, NeuronCalculation, BackpropStage, BackpropNeuronData } from '../types';
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
  stage: CalculationStage,
  data: NeuronCalculation
): OverlayContent {
  switch(stage) {
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
      // Split into two lines if too many terms
      if (terms.length > 3) {
        const mid = Math.ceil(terms.length / 2);
        return {
          title: i18n.t('calculation.dotProduct'),
          color: '#a5b4fc',
          lines: [
            terms.slice(0, mid).join(' + ') + ' +',
            terms.slice(mid).join(' + ') + ` = ${data.dotProduct.toFixed(3)}`
          ]
        };
      }
      return {
        title: i18n.t('calculation.dotProduct'),
        color: '#a5b4fc',
        lines: [terms.join(' + ') + ` = ${data.dotProduct.toFixed(3)}`]
      };
    }

    case 'bias':
      return {
        title: i18n.t('calculation.bias'),
        color: '#fbbf24',
        lines: [`${data.dotProduct.toFixed(3)} + ${data.bias.toFixed(2)} = ${data.withBias.toFixed(3)}`]
      };

    case 'activation':
      return {
        title: i18n.t('calculation.activation'),
        color: '#34d399',
        lines: [`σ(${data.withBias.toFixed(3)}) = ${data.activated.toFixed(3)}`]
      };

    default:
      return { title: '', color: '', lines: [] };
  }
}

// ============================================================================
// Backward Propagation Content
// ============================================================================

function generateErrorContent(
  data: BackpropNeuronData,
  currentLayer: string
): string[] {
  if (currentLayer === 'output') {
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
  } else {
    if (data.nextLayerErrors && data.nextLayerWeights) {
      // Get current neuron label
      let currentNeuronLabel: string;
      if (currentLayer === 'layer2') {
        currentNeuronLabel = `${i18n.t('layers.layer2Prefix')}#${data.neuronIndex + 1}`;
      } else if (currentLayer === 'layer1') {
        currentNeuronLabel = `${i18n.t('layers.layer1Prefix')}#${data.neuronIndex + 1}`;
      } else {
        currentNeuronLabel = `neuron[${data.neuronIndex}]`;
      }
      
      // Get next layer neuron labels
      let nextLayerLabels: string[];
      if (currentLayer === 'layer2') {
        nextLayerLabels = [i18n.t('classes.fail'), i18n.t('classes.pending'), i18n.t('classes.pass')];
      } else if (currentLayer === 'layer1') {
        nextLayerLabels = data.nextLayerErrors.map((_, i) => `${i18n.t('layers.layer2Prefix')}#${i+1}`);
      } else {
        nextLayerLabels = data.nextLayerErrors.map((_, i) => `next[${i}]`);
      }
      
      const content = [
        i18n.t('backprop.hiddenError'),
        '',
        `${currentNeuronLabel}: ${i18n.t('backprop.nextLayerErrors')}`,
        ''
      ];
      
      // Show each connection with clear notation
      data.nextLayerErrors.forEach((nextError, idx) => {
        const weight = data.nextLayerWeights![idx];
        const term = nextError * weight;
        const nextLabel = nextLayerLabels[idx];
        
        // Show: "nextNeuron의 오류 × (현재→next 가중치) = 기여도"
        content.push(
          `${nextLabel}: error=${nextError.toFixed(4)}`
        );
        content.push(
          `  × W[${currentNeuronLabel}→${nextLabel}]=${weight.toFixed(4)}`
        );
        content.push(
          `  = ${term.toFixed(4)}`
        );
        content.push('');
      });
      
      content.push(i18n.t('backprop.sumAll'));
      const contributions = data.nextLayerErrors.map((e, i) => 
        (e * data.nextLayerWeights![i]).toFixed(4)
      );
      content.push(`error = ${contributions.join(' + ')}`);
      content.push(`      = ${data.error.toFixed(4)}`);
      
      return content;
    } else {
      return [
        i18n.t('backprop.hiddenError'),
        '',
        i18n.t('backprop.nextLayerPropagation'),
        i18n.t('backprop.thisNeuron'),
        '',
        `error = Σ(next_error × next_weight)`,
        `      = ${data.error.toFixed(4)}`,
        '',
        i18n.t('backprop.neuronResponsibility')
      ];
    }
  }
}

export function generateBackpropContent(
  stage: BackpropStage,
  data: BackpropNeuronData,
  currentLayer: string
): OverlayContent {
  const y = data.activation;
  const deriv = data.derivative;
  const mostChangedIdx = data.weightDeltas.reduce((max, d, i) =>
    Math.abs(d) > Math.abs(data.weightDeltas[max]) ? i : max, 0
  );
  const inputVal = data.inputs[mostChangedIdx];
  const weightDelta = data.weightDeltas[mostChangedIdx];

  switch(stage) {
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
        lines: [
          `σ'(y) = y × (1 - y)`,
          `σ'(${y.toFixed(3)}) = ${y.toFixed(3)} × (1 - ${y.toFixed(3)})`,
          `= ${deriv.toFixed(4)}`
        ]
      };

    case 'gradient':
      return {
        title: i18n.t('backprop.gradient'),
        color: '#60a5fa',
        lines: [
          `gradient = error × σ'(y)`,
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
          `ΔW = gradient × input × ${i18n.t('controls.learningRate')}(0.1)`,
          ``,
          `${i18n.t('backprop.example')} input[${mostChangedIdx}]${i18n.t('backprop.connectedWeight')}:`,
          `ΔW[${mostChangedIdx}] = ${data.gradient.toFixed(4)} × ${inputVal.toFixed(3)} × 0.1 = ${weightDelta.toFixed(5)}`
        ]
      };

    case 'allWeightDeltas': {
      const lines = [
        i18n.t('backprop.allWeightChanges'),
        ''
      ];
      
      data.inputs.forEach((inputVal, i) => {
        const delta = data.weightDeltas[i];
        const oldWeight = data.oldWeights[i];
        lines.push(
          `W[${i}] = ${oldWeight.toFixed(4)}  →  ΔW[${i}] = η × δ × x[${i}]`
        );
        lines.push(
          `     = 0.1 × ${data.gradient.toFixed(4)} × ${inputVal.toFixed(3)} = ${delta.toFixed(5)}`
        );
      });
      
      lines.push('');
      lines.push(`b = ${data.oldBias.toFixed(4)}  →  Δb = 0.1 × ${data.gradient.toFixed(4)} = ${data.biasDelta.toFixed(5)}`);
      
      return {
        title: i18n.t('backprop.allWeightDeltas'),
        color: '#fcd34d',
        lines
      };
    }

    case 'update': {
      const biasChange = data.newBias - data.oldBias;
      const biasArrow = biasChange > 0 ? '↗' : '↘';
      const lines = [
        i18n.t('backprop.allWeightUpdate'),
        ''
      ];
      
      data.oldWeights.forEach((oldW: number, i: number) => {
        const newW = data.newWeights[i];
        const delta = data.weightDeltas[i];
        const arrow = delta > 0 ? '↗' : '↘';
        lines.push(
          `W[${i}]: ${oldW.toFixed(4)} ${arrow} ${newW.toFixed(4)} (Δ=${delta.toFixed(5)})`
        );
      });
      
      lines.push('');
      lines.push(`Bias: ${data.oldBias.toFixed(4)} ${biasArrow} ${data.newBias.toFixed(4)} (Δ=${biasChange.toFixed(5)})`);
      
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
