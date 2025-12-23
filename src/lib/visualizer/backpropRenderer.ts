// Backpropagation visualization renderer
import type { NodePosition, BackpropNeuronData, BackpropStage, AnimationPhase } from '../types';
import i18n from '../../i18n';

// Helper: Find the node to highlight
function findNodeToHighlight(
  layer: string,
  index: number,
  nodes: NodePosition[][]
): NodePosition | null {
  if (layer === 'layer1' && nodes[1]) {
    return nodes[1][index];
  } else if (layer === 'layer2' && nodes[2]) {
    return nodes[2][index];
  } else if (layer === 'output' && nodes[3]) {
    return nodes[3][index];
  }
  return null;
}

// Helper: Draw error glow around node
function drawErrorGlow(
  ctx: CanvasRenderingContext2D,
  nodeInfo: NodePosition,
  errorMagnitude: number
): void {
  ctx.save();
  const glowSize = Math.min(errorMagnitude * 100 + 20, 60);

  const gradient = ctx.createRadialGradient(
    nodeInfo.centerX, nodeInfo.centerY, 0,
    nodeInfo.centerX, nodeInfo.centerY, nodeInfo.width / 2 + glowSize
  );
  gradient.addColorStop(0, `rgba(239, 68, 68, ${errorMagnitude * 0.8})`);
  gradient.addColorStop(1, 'rgba(239, 68, 68, 0)');

  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(nodeInfo.centerX, nodeInfo.centerY, nodeInfo.width / 2 + glowSize, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// Helper: Generate content for error stage
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
        i18n.t('backprop.nextLayerErrors'),
        ''
      ];
      
      data.nextLayerErrors.forEach((nextError, idx) => {
        const weight = data.nextLayerWeights![idx];
        const term = nextError * weight;
        content.push(
          `${nextLayerLabels[idx]}: ${nextError.toFixed(4)} × ${weight.toFixed(4)} = ${term.toFixed(4)}`
        );
        content.push(
          `          ${i18n.t('backprop.neuronError')}  ${i18n.t('backprop.connectionWeight')}  ${i18n.t('backprop.contribution')}`
        );
      });
      
      content.push('');
      content.push(i18n.t('backprop.sumAll'));
      content.push(`error = ${data.nextLayerErrors.map((e, i) => 
        `${(e * data.nextLayerWeights![i]).toFixed(4)}`
      ).join(' + ')}`);
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

// Helper: Generate content based on stage
function generateStageContent(
  stage: BackpropStage,
  data: BackpropNeuronData,
  currentLayer: string
): { title: string; color: string; content: string[] } {
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
        content: generateErrorContent(data, currentLayer)
      };

    case 'derivative':
      return {
        title: i18n.t('backprop.delta'),
        color: '#a5b4fc',
        content: [
          `σ'(y) = y × (1 - y)`,
          `σ'(${y.toFixed(3)}) = ${y.toFixed(3)} × (1 - ${y.toFixed(3)})`,
          `= ${deriv.toFixed(4)}`
        ]
      };

    case 'gradient':
      return {
        title: i18n.t('backprop.gradient'),
        color: '#60a5fa',
        content: [
          `gradient = error × σ'(y)`,
          `= ${data.error.toFixed(4)} × ${deriv.toFixed(4)}`,
          `= ${data.gradient.toFixed(4)}`
        ]
      };

    case 'weightDelta':
      return {
        title: i18n.t('backprop.weightDelta'),
        color: '#fbbf24',
        content: [
          i18n.t('backprop.weightDeltaCalc'),
          `ΔW = gradient × input × ${i18n.t('controls.learningRate')}(0.1)`,
          ``,
          `${i18n.t('backprop.example')} input[${mostChangedIdx}]${i18n.t('backprop.connectedWeight')}:`,
          `ΔW[${mostChangedIdx}] = ${data.gradient.toFixed(4)} × ${inputVal.toFixed(3)} × 0.1 = ${weightDelta.toFixed(5)}`
        ]
      };

    case 'allWeightDeltas':
      const content = [
        i18n.t('backprop.allWeightChanges'),
        ''
      ];
      
      data.inputs.forEach((inputVal, i) => {
        const delta = data.weightDeltas[i];
        const oldWeight = data.oldWeights[i];
        content.push(
          `W[${i}] = ${oldWeight.toFixed(4)}  →  ΔW[${i}] = η × δ × x[${i}]`
        );
        content.push(
          `     = 0.1 × ${data.gradient.toFixed(4)} × ${inputVal.toFixed(3)} = ${delta.toFixed(5)}`
        );
      });
      
      content.push('');
      content.push(`b = ${data.oldBias.toFixed(4)}  →  Δb = 0.1 × ${data.gradient.toFixed(4)} = ${data.biasDelta.toFixed(5)}`);
      
      return {
        title: i18n.t('backprop.allWeightDeltas'),
        color: '#fcd34d',
        content
      };

    case 'update':
      const biasChange = data.newBias - data.oldBias;
      const biasArrow = biasChange > 0 ? '↗' : '↘';
      const updateContent = [
        i18n.t('backprop.allWeightUpdate'),
        ''
      ];
      
      data.oldWeights.forEach((oldW: number, i: number) => {
        const newW = data.newWeights[i];
        const delta = data.weightDeltas[i];
        const arrow = delta > 0 ? '↗' : '↘';
        updateContent.push(
          `W[${i}]: ${oldW.toFixed(4)} ${arrow} ${newW.toFixed(4)} (Δ=${delta.toFixed(5)})`
        );
      });
      
      updateContent.push('');
      updateContent.push(`Bias: ${data.oldBias.toFixed(4)} ${biasArrow} ${data.newBias.toFixed(4)} (Δ=${biasChange.toFixed(5)})`);
      
      return {
        title: i18n.t('backprop.update'),
        color: '#34d399',
        content: updateContent
      };

    default:
      return { title: '', color: '', content: [] };
  }
}

// Helper: Calculate box position within canvas bounds
function calculateBoxPosition(
  nodeInfo: NodePosition,
  boxWidth: number,
  boxHeight: number,
  canvas: HTMLCanvasElement
): { x: number; y: number } {
  let boxX = nodeInfo.centerX - boxWidth / 2;
  let boxY = nodeInfo.y - boxHeight - 20;

  const margin = 10;

  // Adjust horizontal position
  if (boxX < margin) {
    boxX = margin;
  } else if (boxX + boxWidth > canvas.width - margin) {
    boxX = canvas.width - margin - boxWidth;
  }

  // Adjust vertical position
  if (boxY < margin) {
    boxY = nodeInfo.y + nodeInfo.height + 20;
    if (boxY + boxHeight > canvas.height - margin) {
      boxY = canvas.height - margin - boxHeight;
    }
  } else if (boxY + boxHeight > canvas.height - margin) {
    boxY = canvas.height - margin - boxHeight;
  }

  return { x: boxX, y: boxY };
}

// Helper: Draw overlay box with content
function drawOverlayBox(
  ctx: CanvasRenderingContext2D,
  boxX: number,
  boxY: number,
  boxWidth: number,
  boxHeight: number,
  title: string,
  color: string,
  content: string[]
): void {
  // Background
  ctx.fillStyle = 'rgba(0, 0, 0, 0.65)';
  ctx.beginPath();
  ctx.roundRect(boxX, boxY, boxWidth, boxHeight, 8);
  ctx.fill();

  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.stroke();

  // Title
  ctx.font = 'bold 14px sans-serif';
  ctx.fillStyle = color;
  ctx.textAlign = 'center';
  const textCenterX = boxX + boxWidth / 2;
  ctx.fillText(title, textCenterX, boxY + 25);

  // Content
  ctx.font = 'bold 13px monospace';
  ctx.textAlign = 'center';
  let yOffset = boxY + 50;
  const lineHeight = 22;

  content.forEach((line, idx) => {
    if (idx === content.length - 1) {
      ctx.fillStyle = color;
      ctx.font = 'bold 14px monospace';
    } else {
      ctx.fillStyle = '#cbd5e1';
      ctx.font = '13px monospace';
    }
    ctx.fillText(line, textCenterX, yOffset);
    yOffset += lineHeight;
  });
}

// Main function
export function drawBackpropHighlight(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  nodes: NodePosition[][],
  backpropPhase: AnimationPhase | null,
  currentBackpropData: BackpropNeuronData | null,
  backpropStage: BackpropStage | null
): void {
  if (!backpropPhase) return;

  const { layer, index } = backpropPhase;
  const nodeInfo = findNodeToHighlight(layer, index, nodes);
  if (!nodeInfo) return;

  // Draw error glow
  const errorMagnitude = currentBackpropData ? Math.abs(currentBackpropData.error) : 0.5;
  drawErrorGlow(ctx, nodeInfo, errorMagnitude);

  // Draw information overlay
  if (currentBackpropData && backpropStage) {
    const { title, color, content } = generateStageContent(backpropStage, currentBackpropData, layer);
    
    const boxWidth    = 420;
    const lineHeight  = 22;
    const boxHeight   = 60 + content.length * lineHeight;
    const { x: boxX, y: boxY } = calculateBoxPosition(nodeInfo, boxWidth, boxHeight, canvas);
    drawOverlayBox(ctx, boxX, boxY, boxWidth, boxHeight, title, color, content);
  } else {
    // Fallback label
    ctx.font = 'bold 14px sans-serif';
    ctx.fillStyle = '#ef4444';
    ctx.textAlign = 'center';
    ctx.fillText('◄ BACKPROP', nodeInfo.centerX, nodeInfo.y - 35);
  }
}
