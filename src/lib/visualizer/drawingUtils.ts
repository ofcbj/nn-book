// Drawing utility functions for visualizer
import type { NodePosition, LayerType } from '../types';
import type { LayerName } from '../core';
import i18n from '../../i18n';
import { activationToColor } from './activationColors';
import {
  INPUT_BOX,
  NEURON_BOX,
  LAYER_COLORS,
} from './uiConfig';

export function drawRoundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number
): void {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

export function drawInputVector(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  values: number[],
  inputLabels: string[]
): NodePosition {
  const { width, height, cornerRadius } = INPUT_BOX;
  const centerX = x - width / 2;
  const centerY = y - height / 2;

  drawRoundedRect(ctx, centerX, centerY, width, height, cornerRadius);

  const colors = LAYER_COLORS.input;
  const gradient = ctx.createLinearGradient(centerX, centerY, centerX, centerY + height);
  gradient.addColorStop(0, colors.gradientStart);
  gradient.addColorStop(1, colors.gradientEnd);
  ctx.fillStyle = gradient;
  ctx.fill();

  ctx.strokeStyle = colors.stroke;
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 14px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(i18n.t('layers.input'), x, centerY + 20);

  ctx.font = '12px monospace';
  ctx.textAlign = 'left';
  const startY = centerY + 40;
  values.forEach((val, idx) => {
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText(inputLabels[idx] + ':', centerX + 15, startY + idx * 18);
    ctx.fillStyle = '#60a5fa';
    ctx.font = 'bold 12px monospace';
    ctx.fillText(val.toFixed(2), centerX + 60, startY + idx * 18);
    ctx.font = '12px monospace';
  });

  return { x: centerX, y: centerY, width, height, centerX: x, centerY: y };
}


export function drawNeuronVector(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  weights: number[],
  bias: number,
  activation: number,
  label: string,
  layerType: LayerType,
  isHighlighted: boolean = false,
  isBackpropHighlighted: boolean = false,
  heatmapMode: boolean = false
): NodePosition {
  // Calculate width based on weights and layer type
  const baseWidth = weights.length * NEURON_BOX.weightMultiplier;
  let width = Math.max(NEURON_BOX.minWidth, baseWidth + 40);
  
  // Add extra width based on layer type
  if (layerType === 'layer1') {
    width += NEURON_BOX.extraWidth.layer1;
  } else if (layerType === 'layer2') {
    width += NEURON_BOX.extraWidth.layer2;
  } else if (layerType === 'output') {
    width += NEURON_BOX.extraWidth.output;
  }
  
  // Get height based on layer type
  const height = layerType === 'layer1' 
    ? NEURON_BOX.height.layer1 
    : (layerType === 'layer2' ? NEURON_BOX.height.layer2 : NEURON_BOX.height.output);
  
  const centerX = x - width / 2;
  const centerY = y - height / 2;

  drawRoundedRect(ctx, centerX, centerY, width, height, NEURON_BOX.cornerRadius);

  // Get colors from config based on layer type
  // Note: drawNeuronVector is only called for layer1, layer2, output (not input)
  const colors = LAYER_COLORS[layerType as LayerName];
  
  let gradient: CanvasGradient;
  let strokeColor: string;

  gradient = ctx.createLinearGradient(centerX, centerY, centerX, centerY + height);
  if (isHighlighted) {
    gradient.addColorStop(0, colors.highlightGradientStart);
    gradient.addColorStop(1, colors.highlightGradientEnd);
    strokeColor = colors.highlightStroke;
  } else {
    gradient.addColorStop(0, colors.gradientStart);
    gradient.addColorStop(1, colors.gradientEnd);
    strokeColor = colors.stroke;
  }
  ctx.fillStyle = gradient;

  // Use heatmap color if in heatmap mode (unless highlighted)
  if (heatmapMode && !isHighlighted && !isBackpropHighlighted) {
    const heatColor = activationToColor(activation);
    gradient = ctx.createLinearGradient(centerX, centerY, centerX, centerY + height);
    // Convert rgb to rgba with different alphas for gradient
    const rgbMatch = heatColor.match(/\d+/g);
    if (rgbMatch) {
      const [r, g, b] = rgbMatch.map(Number);
      gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0.8)`);
      gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0.5)`);
    }
    ctx.fillStyle = gradient;
    strokeColor = heatColor;
  }

  ctx.fill();

  // Backprop highlight takes priority over regular highlight
  if (isBackpropHighlighted) {
    ctx.strokeStyle = '#a855f7'; // Purple for backprop
    ctx.lineWidth = 5;
  } else {
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = isHighlighted ? 4 : 2;
  }
  ctx.stroke();

  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 11px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(label, x, centerY + 16);

  ctx.font = '12px monospace';
  ctx.fillStyle = '#cbd5e1';
  ctx.textAlign = 'left';
  ctx.fillText('W:', centerX + 8, centerY + 35);

  ctx.fillStyle = '#a5b4fc';
  const vectorStr = '[' + weights.map(w => w.toFixed(2)).join(', ') + ']';
  const textWidth = ctx.measureText(vectorStr).width;
  if (textWidth > width - 40) {
    ctx.font = '11px monospace';
  }
  ctx.fillText(vectorStr, centerX + 24, centerY + 35);

  ctx.font = '12px monospace';
  ctx.fillStyle = '#cbd5e1';
  ctx.fillText('b:', centerX + 8, centerY + 50);
  ctx.fillStyle = '#fbbf24';
  ctx.fillText(bias.toFixed(2), centerX + 24, centerY + 50);

  ctx.fillStyle = '#34d399';
  ctx.font = 'bold 12px monospace';
  ctx.fillText('ↁE' + activation.toFixed(3), centerX + 70, centerY + 68);

  return { x: centerX, y: centerY, width, height, centerX: x, centerY: y };
}
