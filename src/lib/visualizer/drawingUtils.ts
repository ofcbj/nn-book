// Drawing utility functions for visualizer
import type { NodePosition } from '../types';
import i18n from '../../i18n';

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
  const width = 140;
  const height = 100;
  const centerX = x - width / 2;
  const centerY = y - height / 2;

  drawRoundedRect(ctx, centerX, centerY, width, height, 15);

  const gradient = ctx.createLinearGradient(centerX, centerY, centerX, centerY + height);
  gradient.addColorStop(0, 'rgba(59, 130, 246, 0.3)');
  gradient.addColorStop(1, 'rgba(37, 99, 235, 0.2)');
  ctx.fillStyle = gradient;
  ctx.fill();

  ctx.strokeStyle = '#3b82f6';
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

type LayerType = 'input' | 'layer1' | 'layer2' | 'output';

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
  isBackpropHighlighted: boolean = false
): NodePosition {
  const baseWidth = weights.length * 25;
  let width = Math.max(130, baseWidth + 40);
  if (layerType === 'layer1') {
    width += 30;
  } else if (layerType === 'layer2') {
    width += 60;
  } else if (layerType === 'output') {
    width += 20;
  }
  const height = 90;
  const centerX = x - width / 2;
  const centerY = y - height / 2;

  drawRoundedRect(ctx, centerX, centerY, width, height, 12);

  let gradient: CanvasGradient;
  let strokeColor: string;

  if (layerType === 'layer1') {
    gradient = ctx.createLinearGradient(centerX, centerY, centerX, centerY + height);
    if (isHighlighted) {
      gradient.addColorStop(0, 'rgba(34, 197, 94, 0.9)');
      gradient.addColorStop(1, 'rgba(22, 163, 74, 0.7)');
    } else {
      gradient.addColorStop(0, 'rgba(34, 197, 94, 0.3)');
      gradient.addColorStop(1, 'rgba(22, 163, 74, 0.2)');
    }
    ctx.fillStyle = gradient;
    strokeColor = isHighlighted ? '#4ade80' : '#22c55e';
  } else if (layerType === 'layer2') {
    gradient = ctx.createLinearGradient(centerX, centerY, centerX, centerY + height);
    if (isHighlighted) {
      gradient.addColorStop(0, 'rgba(249, 115, 22, 0.9)');
      gradient.addColorStop(1, 'rgba(234, 88, 12, 0.7)');
    } else {
      gradient.addColorStop(0, 'rgba(249, 115, 22, 0.3)');
      gradient.addColorStop(1, 'rgba(234, 88, 12, 0.2)');
    }
    ctx.fillStyle = gradient;
    strokeColor = isHighlighted ? '#fb923c' : '#f97316';
  } else {
    gradient = ctx.createLinearGradient(centerX, centerY, centerX, centerY + height);
    if (isHighlighted) {
      gradient.addColorStop(0, 'rgba(239, 68, 68, 0.9)');
      gradient.addColorStop(1, 'rgba(220, 38, 38, 0.7)');
    } else {
      gradient.addColorStop(0, 'rgba(239, 68, 68, 0.3)');
      gradient.addColorStop(1, 'rgba(220, 38, 38, 0.2)');
    }
    ctx.fillStyle = gradient;
    strokeColor = isHighlighted ? '#f87171' : '#ef4444';
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
  ctx.fillText('→ ' + activation.toFixed(3), centerX + 70, centerY + 68);

  return { x: centerX, y: centerY, width, height, centerX: x, centerY: y };
}
