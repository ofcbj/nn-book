// Drawing utility functions for visualizer
import type { NodePosition, LayerType } from '../types';
import type { LayerName } from '../core';
import {
  INPUT_BOX,
  NEURON_BOX,
  LAYER_COLORS,
} from './uiConfig';

// =============================================================================
// Backprop Update Data
// =============================================================================

/**
 * Data for displaying value changes during backpropagation.
 * Contains both old and new values for weights and bias.
 */
export interface BackpropUpdateData {
  newWeights: number[];
  newBias: number;
}

// =============================================================================
// Text Rendering Helpers
// =============================================================================

interface LabelStyle {
  bgColor: string;
  textColor: string;
  font: string;
  padding: { x: number; y: number };
  borderRadius: number;
}

const DEFAULT_LABEL_STYLE: LabelStyle = {
  bgColor: 'rgba(0, 0, 0, 0.85)',
  textColor: '#ffffff',
  font: 'bold 10px monospace',
  padding: { x: 4, y: 8 },
  borderRadius: 3,
};

/**
 * Draw text with rounded rectangle background.
 * Consolidates repeated label rendering patterns.
 */
export function drawTextWithBackground(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  style: Partial<LabelStyle> = {}
): void {
  const s = { ...DEFAULT_LABEL_STYLE, ...style };

  ctx.font = s.font;
  const textWidth = ctx.measureText(text).width;

  // Background
  ctx.fillStyle = s.bgColor;
  ctx.beginPath();
  ctx.roundRect(
    x - textWidth / 2 - s.padding.x,
    y - s.padding.y,
    textWidth + s.padding.x * 2,
    s.padding.y * 2 - 2,
    s.borderRadius
  );
  ctx.fill();

  // Text
  ctx.fillStyle = s.textColor;
  ctx.textAlign = 'center';
  ctx.fillText(text, x, y + 3);
}

/** Generic text drawing with configurable alignment and style */
function drawText(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  options: { font?: string; color: string; align?: CanvasTextAlign }
): void {
  ctx.font = options.font ?? '12px monospace';
  ctx.fillStyle = options.color;
  ctx.textAlign = options.align ?? 'left';
  ctx.fillText(text, x, y);
}

/** Format a weight vector for the neuron box: space-separated, no brackets or commas */
export function formatWeights(weights: number[]): string {
  return weights.map(w => w.toFixed(2)).join(' ');
}

const UP_COLOR = '#60a5fa';    // weight increased
const DOWN_COLOR = '#fb7185';  // weight decreased
const SAME_COLOR = '#94a3b8';  // no visible change

/** Colour for a value change: blue up, rose down, gray when the change is below display precision */
function changeColor(oldValue: number, newValue: number): string {
  const delta = newValue - oldValue;
  if (Math.abs(delta) < 0.00005) return SAME_COLOR;
  return delta > 0 ? UP_COLOR : DOWN_COLOR;
}

/**
 * Draw the "→ new weights" row: same layout as the W row, but every value is
 * coloured by the direction it moved so the update reads at a glance.
 */
function drawNewWeightsVector(
  ctx: CanvasRenderingContext2D,
  oldWeights: number[],
  newWeights: number[],
  x: number,
  y: number,
  containerWidth: number,
  maxFontSize: number
): void {
  const tokens = newWeights.map(w => w.toFixed(2));
  const text = tokens.join(' ');
  const available = containerWidth - 24 - 8;
  ctx.textAlign = 'left';
  for (let size = maxFontSize; size >= 8; size--) {
    ctx.font = `bold ${size}px monospace`;
    if (ctx.measureText(text).width <= available) break;
  }
  const space = ctx.measureText(' ').width;
  let cursor = x;
  tokens.forEach((token, i) => {
    ctx.fillStyle = changeColor(oldWeights[i] ?? newWeights[i], newWeights[i]);
    ctx.fillText(token, cursor, y);
    cursor += ctx.measureText(token).width + space;
  });
}

/** Draw weights vector, shrinking the font until it fits inside the box */
function drawWeightsVector(
  ctx: CanvasRenderingContext2D,
  weights: number[],
  x: number,
  y: number,
  containerWidth: number,
  color: string = '#a5b4fc',
  maxFontSize: number = 12
): void {
  const text = formatWeights(weights);
  // The text starts 24px inside the box (after the "W:" label); keep an 8px right margin
  const available = containerWidth - 24 - 8;

  ctx.fillStyle = color;
  ctx.textAlign = 'left';
  for (let size = maxFontSize; size >= 8; size--) {
    ctx.font = `${size}px monospace`;
    if (ctx.measureText(text).width <= available) break;
  }

  ctx.fillText(text, x, y);
}

function drawRoundedRect(
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

/** Width of a neuron box for a layer, given how many weights it shows (scaled down on narrow canvases) */
export function neuronBoxWidth(layer: LayerName, weightCount: number, widthScale: number = 1): number {
  const natural = Math.max(NEURON_BOX.minWidth, weightCount * NEURON_BOX.weightMultiplier + 40) + NEURON_BOX.extraWidth[layer];
  return Math.round(natural * widthScale);
}

/**
 * Draw the input layer as one small box per value, stacked and centred on `y`.
 * Returns one NodePosition per value so connections can leave each box separately.
 */
export function drawInputVector(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  values: number[],
  inputLabels: string[],
  widthScale: number = 1
): NodePosition[] {
  const { height, cornerRadius, spacing } = INPUT_BOX;
  const width = Math.round(INPUT_BOX.width * widthScale);
  const colors = LAYER_COLORS.input;
  const firstCenterY = y - ((values.length - 1) * spacing) / 2;

  return values.map((value, idx) => {
    const centerY = firstCenterY + idx * spacing;
    const left = x - width / 2;
    const top = centerY - height / 2;

    drawRoundedRect(ctx, left, top, width, height, cornerRadius);
    const gradient = ctx.createLinearGradient(left, top, left, top + height);
    gradient.addColorStop(0, colors.gradientStart);
    gradient.addColorStop(1, colors.gradientEnd);
    ctx.fillStyle = gradient;
    ctx.fill();
    ctx.strokeStyle = colors.stroke;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Label (shrunk if a translation is wide), then the value beneath it
    let labelSize = 11;
    for (; labelSize >= 8; labelSize--) {
      ctx.font = `bold ${labelSize}px sans-serif`;
      if (ctx.measureText(inputLabels[idx] ?? '').width <= width - 12) break;
    }
    drawText(ctx, inputLabels[idx] ?? '', x, top + 17, { font: `bold ${labelSize}px sans-serif`, color: '#ffffff', align: 'center' });
    drawText(ctx, value.toFixed(2), x, top + 36, { font: 'bold 13px monospace', color: '#60a5fa', align: 'center' });

    return { x: left, y: top, width, height, centerX: x, centerY };
  });
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
  activationRange?: { min: number; max: number },
  backpropUpdateData?: BackpropUpdateData,
  widthScale: number = 1
): NodePosition {
  // Note: drawNeuronVector is only called for layer1, layer2, output (not input)
  const layer = layerType as LayerName;
  const width = neuronBoxWidth(layer, weights.length, widthScale);
  const height = NEURON_BOX.height + (backpropUpdateData ? NEURON_BOX.backpropExtraHeight : 0);
  const fontSize = NEURON_BOX.fontSize;
  const valueFont = `${fontSize}px monospace`;
  const boldValueFont = `bold ${fontSize}px monospace`;

  const left = x - width / 2;
  const top = y - height / 2;

  drawRoundedRect(ctx, left, top, width, height, NEURON_BOX.cornerRadius);

  const colors = LAYER_COLORS[layer];
  const gradient = ctx.createLinearGradient(left, top, left, top + height);
  let strokeColor: string;

  if (isHighlighted) {
    // Highlighted neurons use full color intensity
    gradient.addColorStop(0, colors.highlightGradientStart);
    gradient.addColorStop(1, colors.highlightGradientEnd);
    strokeColor = colors.highlightStroke;
  } else {
    // Opacity follows the activation, normalised within the layer for contrast
    const minOpacity = 0.3;
    const maxOpacity = 1.0;
    const normalized = activationRange && activationRange.max > activationRange.min
      ? (activation - activationRange.min) / (activationRange.max - activationRange.min)
      : activation;
    const opacity = minOpacity + normalized * (maxOpacity - minOpacity);

    const adjustOpacity = (colorStr: string, alpha: number): string => {
      const rgbaMatch = colorStr.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*[\d.]+)?\)/);
      if (rgbaMatch) {
        const [, r, g, b] = rgbaMatch;
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
      }
      return colorStr;
    };

    gradient.addColorStop(0, adjustOpacity(colors.gradientStart, opacity));
    gradient.addColorStop(1, adjustOpacity(colors.gradientEnd, opacity * 0.7));
    strokeColor = adjustOpacity(colors.stroke, opacity);
  }

  ctx.fillStyle = gradient;
  ctx.fill();

  // Border (purple while this neuron is being back-propagated)
  if (isBackpropHighlighted) {
    ctx.strokeStyle = '#a855f7';
    ctx.lineWidth = 4;
  } else {
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = isHighlighted ? 3 : 2;
  }
  ctx.stroke();

  // Rows: label / W / (→ new W) / b / σ
  const rowStep = fontSize + 3;
  const textX = left + 8;
  const valueX = left + 24;
  let rowY = top + 14;

  drawText(ctx, label, x, rowY, { font: 'bold 11px sans-serif', color: '#ffffff', align: 'center' });

  rowY += rowStep + 1;
  drawText(ctx, 'W:', textX, rowY, { font: valueFont, color: '#cbd5e1' });
  drawWeightsVector(ctx, weights, valueX, rowY, width, undefined, fontSize);

  if (backpropUpdateData) {
    rowY += rowStep;
    drawText(ctx, '→', textX, rowY, { font: boldValueFont, color: '#e2e8f0' });
    drawNewWeightsVector(ctx, weights, backpropUpdateData.newWeights, valueX, rowY, width, fontSize);
  }

  rowY += rowStep;
  drawText(ctx, 'b:', textX, rowY, { font: valueFont, color: '#cbd5e1' });
  if (backpropUpdateData) {
    const oldText = `${bias.toFixed(2)} → `;
    drawText(ctx, oldText, valueX, rowY, { font: valueFont, color: '#fbbf24' });
    ctx.font = valueFont;
    drawText(ctx, backpropUpdateData.newBias.toFixed(2), valueX + ctx.measureText(oldText).width, rowY, {
      font: boldValueFont,
      color: changeColor(bias, backpropUpdateData.newBias),
    });
  } else {
    drawText(ctx, bias.toFixed(2), valueX, rowY, { font: valueFont, color: '#fbbf24' });
  }

  rowY += rowStep;
  drawText(ctx, `σ=${activation.toFixed(3)}`, x + width / 2 - 8, rowY, { font: boldValueFont, color: '#34d399', align: 'right' });

  return { x: left, y: top, width, height, centerX: x, centerY: y };
}
