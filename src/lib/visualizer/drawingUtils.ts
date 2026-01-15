// Drawing utility functions for visualizer
import type { NodePosition, LayerType } from '../types';
import type { LayerName } from '../core';
import i18n from '../../i18n';
import {
  INPUT_BOX,
  NEURON_BOX,
  LAYER_COLORS,
} from './uiConfig';

// =============================================================================
// Text Rendering Helpers
// =============================================================================

export interface LabelStyle {
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

/** Draw weights vector with auto font-size adjustment */
function drawWeightsVector(
  ctx: CanvasRenderingContext2D,
  weights: number[],
  x: number,
  y: number,
  containerWidth: number
): void {
  ctx.font = '12px monospace';
  ctx.fillStyle = '#a5b4fc';
  ctx.textAlign = 'left';
  
  const vectorStr = '[' + weights.map(w => w.toFixed(2)).join(', ') + ']';
  const textWidth = ctx.measureText(vectorStr).width;
  
  // Reduce font size if text is too wide
  if (textWidth > containerWidth - 40) {
    ctx.font = '11px monospace';
  }
  
  ctx.fillText(vectorStr, x, y);
}

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
  activationRange?: { min: number; max: number }
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
    // Highlighted neurons use full color intensity
    gradient.addColorStop(0, colors.highlightGradientStart);
    gradient.addColorStop(1, colors.highlightGradientEnd);
    strokeColor = colors.highlightStroke;
  } else {
    // Apply activation-based opacity to base colors
    // Use layer-specific activation range for dramatic contrast
    const minOpacity = 0.3;
    const maxOpacity = 1.0;

    let opacity: number;
    if (activationRange && activationRange.max > activationRange.min) {
      // Normalize activation within the layer's range
      const normalized = (activation - activationRange.min) / (activationRange.max - activationRange.min);
      opacity = minOpacity + (normalized * (maxOpacity - minOpacity));
    } else {
      // Fallback to global mapping
      opacity = minOpacity + (activation * (maxOpacity - minOpacity));
    }

    // Extract RGB values from the base colors and apply opacity
    const adjustOpacity = (colorStr: string, alpha: number): string => {
      // Extract rgba values if already rgba, otherwise rgb
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

  // === 1. Draw border (highlight if active) ===
  if (isBackpropHighlighted) {
    ctx.strokeStyle = '#a855f7'; // Purple for backprop
    ctx.lineWidth = 5;
  } else {
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = isHighlighted ? 4 : 2;
  }
  ctx.stroke();

  // === 2. Draw neuron label ===
  drawText(ctx, label, x, centerY + 16, {
    font: 'bold 11px sans-serif',
    color: '#ffffff',
    align: 'center'
  });

  // === 3. Draw weights vector (W: [...]) ===
  const weightsY = centerY + 35;
  drawText(ctx, 'W:', centerX + 8, weightsY, { color: '#cbd5e1' });
  drawWeightsVector(ctx, weights, centerX + 24, weightsY, width);

  // === 4. Draw bias value (b: X.XX) ===
  const biasY = centerY + 50;
  drawText(ctx, 'b:', centerX + 8, biasY, { color: '#cbd5e1' });
  drawText(ctx, bias.toFixed(2), centerX + 24, biasY, { color: '#fbbf24' });

  // === 5. Draw activation output (σ=X.XXX) ===
  const activationY = centerY + 68;
  drawText(ctx, `σ=${activation.toFixed(3)}`, centerX + 70, activationY, {
    font: 'bold 12px monospace',
    color: '#34d399'
  });

  return { x: centerX, y: centerY, width, height, centerX: x, centerY: y };
}
