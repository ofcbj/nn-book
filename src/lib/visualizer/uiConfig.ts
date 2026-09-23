/**
 * UI Configuration
 * 
 * Centralized constants for visualizer rendering.
 * Change these values to adjust the visual appearance.
 */

import type { LayerName } from '../core';

// =============================================================================
// Canvas Layout
// =============================================================================

/** Canvas background color */
export const CANVAS_BACKGROUND = '#0a0a0a';

/** Canvas padding */
export const CANVAS_PADDING = {
  left: 40,
  right: 40,
} as const;

/** Minimum horizontal gap between the four columns (input, layer1, layer2, output) */
export const MIN_COLUMN_GAP = 24;

// =============================================================================
// Neuron Box Dimensions
// =============================================================================

/** Base neuron box dimensions */
export const NEURON_BOX = {
  minWidth: 120,
  weightMultiplier: 24,  // width per weight
  cornerRadius: 10,
  /** Box height (label, W, b, σ rows at 11px) */
  height: 68,
  /** Extra height while backprop shows the "→ new weights" row */
  backpropExtraHeight: 14,
  /** Extra width by layer (sized so "W: -0.00 -0.00 ..." fits at 11px, including minus signs) */
  extraWidth: {
    layer1: 30,
    layer2: 60,
    output: 40,
  },
  /** Font size of the value rows */
  fontSize: 11,
} as const;

/** Input vector box dimensions */
/** One box per input value (grade / attitude / response) */
export const INPUT_BOX = {
  width: 112,
  height: 48,
  cornerRadius: 10,
  /** Vertical distance between box centres */
  spacing: 64,
} as const;

// =============================================================================
// Vertical Spacing
// =============================================================================

/** Vertical spacing between neurons in each layer */
export const VERTICAL_SPACING = {
  layer1: 92,
  layer2: 110,
  output: 110,
} as const;

/** Layer name to node array index mapping */
export const LAYER_NODE_INDEX: Record<LayerName, number> = {
  layer1: 1,
  layer2: 2,
  output: 3,
};

// =============================================================================
// Layer Colors
// =============================================================================

export const LAYER_COLORS = {
  input: {
    gradientStart: 'rgba(59, 130, 246, 0.3)',
    gradientEnd: 'rgba(37, 99, 235, 0.2)',
    highlightGradientStart: 'rgba(59, 130, 246, 0.9)',
    highlightGradientEnd: 'rgba(37, 99, 235, 0.7)',
    stroke: '#3b82f6',
    highlightStroke: '#60a5fa',
  },
  layer1: {
    gradientStart: 'rgba(34, 197, 94, 0.3)',
    gradientEnd: 'rgba(22, 163, 74, 0.2)',
    highlightGradientStart: 'rgba(34, 197, 94, 0.9)',
    highlightGradientEnd: 'rgba(22, 163, 74, 0.7)',
    stroke: '#22c55e',
    highlightStroke: '#4ade80',
  },
  layer2: {
    gradientStart: 'rgba(249, 115, 22, 0.3)',
    gradientEnd: 'rgba(234, 88, 12, 0.2)',
    highlightGradientStart: 'rgba(249, 115, 22, 0.9)',
    highlightGradientEnd: 'rgba(234, 88, 12, 0.7)',
    stroke: '#f97316',
    highlightStroke: '#fb923c',
  },
  output: {
    gradientStart: 'rgba(168, 85, 247, 0.3)',
    gradientEnd: 'rgba(147, 51, 234, 0.2)',
    highlightGradientStart: 'rgba(168, 85, 247, 0.9)',
    highlightGradientEnd: 'rgba(147, 51, 234, 0.7)',
    stroke: '#a855f7',
    highlightStroke: '#c084fc',
  },
} as const;

// =============================================================================
// Activation Color Mapping
// =============================================================================

type RGB = [number, number, number];

const ACTIVATION_COLORS = {
  low: [59, 130, 246] as RGB,     // Blue
  mid: [234, 179, 8] as RGB,      // Yellow
  high: [239, 68, 68] as RGB,     // Red
  invalid: [100, 100, 100] as RGB, // Gray
};

function interpolateColor(color1: RGB, color2: RGB, factor: number): RGB {
  return [
    Math.round(color1[0] + (color2[0] - color1[0]) * factor),
    Math.round(color1[1] + (color2[1] - color1[1]) * factor),
    Math.round(color1[2] + (color2[2] - color1[2]) * factor),
  ];
}

/**
 * Converts activation value (0-1) to RGB color string.
 * Uses a diverging color scale: Blue (0) → Yellow (0.5) → Red (1)
 */
export function activationToColor(value: number): string {
  if (isNaN(value) || value === undefined || value === null) {
    const [r, g, b] = ACTIVATION_COLORS.invalid;
    return `rgb(${r}, ${g}, ${b})`;
  }

  const clamped = Math.max(0, Math.min(1, value));
  const rgb = clamped < 0.5
    ? interpolateColor(ACTIVATION_COLORS.low, ACTIVATION_COLORS.mid, clamped / 0.5)
    : interpolateColor(ACTIVATION_COLORS.mid, ACTIVATION_COLORS.high, (clamped - 0.5) / 0.5);

  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

/** Returns color stops for gradient legend */
export function getColorStops(steps: number = 10): Array<{ value: number; color: string }> {
  return Array.from({ length: steps + 1 }, (_, i) => ({
    value: i / steps,
    color: activationToColor(i / steps),
  }));
}
