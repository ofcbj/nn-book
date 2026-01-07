/**
 * UI Configuration
 * 
 * Centralized constants for visualizer rendering.
 * Change these values to adjust the visual appearance.
 */

// =============================================================================
// Canvas Layout
// =============================================================================

/** Canvas background color */
export const CANVAS_BACKGROUND = '#0a0a0a';

/** Canvas padding */
export const CANVAS_PADDING = {
  left: 60,
  right: 80,
} as const;

/** Layer X position ratios (relative to usable width) */
export const LAYER_X_RATIOS = {
  input: 0,      // absolute offset from left
  layer1: 0.32,
  layer2: 0.65,
  output: 1,     // absolute offset from right
} as const;

// =============================================================================
// Neuron Box Dimensions
// =============================================================================

/** Base neuron box dimensions */
export const NEURON_BOX = {
  minWidth: 130,
  weightMultiplier: 25,  // width per weight
  cornerRadius: 12,
  /** Height varies by layer */
  height: {
    layer1: 80,
    layer2: 90,
    output: 90,
  },
  /** Extra width by layer */
  extraWidth: {
    layer1: 30,
    layer2: 60,
    output: 20,
  },
} as const;

/** Input vector box dimensions */
export const INPUT_BOX = {
  width: 140,
  height: 100,
  cornerRadius: 15,
} as const;

// =============================================================================
// Vertical Spacing
// =============================================================================

/** Vertical spacing between neurons in each layer */
export const VERTICAL_SPACING = {
  layer1: 105,
  layer2: 125,
  output: 125,
} as const;

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
// Text Styling
// =============================================================================

export const TEXT_STYLES = {
  label: {
    font: 'bold 11px Arial',
    color: 'white',
  },
  value: {
    font: '10px monospace',
    color: '#e5e5e5',
  },
  weight: {
    font: '9px monospace',
    positiveColor: '#4ade80',
    negativeColor: '#f87171',
  },
} as const;

// =============================================================================
// Connection Lines
// =============================================================================

export const CONNECTION_STYLES = {
  defaultWidth: 1,
  activeWidth: 2.5,
  defaultColor: 'rgba(100, 100, 100, 0.3)',
  activeColor: 'rgba(74, 222, 128, 0.8)',
} as const;
