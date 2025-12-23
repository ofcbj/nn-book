// Color mapping utilities for activation visualization
// Maps activation values (0-1) to color gradients

/**
 * Interpolates between two RGB colors
 */
function interpolateColor(
  color1: [number, number, number],
  color2: [number, number, number],
  factor: number
): [number, number, number] {
  return [
    Math.round(color1[0] + (color2[0] - color1[0]) * factor),
    Math.round(color1[1] + (color2[1] - color1[1]) * factor),
    Math.round(color1[2] + (color2[2] - color1[2]) * factor),
  ];
}

/**
 * Converts activation value (0-1) to RGB color string
 * Uses a diverging color scale:
 * - 0.0 (low): Blue rgb(59, 130, 246)
 * - 0.5 (medium): Yellow rgb(234, 179, 8)
 * - 1.0 (high): Red rgb(239, 68, 68)
 */
export function activationToColor(value: number): string {
  // Handle edge cases
  if (isNaN(value) || value === undefined || value === null) {
    return 'rgb(100, 100, 100)'; // Gray for invalid values
  }

  // Clamp value to [0, 1]
  const clamped = Math.max(0, Math.min(1, value));

  // Color stops
  const blue: [number, number, number] = [59, 130, 246];
  const yellow: [number, number, number] = [234, 179, 8];
  const red: [number, number, number] = [239, 68, 68];

  let rgb: [number, number, number];

  if (clamped < 0.5) {
    // Interpolate between blue and yellow
    const factor = clamped / 0.5;
    rgb = interpolateColor(blue, yellow, factor);
  } else {
    // Interpolate between yellow and red
    const factor = (clamped - 0.5) / 0.5;
    rgb = interpolateColor(yellow, red, factor);
  }

  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

/**
 * Gets a lighter version of the activation color for backgrounds
 */
export function activationToColorWithAlpha(value: number, alpha: number = 0.8): string {
  const rgb = activationToColor(value);
  // Convert rgb(r, g, b) to rgba(r, g, b, alpha)
  return rgb.replace('rgb', 'rgba').replace(')', `, ${alpha})`);
}

/**
 * Returns an array of color stops for creating a gradient legend
 */
export function getColorStops(steps: number = 10): Array<{ value: number; color: string }> {
  const stops: Array<{ value: number; color: string }> = [];
  for (let i = 0; i <= steps; i++) {
    const value = i / steps;
    stops.push({
      value,
      color: activationToColor(value),
    });
  }
  return stops;
}
