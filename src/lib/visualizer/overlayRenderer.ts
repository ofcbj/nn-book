/**
 * Overlay Renderer
 * 
 * Common popup box rendering logic for both Forward and Backward overlays.
 * Separates rendering from content generation for better maintainability.
 */

import type { NodePosition } from '../types';
import type { OverlayContent } from './overlayContentGenerator';

// ============================================================================
// Box Position Calculation
// ============================================================================

export interface BoxPosition {
  x: number;
  y: number;
}

/**
 * Calculate optimal box position avoiding overlap with neuron and canvas bounds.
 */
export function calculateBoxPosition(
  nodeInfo: NodePosition,
  boxWidth: number,
  boxHeight: number,
  canvas: HTMLCanvasElement
): BoxPosition {
  const margin = 10;
  const offset = 15;

  // Default position: above neuron
  let boxX = nodeInfo.centerX - boxWidth / 2;
  let boxY = nodeInfo.y - boxHeight - offset;

  // Check if popup overlaps with neuron box
  const overlapsNeuron = (testX: number, testY: number): boolean => {
    const popupLeft = testX;
    const popupRight = testX + boxWidth;
    const popupTop = testY;
    const popupBottom = testY + boxHeight;

    const neuronLeft = nodeInfo.x;
    const neuronRight = nodeInfo.x + nodeInfo.width;
    const neuronTop = nodeInfo.y;
    const neuronBottom = nodeInfo.y + nodeInfo.height;

    return !(popupRight < neuronLeft || 
             popupLeft > neuronRight || 
             popupBottom < neuronTop || 
             popupTop > neuronBottom);
  };

  // If default position overlaps neuron or is out of bounds above, try below
  if (boxY < margin || overlapsNeuron(boxX, boxY)) {
    boxY = nodeInfo.y + nodeInfo.height + offset;
  }

  // If below also overlaps or out of bounds, try to the right
  if ((boxY + boxHeight > canvas.height - margin) || overlapsNeuron(boxX, boxY)) {
    boxY = nodeInfo.centerY - boxHeight / 2;
    boxX = nodeInfo.x + nodeInfo.width + offset;
  }

  // If right also overlaps or out of bounds, try to the left
  if ((boxX + boxWidth > canvas.width - margin) || overlapsNeuron(boxX, boxY)) {
    boxX = nodeInfo.x - boxWidth - offset;
  }

  // Final boundary adjustments (ensure within canvas)
  if (boxX < margin) boxX = margin;
  if (boxY < margin) boxY = margin;
  if (boxX + boxWidth > canvas.width - margin) boxX = canvas.width - margin - boxWidth;
  if (boxY + boxHeight > canvas.height - margin) boxY = canvas.height - margin - boxHeight;

  return { x: boxX, y: boxY };
}

// ============================================================================
// Box Rendering
// ============================================================================

export interface OverlayBoxOptions {
  boxWidth?: number;
  lineHeight?: number;
  padding?: number;
  titleFontSize?: number;
  contentFontSize?: number;
  borderRadius?: number;
  titlePadding?: number;
}

const DEFAULT_OPTIONS: Required<OverlayBoxOptions> = {
  boxWidth: 420,
  lineHeight: 22,
  padding: 60,
  titleFontSize: 14,
  contentFontSize: 13,
  borderRadius: 8,
  titlePadding: 25,
};

/**
 * Draw an overlay box with title and content lines.
 */
export function drawOverlayBox(
  ctx: CanvasRenderingContext2D,
  boxX: number,
  boxY: number,
  boxWidth: number,
  boxHeight: number,
  content: OverlayContent,
  options: OverlayBoxOptions = {}
): void {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // Background
  ctx.fillStyle = 'rgba(0, 0, 0, 0.65)';
  ctx.beginPath();
  ctx.roundRect(boxX, boxY, boxWidth, boxHeight, opts.borderRadius);
  ctx.fill();

  // Border
  ctx.strokeStyle = content.color;
  ctx.lineWidth = 3;
  ctx.stroke();

  // Title
  ctx.font = `bold ${opts.titleFontSize}px sans-serif`;
  ctx.fillStyle = content.color;
  ctx.textAlign = 'center';
  const textCenterX = boxX + boxWidth / 2;
  ctx.fillText(content.title, textCenterX, boxY + opts.titlePadding);

  // Content lines
  ctx.font = `bold ${opts.contentFontSize}px monospace`;
  ctx.textAlign = 'center';
  let yOffset = boxY + 50;

  content.lines.forEach((line) => {
    // All lines use consistent white color and bold monospace font
    ctx.fillStyle = '#e2e8f0';
    ctx.font = `bold ${opts.contentFontSize}px monospace`;
    ctx.fillText(line, textCenterX, yOffset);
    yOffset += opts.lineHeight;
  });
}

/**
 * Calculate box dimensions based on content.
 */
export function calculateBoxDimensions(
  content: OverlayContent,
  options: OverlayBoxOptions = {}
): { width: number; height: number } {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  return {
    width: opts.boxWidth,
    height: opts.padding + content.lines.length * opts.lineHeight
  };
}

/**
 * Render a complete overlay with automatic positioning.
 */
export function renderOverlay(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  nodeInfo: NodePosition,
  content: OverlayContent,
  options: OverlayBoxOptions = {}
): void {
  const { width: boxWidth, height: boxHeight } = calculateBoxDimensions(content, options);
  const { x: boxX, y: boxY } = calculateBoxPosition(nodeInfo, boxWidth, boxHeight, canvas);
  drawOverlayBox(ctx, boxX, boxY, boxWidth, boxHeight, content, options);
}
