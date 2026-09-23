/**
 * Overlay Renderer
 *
 * Common popup box rendering logic for both Forward and Backward overlays.
 * Separates rendering from content generation for better maintainability.
 */

import type { NodePosition, Viewport } from '../types';
import type { OverlayContent } from './overlayContentGenerator';

// ============================================================================
// Box Position Calculation
// ============================================================================

interface BoxPosition {
  x: number;
  y: number;
}

/**
 * Calculate optimal box position avoiding overlap with neuron and canvas bounds.
 */
export type OverlaySide = 'left' | 'right';

function calculateBoxPosition(
  nodeInfo: NodePosition,
  boxWidth: number,
  boxHeight: number,
  viewport: Viewport,
  preferSide?: OverlaySide
): BoxPosition {
  const margin = 10;
  const offset = 15;

  // Preferred side first (keeps the connections being explained uncovered)
  if (preferSide) {
    const sideX = preferSide === 'right'
      ? nodeInfo.x + nodeInfo.width + offset
      : nodeInfo.x - boxWidth - offset;
    const sideY = Math.min(
      Math.max(nodeInfo.centerY - boxHeight / 2, margin),
      viewport.height - margin - boxHeight
    );
    if (sideX >= margin && sideX + boxWidth <= viewport.width - margin) {
      return { x: sideX, y: sideY };
    }
  }

  // Default position: above neuron
  let boxX = nodeInfo.centerX - boxWidth / 2;
  let boxY = nodeInfo.y - boxHeight - offset;

  const overlapsNeuron = (testX: number, testY: number): boolean => {
    const popupRight = testX + boxWidth;
    const popupBottom = testY + boxHeight;
    const neuronRight = nodeInfo.x + nodeInfo.width;
    const neuronBottom = nodeInfo.y + nodeInfo.height;

    return !(popupRight < nodeInfo.x ||
             testX > neuronRight ||
             popupBottom < nodeInfo.y ||
             testY > neuronBottom);
  };

  // If default position overlaps neuron or is out of bounds above, try below
  if (boxY < margin || overlapsNeuron(boxX, boxY)) {
    boxY = nodeInfo.y + nodeInfo.height + offset;
  }

  // If below also overlaps or out of bounds, try to the right
  if ((boxY + boxHeight > viewport.height - margin) || overlapsNeuron(boxX, boxY)) {
    boxY = nodeInfo.centerY - boxHeight / 2;
    boxX = nodeInfo.x + nodeInfo.width + offset;
  }

  // If right also overlaps or out of bounds, try to the left
  if ((boxX + boxWidth > viewport.width - margin) || overlapsNeuron(boxX, boxY)) {
    boxX = nodeInfo.x - boxWidth - offset;
  }

  // Final boundary adjustments (ensure within canvas)
  if (boxX < margin) boxX = margin;
  if (boxY < margin) boxY = margin;
  if (boxX + boxWidth > viewport.width - margin) boxX = viewport.width - margin - boxWidth;
  if (boxY + boxHeight > viewport.height - margin) boxY = viewport.height - margin - boxHeight;

  return { x: boxX, y: boxY };
}

// ============================================================================
// Box Rendering
// ============================================================================

export interface OverlayBoxOptions {
  /** Try to place the popup on this side of the neuron before the default positions */
  preferSide?: OverlaySide;
  boxWidth?: number;
  lineHeight?: number;
  padding?: number;
  titleFontSize?: number;
  contentFontSize?: number;
  borderRadius?: number;
  titlePadding?: number;
}

const DEFAULT_OPTIONS: Required<Omit<OverlayBoxOptions, 'preferSide'>> = {
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
function drawOverlayBox(
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
  ctx.fillStyle = '#e2e8f0';
  let yOffset = boxY + 50;
  for (const line of content.lines) {
    ctx.fillText(line, textCenterX, yOffset);
    yOffset += opts.lineHeight;
  }
}

/**
 * Calculate box dimensions based on content.
 */
function calculateBoxDimensions(
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
  viewport: Viewport,
  nodeInfo: NodePosition,
  content: OverlayContent,
  options: OverlayBoxOptions = {}
): void {
  const { width: boxWidth, height: boxHeight } = calculateBoxDimensions(content, options);
  const { x: boxX, y: boxY } = calculateBoxPosition(nodeInfo, boxWidth, boxHeight, viewport, options.preferSide);
  drawOverlayBox(ctx, boxX, boxY, boxWidth, boxHeight, content, options);
}
