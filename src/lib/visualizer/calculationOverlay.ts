// Calculation overlay renderer for forward propagation
import type { CalculationStage, NeuronCalculation, NodePosition } from '../types';
import { generateForwardContent } from './overlayContentGenerator';
import { renderOverlay } from './overlayRenderer';

/**
 * Draw calculation overlay for forward propagation.
 * Uses separated content generator and overlay renderer modules.
 */
export function drawCalculationOverlay(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  x: number,
  y: number,
  stage: CalculationStage,
  currentNeuronData: NeuronCalculation | null
): void {
  if (!currentNeuronData) return;

  // Generate content using the content generator
  const content = generateForwardContent(stage, currentNeuronData);
  if (!content.title) return;

  // Create a virtual node position for the overlay renderer
  const nodeInfo: NodePosition = {
    x: x - 40,
    y: y - 40,
    width: 80,
    height: 80,
    centerX: x,
    centerY: y,
  };

  // Render using the common overlay renderer with smaller box width for forward
  renderOverlay(ctx, canvas, nodeInfo, content, {
    boxWidth: 380,
    lineHeight: 20,
    padding: 50,
    titleFontSize: 13,
    contentFontSize: 12,
    borderRadius: 6,
    titlePadding: 20,
  });
}
