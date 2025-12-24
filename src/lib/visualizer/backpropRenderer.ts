// Backpropagation visualization renderer
import type { NodePosition, BackpropNeuronData, BackpropStage, AnimationPhase } from '../types';
import { generateBackpropContent } from './overlayContentGenerator';
import { renderOverlay } from './overlayRenderer';

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Find the node to highlight based on layer and index.
 */
function findNodeToHighlight(
  layer: string,
  index: number,
  nodes: NodePosition[][]
): NodePosition | null {
  if (layer === 'layer1' && nodes[1]) {
    return nodes[1][index];
  } else if (layer === 'layer2' && nodes[2]) {
    return nodes[2][index];
  } else if (layer === 'output' && nodes[3]) {
    return nodes[3][index];
  }
  return null;
}

/**
 * Draw error glow effect around a node.
 */
function drawErrorGlow(
  ctx: CanvasRenderingContext2D,
  nodeInfo: NodePosition,
  errorMagnitude: number
): void {
  ctx.save();
  const glowSize = Math.min(errorMagnitude * 100 + 20, 60);

  const gradient = ctx.createRadialGradient(
    nodeInfo.centerX, nodeInfo.centerY, 0,
    nodeInfo.centerX, nodeInfo.centerY, nodeInfo.width / 2 + glowSize
  );
  gradient.addColorStop(0, `rgba(239, 68, 68, ${errorMagnitude * 0.8})`);
  gradient.addColorStop(1, 'rgba(239, 68, 68, 0)');

  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(nodeInfo.centerX, nodeInfo.centerY, nodeInfo.width / 2 + glowSize, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// ============================================================================
// Main Function
// ============================================================================

/**
 * Draw backpropagation highlight and overlay.
 * Uses separated content generator and overlay renderer modules.
 */
export function drawBackpropHighlight(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  nodes: NodePosition[][],
  backpropPhase: AnimationPhase | null,
  currentBackpropData: BackpropNeuronData | null,
  backpropStage: BackpropStage | null
): void {
  if (!backpropPhase) return;

  const { layer, index } = backpropPhase;
  const nodeInfo = findNodeToHighlight(layer, index, nodes);
  if (!nodeInfo) return;

  // Draw error glow
  const errorMagnitude = currentBackpropData ? Math.abs(currentBackpropData.error) : 0.5;
  drawErrorGlow(ctx, nodeInfo, errorMagnitude);

  // Draw information overlay
  if (currentBackpropData && backpropStage) {
    const content = generateBackpropContent(backpropStage, currentBackpropData, layer);
    renderOverlay(ctx, canvas, nodeInfo, content);
  } else {
    // Fallback label
    ctx.font = 'bold 14px sans-serif';
    ctx.fillStyle = '#ef4444';
    ctx.textAlign = 'center';
    ctx.fillText('◄ BACKPROP', nodeInfo.centerX, nodeInfo.y - 35);
  }
}
