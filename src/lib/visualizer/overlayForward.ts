// Forward propagation overlay renderer
import type { NodePosition, Viewport } from '../types';
import type { AnimationState } from '../animation';
import { generateForwardContent } from './overlayContentGenerator';
import { renderOverlay } from './overlayRenderer';
import { LAYER_NODE_INDEX } from './uiConfig';

/**
 * Draw forward propagation calculation overlay.
 * Shows the step-by-step calculation for the currently animating neuron.
 */
export function drawForwardOverlay(
  ctx: CanvasRenderingContext2D,
  viewport: Viewport,
  nodes: NodePosition[][],
  animationState: AnimationState
): void {
  if (animationState.type !== 'forward_animating' || !animationState.neuronData) return;

  const { layer, neuronIndex, stage, neuronData } = animationState;

  const nodeInfo = nodes[LAYER_NODE_INDEX[layer]]?.[neuronIndex];
  if (!nodeInfo) return;

  const content = generateForwardContent(stage, neuronData, layer);
  if (!content.title) return;

  renderOverlay(ctx, viewport, nodeInfo, content, {
    boxWidth: 380,
    lineHeight: 20,
    padding: 50,
    titleFontSize: 13,
    contentFontSize: 12,
    borderRadius: 6,
    titlePadding: 20,
  });
}
