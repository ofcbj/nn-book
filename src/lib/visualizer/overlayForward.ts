// Forward propagation overlay renderer
import type { NodePosition } from '../types';
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
  canvas: HTMLCanvasElement,
  nodes: NodePosition[][],
  animationState: AnimationState
): void {
  if (animationState.type !== 'forward_animating' || !animationState.neuronData) return;

  const { layer, neuronIndex, stage, neuronData } = animationState;

  // Find the animating neuron's position
  const layerIdx = LAYER_NODE_INDEX[layer];
  if (layerIdx === undefined || !nodes[layerIdx]) return;

  const nodeInfo = nodes[layerIdx][neuronIndex];
  if (!nodeInfo) return;

  // Generate and render overlay content
  const content = generateForwardContent(stage, neuronData);
  if (!content.title) return;

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
