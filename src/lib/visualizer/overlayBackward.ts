// Backward propagation overlay renderer
import type { NodePosition, BackwardCalculation, BackwardSteps, Viewport } from '../types';
import type { AnimationState } from '../animation';
import type { NeuralNetwork, LayerName } from '../core';
import { LAYER_NAMES, DEFAULT_LEARNING_RATE } from '../core';
import { generateBackpropContent } from './overlayContentGenerator';
import { renderOverlay } from './overlayRenderer';
import { drawTextWithBackground } from './drawingUtils';
import { LAYER_NODE_INDEX } from './uiConfig';

// ============================================================================
// Helper Functions
// ============================================================================

/** Node array of the layer after `layer`, or null for the output layer */
function getNextLayerNodes(layer: LayerName, nodes: NodePosition[][]): NodePosition[] | null {
  const nextLayer = LAYER_NAMES[LAYER_NAMES.indexOf(layer) + 1];
  return nextLayer ? nodes[LAYER_NODE_INDEX[nextLayer]] ?? null : null;
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
  gradient.addColorStop(0, `rgba(239, 68, 68, ${Math.min(errorMagnitude * 0.8, 1)})`);
  gradient.addColorStop(1, 'rgba(239, 68, 68, 0)');

  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(nodeInfo.centerX, nodeInfo.centerY, nodeInfo.width / 2 + glowSize, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

/**
 * Draw highlighted connections from the current hidden neuron to the next layer.
 * Shows which weights carry the next layer's δ back to this neuron.
 */
function drawBackwardConnections(
  ctx: CanvasRenderingContext2D,
  currentNode: NodePosition,
  nextLayerNodes: NodePosition[],
  nextLayerDeltas: number[] | undefined,
  nextLayerWeights: number[] | undefined
): void {
  ctx.save();

  nextLayerNodes.forEach((nextNode, idx) => {
    const startX = currentNode.centerX + currentNode.width / 2;
    const startY = currentNode.centerY;
    const endX = nextNode.centerX - nextNode.width / 2;
    const endY = nextNode.centerY;

    // Connection line with red glow
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)';
    ctx.lineWidth = 4;
    ctx.shadowColor = 'rgba(239, 68, 68, 0.9)';
    ctx.shadowBlur = 12;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Weight label on the connection line
    const weight = nextLayerWeights?.[idx];
    if (weight !== undefined) {
      drawTextWithBackground(ctx, `W=${weight.toFixed(3)}`, (startX + endX) / 2, (startY + endY) / 2, {
        bgColor: 'rgba(0, 0, 0, 0.85)',
        textColor: '#fbbf24',
      });
    }
  });

  // δ labels on the next-layer neurons
  nextLayerDeltas?.forEach((delta, idx) => {
    const nextNode = nextLayerNodes[idx];
    if (!nextNode) return;
    drawTextWithBackground(ctx, `δ=${delta.toFixed(4)}`, nextNode.centerX, nextNode.y + nextNode.height + 18, {
      bgColor: 'rgba(239, 68, 68, 0.9)',
      font: 'bold 11px monospace',
      padding: { x: 6, y: 10 },
      borderRadius: 4,
    });
  });

  ctx.restore();
}

/**
 * Draw δ labels under every neuron during backpropagation.
 */
function drawAllDeltaLabels(
  ctx: CanvasRenderingContext2D,
  nodes: NodePosition[][],
  allBackpropData: BackwardSteps
): void {
  ctx.save();

  for (const layer of LAYER_NAMES) {
    const layerNodes = nodes[LAYER_NODE_INDEX[layer]] ?? [];
    const data: BackwardCalculation[] = allBackpropData[layer];

    layerNodes.forEach((node, idx) => {
      const neuron = data[idx];
      if (!neuron) return;
      const magnitude = Math.min(Math.abs(neuron.gradient) * 2, 1);
      drawTextWithBackground(ctx, `δ=${neuron.gradient.toFixed(3)}`, node.centerX, node.y + node.height + 14, {
        bgColor: `rgba(239, 68, 68, ${0.5 + magnitude * 0.4})`,
      });
    });
  }

  ctx.restore();
}

// ============================================================================
// Main Function
// ============================================================================

/**
 * Draw backward propagation overlay visualization.
 * Shows δ values, connection weights, and backprop calculations.
 */
export function drawBackwardOverlay(
  ctx: CanvasRenderingContext2D,
  viewport: Viewport,
  nodes: NodePosition[][],
  nn: NeuralNetwork,
  animationState: AnimationState,
  learningRate: number = DEFAULT_LEARNING_RATE
): void {
  if (animationState.type !== 'backward_animating' &&
      animationState.type !== 'showing_backprop_modal') {
    return;
  }

  // Persistent δ labels on all neurons
  if (nn.lastBackwardSteps) {
    drawAllDeltaLabels(ctx, nodes, nn.lastBackwardSteps);
  }

  if (animationState.type !== 'backward_animating') return;

  const { layer, neuronIndex, neuronData, stage } = animationState;
  const nodeInfo = nodes[LAYER_NODE_INDEX[layer]]?.[neuronIndex];
  if (!nodeInfo) return;

  // Hidden layers: show the connections that carry the next layer's δ back here
  if (neuronData) {
    const nextLayerNodes = getNextLayerNodes(layer, nodes);
    if (nextLayerNodes && nextLayerNodes.length > 0) {
      drawBackwardConnections(ctx, nodeInfo, nextLayerNodes, neuronData.nextLayerDeltas, neuronData.nextLayerWeights);
    }
  }

  // Error glow
  const errorMagnitude = neuronData ? Math.abs(neuronData.error) : 0.5;
  drawErrorGlow(ctx, nodeInfo, errorMagnitude);

  // Information overlay
  if (neuronData) {
    const content = generateBackpropContent(stage, neuronData, layer, learningRate);
    renderOverlay(ctx, viewport, nodeInfo, content);
  } else {
    ctx.font = 'bold 14px sans-serif';
    ctx.fillStyle = '#ef4444';
    ctx.textAlign = 'center';
    ctx.fillText('◄ BACKPROP', nodeInfo.centerX, nodeInfo.y - 35);
  }
}
