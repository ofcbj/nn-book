// Backward propagation overlay renderer
import type { NodePosition, BackwardCalculation, BackwardSteps } from '../types';
import type { AnimationState } from '../animation';
import type { NeuralNetwork } from '../core';
import { generateBackpropContent } from './overlayContentGenerator';
import { renderOverlay } from './overlayRenderer';
import { drawTextWithBackground } from './drawingUtils';
import { LAYER_NODE_INDEX } from './uiConfig';

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Get the highlighted node based on layer and index.
 */
function getHighlightedNode(
  layer: string,
  index: number,
  nodes: NodePosition[][]
): NodePosition | null {
  const layerIdx = LAYER_NODE_INDEX[layer];
  return layerIdx !== undefined && nodes[layerIdx] ? nodes[layerIdx][index] : null;
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

/**
 * Draw highlighted connections from current neuron to next layer during error stage.
 * This helps visualize which weights are being used in the error calculation.
 * Also draws labels showing the original error values and weights.
 */
function drawBackwardConnections(
  ctx: CanvasRenderingContext2D,
  currentNode: NodePosition,
  nextLayerNodes: NodePosition[],
  nextLayerErrors: number[] | undefined,
  nextLayerWeights: number[] | undefined
): void {
  ctx.save();
  
  nextLayerNodes.forEach((nextNode, idx) => {
    const startX = currentNode.centerX + currentNode.width / 2;
    const startY = currentNode.centerY;
    const endX = nextNode.centerX - nextNode.width / 2;
    const endY = nextNode.centerY;
    
    // Draw connection line with red glow
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    
    // Glow effect
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)';
    ctx.lineWidth = 4;
    ctx.shadowColor = 'rgba(239, 68, 68, 0.9)';
    ctx.shadowBlur = 12;
    ctx.stroke();
    ctx.shadowBlur = 0;
    
    // Draw weight label on connection line
    if (nextLayerWeights && nextLayerWeights[idx] !== undefined) {
      const midX = (startX + endX) / 2;
      const midY = (startY + endY) / 2;
      drawTextWithBackground(ctx, `W=${nextLayerWeights[idx].toFixed(3)}`, midX, midY, {
        bgColor: 'rgba(0, 0, 0, 0.85)',
        textColor: '#fbbf24',
      });
    }
  });
  
  // Draw original error labels on next layer neurons
  if (nextLayerErrors) {
    nextLayerNodes.forEach((nextNode, idx) => {
      if (nextLayerErrors[idx] !== undefined) {
        const labelX = nextNode.centerX;
        const labelY = nextNode.y + nextNode.height + 18;
        drawTextWithBackground(ctx, `err=${nextLayerErrors[idx].toFixed(4)}`, labelX, labelY, {
          bgColor: 'rgba(239, 68, 68, 0.9)',
          font: 'bold 11px monospace',
          padding: { x: 6, y: 10 },
          borderRadius: 4,
        });
      }
    });
  }

  ctx.restore();
}

/**
 * Draw error labels on all neurons during backpropagation.
 * This provides persistent visualizer of error values.
 */
function drawAllErrorLabels(
  ctx: CanvasRenderingContext2D,
  nodes: NodePosition[][],
  allBackpropData: BackwardSteps
): void {
  ctx.save();
  
  const layerData: { nodes: NodePosition[], data: BackwardCalculation[] }[] = [
    { nodes: nodes[1] || [], data: allBackpropData.layer1 },
    { nodes: nodes[2] || [], data: allBackpropData.layer2 },
    { nodes: nodes[3] || [], data: allBackpropData.output },
  ];
  
  layerData.forEach(({ nodes: layerNodes, data }) => {
    layerNodes.forEach((node, idx) => {
      if (data[idx]) {
        const errorValue = data[idx].error;
        const labelX = node.centerX;
        const labelY = node.y + node.height + 14;
        const errorMagnitude = Math.min(Math.abs(errorValue) * 2, 1);

        drawTextWithBackground(ctx, `δ=${errorValue.toFixed(3)}`, labelX, labelY, {
          bgColor: `rgba(239, 68, 68, ${0.5 + errorMagnitude * 0.4})`,
        });
      }
    });
  });
  
  ctx.restore();
}

// ============================================================================
// Main Function
// ============================================================================

/**
 * Draw backward propagation overlay visualization.
 * Shows error values, connection weights, and backprop calculations.
 */
export function drawBackwardOverlay(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  nodes: NodePosition[][],
  nn: NeuralNetwork,
  animationState: AnimationState,
  learningRate: number = 0.25
): void {
  // Only show backprop highlights during backward animation
  if (animationState.type !== 'backward_animating' && 
      animationState.type !== 'showing_backprop_modal') {
    return;
  }

  // Draw persistent error labels on all neurons if we have backprop data
  if (nn.lastBackwardSteps) {
    drawAllErrorLabels(ctx, nodes, nn.lastBackwardSteps);
  }
  
  if (animationState.type !== 'backward_animating') return;

  const { layer, neuronIndex, neuronData, stage } = animationState;
  const nodeInfo = getHighlightedNode(layer, neuronIndex, nodes);
  if (!nodeInfo) return;

  // Draw connection lines to next layer for hidden layers during all backprop stages
  // Backpropagation always considers error from the next layer
  if (layer !== 'output' && neuronData) {
    // Map layer to next layer nodes
    const nextLayerMap: Record<string, NodePosition[]> = {
      layer1: nodes[2] || [],  // layer2 nodes
      layer2: nodes[3] || []   // output nodes
    };
    
    const nextLayerNodes = nextLayerMap[layer];
    
    if (nextLayerNodes.length > 0) {
      drawBackwardConnections(
        ctx,
        nodeInfo,
        nextLayerNodes,
        neuronData.nextLayerErrors,
        neuronData.nextLayerWeights
      );
    }
  }

  // Draw error glow
  const errorMagnitude = neuronData ? Math.abs(neuronData.error) : 0.5;
  drawErrorGlow(ctx, nodeInfo, errorMagnitude);

  // Draw information overlay
  if (neuronData && stage) {
    const content = generateBackpropContent(stage, neuronData, layer, learningRate);
    renderOverlay(ctx, canvas, nodeInfo, content);
  } else {
    // Fallback label
    ctx.font = 'bold 14px sans-serif';
    ctx.fillStyle = '#ef4444';
    ctx.textAlign = 'center';
    ctx.fillText('◄ BACKPROP', nodeInfo.centerX, nodeInfo.y - 35);
  }
}

