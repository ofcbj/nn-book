// Backpropagation visualizer renderer
import type { NodePosition, BackpropNeuronData, BackpropStage, AnimationPhase, BackpropSteps } from '../types';
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
  const layerIndexMap: Record<string, number> = {
    layer1: 1,
    layer2: 2,
    output: 3
  };

  const layerIdx = layerIndexMap[layer];
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
function drawBackpropConnections(
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
      
      // Background for weight label
      ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
      const weightText = `W=${nextLayerWeights[idx].toFixed(3)}`;
      const textWidth = ctx.measureText(weightText).width;
      ctx.fillRect(midX - textWidth / 2 - 4, midY - 8, textWidth + 8, 16);
      
      // Weight text
      ctx.font = 'bold 10px monospace';
      ctx.fillStyle = '#fbbf24';
      ctx.textAlign = 'center';
      ctx.fillText(weightText, midX, midY + 4);
    }
  });
  
  // Draw original error labels on next layer neurons
  if (nextLayerErrors) {
    nextLayerNodes.forEach((nextNode, idx) => {
      if (nextLayerErrors[idx] !== undefined) {
        const errorValue = nextLayerErrors[idx];
        
        // Position label below the neuron
        const labelX = nextNode.centerX;
        const labelY = nextNode.y + nextNode.height + 18;
        
        // Background
        ctx.fillStyle = 'rgba(239, 68, 68, 0.9)';
        const errorText = `err=${errorValue.toFixed(4)}`;
        ctx.font = 'bold 11px monospace';
        const textWidth = ctx.measureText(errorText).width;
        
        // Rounded rectangle background
        ctx.beginPath();
        ctx.roundRect(labelX - textWidth / 2 - 6, labelY - 10, textWidth + 12, 18, 4);
        ctx.fill();
        
        // Error text
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.fillText(errorText, labelX, labelY + 3);
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
  allBackpropData: BackpropSteps
): void {
  ctx.save();
  
  const layerData: { nodes: NodePosition[], data: BackpropNeuronData[] }[] = [
    { nodes: nodes[1] || [], data: allBackpropData.layer1 },
    { nodes: nodes[2] || [], data: allBackpropData.layer2 },
    { nodes: nodes[3] || [], data: allBackpropData.output },
  ];
  
  layerData.forEach(({ nodes: layerNodes, data }) => {
    layerNodes.forEach((node, idx) => {
      if (data[idx]) {
        const errorValue = data[idx].error;
        
        // Position label below the neuron
        const labelX = node.centerX;
        const labelY = node.y + node.height + 14;
        
        // Background with semi-transparent red
        const errorMagnitude = Math.min(Math.abs(errorValue) * 2, 1);
        ctx.fillStyle = `rgba(239, 68, 68, ${0.5 + errorMagnitude * 0.4})`;
        const errorText = `δ=${errorValue.toFixed(3)}`;
        ctx.font = 'bold 10px monospace';
        const textWidth = ctx.measureText(errorText).width;
        
        // Rounded rectangle background
        ctx.beginPath();
        ctx.roundRect(labelX - textWidth / 2 - 4, labelY - 8, textWidth + 8, 14, 3);
        ctx.fill();
        
        // Error text
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.fillText(errorText, labelX, labelY + 2);
      }
    });
  });
  
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
  backpropStage: BackpropStage | null,
  allBackpropData: BackpropSteps | null = null
): void {
  // Draw persistent error labels on all neurons if we have backprop data
  if (allBackpropData) {
    drawAllErrorLabels(ctx, nodes, allBackpropData);
  }
  
  if (!backpropPhase) return;

  const { layer, index } = backpropPhase;
  const nodeInfo = findNodeToHighlight(layer, index, nodes);
  if (!nodeInfo) return;

  // Draw connection lines during 'error' stage for hidden layers
  if (backpropStage === 'error' && layer !== 'output' && currentBackpropData) {
    // Map layer to next layer nodes
    const nextLayerMap: Record<string, NodePosition[]> = {
      layer1: nodes[2] || [],  // layer2 nodes
      layer2: nodes[3] || []   // output nodes
    };
    
    const nextLayerNodes = nextLayerMap[layer];
    
    if (nextLayerNodes.length > 0) {
      drawBackpropConnections(
        ctx,
        nodeInfo,
        nextLayerNodes,
        currentBackpropData.nextLayerErrors,
        currentBackpropData.nextLayerWeights
      );
    }
  }

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


