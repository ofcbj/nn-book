// Backward propagation overlay renderer
import type { NodePosition, BackwardCalculation, BackwardSteps, Viewport } from '../types';
import type { AnimationState } from '../animation';
import type { NeuralNetwork, LayerName } from '../core';
import { LAYER_NAMES, DEFAULT_LEARNING_RATE } from '../core';
import { generateBackpropContent } from './overlayContentGenerator';
import { renderOverlay } from './overlayRenderer';
import { backwardFocus } from './connectionRenderer';
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
  const strength = Math.min(errorMagnitude, 1);
  const glowSize = 16 + strength * 24;

  const gradient = ctx.createRadialGradient(
    nodeInfo.centerX, nodeInfo.centerY, nodeInfo.width / 2,
    nodeInfo.centerX, nodeInfo.centerY, nodeInfo.width / 2 + glowSize
  );
  gradient.addColorStop(0, `rgba(239, 68, 68, ${(0.15 + strength * 0.35).toFixed(3)})`);
  gradient.addColorStop(1, 'rgba(239, 68, 68, 0)');

  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(nodeInfo.centerX, nodeInfo.centerY, nodeInfo.width / 2 + glowSize, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

/** Blue for "increase / should have been higher", rose for "decrease / should have been lower" */
function signRgb(value: number): string {
  return value >= 0 ? '96, 165, 250' : '251, 113, 133';
}

function signedFixed(value: number, digits: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
}

/** Weight changes are often ~0.0005; show enough decimals that they do not all read "+0.001" */
export function formatDelta(value: number): string {
  const abs = Math.abs(value);
  const digits = abs >= 0.01 ? 3 : abs >= 0.001 ? 4 : 5;
  return signedFixed(value, digits);
}

/**
 * Error stages: label each outgoing connection with its share of the blame, δ_next · w.
 * (The lines themselves are drawn and emphasised by connectionRenderer.)
 */
function drawOutgoingContributionLabels(
  ctx: CanvasRenderingContext2D,
  currentNode: NodePosition,
  nextLayerNodes: NodePosition[],
  nextLayerDeltas: number[] | undefined,
  nextLayerWeights: number[] | undefined
): void {
  if (!nextLayerDeltas || !nextLayerWeights) return;
  ctx.save();
  nextLayerNodes.forEach((nextNode, idx) => {
    const delta = nextLayerDeltas[idx];
    const weight = nextLayerWeights[idx];
    if (delta === undefined || weight === undefined) return;
    const contribution = delta * weight;
    const startX = currentNode.centerX + currentNode.width / 2;
    const endX = nextNode.centerX - nextNode.width / 2;
    const midX = (startX + endX) / 2;
    const midY = (currentNode.centerY + nextNode.centerY) / 2;
    drawTextWithBackground(ctx, `δ·w=${signedFixed(contribution, 3)}`, midX, midY, {
      bgColor: `rgba(${signRgb(contribution)}, 0.9)`,
      textColor: '#0f172a',
    });
  });
  ctx.restore();
}

/**
 * ΔW stages: label each incoming connection with how much its weight changes.
 * (connectionRenderer draws the matching blue/rose halo on the line.)
 */
function drawIncomingDeltaLabels(
  ctx: CanvasRenderingContext2D,
  currentNode: NodePosition,
  previousLayerNodes: NodePosition[],
  weightDeltas: number[]
): void {
  ctx.save();
  previousLayerNodes.forEach((prevNode, idx) => {
    const delta = weightDeltas[idx];
    if (delta === undefined) return;
    const startX = prevNode.centerX + prevNode.width / 2;
    const endX = currentNode.centerX - currentNode.width / 2;
    // Place labels at 60% of the way so they do not pile up near the fan-in point
    const t = 0.6;
    const labelX = startX + (endX - startX) * t;
    const labelY = prevNode.centerY + (currentNode.centerY - prevNode.centerY) * t;
    drawTextWithBackground(ctx, `ΔW=${formatDelta(delta)}`, labelX, labelY, {
      bgColor: `rgba(${signRgb(delta)}, 0.9)`,
      textColor: '#0f172a',
    });
  });
  ctx.restore();
}

/**
 * δ gauge under every neuron: a two-sided bar centred on zero. Blue to the right
 * = "should have fired more → incoming weights increase", rose to the left =
 * "fired too much → weights decrease". Length is normalised within each layer
 * so the most responsible neuron of a layer fills its bar.
 */
function drawDeltaGauges(
  ctx: CanvasRenderingContext2D,
  nodes: NodePosition[][],
  allBackpropData: BackwardSteps,
  activeNeuron: { layer: LayerName; index: number } | null
): void {
  ctx.save();
  const gaugeHeight = 6;

  for (const layer of LAYER_NAMES) {
    const layerNodes = nodes[LAYER_NODE_INDEX[layer]] ?? [];
    const data: BackwardCalculation[] = allBackpropData[layer];
    const maxAbs = Math.max(...data.map(n => Math.abs(n.gradient)), 1e-9);

    layerNodes.forEach((node, idx) => {
      const neuron = data[idx];
      if (!neuron) return;
      const delta = neuron.gradient;
      const norm = Math.abs(delta) / maxAbs;
      const trackWidth = node.width * 0.8;
      const halfTrack = trackWidth / 2;
      const trackX = node.centerX - halfTrack;
      const trackY = node.y + node.height + 5;

      // Track
      ctx.fillStyle = 'rgba(148, 163, 184, 0.25)';
      ctx.beginPath();
      ctx.roundRect(trackX, trackY, trackWidth, gaugeHeight, 3);
      ctx.fill();

      // Fill from the centre toward the sign's side
      const fillWidth = Math.max(2, halfTrack * norm);
      const fillX = delta >= 0 ? node.centerX : node.centerX - fillWidth;
      ctx.fillStyle = `rgba(${signRgb(delta)}, ${(0.45 + 0.55 * norm).toFixed(3)})`;
      ctx.beginPath();
      ctx.roundRect(fillX, trackY, fillWidth, gaugeHeight, 3);
      ctx.fill();

      // Zero tick
      ctx.fillStyle = 'rgba(226, 232, 240, 0.8)';
      ctx.fillRect(node.centerX - 0.5, trackY - 2, 1, gaugeHeight + 4);

      // Value only for the neuron being explained
      if (activeNeuron && activeNeuron.layer === layer && activeNeuron.index === idx) {
        drawTextWithBackground(ctx, `δ=${signedFixed(delta, 4)}`, node.centerX, trackY + gaugeHeight + 12, {
          bgColor: 'rgba(15, 23, 42, 0.9)',
          textColor: `rgb(${signRgb(delta)})`,
        });
      }
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

  const activeNeuron = animationState.type === 'backward_animating'
    ? { layer: animationState.layer, index: animationState.neuronIndex }
    : null;

  // δ gauges under every neuron (persist while the summary modal is open)
  if (nn.lastBackwardSteps) {
    drawDeltaGauges(ctx, nodes, nn.lastBackwardSteps, activeNeuron);
  }

  if (animationState.type !== 'backward_animating') return;

  const { layer, neuronIndex, neuronData, stage } = animationState;
  const nodeInfo = nodes[LAYER_NODE_INDEX[layer]]?.[neuronIndex];
  if (!nodeInfo) return;

  if (neuronData) {
    if (backwardFocus(stage) === 'outgoing') {
      // Where the blame comes from: contributions of the next layer's δ through each weight
      const nextLayerNodes = getNextLayerNodes(layer, nodes);
      if (nextLayerNodes && nextLayerNodes.length > 0) {
        drawOutgoingContributionLabels(ctx, nodeInfo, nextLayerNodes, neuronData.nextLayerDeltas, neuronData.nextLayerWeights);
      }
    } else {
      // What changes: ΔW of every incoming weight
      const previousLayerNodes = nodes[LAYER_NODE_INDEX[layer] - 1] ?? [];
      drawIncomingDeltaLabels(ctx, nodeInfo, previousLayerNodes, neuronData.weightDeltas);
    }
  }

  // Error glow
  const errorMagnitude = neuronData ? Math.abs(neuronData.error) : 0.5;
  drawErrorGlow(ctx, nodeInfo, errorMagnitude);

  // Information overlay, placed away from the connections being explained
  if (neuronData) {
    const content = generateBackpropContent(stage, neuronData, layer, learningRate);
    renderOverlay(ctx, viewport, nodeInfo, content, {
      preferSide: backwardFocus(stage) === 'outgoing' ? 'left' : 'right',
    });
  } else {
    ctx.font = 'bold 14px sans-serif';
    ctx.fillStyle = '#ef4444';
    ctx.textAlign = 'center';
    ctx.fillText('◄ BACKPROP', nodeInfo.centerX, nodeInfo.y - 35);
  }
}
