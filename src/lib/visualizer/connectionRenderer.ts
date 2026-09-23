// Connection rendering between neural network layers.
// Each line encodes its weight: thickness = |w|, colour = sign (blue +, rose −).
import type { NodePosition, BackwardStage } from '../types';
import type { AnimationState } from '../animation';
import { LAYER_NAMES, LAYER_SIZES, type LayerName } from '../core';

// ============================================================================
// Types
// ============================================================================

/** weights[to][j][i] = weight from neuron i of the previous layer into neuron j of layer `to` */
export type LayerWeights = Record<LayerName, number[][]>;

export interface ConnectionStyle {
  color: string;
  lineWidth: number;
}

// ============================================================================
// Constants
// ============================================================================

const POSITIVE_RGB = '96, 165, 250';  // blue
const NEGATIVE_RGB = '251, 113, 133'; // rose
const MIN_WIDTH = 0.75;
const MAX_WIDTH = 4;
const MIN_ALPHA = 0.3;
const MAX_ALPHA = 0.9;
/**
 * While a neuron is animating, the lines it is using are drawn fully opaque with a glow
 * (forward: its incoming lines; backward: its outgoing lines, which carry the next layer's δ)…
 */
const ACTIVE_ALPHA = 1;
/** …and every other line fades back so the active ones stand out */
const INACTIVE_DIM = 0.35;

// ============================================================================
// Weight encoding
// ============================================================================

/**
 * Map a weight to a line style.
 * `scale` is the |w| that maps to full thickness/opacity (at least 1 so tiny
 * initial weights stay thin instead of being blown up by normalisation).
 */
export function weightToStyle(weight: number, scale: number, alphaMultiplier: number = 1): ConnectionStyle {
  const norm = Math.min(Math.abs(weight) / Math.max(scale, 1), 1);
  const alpha = (MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * norm) * alphaMultiplier;
  return {
    color: `rgba(${weightRgb(weight)}, ${alpha.toFixed(3)})`,
    lineWidth: MIN_WIDTH + (MAX_WIDTH - MIN_WIDTH) * norm,
  };
}

/** Blue for positive weights, rose for negative */
export function weightRgb(weight: number): string {
  return weight >= 0 ? POSITIVE_RGB : NEGATIVE_RGB;
}

/** Largest |w| in a layer's weight matrix */
export function maxAbsWeight(weights: number[][]): number {
  let max = 0;
  for (const row of weights) {
    for (const w of row) {
      max = Math.max(max, Math.abs(w));
    }
  }
  return max;
}

// ============================================================================
// Drawing
// ============================================================================

interface Anchor {
  x: number;
  y: number;
}

/** Two arrowheads along the line from `start` (the neuron) toward `end` (its input), at 35% and 70% */
function drawBackwardArrows(ctx: CanvasRenderingContext2D, start: Anchor, end: Anchor, color: string, size: number): void {
  const angle = Math.atan2(end.y - start.y, end.x - start.x);
  ctx.fillStyle = color;
  for (const t of [0.35, 0.7]) {
    const tipX = start.x + (end.x - start.x) * t;
    const tipY = start.y + (end.y - start.y) * t;
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(tipX - size * Math.cos(angle - 0.5), tipY - size * Math.sin(angle - 0.5));
    ctx.lineTo(tipX - size * Math.cos(angle + 0.5), tipY - size * Math.sin(angle + 0.5));
    ctx.closePath();
    ctx.fill();
  }
}

/** Right edge, vertical centre of source node i */
function sourceAnchor(sourceNodes: NodePosition[], i: number): Anchor {
  const node = sourceNodes[i];
  return { x: node.centerX + node.width / 2, y: node.centerY };
}

/** Backprop stages that explain where the error came from (next layer) vs. what changes (incoming weights) */
const OUTGOING_STAGES: readonly BackwardStage[] = ['error', 'derivative', 'gradient'];

/** During backprop, which side of the current neuron the explanation is about */
export function backwardFocus(stage: BackwardStage): 'outgoing' | 'incoming' {
  return OUTGOING_STAGES.includes(stage) ? 'outgoing' : 'incoming';
}

/**
 * Is the connection from source neuron i to target neuron j part of the current calculation?
 * Forward: the lines feeding the neuron being computed.
 * Backward, error stages: the lines leaving the neuron (its error is Σ δ_next · w over them).
 * Backward, ΔW stages: the lines feeding the neuron (these are the weights being adjusted).
 */
function isConnectionActive(
  animationState: AnimationState,
  sourceLayer: LayerName | 'input',
  sourceIndex: number,
  targetLayer: LayerName,
  targetIndex: number
): boolean {
  if (animationState.type === 'forward_animating') {
    return animationState.layer === targetLayer && animationState.neuronIndex === targetIndex;
  }
  if (animationState.type === 'backward_animating') {
    return backwardFocus(animationState.stage) === 'outgoing'
      ? animationState.layer === sourceLayer && animationState.neuronIndex === sourceIndex
      : animationState.layer === targetLayer && animationState.neuronIndex === targetIndex;
  }
  return false;
}

/** ΔW of the incoming weights of the neuron being back-propagated, when a ΔW stage is showing */
function incomingWeightDeltas(animationState: AnimationState): number[] | null {
  if (animationState.type !== 'backward_animating' || backwardFocus(animationState.stage) !== 'incoming') return null;
  return animationState.neuronData?.weightDeltas ?? null;
}

function drawLayerConnections(
  ctx: CanvasRenderingContext2D,
  sourceNodes: NodePosition[],
  targetNodes: NodePosition[],
  sourceLayer: LayerName | 'input',
  targetLayer: LayerName,
  weights: number[][],
  animationState: AnimationState,
  somethingIsAnimating: boolean
): void {
  const scale = maxAbsWeight(weights);
  const deltas = incomingWeightDeltas(animationState);
  const maxDelta = deltas ? Math.max(...deltas.map(Math.abs), 1e-9) : 0;

  for (let j = 0; j < targetNodes.length; j++) {
    const to = targetNodes[j];
    const neuronWeights = weights[j] ?? [];

    for (let i = 0; i < neuronWeights.length; i++) {
      const from = sourceAnchor(sourceNodes, i);
      const weight = neuronWeights[i];
      const active = isConnectionActive(animationState, sourceLayer, i, targetLayer, j);
      // Colour (sign) and thickness (|w|) never change; only emphasis does
      const style = weightToStyle(weight, scale, somethingIsAnimating && !active ? INACTIVE_DIM : 1);

      // ΔW halo: blue = this weight grows, rose = it shrinks; thickness ∝ |ΔW| within the neuron.
      // Arrowheads run from the neuron back toward its inputs: its δ is what re-tunes these weights.
      if (active && deltas) {
        const delta = deltas[i] ?? 0;
        const norm = Math.abs(delta) / maxDelta;
        const toX = to.centerX - to.width / 2;
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(toX, to.centerY);
        ctx.strokeStyle = `rgba(${weightRgb(delta)}, ${(0.3 + 0.4 * norm).toFixed(3)})`;
        ctx.lineWidth = style.lineWidth + 6 + 10 * norm;
        ctx.shadowBlur = 0;
        ctx.stroke();
        drawBackwardArrows(ctx, { x: toX, y: to.centerY }, from, `rgba(${weightRgb(delta)}, 0.95)`, 5 + 5 * norm);
      }

      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.centerX - to.width / 2, to.centerY);
      ctx.lineWidth = style.lineWidth;

      if (active) {
        ctx.strokeStyle = `rgba(${weightRgb(weight)}, ${ACTIVE_ALPHA})`;
        ctx.shadowColor = ctx.strokeStyle;
        ctx.shadowBlur = 10;
      } else {
        ctx.strokeStyle = style.color;
        ctx.shadowBlur = 0;
      }

      ctx.stroke();
      ctx.shadowBlur = 0;
    }
  }
}

// ============================================================================
// Main Export
// ============================================================================

/**
 * Draw all connections. `nodes` is [input, layer1, layer2, output].
 * Every line keeps its sign colour and |w| thickness; while a neuron is animating
 * (forward or backward) the lines it is using are emphasised and the rest are dimmed.
 */
export function drawConnections(
  ctx: CanvasRenderingContext2D,
  nodes: NodePosition[][],
  animationState: AnimationState,
  weights: LayerWeights
): void {
  const animating = animationState.type === 'forward_animating' || animationState.type === 'backward_animating';

  LAYER_NAMES.forEach((targetLayer, idx) => {
    const sourceNodes = nodes[idx];
    const targetNodes = nodes[idx + 1];
    if (!sourceNodes || !targetNodes) return;

    drawLayerConnections(
      ctx,
      sourceNodes,
      targetNodes.slice(0, LAYER_SIZES[targetLayer]),
      idx === 0 ? 'input' : LAYER_NAMES[idx - 1],
      targetLayer,
      weights[targetLayer],
      animationState,
      animating
    );
  });
}
