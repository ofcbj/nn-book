// Network rendering module - draws the base neural network structure
import type { ForwardSteps, NodePosition, AnimationPhase, ForwardCalculation, LayerType } from '../types';
import type { AnimationState } from '../animation';
import type { NeuralNetwork } from '../core';
import { LAYER_SIZES } from '../core';
import { drawInputVector, drawNeuronVector } from './drawingUtils';
import { CANVAS_BACKGROUND, CANVAS_PADDING, VERTICAL_SPACING } from './uiConfig';
import i18n from '../../i18n';

// =============================================================================
// Types
// =============================================================================

interface LayerConfig {
  layerName: LayerType;
  neurons: ForwardCalculation[];
  x: number;
  neuronCount: number;
  verticalSpacing: number;
  getLabel: (index: number) => string;
}

interface DrawContext {
  ctx: CanvasRenderingContext2D;
  height: number;
  animationState: AnimationState;
  activationRange: { min: number; max: number };
}

export interface OverlayCallbacks {
  drawConnections: (ctx: CanvasRenderingContext2D, nodes: NodePosition[][], animationState: AnimationState) => void;
  drawForwardOverlay?: (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, nodes: NodePosition[][], animationState: AnimationState) => void;
  drawBackwardOverlay?: (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, nodes: NodePosition[][], nn: NeuralNetwork, animationState: AnimationState) => void;
}

// =============================================================================
// Helper Functions
// =============================================================================

function getAnimatingNeuron(state: AnimationState): AnimationPhase | null {
  if (state.type === 'forward_animating' || state.type === 'backward_animating') {
    return { layer: state.layer, index: state.neuronIndex };
  }
  return null;
}

function isForwardMode(state: AnimationState): boolean {
  return state.type === 'forward_animating' || state.type === 'showing_loss_modal';
}

function isBackwardMode(state: AnimationState): boolean {
  return state.type === 'backward_animating' || state.type === 'showing_backprop_modal';
}

function drawLayerNeurons(config: LayerConfig, context: DrawContext): NodePosition[] {
  const { layerName, neurons, x, neuronCount, verticalSpacing, getLabel } = config;
  const { ctx, height, animationState, activationRange } = context;

  const nodes: NodePosition[] = [];
  const totalHeight = (neuronCount - 1) * verticalSpacing;
  const startY = (height - totalHeight) / 2;

  const animatingNeuron = getAnimatingNeuron(animationState);
  const isForward = isForwardMode(animationState);
  const isBackward = isBackwardMode(animationState);

  for (let i = 0; i < neuronCount; i++) {
    const neuron = neurons[i];
    const y = startY + i * verticalSpacing;
    const isAnimating = animatingNeuron?.layer === layerName && animatingNeuron.index === i;

    const node = drawNeuronVector(
      ctx, x, y,
      neuron.weights, neuron.bias, neuron.activated,
      getLabel(i), layerName,
      isAnimating && isForward || false,
      isAnimating && isBackward || false,
      activationRange
    );

    nodes.push(node);
  }

  return nodes;
}

// =============================================================================
// Main Export
// =============================================================================

/**
 * Draw the neural network structure and overlays based on animation state.
 */
export function drawNetwork(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  nn: NeuralNetwork,
  steps: ForwardSteps | null,
  inputLabels: string[],
  animationState: AnimationState,
  callbacks: OverlayCallbacks
): NodePosition[][] {
  const width = canvas.width;
  const height = canvas.height;

  // Clear canvas
  ctx.fillStyle = CANVAS_BACKGROUND;
  ctx.fillRect(0, 0, width, height);

  if (!steps) return [];

  const nodes: NodePosition[][] = [];

  // Calculate dynamic positions based on canvas width
  const paddingLeft = CANVAS_PADDING.left;
  const paddingRight = CANVAS_PADDING.right;
  const usableWidth = width - paddingLeft - paddingRight;

  const inputX = paddingLeft + 30;
  const layer1X = paddingLeft + usableWidth * 0.32;
  const layer2X = paddingLeft + usableWidth * 0.65;
  const outputX = width - paddingRight - 10;

  // Draw input layer
  const inputNode = drawInputVector(ctx, inputX, height / 2, steps.input, inputLabels);
  nodes.push([inputNode]);

  // Layer configurations
  const classNames = [i18n.t('classes.fail'), i18n.t('classes.pending'), i18n.t('classes.pass')];
  const layerConfigs = [
    { name: 'layer1' as const, data: steps.layer1, x: layer1X, getLabel: (i: number) => `${i18n.t('layers.layer1Prefix')} #${i + 1}` },
    { name: 'layer2' as const, data: steps.layer2, x: layer2X, getLabel: (i: number) => `${i18n.t('layers.layer2Prefix')} #${i + 1}` },
    { name: 'output' as const, data: steps.output, x: outputX, getLabel: (i: number) => classNames[i] },
  ];

  // Draw all layers
  layerConfigs.forEach(({ name, data, x, getLabel }) => {
    const activations = data.map(n => n.activated);
    const layerNodes = drawLayerNeurons({
      layerName: name,
      neurons: data,
      x,
      neuronCount: LAYER_SIZES[name],
      verticalSpacing: VERTICAL_SPACING[name],
      getLabel,
    }, {
      ctx, height, animationState,
      activationRange: { min: Math.min(...activations), max: Math.max(...activations) },
    });
    nodes.push(layerNodes);
  });

  // Draw connections
  callbacks.drawConnections(ctx, nodes, animationState);

  // Draw overlays based on animation state
  const { type } = animationState;

  if (type === 'forward_animating' && callbacks.drawForwardOverlay) {
    callbacks.drawForwardOverlay(ctx, canvas, nodes, animationState);
  }

  if ((type === 'backward_animating' || type === 'showing_backprop_modal') && callbacks.drawBackwardOverlay) {
    callbacks.drawBackwardOverlay(ctx, canvas, nodes, nn, animationState);
  }

  return nodes;
}
