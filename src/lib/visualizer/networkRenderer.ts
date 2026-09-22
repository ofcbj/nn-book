// Network rendering module - draws the base neural network structure
import type { ForwardSteps, NodePosition, ForwardCalculation, LayerType, BackwardCalculation, Viewport } from '../types';
import type { AnimationState } from '../animation';
import { checkMode, getAnimatingNeuron } from '../animation';
import type { NeuralNetwork } from '../core';
import { LAYER_SIZES } from '../core';
import { drawInputVector, drawNeuronVector, type BackpropUpdateData } from './drawingUtils';
import { CANVAS_BACKGROUND, CANVAS_PADDING, VERTICAL_SPACING } from './uiConfig';
import i18n from '../../i18n';

// =============================================================================
// Types
// =============================================================================

interface LayerConfig {
  layerName        : LayerType;
  neurons          : ForwardCalculation[];
  x                : number;
  neuronCount      : number;
  verticalSpacing  : number;
  getLabel         : (index: number) => string;
  backpropData?    : BackwardCalculation[];  // Backprop data for displaying weight/bias updates
}

interface DrawContext {
  ctx            : CanvasRenderingContext2D;
  height         : number;
  animationState : AnimationState;
  activationRange: { min: number; max: number };
}

export interface OverlayCallbacks {
  drawConnections: (ctx: CanvasRenderingContext2D, nodes: NodePosition[][], animationState: AnimationState) => void;
  drawForwardOverlay?: (ctx: CanvasRenderingContext2D, viewport: Viewport, nodes: NodePosition[][], animationState: AnimationState) => void;
  drawBackwardOverlay?: (ctx: CanvasRenderingContext2D, viewport: Viewport, nodes: NodePosition[][], nn: NeuralNetwork, animationState: AnimationState) => void;
}

// =============================================================================
// Helper Functions
// =============================================================================

function drawLayerNeurons(config: LayerConfig, context: DrawContext): NodePosition[] {
  const { layerName, neurons, x, neuronCount, verticalSpacing, getLabel, backpropData } = config;
  const { ctx, height, animationState, activationRange } = context;

  const nodes: NodePosition[] = [];
  const totalHeight = (neuronCount - 1) * verticalSpacing;
  const startY = (height - totalHeight) / 2;

  const animatingNeuron = getAnimatingNeuron(animationState);
  const isForward   = checkMode(animationState, 'forward');
  const isBackward  = checkMode(animationState, 'backward');

  for (let i = 0; i < neuronCount; i++) {
    const neuron = neurons[i];
    const y = startY + i * verticalSpacing;
    const isAnimating = animatingNeuron?.layer === layerName && animatingNeuron.index === i;

    // During backprop the network's weights are already updated, so show the
    // pre-update values from the backprop data and the new values beneath them.
    let backpropUpdateData: BackpropUpdateData | undefined;
    let displayWeights = neuron.weights;
    let displayBias = neuron.bias;

    if (isBackward && backpropData && backpropData[i]) {
      backpropUpdateData = {
        newWeights: backpropData[i].newWeights,
        newBias: backpropData[i].newBias,
      };
      displayWeights = backpropData[i].oldWeights;
      displayBias = backpropData[i].oldBias;
    }

    const node = drawNeuronVector(
      ctx, x, y,
      displayWeights, displayBias, neuron.activated,
      getLabel(i), layerName,
      isAnimating && isForward,
      isAnimating && isBackward,
      activationRange,
      backpropUpdateData
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
 * Returns node positions per layer: [input, layer1, layer2, output].
 */
export function drawNetwork(
  ctx: CanvasRenderingContext2D,
  viewport: Viewport,
  nn: NeuralNetwork,
  steps: ForwardSteps | null,
  inputLabels: string[],
  animationState: AnimationState,
  callbacks: OverlayCallbacks
): NodePosition[][] {
  const { width, height } = viewport;

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

  // Get backprop data if in backward mode
  const backwardSteps = checkMode(animationState, 'backward') ? nn.lastBackwardSteps : null;

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
      backpropData: backwardSteps ? backwardSteps[name] : undefined,
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
    callbacks.drawForwardOverlay(ctx, viewport, nodes, animationState);
  }

  if ((type === 'backward_animating' || type === 'showing_backprop_modal') && callbacks.drawBackwardOverlay) {
    callbacks.drawBackwardOverlay(ctx, viewport, nodes, nn, animationState);
  }

  return nodes;
}
