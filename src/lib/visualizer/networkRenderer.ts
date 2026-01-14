// Network rendering module
import type { ForwardSteps, NodePosition, AnimationPhase, NeuronCalculation, ForwardStage, LayerType } from '../types';
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
  neurons: NeuronCalculation[];
  x: number;
  neuronCount: number;
  verticalSpacing: number;
  getLabel: (index: number) => string;
}

interface DrawContext {
  ctx: CanvasRenderingContext2D;
  height: number;
  animationState: AnimationState;
  drawCalculationOverlay: ((animationState: AnimationState, ctx: CanvasRenderingContext2D, x: number, y: number) => void) | null;
  activationRange: { min: number; max: number };
}

// Helper functions to extract data from AnimationState
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

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Draw a layer of neurons with common logic
 */
function drawLayerNeurons(config: LayerConfig, context: DrawContext): NodePosition[] {
  const { layerName, neurons, x, neuronCount, verticalSpacing, getLabel } = config;
  const { ctx, height, animationState, drawCalculationOverlay, activationRange } = context;

  const nodes: NodePosition[] = [];
  const totalHeight = (neuronCount - 1) * verticalSpacing;
  const startY = (height - totalHeight) / 2;

  const animatingNeuron = getAnimatingNeuron(animationState);
  const isForward = isForwardMode(animationState);
  const isBackward = isBackwardMode(animationState);

  for (let i = 0; i < neuronCount; i++) {
    const neuron = neurons[i];
    const y = startY + i * verticalSpacing;

    // Check if this neuron is currently animating
    const isAnimating = animatingNeuron?.layer === layerName && animatingNeuron.index === i;

    const node = drawNeuronVector(
      ctx,
      x,
      y,
      neuron.weights,
      neuron.bias,
      neuron.activated,
      getLabel(i),
      layerName,
      isAnimating && isForward || false,  // highlighted only in forward mode
      isAnimating && isBackward || false,  // backprop highlighted only in backward mode
      activationRange
    );

    nodes.push(node);

    // Show calculation overlay for forward animation only
    if (isAnimating && isForward && drawCalculationOverlay) {
      drawCalculationOverlay(animationState, ctx, node.centerX, node.centerY);
    }
  }

  return nodes;
}

// =============================================================================
// Main Export
// =============================================================================

export function drawNetwork(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  nn: NeuralNetwork,
  steps: ForwardSteps | null,
  inputLabels: string[],
  animationState: AnimationState,
  drawConnectionsVector: (animationState: AnimationState, ctx: CanvasRenderingContext2D, nodes: NodePosition[][], nn: NeuralNetwork) => void,
  drawLossOverlay: ((ctx: CanvasRenderingContext2D, width: number, height: number) => void) | null,
  drawBackpropHighlight: ((nodes: NodePosition[][], nn: NeuralNetwork) => void) | null,
  drawCalculationOverlay: ((animationState: AnimationState, ctx: CanvasRenderingContext2D, x: number, y: number) => void) | null,
  nnForBackprop: NeuralNetwork
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

  // Shared drawing context
  const drawContext: Omit<DrawContext, 'activationRange'> = {
    ctx,
    height,
    animationState,
    drawCalculationOverlay,
  };

  // Input layer
  const inputNode = drawInputVector(ctx, inputX, height / 2, steps.input, inputLabels);
  nodes.push([inputNode]);

  // Hidden and output layers configuration
  const classNames = [i18n.t('classes.fail'), i18n.t('classes.pending'), i18n.t('classes.pass')];
  
  const layerConfigs = [
    { 
      name: 'layer1' as const, 
      data: steps.layer1, 
      x: layer1X, 
      getLabel: (i: number) => `${i18n.t('layers.layer1Prefix')} #${i + 1}` 
    },
    { 
      name: 'layer2' as const, 
      data: steps.layer2, 
      x: layer2X, 
      getLabel: (i: number) => `${i18n.t('layers.layer2Prefix')} #${i + 1}` 
    },
    { 
      name: 'output' as const, 
      data: steps.output, 
      x: outputX, 
      getLabel: (i: number) => classNames[i] 
    }
  ];

  // Draw all layers
  layerConfigs.forEach(({ name, data, x, getLabel }) => {
    // Calculate activation range for this layer
    const activations = data.map(n => n.activated);
    const minActivation = Math.min(...activations);
    const maxActivation = Math.max(...activations);

    const layerNodes = drawLayerNeurons({
      layerName: name,
      neurons: data,
      x,
      neuronCount: LAYER_SIZES[name],
      verticalSpacing: VERTICAL_SPACING[name],
      getLabel,
    }, {
      ...drawContext,
      activationRange: { min: minActivation, max: maxActivation }
    });
    nodes.push(layerNodes);
  });

  // Draw connections between layers
  drawConnectionsVector(animationState, ctx, nodes, nn);

  // Draw overlays
  if (drawLossOverlay) {
    drawLossOverlay(ctx, width, height);
  }

  if (drawBackpropHighlight) {
    drawBackpropHighlight(nodes, nnForBackprop);
  }

  return nodes;
}
