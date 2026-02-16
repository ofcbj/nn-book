// Network rendering module - draws the base neural network structure
import type { ForwardSteps, NodePosition, AnimationPhase, ForwardCalculation, LayerType, BackwardCalculation } from '../types';
import type { AnimationState } from '../animation';
import { checkMode } from '../animation';
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

    // Get backprop update data if in backward mode
    let backpropUpdateData: BackpropUpdateData | undefined;
    // Use oldWeights/oldBias from backprop data when in backward mode
    // because the network's weights are already updated after train()
    let displayWeights = neuron.weights;
    let displayBias = neuron.bias;
    
    if (isBackward && backpropData && backpropData[i]) {
      backpropUpdateData = {
        newWeights: backpropData[i].newWeights,
        newBias: backpropData[i].newBias,
      };
      // Override display values with old weights/bias from backprop data
      displayWeights = backpropData[i].oldWeights;
      displayBias = backpropData[i].oldBias;
    }

    const node = drawNeuronVector(
      ctx, x, y,
      displayWeights, displayBias, neuron.activated,
      getLabel(i), layerName,
      isAnimating && isForward || false,
      isAnimating && isBackward || false,
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

  // Get backprop data if in backward mode
  const isBackward = checkMode(animationState, 'backward');
  const backwardSteps = isBackward ? nn.lastBackwardSteps : null;

  // Draw all layers
  layerConfigs.forEach(({ name, data, x, getLabel }) => {
    const activations = data.map(n => n.activated);
    
    // Get backprop data for this layer
    const backpropData = backwardSteps ? backwardSteps[name] : undefined;
    
    const layerNodes = drawLayerNeurons({
      layerName: name,
      neurons: data,
      x,
      neuronCount: LAYER_SIZES[name],
      verticalSpacing: VERTICAL_SPACING[name],
      getLabel,
      backpropData,
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
