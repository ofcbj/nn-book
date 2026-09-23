// Network rendering module - draws the base neural network structure
import type { ForwardSteps, NodePosition, ForwardCalculation, LayerType, BackwardCalculation, Viewport } from '../types';
import type { AnimationState } from '../animation';
import { checkMode, getAnimatingNeuron } from '../animation';
import type { NeuralNetwork, LayerName } from '../core';
import { drawInputVector, drawNeuronVector, neuronBoxWidth, type BackpropUpdateData } from './drawingUtils';
import type { LayerWeights } from './connectionRenderer';
import { LAYER_SIZES, LAYER_NAMES, INPUT_SIZE, BACKWARD_LAYER_ORDER } from '../core';
import { CANVAS_BACKGROUND, CANVAS_PADDING, VERTICAL_SPACING, INPUT_BOX, MIN_COLUMN_GAP } from './uiConfig';
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
  /** Whether backprop has already reached this neuron's update stage */
  isUpdated        : (index: number) => boolean;
}

interface DrawContext {
  ctx            : CanvasRenderingContext2D;
  height         : number;
  animationState : AnimationState;
  activationRange: { min: number; max: number };
  /** < 1 when the canvas is too narrow for the natural box widths */
  widthScale     : number;
}

export interface OverlayCallbacks {
  drawConnections: (ctx: CanvasRenderingContext2D, nodes: NodePosition[][], animationState: AnimationState, weights: LayerWeights) => void;
  drawForwardOverlay?: (ctx: CanvasRenderingContext2D, viewport: Viewport, nodes: NodePosition[][], animationState: AnimationState) => void;
  drawBackwardOverlay?: (ctx: CanvasRenderingContext2D, viewport: Viewport, nodes: NodePosition[][], nn: NeuralNetwork, animationState: AnimationState) => void;
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Has backprop already applied this neuron's weight update?
 * Backprop walks output → layer2 → layer1, each layer from its last neuron to its first,
 * and a neuron's weights flip to their new values at its 'update' stage.
 */
function makeIsUpdated(animationState: AnimationState): (layer: LayerName, index: number) => boolean {
  if (animationState.type === 'showing_backprop_modal') return () => true;
  if (animationState.type !== 'backward_animating') return () => false;
  const { layer: currentLayer, neuronIndex, stage } = animationState;
  const currentPos = BACKWARD_LAYER_ORDER.indexOf(currentLayer);
  return (layer, index) => {
    const pos = BACKWARD_LAYER_ORDER.indexOf(layer);
    if (pos < currentPos) return true;
    if (pos > currentPos) return false;
    if (index > neuronIndex) return true;
    return index === neuronIndex && stage === 'update';
  };
}

function drawLayerNeurons(config: LayerConfig, context: DrawContext): NodePosition[] {
  const { layerName, neurons, x, neuronCount, verticalSpacing, getLabel, backpropData, isUpdated } = config;
  const { ctx, height, animationState, activationRange, widthScale } = context;

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

    // During backprop the network already holds the new weights. Show the pre-update
    // values, and reveal "→ new" only once backprop has actually reached this neuron's
    // update stage, so the changes appear neuron by neuron in backward order.
    let backpropUpdateData: BackpropUpdateData | undefined;
    let displayWeights = neuron.weights;
    let displayBias = neuron.bias;

    if (isBackward && backpropData && backpropData[i]) {
      displayWeights = backpropData[i].oldWeights;
      displayBias = backpropData[i].oldBias;
      if (isUpdated(i)) {
        backpropUpdateData = {
          newWeights: backpropData[i].newWeights,
          newBias: backpropData[i].newBias,
        };
      }
    }

    const node = drawNeuronVector(
      ctx, x, y,
      displayWeights, displayBias, neuron.activated,
      getLabel(i), layerName,
      isAnimating && isForward,
      isAnimating && isBackward,
      activationRange,
      backpropUpdateData,
      widthScale
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

  // Column layout: the four columns have different box widths, so place them by
  // their edges with one equal gap between neighbours (centred within the canvas).
  // If even the minimum gaps do not fit, scale every box width down together.
  const usableWidth = width - CANVAS_PADDING.left - CANVAS_PADDING.right;
  const naturalWidths = [
    INPUT_BOX.width,
    neuronBoxWidth('layer1', INPUT_SIZE),
    neuronBoxWidth('layer2', LAYER_SIZES.layer1),
    neuronBoxWidth('output', LAYER_SIZES.layer2),
  ];
  const gapCount = naturalWidths.length - 1;
  const naturalTotal = naturalWidths.reduce((sum, w) => sum + w, 0);
  const widthScale = Math.min(1, Math.max(0.6, (usableWidth - MIN_COLUMN_GAP * gapCount) / naturalTotal));
  const columnWidths = naturalWidths.map(w => Math.round(w * widthScale));
  const totalBoxWidth = columnWidths.reduce((sum, w) => sum + w, 0);
  const gap = Math.max(MIN_COLUMN_GAP, (usableWidth - totalBoxWidth) / gapCount);
  const contentWidth = totalBoxWidth + gap * (columnWidths.length - 1);
  let cursor = Math.max(CANVAS_PADDING.left, (width - contentWidth) / 2);
  const columnCenters = columnWidths.map(w => {
    const center = cursor + w / 2;
    cursor += w + gap;
    return center;
  });
  const [inputX, layer1X, layer2X, outputX] = columnCenters;

  // Draw input layer (one box per value)
  nodes.push(drawInputVector(ctx, inputX, height / 2, steps.input, inputLabels, widthScale));

  // Layer configurations
  const classNames = [i18n.t('classes.fail'), i18n.t('classes.pending'), i18n.t('classes.pass')];
  const layerConfigs = [
    { name: 'layer1' as const, data: steps.layer1, x: layer1X, getLabel: (i: number) => `${i18n.t('layers.layer1Prefix')} #${i + 1}` },
    { name: 'layer2' as const, data: steps.layer2, x: layer2X, getLabel: (i: number) => `${i18n.t('layers.layer2Prefix')} #${i + 1}` },
    { name: 'output' as const, data: steps.output, x: outputX, getLabel: (i: number) => classNames[i] },
  ];

  // Get backprop data if in backward mode
  const backwardSteps = checkMode(animationState, 'backward') ? nn.lastBackwardSteps : null;
  const isUpdated = makeIsUpdated(animationState);

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
      isUpdated: (i) => isUpdated(name, i),
    }, {
      ctx, height, animationState,
      activationRange: { min: Math.min(...activations), max: Math.max(...activations) },
      widthScale,
    });
    nodes.push(layerNodes);
  });

  // Draw connections. During backprop the network already holds the updated
  // weights, so a neuron's incoming lines switch from its old to its new weights
  // at the moment its update is shown.
  const connectionWeights = {} as LayerWeights;
  for (const name of LAYER_NAMES) {
    connectionWeights[name] = backwardSteps
      ? backwardSteps[name].map((n, i) => (isUpdated(name, i) ? n.newWeights : n.oldWeights))
      : steps[name].map(n => n.weights);
  }
  callbacks.drawConnections(ctx, nodes, animationState, connectionWeights);

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
