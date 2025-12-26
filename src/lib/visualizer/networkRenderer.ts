// Network rendering module
import type { CalculationSteps, NodePosition, AnimationPhase, NeuronCalculation, CalculationStage, LayerType } from '../types';
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
  highlightedNeuron: AnimationPhase | null;
  backpropPhase: AnimationPhase | null;
  calculationStage: CalculationStage | null;
  currentNeuronData: NeuronCalculation | null;
  drawCalculationOverlay: ((ctx: CanvasRenderingContext2D, x: number, y: number, stage: CalculationStage, neuronData: NeuronCalculation | null) => void) | null;
  heatmapMode: boolean;
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Draw a layer of neurons with common logic
 */
function drawLayerNeurons(config: LayerConfig, context: DrawContext): NodePosition[] {
  const { layerName, neurons, x, neuronCount, verticalSpacing, getLabel } = config;
  const { ctx, height, highlightedNeuron, backpropPhase, calculationStage, currentNeuronData, drawCalculationOverlay, heatmapMode } = context;

  const nodes: NodePosition[] = [];
  const totalHeight = (neuronCount - 1) * verticalSpacing;
  const startY = (height - totalHeight) / 2;

  for (let i = 0; i < neuronCount; i++) {
    const neuron = neurons[i];
    const y = startY + i * verticalSpacing;

    const isHighlighted = highlightedNeuron?.layer === layerName && highlightedNeuron.index === i;
    const isBackpropHighlighted = backpropPhase?.layer === layerName && backpropPhase.index === i;

    const node = drawNeuronVector(
      ctx,
      x,
      y,
      neuron.weights,
      neuron.bias,
      neuron.activated,
      getLabel(i),
      layerName,
      isHighlighted || false,
      isBackpropHighlighted || false,
      heatmapMode
    );

    // Show calculation overlay for highlighted neuron
    if (isHighlighted && calculationStage && currentNeuronData && drawCalculationOverlay) {
      drawCalculationOverlay(ctx, x, y, calculationStage, currentNeuronData);
    }

    nodes.push(node);
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
  steps: CalculationSteps | null,
  inputLabels: string[],
  highlightedNeuron: AnimationPhase | null,
  backpropPhase: AnimationPhase | null,
  calculationStage: CalculationStage | null,
  drawConnectionsVector: (ctx: CanvasRenderingContext2D, nodes: NodePosition[][], nn: NeuralNetwork) => void,
  drawLossOverlay: ((ctx: CanvasRenderingContext2D, width: number, height: number) => void) | null,
  drawBackpropHighlight: ((ctx: CanvasRenderingContext2D, nodes: NodePosition[][]) => void) | null,
  drawCalculationOverlay: ((ctx: CanvasRenderingContext2D, x: number, y: number, stage: CalculationStage, neuronData: NeuronCalculation | null) => void) | null,
  currentNeuronData: NeuronCalculation | null,
  heatmapMode: boolean = false
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
  const drawContext: DrawContext = {
    ctx,
    height,
    highlightedNeuron,
    backpropPhase,
    calculationStage,
    currentNeuronData,
    drawCalculationOverlay,
    heatmapMode,
  };

  // Input layer
  const inputNode = drawInputVector(ctx, inputX, height / 2, steps.input, inputLabels);
  nodes.push([inputNode]);

  // Layer 1 - 5 neurons
  const layer1Nodes = drawLayerNeurons({
    layerName: 'layer1',
    neurons: steps.layer1,
    x: layer1X,
    neuronCount: LAYER_SIZES.layer1,
    verticalSpacing: VERTICAL_SPACING.layer1,
    getLabel: (i) => `${i18n.t('layers.layer1Prefix')} #${i + 1}`,
  }, drawContext);
  nodes.push(layer1Nodes);

  // Layer 2 - 3 neurons
  const layer2Nodes = drawLayerNeurons({
    layerName: 'layer2',
    neurons: steps.layer2,
    x: layer2X,
    neuronCount: LAYER_SIZES.layer2,
    verticalSpacing: VERTICAL_SPACING.layer2,
    getLabel: (i) => `${i18n.t('layers.layer2Prefix')} #${i + 1}`,
  }, drawContext);
  nodes.push(layer2Nodes);

  // Output layer - 3 neurons
  const classNames = [i18n.t('classes.fail'), i18n.t('classes.pending'), i18n.t('classes.pass')];
  const outputNodes = drawLayerNeurons({
    layerName: 'output',
    neurons: steps.output,
    x: outputX,
    neuronCount: LAYER_SIZES.output,
    verticalSpacing: VERTICAL_SPACING.output,
    getLabel: (i) => classNames[i],
  }, drawContext);
  nodes.push(outputNodes);

  // Draw connections between layers
  drawConnectionsVector(ctx, nodes, nn);

  // Draw overlays
  if (drawLossOverlay) {
    drawLossOverlay(ctx, width, height);
  }

  if (drawBackpropHighlight) {
    drawBackpropHighlight(ctx, nodes);
  }

  return nodes;
}
