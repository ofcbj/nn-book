// Network rendering module
import type { ForwardSteps, NodePosition, AnimationPhase, NeuronCalculation, ForwardStage, LayerType } from '../types';
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
  ForwardStage: ForwardStage | null;
  currentNeuronData: NeuronCalculation | null;
  drawCalculationOverlay: ((ctx: CanvasRenderingContext2D, x: number, y: number, stage: ForwardStage, neuronData: NeuronCalculation | null) => void) | null;
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
  const { ctx, height, highlightedNeuron, backpropPhase, ForwardStage, currentNeuronData, drawCalculationOverlay, heatmapMode } = context;

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
    if (isHighlighted && ForwardStage && currentNeuronData && drawCalculationOverlay) {
      drawCalculationOverlay(ctx, x, y, ForwardStage, currentNeuronData);
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
  steps: ForwardSteps | null,
  inputLabels: string[],
  highlightedNeuron: AnimationPhase | null,
  backpropPhase: AnimationPhase | null,
  ForwardStage: ForwardStage | null,
  drawConnectionsVector: (ctx: CanvasRenderingContext2D, nodes: NodePosition[][], nn: NeuralNetwork) => void,
  drawLossOverlay: ((ctx: CanvasRenderingContext2D, width: number, height: number) => void) | null,
  drawBackpropHighlight: ((ctx: CanvasRenderingContext2D, nodes: NodePosition[][]) => void) | null,
  drawCalculationOverlay: ((ctx: CanvasRenderingContext2D, x: number, y: number, stage: ForwardStage, neuronData: NeuronCalculation | null) => void) | null,
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
    ForwardStage,
    currentNeuronData,
    drawCalculationOverlay,
    heatmapMode,
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
    const layerNodes = drawLayerNeurons({
      layerName: name,
      neurons: data,
      x,
      neuronCount: LAYER_SIZES[name],
      verticalSpacing: VERTICAL_SPACING[name],
      getLabel,
    }, drawContext);
    nodes.push(layerNodes);
  });

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
