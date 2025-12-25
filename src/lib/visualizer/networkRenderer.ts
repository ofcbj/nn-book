// Network rendering module
import type { CalculationSteps, NodePosition, AnimationPhase, NeuronCalculation, CalculationStage } from '../types';
import type { NeuralNetwork } from '../core';
import { drawInputVector, drawNeuronVector } from './drawingUtils';
import i18n from '../../i18n';



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
  drawCalculationOverlay: ((ctx: CanvasRenderingContext2D, x: number, y: number, stage: any, neuronData: NeuronCalculation | null) => void) | null,
  currentNeuronData: NeuronCalculation | null,
  heatmapMode: boolean = false
): NodePosition[][] {
  const width = canvas.width;
  const height = canvas.height;

  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(0, 0, width, height);

  if (!steps) return [];

  const nodes: NodePosition[][] = [];

  // Calculate dynamic positions based on canvas width
  // We have 4 layers: input, layer1, layer2, output
  const paddingLeft = 60;
  const paddingRight = 80;
  const usableWidth = width - paddingLeft - paddingRight;

  // Position layers with better spacing
  const inputX = paddingLeft + 30;
  const layer1X = paddingLeft + usableWidth * 0.32;
  const layer2X = paddingLeft + usableWidth * 0.65;
  const outputX = width - paddingRight - 10;

  const inputNode = drawInputVector(ctx, inputX, height / 2, steps.input, inputLabels);
  nodes.push([inputNode]);

  // Layer 1 - 5 neurons with tighter spacing
  const layer1Nodes: NodePosition[] = [];
  const layer1VerticalSpacing = 105;
  const layer1TotalHeight = (5 - 1) * layer1VerticalSpacing;
  const layer1StartY = (height - layer1TotalHeight) / 2;
  for (let i = 0; i < 5; i++) {
    const neuron = steps.layer1[i];
    const isHighlighted = highlightedNeuron &&
                          highlightedNeuron.layer === 'layer1' &&
                          highlightedNeuron.index === i;
    const isBackpropHighlighted = backpropPhase &&
                                   backpropPhase.layer === 'layer1' &&
                                   backpropPhase.index === i;
    const node = drawNeuronVector(
      ctx,
      layer1X,
      layer1StartY + i * layer1VerticalSpacing,
      neuron.weights,
      neuron.bias,
      neuron.activated,
      `${i18n.t('layers.layer1Prefix')} #${i + 1}`,
      'layer1',
      isHighlighted || false,
      isBackpropHighlighted || false,
      heatmapMode
    );
    
    // Show calculation overlay for highlighted neuron
    if (highlightedNeuron && highlightedNeuron.layer === 'layer1' && highlightedNeuron.index === i && 
        calculationStage && currentNeuronData && drawCalculationOverlay) {
      drawCalculationOverlay(ctx, layer1X, layer1StartY + i * layer1VerticalSpacing, calculationStage as any, currentNeuronData);
    }
    
    layer1Nodes.push(node);
  }
  nodes.push(layer1Nodes);

  // Layer 2 - 3 neurons with better spacing
  const layer2Nodes: NodePosition[] = [];
  const layer2VerticalSpacing = 125;
  const layer2TotalHeight = (3 - 1) * layer2VerticalSpacing;
  const layer2StartY = (height - layer2TotalHeight) / 2;
  for (let i = 0; i < 3; i++) {
    const neuron = steps.layer2[i];
    const isHighlighted = highlightedNeuron &&
                          highlightedNeuron.layer === 'layer2' &&
                          highlightedNeuron.index === i;
    const isBackpropHighlighted = backpropPhase &&
                                   backpropPhase.layer === 'layer2' &&
                                   backpropPhase.index === i;
    const node = drawNeuronVector(
      ctx,
      layer2X,
      layer2StartY + i * layer2VerticalSpacing,
      neuron.weights,
      neuron.bias,
      neuron.activated,
      `${i18n.t('layers.layer2Prefix')} #${i + 1}`,
      'layer2',
      isHighlighted || false,
      isBackpropHighlighted || false,
      heatmapMode
    );
    
    // Show calculation overlay for highlighted neuron
    if (highlightedNeuron && highlightedNeuron.layer === 'layer2' && highlightedNeuron.index === i && 
        calculationStage && currentNeuronData && drawCalculationOverlay) {
      drawCalculationOverlay(ctx, layer2X, layer2StartY + i * layer2VerticalSpacing, calculationStage as any, currentNeuronData);
    }
    
    layer2Nodes.push(node);
  }
  nodes.push(layer2Nodes);

  // Output layer - 3 neurons with better spacing
  const outputNodes: NodePosition[] = [];
  const classNames = [i18n.t('classes.fail'), i18n.t('classes.pending'), i18n.t('classes.pass')];
  const outputVerticalSpacing = 125;
  const outputTotalHeight = (3 - 1) * outputVerticalSpacing;
  const outputStartY = (height - outputTotalHeight) / 2;

  for (let i = 0; i < 3; i++) {
    const output = steps.output[i];
    const isHighlighted = highlightedNeuron &&
                          highlightedNeuron.layer === 'output' &&
                          highlightedNeuron.index === i;
    const isBackpropHighlighted = backpropPhase &&
                                   backpropPhase.layer === 'output' &&
                                   backpropPhase.index === i;
    const outputNode = drawNeuronVector(
      ctx,
      outputX,
      outputStartY + i * outputVerticalSpacing,
      output.weights,
      output.bias,
      output.activated,
      classNames[i],
      'output',
      isHighlighted || false,
      isBackpropHighlighted || false,
      heatmapMode
    );
    
    // Show calculation overlay for highlighted neuron
    if (highlightedNeuron && highlightedNeuron.layer === 'output' && highlightedNeuron.index === i && 
        calculationStage && currentNeuronData && drawCalculationOverlay) {
      drawCalculationOverlay(ctx, outputX, outputStartY + i * outputVerticalSpacing, calculationStage as any, currentNeuronData);
    }
    
    outputNodes.push(outputNode);
  }
  nodes.push(outputNodes);

  drawConnectionsVector(ctx, nodes, nn);

  if (drawLossOverlay) {
    drawLossOverlay(ctx, width, height);
  }

  if (drawBackpropHighlight) {
    drawBackpropHighlight(ctx, nodes);
  }

  return nodes;
}
