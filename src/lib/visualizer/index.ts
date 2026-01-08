// Visualizer for React - Canvas-based visualizer
// Modified to work with React refs instead of direct DOM queries

import type { ForwardSteps, NeuronCalculation, AnimationPhase, ForwardStage, NodePosition, LossDisplayData, BackpropNeuronData, BackpropStage, BackpropSteps } from '../types';
import type { NeuralNetwork, LayerName } from '../core';
import i18n from '../../i18n';
import { activationToColor } from './activationColors';
import { drawBackpropHighlight } from './backpropRenderer';
import { drawNetwork } from './networkRenderer';
import { drawCalculationOverlay as drawCalcOverlay } from './calculationOverlay';

export class Visualizer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  get inputLabels(): string[] {
    return [i18n.t('controls.grade'), i18n.t('controls.attitude'), i18n.t('controls.response')];
  }
  highlightedNeuron: AnimationPhase | null = null;

  // Calculation animation properties
  ForwardStage: ForwardStage | null = null;
  activeConnections: number[] = [];
  currentNeuronData: NeuronCalculation | null = null;

  // Backpropagation visualizer
  showLoss: LossDisplayData | null = null;
  backpropPhase: AnimationPhase | null = null;
  currentBackpropData: BackpropNeuronData | null = null;
  backpropStage: BackpropStage | null = null;
  allBackpropData: BackpropSteps | null = null;  // All backprop data for persistent error labels

  // Store node positions for click detection
  private lastNodes: NodePosition[][] = [];

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Failed to get 2D rendering context');
    }
    this.ctx = ctx;
    this.resizeCanvas();
  }

  resizeCanvas(): void {
    this.canvas.width = this.canvas.offsetWidth;
    this.canvas.height = this.canvas.offsetHeight;
  }

  /**
   * Set visualizer state for forward propagation animation.
   */
  setForwardAnimationState(
    layer: AnimationPhase['layer'],
    index: number,
    stage: ForwardStage,
    neuronData: NeuronCalculation | null
  ): void {
    this.highlightedNeuron = { layer, index };
    this.ForwardStage = stage;
    this.currentNeuronData = neuronData;
    this.backpropPhase = null;
    this.currentBackpropData = null;
    this.backpropStage = null;
    this.allBackpropData = null;
  }

  /**
   * Set visualizer state for backward propagation animation.
   */
  setBackwardAnimationState(
    layer: AnimationPhase['layer'],
    index: number,
    stage: BackpropStage,
    neuronData: BackpropNeuronData | null,
    allBackpropData: BackpropSteps | null
  ): void {
    this.highlightedNeuron = null;
    this.ForwardStage = null;
    this.currentNeuronData = null;
    this.backpropPhase = { layer, index };
    this.currentBackpropData = neuronData;
    this.backpropStage = stage;
    this.allBackpropData = allBackpropData;
  }

  /**
   * Clear all animation state.
   */
  clearAnimationState(): void {
    this.highlightedNeuron = null;
    this.ForwardStage = null;
    this.currentNeuronData = null;
    this.backpropPhase = null;
    this.currentBackpropData = null;
    this.backpropStage = null;
    this.allBackpropData = null;
  }

  private drawCalculationOverlay(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    stage: ForwardStage,
    neuronData: NeuronCalculation | null
  ): void {
    drawCalcOverlay(ctx, this.canvas, x, y, stage, neuronData);
  }

  drawNetwork(nn: NeuralNetwork, steps: ForwardSteps | null): void {
    const nodes = drawNetwork(
      this.ctx,
      this.canvas,
      nn,
      steps,
      this.inputLabels,
      this.highlightedNeuron,
      this.backpropPhase,
      this.ForwardStage,
      this.drawConnectionsVector.bind(this),
      this.showLoss ? this.drawLossOverlay.bind(this) : null,
      this.backpropPhase ? this.drawBackpropHighlight.bind(this) : null,
      this.drawCalculationOverlay.bind(this),
      this.currentNeuronData
    );
    // Store nodes for click detection
    this.lastNodes = nodes;
  }

  private drawConnectionsVector(
    ctx: CanvasRenderingContext2D, 
    nodes: NodePosition[][], 
    _nn: NeuralNetwork
  ): void {
    // Color themes for each layer connection
    const colors = {
      input: { active: 'rgba(96, 165, 250, 0.9)', shadow: 'rgba(96, 165, 250, 0.8)' },
      layer1: { active: 'rgba(52, 211, 153, 0.9)', shadow: 'rgba(52, 211, 153, 0.8)' },
      layer2: { active: 'rgba(251, 146, 60, 0.9)', shadow: 'rgba(251, 146, 60, 0.8)' }
    };
    const inactiveColor = 'rgba(100, 116, 139, 0.4)';

    // Define connections: from → to with counts
    const connections = [
      { from: 'input', to: 'layer1', fromCount: 1, toCount: 5, theme: colors.input },
      { from: 'layer1', to: 'layer2', fromCount: 5, toCount: 3, theme: colors.layer1 },
      { from: 'layer2', to: 'output', fromCount: 3, toCount: 3, theme: colors.layer2 }
    ] as const;

    // Draw all connections
    connections.forEach(({ from, to, fromCount, toCount, theme }, idx) => {
      this.drawLayerConnections(ctx, nodes, {
        fromLayerIdx: idx,
        toLayerIdx: idx + 1,
        fromLayer: from,
        toLayer: to,
        fromCount,
        toCount,
        activeColor: theme.active,
        activeShadow: theme.shadow,
        inactiveColor
      });
    });
  }

  /**
   * Helper: Draw connections between two layers
   */
  private drawLayerConnections(
    ctx: CanvasRenderingContext2D,
    nodes: NodePosition[][],
    config: {
      fromLayerIdx: number;
      toLayerIdx: number;
      fromLayer: string;
      toLayer: string;
      fromCount: number;
      toCount: number;
      activeColor: string;
      activeShadow: string;
      inactiveColor: string;
    }
  ): void {
    const { fromLayerIdx, toLayerIdx, fromLayer, toLayer, fromCount, toCount, activeColor, activeShadow, inactiveColor } = config;

    for (let i = 0; i < fromCount; i++) {
      for (let j = 0; j < toCount; j++) {
        const from = nodes[fromLayerIdx][i];
        const to = nodes[toLayerIdx][j];
        
        const isActive = this.isConnectionActive(fromLayer, i, toLayer, j);
        
        ctx.beginPath();
        ctx.moveTo(from.centerX + from.width / 2, from.centerY);
        ctx.lineTo(to.centerX - to.width / 2, to.centerY);
        
        if (isActive) {
          ctx.strokeStyle = activeColor;
          ctx.lineWidth = 3;
          ctx.shadowColor = activeShadow;
          ctx.shadowBlur = 10;
        } else {
          ctx.strokeStyle = inactiveColor;
          ctx.lineWidth = 1;
          ctx.shadowBlur = 0;
        }
        
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
    }
  }

  private isConnectionActive(
    _fromLayer: string, 
    _fromIndex: number, 
    toLayer: string, 
    toIndex: number
  ): boolean {
    // Show active connections when any calculation stage is active (not just 'connections')
    // This keeps connection lines visible during the entire calculation popup display
    if (!this.highlightedNeuron) return false;
    
    // Check if this connection leads to the highlighted neuron
    return this.highlightedNeuron.layer === toLayer && 
           this.highlightedNeuron.index === toIndex;
  }

  private drawLossOverlay(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    if (!this.showLoss) return;
    
    const { targetClass, targetName, predictions, loss } = this.showLoss;
    const classNames = [i18n.t('classes.fail'), i18n.t('classes.pending'), i18n.t('classes.pass')];
    
    ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
    ctx.fillRect(width/2 - 250, height/2 - 150, 500, 300);
    
    ctx.strokeStyle = '#60a5fa';
    ctx.lineWidth = 3;
    ctx.strokeRect(width/2 - 250, height/2 - 150, 500, 300);
    
    ctx.font = 'bold 24px sans-serif';
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.fillText('📊 Loss Calculation', width/2, height/2 - 110);
    
    ctx.font = 'bold 18px sans-serif';
    ctx.fillStyle = '#22c55e';
    ctx.fillText(`Target: ${targetName}`, width/2, height/2 - 70);
    
    ctx.font = '16px monospace';
    ctx.textAlign = 'left';
    predictions.forEach((prob, i) => {
      const y = height/2 - 30 + i * 30;
      const color = i === targetClass ? '#22c55e' : '#64748b';
      ctx.fillStyle = color;
      const barWidth = prob * 200;
      ctx.fillRect(width/2 - 100, y, barWidth, 20);
      
      ctx.fillStyle = '#ffffff';
      ctx.fillText(`${classNames[i]}: ${(prob * 100).toFixed(1)}%`, width/2 + 110, y + 15);
    });
    
    ctx.font = 'bold 20px sans-serif';
    ctx.fillStyle = '#fbbf24';
    ctx.textAlign = 'center';
    ctx.fillText(`Cross-Entropy Loss: ${loss.toFixed(4)}`, width/2, height/2 + 80);
  }

  private drawBackpropHighlight(ctx: CanvasRenderingContext2D, nodes: NodePosition[][]): void {
    drawBackpropHighlight(
      ctx,
      this.canvas,
      nodes,
      this.backpropPhase,
      this.currentBackpropData,
      this.backpropStage,
      this.allBackpropData
    );
  }

  update(nn: NeuralNetwork): void {
    const steps = nn.getCalculationSteps();
    this.drawNetwork(nn, steps);
  }

  getActivationColor(value: number): string {
    return activationToColor(value);
  }

  // Find neuron at given canvas coordinates
  public findNeuronAtPosition(x: number, y: number): {
    layer: LayerName;
    index: number;
  } | null {
    for (const [layerIndex, layerNodes] of this.lastNodes.entries()) {
      // Skip input layer (index 0)
      if (layerIndex === 0) continue;
      
      for (const [nodeIndex, node] of layerNodes.entries()) {
        if (x >= node.x && x <= node.x + node.width &&
            y >= node.y && y <= node.y + node.height) {
          
          // Determine layer name
          let layer: LayerName;
          if (layerIndex === 1) layer = 'layer1';
          else if (layerIndex === 2) layer = 'layer2';
          else if (layerIndex === 3) layer = 'output';
          else continue;
          
          return { layer, index: nodeIndex };
        }
      }
    }
    return null;
  }
}
