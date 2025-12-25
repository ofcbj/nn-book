// Visualizer for React - Canvas-based visualization
// Modified to work with React refs instead of direct DOM queries

import type { CalculationSteps, NeuronCalculation, AnimationPhase, CalculationStage, NodePosition, LossDisplayData, BackpropNeuronData, BackpropStage, BackpropSteps } from '../types';
import type { NeuralNetwork } from '../core';
import i18n from '../../i18n';
import { activationToColor } from './activationColors';
import { drawBackpropHighlight } from './backpropRenderer';
import { drawNetwork } from './networkRenderer';
import { drawCalculationOverlay as drawCalcOverlay } from './calculationOverlay';

export class Visualizer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private heatmapMode: boolean = false;

  get inputLabels(): string[] {
    return [i18n.t('controls.grade'), i18n.t('controls.attitude'), i18n.t('controls.response')];
  }
  highlightedNeuron: AnimationPhase | null = null;

  // Calculation animation properties
  calculationStage: CalculationStage | null = null;
  activeConnections: number[] = [];
  currentNeuronData: NeuronCalculation | null = null;

  // Backpropagation visualization
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

  private drawCalculationOverlay(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    stage: CalculationStage,
    neuronData: NeuronCalculation | null
  ): void {
    drawCalcOverlay(ctx, this.canvas, x, y, stage, neuronData);
  }

  drawNetwork(nn: NeuralNetwork, steps: CalculationSteps | null): void {
    const nodes = drawNetwork(
      this.ctx,
      this.canvas,
      nn,
      steps,
      this.inputLabels,
      this.highlightedNeuron,
      this.backpropPhase,
      this.calculationStage,
      this.drawConnectionsVector.bind(this),
      this.showLoss ? this.drawLossOverlay.bind(this) : null,
      this.backpropPhase ? this.drawBackpropHighlight.bind(this) : null,
      this.drawCalculationOverlay.bind(this),
      this.currentNeuronData,
      this.heatmapMode
    );
    // Store nodes for click detection
    this.lastNodes = nodes;
  }

  private drawConnectionsVector(
    ctx: CanvasRenderingContext2D, 
    nodes: NodePosition[][], 
    _nn: NeuralNetwork
  ): void {
    for (let i = 0; i < 5; i++) {
      const from = nodes[0][0];
      const to = nodes[1][i];
      
      const isActive = this.isConnectionActive('input', 0, 'layer1', i);
      
      ctx.beginPath();
      ctx.moveTo(from.centerX + from.width / 2, from.centerY);
      ctx.lineTo(to.centerX - to.width / 2, to.centerY);
      
      if (isActive) {
        ctx.strokeStyle = 'rgba(96, 165, 250, 0.9)';
        ctx.lineWidth = 3;
        ctx.shadowColor = 'rgba(96, 165, 250, 0.8)';
        ctx.shadowBlur = 10;
      } else {
        ctx.strokeStyle = 'rgba(100, 116, 139, 0.4)';
        ctx.lineWidth = 1;
        ctx.shadowBlur = 0;
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
    }
    
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 5; j++) {
        const from = nodes[1][j];
        const to = nodes[2][i];
        
        const isActive = this.isConnectionActive('layer1', j, 'layer2', i);
        
        ctx.beginPath();
        ctx.moveTo(from.centerX + from.width / 2, from.centerY);
        ctx.lineTo(to.centerX - to.width / 2, to.centerY);
        
        if (isActive) {
          ctx.strokeStyle = 'rgba(52, 211, 153, 0.9)';
          ctx.lineWidth = 3;
          ctx.shadowColor = 'rgba(52, 211, 153, 0.8)';
          ctx.shadowBlur = 10;
        } else {
          ctx.strokeStyle = 'rgba(100, 116, 139, 0.3)';
          ctx.lineWidth = 1;
          ctx.shadowBlur = 0;
        }
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
    }
    
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const from = nodes[2][i];
        const to = nodes[3][j];
        
        const isActive = this.isConnectionActive('layer2', i, 'output', j);
        
        ctx.beginPath();
        ctx.moveTo(from.centerX + from.width / 2, from.centerY);
        ctx.lineTo(to.centerX - to.width / 2, to.centerY);
        
        if (isActive) {
          ctx.strokeStyle = 'rgba(251, 146, 60, 0.9)';
          ctx.lineWidth = 3;
          ctx.shadowColor = 'rgba(251, 146, 60, 0.8)';
          ctx.shadowBlur = 10;
        } else {
          ctx.strokeStyle = 'rgba(100, 116, 139, 0.4)';
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

  setHeatmapMode(enabled: boolean): void {
    this.heatmapMode = enabled;
  }

  getHeatmapMode(): boolean {
    return this.heatmapMode;
  }

  getActivationColor(value: number): string {
    return activationToColor(value);
  }

  // Find neuron at given canvas coordinates
  public findNeuronAtPosition(x: number, y: number): {
    layer: 'layer1' | 'layer2' | 'output';
    index: number;
  } | null {
    for (const [layerIndex, layerNodes] of this.lastNodes.entries()) {
      // Skip input layer (index 0)
      if (layerIndex === 0) continue;
      
      for (const [nodeIndex, node] of layerNodes.entries()) {
        if (x >= node.x && x <= node.x + node.width &&
            y >= node.y && y <= node.y + node.height) {
          
          // Determine layer name
          let layer: 'layer1' | 'layer2' | 'output';
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
