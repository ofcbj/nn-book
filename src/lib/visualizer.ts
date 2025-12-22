// Visualizer for React - Canvas-based visualization
// Modified to work with React refs instead of direct DOM queries

import type { CalculationSteps, NeuronCalculation, AnimationPhase, CalculationStage, NodePosition, LossDisplayData } from './types';
import type { NeuralNetwork } from './network';

type LayerType = 'input' | 'layer1' | 'layer2' | 'output';

export class Visualizer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  
  inputLabels: string[] = ['성적', '태도', '응답수준'];
  highlightedNeuron: AnimationPhase | null = null;
  
  // Calculation animation properties
  calculationStage: CalculationStage | null = null;
  intermediateValue: number | null = null;
  activeConnections: number[] = [];
  currentNeuronData: NeuronCalculation | null = null;
  
  // Backpropagation visualization
  showLoss: LossDisplayData | null = null;
  backpropPhase: AnimationPhase | null = null;

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

  private drawRoundedRect(
    ctx: CanvasRenderingContext2D, 
    x: number, 
    y: number, 
    width: number, 
    height: number, 
    radius: number
  ): void {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }

  private drawInputVector(
    ctx: CanvasRenderingContext2D, 
    x: number, 
    y: number, 
    values: number[]
  ): NodePosition {
    const width = 140;
    const height = 100;
    const centerX = x - width / 2;
    const centerY = y - height / 2;
    
    this.drawRoundedRect(ctx, centerX, centerY, width, height, 15);
    
    const gradient = ctx.createLinearGradient(centerX, centerY, centerX, centerY + height);
    gradient.addColorStop(0, 'rgba(59, 130, 246, 0.3)');
    gradient.addColorStop(1, 'rgba(37, 99, 235, 0.2)');
    ctx.fillStyle = gradient;
    ctx.fill();
    
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('면접자', x, centerY + 20);
    
    ctx.font = '12px monospace';
    ctx.textAlign = 'left';
    const startY = centerY + 40;
    values.forEach((val, idx) => {
      ctx.fillStyle = '#cbd5e1';
      ctx.fillText(this.inputLabels[idx] + ':', centerX + 15, startY + idx * 18);
      ctx.fillStyle = '#60a5fa';
      ctx.font = 'bold 12px monospace';
      ctx.fillText(val.toFixed(2), centerX + 80, startY + idx * 18);
      ctx.font = '12px monospace';
    });
    
    return { x: centerX, y: centerY, width, height, centerX: x, centerY: y };
  }

  private drawNeuronVector(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    weights: number[],
    bias: number,
    activation: number,
    label: string,
    layerType: LayerType,
    isHighlighted: boolean = false
  ): NodePosition {
    const baseWidth = weights.length * 25;
    let width = Math.max(130, baseWidth + 40);
    if (layerType === 'layer1') {
      width += 30;
    } else if (layerType === 'layer2') {
      width += 60;
    } else if (layerType === 'output') {
      width += 20;
    }
    const height = 90;
    const centerX = x - width / 2;
    const centerY = y - height / 2;
    
    this.drawRoundedRect(ctx, centerX, centerY, width, height, 12);
    
    let gradient: CanvasGradient;
    let strokeColor: string;
    
    if (layerType === 'layer1') {
      gradient = ctx.createLinearGradient(centerX, centerY, centerX, centerY + height);
      if (isHighlighted) {
        gradient.addColorStop(0, 'rgba(34, 197, 94, 0.9)');
        gradient.addColorStop(1, 'rgba(22, 163, 74, 0.7)');
      } else {
        gradient.addColorStop(0, 'rgba(34, 197, 94, 0.3)');
        gradient.addColorStop(1, 'rgba(22, 163, 74, 0.2)');
      }
      ctx.fillStyle = gradient;
      strokeColor = isHighlighted ? '#4ade80' : '#22c55e';
    } else if (layerType === 'layer2') {
      gradient = ctx.createLinearGradient(centerX, centerY, centerX, centerY + height);
      if (isHighlighted) {
        gradient.addColorStop(0, 'rgba(249, 115, 22, 0.9)');
        gradient.addColorStop(1, 'rgba(234, 88, 12, 0.7)');
      } else {
        gradient.addColorStop(0, 'rgba(249, 115, 22, 0.3)');
        gradient.addColorStop(1, 'rgba(234, 88, 12, 0.2)');
      }
      ctx.fillStyle = gradient;
      strokeColor = isHighlighted ? '#fb923c' : '#f97316';
    } else {
      gradient = ctx.createLinearGradient(centerX, centerY, centerX, centerY + height);
      if (isHighlighted) {
        gradient.addColorStop(0, 'rgba(239, 68, 68, 0.9)');
        gradient.addColorStop(1, 'rgba(220, 38, 38, 0.7)');
      } else {
        gradient.addColorStop(0, 'rgba(239, 68, 68, 0.3)');
        gradient.addColorStop(1, 'rgba(220, 38, 38, 0.2)');
      } 
      ctx.fillStyle = gradient;
      strokeColor = isHighlighted ? '#f87171' : '#ef4444';
    }
    
    ctx.fill();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = isHighlighted ? 4 : 2;
    ctx.stroke();
    
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(label, x, centerY + 16);
    
    ctx.font = '12px monospace';
    ctx.fillStyle = '#cbd5e1';
    ctx.textAlign = 'left';
    ctx.fillText('W:', centerX + 8, centerY + 35);
    
    ctx.fillStyle = '#a5b4fc';
    const vectorStr = '[' + weights.map(w => w.toFixed(2)).join(', ') + ']';
    const textWidth = ctx.measureText(vectorStr).width;
    if (textWidth > width - 40) {
      ctx.font = '11px monospace';
    }
    ctx.fillText(vectorStr, centerX + 24, centerY + 35);
    
    ctx.font = '12px monospace';
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText('b:', centerX + 8, centerY + 50);
    ctx.fillStyle = '#fbbf24';
    ctx.fillText(bias.toFixed(2), centerX + 24, centerY + 50);
    
    ctx.fillStyle = '#34d399';
    ctx.font = 'bold 12px monospace';
    ctx.fillText('→ ' + activation.toFixed(3), centerX + 70, centerY + 68);
    
    if (this.highlightedNeuron && 
        this.highlightedNeuron.layer === layerType && 
        this.highlightedNeuron.index !== undefined) {
      const neuronIndex = label.includes('#') ? parseInt(label.split('#')[1]) - 1 : 0;
      if (this.highlightedNeuron.index === neuronIndex && this.calculationStage && this.intermediateValue !== null) {
        this.drawCalculationOverlay(ctx, x, centerY - 20, this.calculationStage, this.intermediateValue);
      }
    }
    
    return { x: centerX, y: centerY, width, height, centerX: x, centerY: y };
  }

  private drawCalculationOverlay(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    stage: CalculationStage,
    value: number
  ): void {
    if (!this.currentNeuronData) return;
    
    let text = '';
    let color = '';
    const data = this.currentNeuronData;
    
    switch(stage) {
      case 'connections':
        text = '입력 연결 계산 중...';
        color = '#60a5fa';
        break;
        
      case 'dotProduct':
        const terms = data.inputs.map((input, i) => 
          `${input.toFixed(2)}×${data.weights[i].toFixed(2)}`
        );
        text = terms.join(' + ') + ` = ${value.toFixed(3)}`;
        color = '#a5b4fc';
        break;
        
      case 'bias':
        text = `${data.dotProduct.toFixed(3)} + ${data.bias.toFixed(2)} = ${value.toFixed(3)}`;
        color = '#fbbf24';
        break;
        
      case 'activation':
        text = `σ(${data.withBias.toFixed(3)}) = ${value.toFixed(3)}`;
        color = '#34d399';
        break;
    }
    
    if (!text) return;
    
    ctx.font = 'bold 13px monospace';
    const metrics = ctx.measureText(text);
    const maxWidth = 600;
    
    let lines = [text];
    if (metrics.width > maxWidth && stage === 'dotProduct') {
      const termsPerLine = Math.ceil(data.inputs.length / 2);
      const line1Terms = data.inputs.slice(0, termsPerLine).map((input, i) => 
        `${input.toFixed(2)}×${data.weights[i].toFixed(2)}`
      );
      const line2Terms = data.inputs.slice(termsPerLine).map((input, i) => 
        `${input.toFixed(2)}×${data.weights[i + termsPerLine].toFixed(2)}`
      );
      
      lines = [
        line1Terms.join(' + ') + ' +',
        line2Terms.join(' + ') + ` = ${value.toFixed(3)}`
      ];
    }
    
    const padding = 10;
    const lineHeight = 20;
    const maxLineWidth = Math.max(...lines.map(line => ctx.measureText(line).width));
    const boxWidth = maxLineWidth + padding * 2;
    const boxHeight = lineHeight * lines.length + padding * 2;
    
    ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
    ctx.beginPath();
    ctx.roundRect(x - boxWidth / 2, y - boxHeight / 2, boxWidth, boxHeight, 6);
    ctx.fill();
    
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
    
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    lines.forEach((line, i) => {
      const lineY = y - boxHeight / 2 + padding + lineHeight / 2 + i * lineHeight;
      ctx.fillText(line, x, lineY);
    });
  }

  drawNetwork(nn: NeuralNetwork, steps: CalculationSteps | null): void {
    const ctx = this.ctx;
    const width = this.canvas.width;
    const height = this.canvas.height;

    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, height);

    if (!steps) return;

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

    const inputNode = this.drawInputVector(ctx, inputX, height / 2, steps.input);
    nodes.push([inputNode]);

    // Layer 1 - 5 neurons with tighter spacing
    const layer1Nodes: NodePosition[] = [];
    const layer1VerticalSpacing = 95;
    const layer1TotalHeight = (5 - 1) * layer1VerticalSpacing;
    const layer1StartY = (height - layer1TotalHeight) / 2;
    for (let i = 0; i < 5; i++) {
      const neuron = steps.layer1[i];
      const isHighlighted = this.highlightedNeuron &&
                            this.highlightedNeuron.layer === 'layer1' &&
                            this.highlightedNeuron.index === i;
      const node = this.drawNeuronVector(
        ctx,
        layer1X,
        layer1StartY + i * layer1VerticalSpacing,
        neuron.weights,
        neuron.bias,
        neuron.activated,
        `1차 #${i + 1}`,
        'layer1',
        isHighlighted || false
      );
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
      const isHighlighted = this.highlightedNeuron &&
                            this.highlightedNeuron.layer === 'layer2' &&
                            this.highlightedNeuron.index === i;
      const node = this.drawNeuronVector(
        ctx,
        layer2X,
        layer2StartY + i * layer2VerticalSpacing,
        neuron.weights,
        neuron.bias,
        neuron.activated,
        `2차 #${i + 1}`,
        'layer2',
        isHighlighted || false
      );
      layer2Nodes.push(node);
    }
    nodes.push(layer2Nodes);

    // Output layer - 3 neurons with better spacing
    const outputNodes: NodePosition[] = [];
    const classNames = ['불합격', '보류', '합격'];
    const outputVerticalSpacing = 125;
    const outputTotalHeight = (3 - 1) * outputVerticalSpacing;
    const outputStartY = (height - outputTotalHeight) / 2;

    for (let i = 0; i < 3; i++) {
      const output = steps.output[i];
      const isHighlighted = this.highlightedNeuron &&
                            this.highlightedNeuron.layer === 'output' &&
                            this.highlightedNeuron.index === i;
      const outputNode = this.drawNeuronVector(
        ctx,
        outputX,
        outputStartY + i * outputVerticalSpacing,
        output.weights,
        output.bias,
        output.activated,
        classNames[i],
        'output',
        isHighlighted || false
      );
      outputNodes.push(outputNode);
    }
    nodes.push(outputNodes);

    this.drawConnectionsVector(ctx, nodes, nn);

    if (this.showLoss) {
      this.drawLossOverlay(ctx, width, height);
    }

    if (this.backpropPhase) {
      this.drawBackpropHighlight(ctx, nodes);
    }
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
    if (this.calculationStage !== 'connections') return false;
    if (!this.highlightedNeuron) return false;
    
    return this.highlightedNeuron.layer === toLayer && 
           this.highlightedNeuron.index === toIndex;
  }

  private drawLossOverlay(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    if (!this.showLoss) return;
    
    const { targetClass, targetName, predictions, loss } = this.showLoss;
    const classNames = ['불합격', '보류', '합격'];
    
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
    if (!this.backpropPhase) return;
    
    const { layer, index } = this.backpropPhase;
    let nodeInfo: NodePosition | null = null;
    
    if (layer === 'layer1' && nodes[1]) {
      nodeInfo = nodes[1][index];
    } else if (layer === 'layer2' && nodes[2]) {
      nodeInfo = nodes[2][index];
    } else if (layer === 'output' && nodes[3]) {
      nodeInfo = nodes[3][index];
    }
    
    if (!nodeInfo) return;
    
    ctx.save();
    const gradient = ctx.createRadialGradient(
      nodeInfo.centerX, nodeInfo.centerY, 0,
      nodeInfo.centerX, nodeInfo.centerY, nodeInfo.width / 2 + 20
    );
    gradient.addColorStop(0, 'rgba(239, 68, 68, 0.6)');
    gradient.addColorStop(1, 'rgba(239, 68, 68, 0)');
    
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(nodeInfo.centerX, nodeInfo.centerY, nodeInfo.width / 2 + 25, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    
    ctx.font = 'bold 14px sans-serif';
    ctx.fillStyle = '#ef4444';
    ctx.textAlign = 'center';
    ctx.fillText('◄ BACKPROP', nodeInfo.centerX, nodeInfo.y - 35);
  }

  update(nn: NeuralNetwork): void {
    const steps = nn.getCalculationSteps();
    this.drawNetwork(nn, steps);
  }
}
