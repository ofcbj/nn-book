// Visualizer for React - Canvas-based neural network visualizer
import type { ForwardSteps, NodePosition } from '../types';
import type { AnimationState } from '../animation';
import type { NeuralNetwork, LayerName } from '../core';
import i18n from '../../i18n';
import { activationToColor } from './uiConfig';
import { drawNetwork } from './networkRenderer';
import { drawConnections } from './connectionRenderer';
import { drawForwardOverlay } from './overlayForward';
import { drawBackwardOverlay } from './overlayBackward';

export class Visualizer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private lastNodes: NodePosition[][] = [];

  get inputLabels(): string[] {
    return [i18n.t('controls.grade'), i18n.t('controls.attitude'), i18n.t('controls.response')];
  }

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
   * Main render method - draws the network and appropriate overlays
   */
  draw(nn: NeuralNetwork, steps: ForwardSteps | null, animationState: AnimationState, learningRate: number = 0.25): void {
    const nodes = drawNetwork(
      this.ctx,
      this.canvas,
      nn,
      steps,
      this.inputLabels,
      animationState,
      {
        drawConnections,
        drawForwardOverlay,
        drawBackwardOverlay: (ctx, canvas, nodes, nn, animState) => 
          drawBackwardOverlay(ctx, canvas, nodes, nn, animState, learningRate),
      }
    );
    this.lastNodes = nodes;
  }

  /**
   * Convenience method that gets steps from network and draws
   */
  update(nn: NeuralNetwork, animationState: AnimationState, learningRate: number = 0.25): void {
    const steps = nn.getForwardSteps();
    this.draw(nn, steps, animationState, learningRate);
  }

  getActivationColor(value: number): string {
    return activationToColor(value);
  }

  /**
   * Find neuron at given canvas coordinates (for click detection)
   */
  public findNeuronAtPosition(x: number, y: number): { layer: LayerName; index: number } | null {
    for (const [layerIndex, layerNodes] of this.lastNodes.entries()) {
      if (layerIndex === 0) continue; // Skip input layer

      for (const [nodeIndex, node] of layerNodes.entries()) {
        if (x >= node.x && x <= node.x + node.width &&
            y >= node.y && y <= node.y + node.height) {
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
