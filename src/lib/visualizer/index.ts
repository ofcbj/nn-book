// Visualizer for React - Canvas-based neural network visualizer
import type { ForwardSteps, NodePosition, Viewport } from '../types';
import type { AnimationState } from '../animation';
import type { NeuralNetwork, LayerName } from '../core';
import { LAYER_NAMES, DEFAULT_LEARNING_RATE } from '../core';
import i18n from '../../i18n';
import { drawNetwork } from './networkRenderer';
import { drawConnections } from './connectionRenderer';
import { drawForwardOverlay } from './overlayForward';
import { drawBackwardOverlay } from './overlayBackward';

export class Visualizer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private lastNodes: NodePosition[][] = [];
  /** Logical (CSS pixel) size of the drawing area */
  private viewport: Viewport = { width: 0, height: 0 };

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

  /**
   * Match the canvas bitmap to its CSS size, scaled by devicePixelRatio
   * so the drawing stays sharp on high-DPI screens. All drawing code
   * keeps working in logical (CSS) pixels.
   */
  resizeCanvas(): void {
    const dpr = window.devicePixelRatio || 1;
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    this.viewport = { width, height };
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /**
   * Main render method - draws the network and appropriate overlays
   */
  draw(nn: NeuralNetwork, steps: ForwardSteps | null, animationState: AnimationState, learningRate: number = DEFAULT_LEARNING_RATE): void {
    this.lastNodes = drawNetwork(
      this.ctx,
      this.viewport,
      nn,
      steps,
      this.inputLabels,
      animationState,
      {
        drawConnections,
        drawForwardOverlay,
        drawBackwardOverlay: (ctx, viewport, nodes, network, animState) =>
          drawBackwardOverlay(ctx, viewport, nodes, network, animState, learningRate),
      }
    );
  }

  /**
   * Convenience method that gets steps from network and draws
   */
  update(nn: NeuralNetwork, animationState: AnimationState, learningRate: number = DEFAULT_LEARNING_RATE): void {
    this.draw(nn, nn.getForwardSteps(), animationState, learningRate);
  }

  /**
   * Find the processing-layer neuron at the given logical canvas coordinates (for click detection)
   */
  findNeuronAtPosition(x: number, y: number): { layer: LayerName; index: number } | null {
    // lastNodes[0] is the input box; processing layers follow in LAYER_NAMES order
    for (const [i, layer] of LAYER_NAMES.entries()) {
      const layerNodes = this.lastNodes[i + 1];
      if (!layerNodes) continue;
      for (const [index, node] of layerNodes.entries()) {
        if (x >= node.x && x <= node.x + node.width &&
            y >= node.y && y <= node.y + node.height) {
          return { layer, index };
        }
      }
    }
    return null;
  }
}
