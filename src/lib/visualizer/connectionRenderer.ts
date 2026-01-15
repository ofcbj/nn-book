// Connection rendering between neural network layers
import type { NodePosition } from '../types';
import type { AnimationState } from '../animation';
import { LAYER_SIZES } from '../core';

// ============================================================================
// Types
// ============================================================================

interface ConnectionTheme {
  active: string;
  shadow: string;
}

interface LayerConnectionConfig {
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

// ============================================================================
// Constants
// ============================================================================

const CONNECTION_COLORS: Record<string, ConnectionTheme> = {
  input: { active: 'rgba(96, 165, 250, 0.9)', shadow: 'rgba(96, 165, 250, 0.8)' },
  layer1: { active: 'rgba(52, 211, 153, 0.9)', shadow: 'rgba(52, 211, 153, 0.8)' },
  layer2: { active: 'rgba(251, 146, 60, 0.9)', shadow: 'rgba(251, 146, 60, 0.8)' },
};

const INACTIVE_COLOR = 'rgba(100, 116, 139, 0.4)';

// ============================================================================
// Helper Functions
// ============================================================================

function isConnectionActive(
  animationState: AnimationState,
  toLayer: string,
  toIndex: number
): boolean {
  if (animationState.type !== 'forward_animating' &&
      animationState.type !== 'backward_animating') {
    return false;
  }
  return animationState.layer === toLayer && animationState.neuronIndex === toIndex;
}

function drawLayerConnections(
  ctx: CanvasRenderingContext2D,
  nodes: NodePosition[][],
  animationState: AnimationState,
  config: LayerConnectionConfig
): void {
  const {
    fromLayerIdx, toLayerIdx, toLayer,
    fromCount, toCount,
    activeColor, activeShadow, inactiveColor
  } = config;

  for (let i = 0; i < fromCount; i++) {
    for (let j = 0; j < toCount; j++) {
      const from = nodes[fromLayerIdx][i];
      const to = nodes[toLayerIdx][j];
      const isActive = isConnectionActive(animationState, toLayer, j);

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

// ============================================================================
// Main Export
// ============================================================================

export function drawConnections(
  ctx: CanvasRenderingContext2D,
  nodes: NodePosition[][],
  animationState: AnimationState
): void {
  const connections = [
    { from: 'input', to: 'layer1', fromCount: 1, toCount: LAYER_SIZES.layer1, theme: CONNECTION_COLORS.input },
    { from: 'layer1', to: 'layer2', fromCount: LAYER_SIZES.layer1, toCount: LAYER_SIZES.layer2, theme: CONNECTION_COLORS.layer1 },
    { from: 'layer2', to: 'output', fromCount: LAYER_SIZES.layer2, toCount: LAYER_SIZES.output, theme: CONNECTION_COLORS.layer2 },
  ] as const;

  connections.forEach(({ from, to, fromCount, toCount, theme }, idx) => {
    drawLayerConnections(ctx, nodes, animationState, {
      fromLayerIdx: idx,
      toLayerIdx: idx + 1,
      fromLayer: from,
      toLayer: to,
      fromCount,
      toCount,
      activeColor: theme.active,
      activeShadow: theme.shadow,
      inactiveColor: INACTIVE_COLOR,
    });
  });
}
