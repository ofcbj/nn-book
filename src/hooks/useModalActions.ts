/**
 * Modal Actions Hook
 *
 * Manages modal lifecycle and interactions:
 * - Loss Modal: Triggers backward propagation animation
 * - Backprop Modal: Shows backpropagation summary
 *
 * Separated from useAnimationEngine to reduce complexity and improve maintainability.
 */

import { useCallback, RefObject } from 'react';
import type { NeuralNetwork } from '../lib/core';
import type { UseNetworkStateReturn } from './useNetworkState';
import { createSnapshot, compareSnapshots } from '../lib/core/networkSnapshot';

export interface UseModalActionsParams {
  nnRef: RefObject<NeuralNetwork>;
  state: UseNetworkStateReturn;
  animateBackwardPropagation: (speedOverride: number) => Promise<boolean>;
  sleep: (ms: number, overrideSpeed?: number) => Promise<void>;
  computeAndRefreshDisplay: () => void;
  closeLossModalAction: () => void;
  closeBackpropModalAction: () => void;
  refreshDisplayOnly: () => void;
  animationSpeed: number;
}

export interface UseModalActionsReturn {
  closeLossModal: () => Promise<void>;
  closeBackpropModal: () => void;
}

export function useModalActions(params: UseModalActionsParams): UseModalActionsReturn {
  const {
    nnRef,
    state,
    animateBackwardPropagation,
    sleep,
    computeAndRefreshDisplay,
    closeLossModalAction,
    closeBackpropModalAction,
    refreshDisplayOnly,
    animationSpeed,
  } = params;

  // Helper: Create weight comparison data after training
  const createWeightComparisonAfterTraining = useCallback((
    oldSnapshot: ReturnType<typeof createSnapshot>,
    newSnapshot: ReturnType<typeof createSnapshot>
  ) => {
    const comparisonData = compareSnapshots(oldSnapshot, newSnapshot, state.stats.learningRate);
    state.modalSetters.setWeightComparisonData(comparisonData);
  }, [state.stats.learningRate, state.modalSetters]);

  // Close loss modal - start backward propagation
  const closeLossModal = useCallback(async () => {
    state.modalSetters.setLossModalData(null);
    closeLossModalAction();

    const nn = nnRef.current;

    const oldSnapshot = createSnapshot(nn);

    const completed = await animateBackwardPropagation(animationSpeed);
    await sleep(500, animationSpeed);

    // Update stats and weight comparison data if animation completed
    // (even if interrupted by clicking neurons during animation)
    if (completed) {
      const newSnapshot = createSnapshot(nn);
      createWeightComparisonAfterTraining(oldSnapshot, newSnapshot);
      state.statsSetters.setEpoch(prev => prev + 1);
      state.statsSetters.setLoss(nn.lastLoss);
    }

    computeAndRefreshDisplay();
  }, [
    state.modalSetters,
    state.statsSetters,
    animationSpeed,
    createWeightComparisonAfterTraining,
    closeLossModalAction,
    animateBackwardPropagation,
    sleep,
    computeAndRefreshDisplay,
    nnRef,
  ]);

  // Close backprop modal
  const closeBackpropModal = useCallback(() => {
    state.modalSetters.setBackpropSummaryData(null);
    closeBackpropModalAction();
    refreshDisplayOnly();
  }, [state.modalSetters, closeBackpropModalAction, refreshDisplayOnly]);

  return {
    closeLossModal,
    closeBackpropModal,
  };
}
