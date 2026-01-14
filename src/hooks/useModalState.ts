/**
 * Generic Modal State Hook
 *
 * Provides consistent modal state management with open/close actions.
 * Eliminates pattern duplication across multiple modals.
 *
 * This is a reusable utility hook for managing modal state.
 * For specific modal business logic, see useModalActions.
 */

import { useState, useCallback } from 'react';

export interface ModalState<T> {
  show: boolean;
  data: T | null;
}

export interface ModalActions<T> {
  open: (data: T) => void;
  close: () => void;
  setData: (data: T | null) => void; // Set data without opening modal
}

export type UseModalReturn<T> = ModalState<T> & ModalActions<T>;

/**
 * Generic modal state management hook
 *
 * @returns Modal state (show, data) and actions (open, close, setData)
 *
 * @example
 * const lossModal = useModal<LossData>();
 * lossModal.open({ targetClass: 2, predictions: [0.1, 0.2, 0.7], loss: 0.5 });
 * lossModal.setData({ ... }); // Set data without opening
 * lossModal.close();
 */
export function useModal<T>(): UseModalReturn<T> {
  const [data, setDataInternal] = useState<T | null>(null);
  const [show, setShow] = useState(false);

  const open = useCallback((newData: T) => {
    setDataInternal(newData);
    setShow(true);
  }, []);

  const close = useCallback(() => {
    setShow(false);
    // Keep data available even when closed, for "View" button
  }, []);

  const setData = useCallback((newData: T | null) => {
    setDataInternal(newData);
    // Don't change show state
  }, []);

  return {
    show,
    data,
    open,
    close,
    setData,
  };
}
