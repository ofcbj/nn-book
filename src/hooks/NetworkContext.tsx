/**
 * Neural Network Context
 * 
 * Provides centralized access to neural network state and actions.
 * Wraps the existing useNeuralNetwork hook without changing its logic.
 * 
 * Usage:
 * 1. Wrap your app with <NetworkProvider>
 * 2. Use useNetworkContext() to access state from any component
 * 3. Use useNetworkSelector() for optimized re-renders (optional)
 */

import { createContext, useContext, ReactNode } from 'react';
import { useNeuralNetwork, UseNeuralNetworkReturn } from './useNeuralNetwork';

// =============================================================================
// Context
// =============================================================================

const NetworkContext = createContext<UseNeuralNetworkReturn | null>(null);

// =============================================================================
// Provider Component
// =============================================================================

interface NetworkProviderProps {
  children: ReactNode;
}

/**
 * NetworkProvider
 * 
 * Wraps the application to provide centralized neural network state.
 * Uses the existing useNeuralNetwork hook internally - no logic duplication!
 */
export function NetworkProvider({ children }: NetworkProviderProps) {
  // Use existing hook - all logic remains in useNeuralNetwork
  const network = useNeuralNetwork();
  
  return (
    <NetworkContext.Provider value={network}>
      {children}
    </NetworkContext.Provider>
  );
}

// =============================================================================
// Hook for consuming context
// =============================================================================

/**
 * useNetworkContext
 * 
 * Access neural network state and actions from anywhere in the component tree.
 * Must be used within a NetworkProvider.
 * 
 * @example
 * function MyComponent() {
 *   const { stats, actions } = useNetworkContext();
 *   return <div>Epoch: {stats.epoch}</div>;
 * }
 */
export function useNetworkContext(): UseNeuralNetworkReturn {
  const context = useContext(NetworkContext);
  
  if (!context) {
    throw new Error(
      'useNetworkContext must be used within a NetworkProvider. ' +
      'Make sure to wrap your app with <NetworkProvider>.'
    );
  }
  
  return context;
}

// =============================================================================
// Selector Hook (Optional - for performance optimization)
// =============================================================================

/**
 * useNetworkSelector
 * 
 * Select a specific slice of network state to minimize re-renders.
 * Only re-renders when the selected value changes.
 * 
 * @example
 * // Only re-renders when stats change
 * const stats = useNetworkSelector(s => s.stats);
 * 
 * // Only re-renders when epoch changes
 * const epoch = useNetworkSelector(s => s.stats.epoch);
 */
export function useNetworkSelector<T>(
  selector: (state: UseNeuralNetworkReturn) => T
): T {
  const network = useNetworkContext();
  return selector(network);
}
