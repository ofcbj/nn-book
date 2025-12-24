/**
 * Neural Network Hook - Refactored with Animation State Machine
 * 
 * This hook manages the neural network instance and integrates with the
 * animation state machine for predictable state transitions.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { NeuralNetwork } from '../lib/network';
import { Visualizer } from '../lib/visualizer';
import type { CalculationSteps, BackpropSummaryData, WeightComparisonData, CalculationStage, BackpropStage } from '../lib/types';
import type { ActivationData } from '../components/ActivationHeatmap';
import { createWeightComparisonData } from '../lib/weightComparison';
import {
  useAnimationStateMachine,
  getNextForwardStage,
  getNextBackpropStage,
  getNextForwardNeuron,
  getNextBackwardNeuron,
  FORWARD_STAGES,
  BACKPROP_STAGES,
} from './useAnimationStateMachine';
import {
  runAnimationLoop,
  forwardNeuronIndices,
  backwardNeuronIndices,
  FORWARD_STAGE_DURATIONS,
  BACKWARD_STAGE_DURATIONS,
} from '../lib/animationLoop';

interface UseNeuralNetworkReturn {
  // Neural network
  nn: NeuralNetwork;
  visualizer: Visualizer | null;
  setVisualizer: (v: Visualizer) => void;

  // Input values
  grade: number;
  attitude: number;
  response: number;
  targetValue: number;
  learningRate: number;
  animationSpeed: number;
  isManualMode: boolean;

  // Setters
  setGrade: (v: number) => void;
  setAttitude: (v: number) => void;
  setResponse: (v: number) => void;
  setTargetValue: (v: number) => void;
  setLearningRate: (v: number) => void;
  setAnimationSpeed: (v: number) => void;
  setIsManualMode: (v: boolean) => void;
  nextStep: () => void;

  // Stats
  epoch: number;
  loss: number;
  output: number[] | null;
  steps: CalculationSteps | null;

  // Training state
  isTraining: boolean;
  isAnimating: boolean;

  // Loss modal
  showLossModal: boolean;
  lossModalData: { targetClass: number; predictions: number[]; loss: number } | null;

  // Backprop summary modal
  showBackpropModal: boolean;
  backpropSummaryData: BackpropSummaryData | null;

  // Heatmap visualization
  showCanvasHeatmap: boolean;
  showGridHeatmap: boolean;
  activations: ActivationData | null;
  toggleCanvasHeatmap: () => void;
  toggleGridHeatmap: () => void;

  // Weight comparison
  showComparisonModal: boolean;
  weightComparisonData: WeightComparisonData | null;
  openComparisonModal: () => void;
  closeComparisonModal: () => void;

  // Actions
  trainOneStepWithAnimation: () => Promise<void>;
  toggleTraining: () => void;
  reset: () => void;
  closeLossModal: () => void;
  closeBackpropModal: () => void;
  updateVisualization: () => void;
  handleCanvasClick: (x?: number, y?: number) => void;
}

export function useNeuralNetwork(): UseNeuralNetworkReturn {
  // =========================================================================
  // Core Refs
  // =========================================================================
  const nnRef = useRef(new NeuralNetwork());
  const visualizerRef = useRef<Visualizer | null>(null);
  const trainingIntervalRef = useRef<number | undefined>(undefined);
  
  // =========================================================================
  // Animation State Machine
  // =========================================================================
  const animationMachine = useAnimationStateMachine();
  
  // =========================================================================
  // Local State (non-animation related)
  // =========================================================================
  const [grade, setGrade] = useState(0.7);
  const [attitude, setAttitude] = useState(0.5);
  const [response, setResponse] = useState(0.8);
  const [targetValue, setTargetValue] = useState(2);
  const [learningRate, setLearningRate] = useState(0.1);
  
  // Animation speed is managed both locally (for UI) and in state machine
  const [animationSpeed, setAnimationSpeedLocal] = useState(1.0);
  const [isManualMode, setIsManualMode] = useState(false);

  // Stats
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(0);
  const [output, setOutput] = useState<number[] | null>(null);
  const [steps, setSteps] = useState<CalculationSteps | null>(null);
  
  // Training state
  const [isTraining, setIsTraining] = useState(false);
  
  // Loss modal (derived from state machine)
  const [lossModalData, setLossModalData] = useState<{ targetClass: number; predictions: number[]; loss: number } | null>(null);
  
  // Backprop summary
  const [backpropSummaryData, setBackpropSummaryData] = useState<BackpropSummaryData | null>(null);

  // Heatmap
  const [showCanvasHeatmap, setShowCanvasHeatmap] = useState(false);
  const [showGridHeatmap, setShowGridHeatmap] = useState(true);
  const [activations, setActivations] = useState<ActivationData | null>(null);

  // Weight comparison
  const [showComparisonModal, setShowComparisonModal] = useState(false);
  const [weightComparisonData, setWeightComparisonData] = useState<WeightComparisonData | null>(null);

  // Ref to signal animation should stop
  const shouldStopRef = useRef(false);
  
  // Ref for continueFromJumpedPosition function (to be called from useEffect)
  const continueFromJumpedPositionRef = useRef<(() => Promise<void>) | null>(null);
  
  // =========================================================================
  // Sync animation speed with state machine
  // =========================================================================
  const setAnimationSpeed = useCallback((speed: number) => {
    setAnimationSpeedLocal(speed);
    animationMachine.setSpeed(speed);
  }, [animationMachine]);
  
  // Sync local animationSpeed with state machine's speed
  // This ensures UI reflects pauses triggered by jumpToNeuron, pause(), etc.
  useEffect(() => {
    const machineSpeed = animationMachine.state.speed;
    if (machineSpeed !== animationSpeed) {
      setAnimationSpeedLocal(machineSpeed);
    }
  }, [animationMachine.state.speed, animationSpeed]);
  
  // =========================================================================
  // Visualizer Setup
  // =========================================================================
  const setVisualizer = useCallback((v: Visualizer) => {
    visualizerRef.current = v;
  }, []);

  // =========================================================================
  // Update Visualization (sync visualizer with state machine)
  // =========================================================================
  const updateVisualization = useCallback(() => {
    const nn = nnRef.current;
    const inputs = [grade, attitude, response];
    nn.feedforward(inputs);
    
    if (nn.lastOutput) {
      setOutput(nn.lastOutput.toArray());
    }
    setSteps(nn.getCalculationSteps());
    
    // Update activations for heatmap
    if (nn.lastInput && nn.lastHidden1 && nn.lastHidden2 && nn.lastOutput) {
      setActivations({
        input: nn.lastInput.toArray(),
        layer1: nn.lastHidden1.toArray(),
        layer2: nn.lastHidden2.toArray(),
        output: nn.lastOutput.toArray(),
      });
    }
    
    if (visualizerRef.current) {
      // Sync visualizer state from state machine
      const state = animationMachine.state;
      
      if (state.type === 'forward_animating') {
        visualizerRef.current.highlightedNeuron = { layer: state.layer, index: state.neuronIndex };
        visualizerRef.current.calculationStage = state.stage;
        visualizerRef.current.currentNeuronData = state.neuronData;
        visualizerRef.current.backpropPhase = null;
        visualizerRef.current.currentBackpropData = null;
        visualizerRef.current.backpropStage = null;
        visualizerRef.current.allBackpropData = null;
      } else if (state.type === 'backward_animating') {
        visualizerRef.current.highlightedNeuron = null;
        visualizerRef.current.calculationStage = null;
        visualizerRef.current.currentNeuronData = null;
        visualizerRef.current.backpropPhase = { layer: state.layer, index: state.neuronIndex };
        visualizerRef.current.currentBackpropData = state.neuronData;
        visualizerRef.current.backpropStage = state.stage;
        // Set all backprop data for persistent error labels
        visualizerRef.current.allBackpropData = nn.lastBackpropSteps;
      } else {
        // Clear all highlights
        visualizerRef.current.highlightedNeuron = null;
        visualizerRef.current.calculationStage = null;
        visualizerRef.current.currentNeuronData = null;
        visualizerRef.current.backpropPhase = null;
        visualizerRef.current.currentBackpropData = null;
        visualizerRef.current.backpropStage = null;
        visualizerRef.current.allBackpropData = null;
      }
      
      visualizerRef.current.update(nn);
    }
  }, [grade, attitude, response, animationMachine.state]);
  
  // =========================================================================
  // Sleep utility that respects pause state
  // =========================================================================
  const sleep = useCallback(async (ms: number, overrideSpeed?: number): Promise<void> => {
    // Use override speed if provided, otherwise use local animationSpeed state
    const effectiveSpeed = overrideSpeed ?? animationSpeed;
    
    if (isManualMode || effectiveSpeed === 0) {
      // Wait for user to advance
      await animationMachine.waitForNextStep();
    } else {
      await new Promise(resolve => setTimeout(resolve, ms / effectiveSpeed));
    }
  }, [animationMachine, isManualMode, animationSpeed]);
  
  // =========================================================================
  // Forward Propagation Animation
  // =========================================================================
  const animateForwardPropagation = useCallback(async () => {
    const nn = nnRef.current;
    
    await runAnimationLoop({
      mode: 'forward',
      layers: ['layer1', 'layer2', 'output'],
      getNeuronIndices: forwardNeuronIndices,
      stages: FORWARD_STAGES,
      stageDurations: FORWARD_STAGE_DURATIONS,
      getData: () => {
        const calcSteps = nn.getCalculationSteps();
        if (!calcSteps) return null;
        return { layer1: calcSteps.layer1, layer2: calcSteps.layer2, output: calcSteps.output };
      },
      onTick: (layer, neuronIndex, stage, data) => {
        animationMachine.forwardTick(layer, neuronIndex, stage, data);
      },
      onComplete: () => {
        animationMachine.forwardComplete();
      },
      shouldStop: () => shouldStopRef.current,
      sleep,
      updateVisualization,
    });
  }, [animationMachine, sleep, updateVisualization]);
  
  // =========================================================================
  // Backward Propagation Animation
  // =========================================================================
  const animateBackwardPropagation = useCallback(async (speedOverride: number = 1.0) => {
    const nn = nnRef.current;
    const backpropData = nn.lastBackpropSteps;
    if (!backpropData) return;
    
    await runAnimationLoop({
      mode: 'backward',
      layers: ['output', 'layer2', 'layer1'], // Reverse order
      getNeuronIndices: backwardNeuronIndices,
      stages: BACKPROP_STAGES,
      stageDurations: BACKWARD_STAGE_DURATIONS,
      getData: () => {
        return { layer1: backpropData.layer1, layer2: backpropData.layer2, output: backpropData.output };
      },
      onTick: (layer, neuronIndex, stage, data) => {
        animationMachine.backwardTick(layer, neuronIndex, stage, data);
      },
      onStageComplete: (layer, neuronIndex, stage, data) => {
        // After 'update' stage, apply weight updates
        if (stage === 'update') {
          if (layer === 'output') {
            nn.weights_hidden2_output.data[neuronIndex] = data.newWeights;
            nn.bias_output.data[neuronIndex][0] = data.newBias;
          } else if (layer === 'layer2') {
            nn.weights_hidden1_hidden2.data[neuronIndex] = data.newWeights;
            nn.bias_hidden2.data[neuronIndex][0] = data.newBias;
          } else if (layer === 'layer1') {
            nn.weights_input_hidden1.data[neuronIndex] = data.newWeights;
            nn.bias_hidden1.data[neuronIndex][0] = data.newBias;
          }
          
          // Re-run feedforward to update calculation steps
          nn.feedforward(nn.lastInput!.toArray());
          updateVisualization();
        }
      },
      onComplete: () => {
        animationMachine.backwardComplete();
      },
      shouldStop: () => shouldStopRef.current,
      sleep,
      updateVisualization,
      speedOverride,
    });
    
    // Collect summary data
    const summaryData: BackpropSummaryData = {
      oldWeights: {
        layer1: backpropData.layer1.map(n => [...n.oldWeights]),
        layer2: backpropData.layer2.map(n => [...n.oldWeights]),
        output: backpropData.output.map(n => [...n.oldWeights]),
      },
      newWeights: {
        layer1: backpropData.layer1.map(n => [...n.newWeights]),
        layer2: backpropData.layer2.map(n => [...n.newWeights]),
        output: backpropData.output.map(n => [...n.newWeights]),
      },
      oldBiases: {
        layer1: backpropData.layer1.map(n => n.oldBias),
        layer2: backpropData.layer2.map(n => n.oldBias),
        output: backpropData.output.map(n => n.oldBias),
      },
      newBiases: {
        layer1: backpropData.layer1.map(n => n.newBias),
        layer2: backpropData.layer2.map(n => n.newBias),
        output: backpropData.output.map(n => n.newBias),
      },
      learningRate: learningRate,
      totalWeightsUpdated:
        backpropData.layer1.reduce((sum, n) => sum + n.oldWeights.length, 0) +
        backpropData.layer2.reduce((sum, n) => sum + n.oldWeights.length, 0) +
        backpropData.output.reduce((sum, n) => sum + n.oldWeights.length, 0),
    };
    
    setBackpropSummaryData(summaryData);
  }, [animationMachine, sleep, updateVisualization, learningRate]);
  
  // =========================================================================
  // Continue Animation from Jumped Position
  // =========================================================================
  const continueFromJumpedPosition = useCallback(async () => {
    const state = animationMachine.state;
    if (state.type !== 'forward_animating' && state.type !== 'backward_animating') return;
    
    const nn = nnRef.current;
    
    if (state.type === 'forward_animating') {
      const calcSteps = nn.getCalculationSteps();
      if (!calcSteps) return;
      
      const layers: Array<'layer1' | 'layer2' | 'output'> = ['layer1', 'layer2', 'output'];
      const layerSizes = { layer1: 5, layer2: 3, output: 3 };
      const layerData = { layer1: calcSteps.layer1, layer2: calcSteps.layer2, output: calcSteps.output };
      
      const baseDelay = 400;
      const connectionDelay = 150;
      const stageDurations: Record<CalculationStage, number> = {
        connections: connectionDelay,
        dotProduct: baseDelay,
        bias: baseDelay,
        activation: baseDelay,
      };
      
      let startLayerIdx = layers.indexOf(state.layer);
      let startNeuronIdx = state.neuronIndex + 1; // Start from next neuron
      
      for (let layerIdx = startLayerIdx; layerIdx < layers.length; layerIdx++) {
        const layer = layers[layerIdx];
        const startIdx = layerIdx === startLayerIdx ? startNeuronIdx : 0;
        
        for (let neuronIndex = startIdx; neuronIndex < layerSizes[layer]; neuronIndex++) {
          if (shouldStopRef.current) return;
          
          const neuronData = layerData[layer][neuronIndex];
          
          for (const stage of FORWARD_STAGES) {
            if (shouldStopRef.current) return;
            
            animationMachine.forwardTick(layer, neuronIndex, stage, neuronData);
            updateVisualization();
            await sleep(stageDurations[stage]);
          }
        }
      }
      
      animationMachine.forwardComplete();
      
    } else if (state.type === 'backward_animating') {
      const backpropData = nn.lastBackpropSteps;
      if (!backpropData) return;
      
      const layers: Array<'layer1' | 'layer2' | 'output'> = ['output', 'layer2', 'layer1'];
      const layerStartIndices = { output: 2, layer2: 2, layer1: 4 };
      const layerData = { layer1: backpropData.layer1, layer2: backpropData.layer2, output: backpropData.output };
      
      const stageDurations: Record<BackpropStage, number> = {
        error: 300,
        derivative: 350,
        gradient: 350,
        weightDelta: 350,
        allWeightDeltas: 400,
        update: 300,
      };
      
      let startLayerIdx = layers.indexOf(state.layer);
      let startNeuronIdx = state.neuronIndex - 1; // Start from previous neuron (backward order)
      
      for (let layerIdx = startLayerIdx; layerIdx < layers.length; layerIdx++) {
        const layer = layers[layerIdx];
        const startIdx = layerIdx === startLayerIdx ? startNeuronIdx : layerStartIndices[layer];
        
        for (let neuronIndex = startIdx; neuronIndex >= 0; neuronIndex--) {
          if (shouldStopRef.current) return;
          
          const neuronData = layerData[layer][neuronIndex];
          
          for (const stage of BACKPROP_STAGES) {
            if (shouldStopRef.current) return;
            
            animationMachine.backwardTick(layer, neuronIndex, stage, neuronData);
            updateVisualization();
            await sleep(stageDurations[stage]);
            
            if (stage === 'update') {
              if (layer === 'output') {
                nn.weights_hidden2_output.data[neuronIndex] = neuronData.newWeights;
                nn.bias_output.data[neuronIndex][0] = neuronData.newBias;
              } else if (layer === 'layer2') {
                nn.weights_hidden1_hidden2.data[neuronIndex] = neuronData.newWeights;
                nn.bias_hidden2.data[neuronIndex][0] = neuronData.newBias;
              } else if (layer === 'layer1') {
                nn.weights_input_hidden1.data[neuronIndex] = neuronData.newWeights;
                nn.bias_hidden1.data[neuronIndex][0] = neuronData.newBias;
              }
              nn.feedforward(nn.lastInput!.toArray());
              updateVisualization();
            }
          }
        }
      }
      
      animationMachine.backwardComplete();
      
      // Summary data
      const summaryData: BackpropSummaryData = {
        oldWeights: {
          layer1: backpropData.layer1.map(n => [...n.oldWeights]),
          layer2: backpropData.layer2.map(n => [...n.oldWeights]),
          output: backpropData.output.map(n => [...n.oldWeights]),
        },
        newWeights: {
          layer1: backpropData.layer1.map(n => [...n.newWeights]),
          layer2: backpropData.layer2.map(n => [...n.newWeights]),
          output: backpropData.output.map(n => [...n.newWeights]),
        },
        oldBiases: {
          layer1: backpropData.layer1.map(n => n.oldBias),
          layer2: backpropData.layer2.map(n => n.oldBias),
          output: backpropData.output.map(n => n.oldBias),
        },
        newBiases: {
          layer1: backpropData.layer1.map(n => n.newBias),
          layer2: backpropData.layer2.map(n => n.newBias),
          output: backpropData.output.map(n => n.newBias),
        },
        learningRate: learningRate,
        totalWeightsUpdated:
          backpropData.layer1.reduce((sum, n) => sum + n.oldWeights.length, 0) +
          backpropData.layer2.reduce((sum, n) => sum + n.oldWeights.length, 0) +
          backpropData.output.reduce((sum, n) => sum + n.oldWeights.length, 0),
      };
      setBackpropSummaryData(summaryData);
    }
  }, [animationMachine, animationSpeed, sleep, updateVisualization, learningRate]);
  
  // Store reference to continueFromJumpedPosition for useEffect
  continueFromJumpedPositionRef.current = continueFromJumpedPosition;
  
  // Track previous speed to detect resume from pause  
  const prevSpeedRef = useRef(animationMachine.state.speed);
  
  // Effect to resume animation when speed changes from 0 to > 0 while jumped
  useEffect(() => {
    const currentSpeed = animationMachine.state.speed;
    const wasZero = prevSpeedRef.current === 0;
    const isNowPositive = currentSpeed > 0;
    const isJumped = animationMachine.state.isJumped;
    
    if (wasZero && isNowPositive && isJumped) {
      // Speed was increased from paused/jumped state, continue animation
      shouldStopRef.current = false;
      if (continueFromJumpedPositionRef.current) {
        continueFromJumpedPositionRef.current();
      }
    }
    
    prevSpeedRef.current = currentSpeed;
  }, [animationMachine.state.speed, animationMachine.state.isJumped]);
  
  // =========================================================================
  // Train One Step With Animation
  // =========================================================================
  const trainOneStepWithAnimation = useCallback(async () => {
    if (animationMachine.isAnimating) {
      shouldStopRef.current = true;
      return;
    }
    
    shouldStopRef.current = false;
    animationMachine.startTraining();
    
    const nn = nnRef.current;
    const inputs = [grade, attitude, response];
    const targetOneHot = [0, 0, 0];
    targetOneHot[targetValue] = 1;
    
    // Forward pass
    nn.feedforward(inputs);
    await animateForwardPropagation();
    
    if (shouldStopRef.current) return;
    
    // Backup old weights
    const oldWeights = {
      layer1: JSON.parse(JSON.stringify(nn.weights_input_hidden1.data)),
      layer2: JSON.parse(JSON.stringify(nn.weights_hidden1_hidden2.data)),
      output: JSON.parse(JSON.stringify(nn.weights_hidden2_output.data))
    };
    const oldBiases = {
      layer1: JSON.parse(JSON.stringify(nn.bias_hidden1.data)),
      layer2: JSON.parse(JSON.stringify(nn.bias_hidden2.data)),
      output: JSON.parse(JSON.stringify(nn.bias_output.data))
    };
    
    // Train
    nn.train(inputs, targetOneHot);
    const predictions = nn.lastOutput?.toArray() || [0, 0, 0];
    const currentLoss = nn.lastLoss;
    
    // Restore old weights for backprop visualization
    nn.weights_input_hidden1.data = oldWeights.layer1;
    nn.weights_hidden1_hidden2.data = oldWeights.layer2;
    nn.weights_hidden2_output.data = oldWeights.output;
    nn.bias_hidden1.data = oldBiases.layer1;
    nn.bias_hidden2.data = oldBiases.layer2;
    nn.bias_output.data = oldBiases.output;
    nn.feedforward(inputs);
    
    setLossModalData({ targetClass: targetValue, predictions, loss: currentLoss });
  }, [grade, attitude, response, targetValue, animateForwardPropagation, animationMachine]);
  
  // =========================================================================
  // Close Loss Modal - Start Backward Propagation
  // =========================================================================
  const closeLossModal = useCallback(async () => {
    setLossModalData(null);
    animationMachine.closeLossModal();
    
    // Ensure animation speed is at least 1.0 for backprop
    if (animationSpeed === 0) {
      setAnimationSpeedLocal(1.0);
    }
    
    const nn = nnRef.current;
    shouldStopRef.current = false;
    
    // Store old weights for comparison
    const oldWeights = {
      layer1: JSON.parse(JSON.stringify(nn.weights_input_hidden1.data)),
      layer2: JSON.parse(JSON.stringify(nn.weights_hidden1_hidden2.data)),
      output: JSON.parse(JSON.stringify(nn.weights_hidden2_output.data))
    };
    const oldBiases = {
      layer1: nn.bias_hidden1.data.map(row => row[0]),
      layer2: nn.bias_hidden2.data.map(row => row[0]),
      output: nn.bias_output.data.map(row => row[0])
    };
    
    // Use speed of 1.0 if current speed is 0 (e.g., came from clicked pause)
    const backpropSpeed = animationSpeed > 0 ? animationSpeed : 1.0;
    await animateBackwardPropagation(backpropSpeed);
    await sleep(500, backpropSpeed);
    
    // Collect new weights
    const newWeights = {
      layer1: JSON.parse(JSON.stringify(nn.weights_input_hidden1.data)),
      layer2: JSON.parse(JSON.stringify(nn.weights_hidden1_hidden2.data)),
      output: JSON.parse(JSON.stringify(nn.weights_hidden2_output.data))
    };
    const newBiases = {
      layer1: nn.bias_hidden1.data.map(row => row[0]),
      layer2: nn.bias_hidden2.data.map(row => row[0]),
      output: nn.bias_output.data.map(row => row[0])
    };
    
    const comparisonData = createWeightComparisonData(oldWeights, newWeights, oldBiases, newBiases, learningRate);
    setWeightComparisonData(comparisonData);
    
    setEpoch(prev => prev + 1);
    setLoss(nn.lastLoss);
    updateVisualization();
  }, [animationMachine, animateBackwardPropagation, sleep, learningRate, updateVisualization]);
  
  // =========================================================================
  // Close Backprop Modal
  // =========================================================================
  const closeBackpropModal = useCallback(() => {
    setBackpropSummaryData(null);
    animationMachine.closeBackpropModal();
    updateVisualization();
  }, [animationMachine, updateVisualization]);
  
  // =========================================================================
  // Handle Canvas Click
  // =========================================================================
  const handleCanvasClick = useCallback((x?: number, y?: number) => {
    if (!animationMachine.isAnimating) return;
    
    const visualizer = visualizerRef.current;
    const nn = nnRef.current;
    const state = animationMachine.state;
    
    // If coordinates provided, check if clicking on a neuron
    if (x !== undefined && y !== undefined && visualizer) {
      const neuron = visualizer.findNeuronAtPosition(x, y);
      
      if (neuron && (neuron.layer === 'layer1' || neuron.layer === 'layer2' || neuron.layer === 'output')) {
        // Check if clicking on the same neuron that's already selected (works for both forward and backward)
        const isSameNeuronForward = state.type === 'forward_animating' && 
                                    state.layer === neuron.layer && 
                                    state.neuronIndex === neuron.index;
        const isSameNeuronBackward = state.type === 'backward_animating' && 
                                     state.layer === neuron.layer && 
                                     state.neuronIndex === neuron.index;
        
        if (isSameNeuronForward) {
          // Same neuron clicked in forward mode - advance to next stage
          const nextStage = getNextForwardStage(state.stage);
          if (nextStage) {
            animationMachine.forwardTick(neuron.layer, neuron.index, nextStage, state.neuronData);
            updateVisualization();
          } else {
            // All stages done for this neuron - move to next neuron manually (stay paused)
            const nextNeuron = getNextForwardNeuron(neuron.layer, neuron.index);
            if (nextNeuron) {
              // Get neuron data for the next neuron
              const calcSteps = nn.getCalculationSteps();
              if (calcSteps) {
                const layerData = { layer1: calcSteps.layer1, layer2: calcSteps.layer2, output: calcSteps.output };
                const nextNeuronData = layerData[nextNeuron.layer as 'layer1' | 'layer2' | 'output'][nextNeuron.index];
                // Jump to next neuron (this sets speed to 0)
                animationMachine.jumpToNeuron(nextNeuron.layer, nextNeuron.index);
                animationMachine.forwardTick(nextNeuron.layer, nextNeuron.index, 'dotProduct', nextNeuronData);
                updateVisualization();
              }
            } else {
              // No next neuron - forward pass complete
              // Need to run train() to prepare backprop data
              const inputs = [grade, attitude, response];
              const targetOneHot = [0, 0, 0];
              targetOneHot[targetValue] = 1;
              
              // Backup old weights
              const oldWeights = {
                layer1: JSON.parse(JSON.stringify(nn.weights_input_hidden1.data)),
                layer2: JSON.parse(JSON.stringify(nn.weights_hidden1_hidden2.data)),
                output: JSON.parse(JSON.stringify(nn.weights_hidden2_output.data))
              };
              const oldBiases = {
                layer1: JSON.parse(JSON.stringify(nn.bias_hidden1.data)),
                layer2: JSON.parse(JSON.stringify(nn.bias_hidden2.data)),
                output: JSON.parse(JSON.stringify(nn.bias_output.data))
              };
              
              // Train to prepare backprop data
              nn.train(inputs, targetOneHot);
              const predictions = nn.lastOutput?.toArray() || [0, 0, 0];
              const currentLoss = nn.lastLoss;
              
              // Restore old weights for backprop visualization
              nn.weights_input_hidden1.data = oldWeights.layer1;
              nn.weights_hidden1_hidden2.data = oldWeights.layer2;
              nn.weights_hidden2_output.data = oldWeights.output;
              nn.bias_hidden1.data = oldBiases.layer1;
              nn.bias_hidden2.data = oldBiases.layer2;
              nn.bias_output.data = oldBiases.output;
              nn.feedforward(inputs);
              
              // Show loss modal
              animationMachine.forwardComplete();
              setLossModalData({ targetClass: targetValue, predictions, loss: currentLoss });
            }
          }
          return;
        }
        
        if (isSameNeuronBackward) {
          // Same neuron clicked in backward mode - advance to next backprop stage
          const nextStage = getNextBackpropStage(state.stage);
          if (nextStage) {
            animationMachine.backwardTick(neuron.layer, neuron.index, nextStage, state.neuronData);
            updateVisualization();
          } else {
            // All stages done for this neuron - move to next neuron in backward order
            const nextNeuron = getNextBackwardNeuron(neuron.layer, neuron.index);
            if (nextNeuron) {
              const backpropData = nn.lastBackpropSteps;
              if (backpropData) {
                const layerData = { layer1: backpropData.layer1, layer2: backpropData.layer2, output: backpropData.output };
                const nextNeuronData = layerData[nextNeuron.layer as 'layer1' | 'layer2' | 'output'][nextNeuron.index];
                animationMachine.jumpToNeuron(nextNeuron.layer, nextNeuron.index);
                animationMachine.backwardTick(nextNeuron.layer, nextNeuron.index, 'error', nextNeuronData);
                updateVisualization();
              }
            } else {
              // No next neuron - backward pass complete
              animationMachine.backwardComplete();
            }
          }
          return;
        }
        
        // Different neuron clicked - jump to it
        // Stop current animation
        shouldStopRef.current = true;
        
        // Jump to this neuron
        animationMachine.jumpToNeuron(neuron.layer, neuron.index);
        
        // Set up neuron data based on current mode
        if (state.type === 'forward_animating') {
          const calcSteps = nn.getCalculationSteps();
          if (calcSteps) {
            const layerData = { layer1: calcSteps.layer1, layer2: calcSteps.layer2, output: calcSteps.output };
            const neuronData = layerData[neuron.layer][neuron.index];
            animationMachine.forwardTick(neuron.layer, neuron.index, 'dotProduct', neuronData);
          }
        } else if (state.type === 'backward_animating') {
          const backpropData = nn.lastBackpropSteps;
          if (backpropData) {
            const layerData = { layer1: backpropData.layer1, layer2: backpropData.layer2, output: backpropData.output };
            const neuronData = layerData[neuron.layer][neuron.index];
            animationMachine.backwardTick(neuron.layer, neuron.index, 'error', neuronData);
          }
        }
        
        updateVisualization();
        return;
      }
    }
    
    // Click on empty space - pause/resume or next step
    if (animationMachine.state.speed > 0) {
      animationMachine.pause();
    } else {
      // We're paused - check if jumped to a neuron
      if (state.isJumped) {
        // Advance through stages of current neuron
        if (state.type === 'forward_animating') {
          const nextStage = getNextForwardStage(state.stage);
          if (nextStage) {
            animationMachine.forwardTick(state.layer, state.neuronIndex, nextStage, state.neuronData);
            updateVisualization();
          } else {
            // All stages done - move to next neuron while staying paused
            const nextNeuron = getNextForwardNeuron(state.layer, state.neuronIndex);
            if (nextNeuron) {
              // Get neuron data for the next neuron
              const calcSteps = nn.getCalculationSteps();
              if (calcSteps) {
                const layerData = { layer1: calcSteps.layer1, layer2: calcSteps.layer2, output: calcSteps.output };
                const nextNeuronData = layerData[nextNeuron.layer as 'layer1' | 'layer2' | 'output'][nextNeuron.index];
                // Jump to next neuron (stays paused with speed=0)
                animationMachine.jumpToNeuron(nextNeuron.layer, nextNeuron.index);
                animationMachine.forwardTick(nextNeuron.layer, nextNeuron.index, 'dotProduct', nextNeuronData);
                updateVisualization();
              }
            } else {
              // No more neurons - forward pass complete
              const inputs = [grade, attitude, response];
              const targetOneHot = [0, 0, 0];
              targetOneHot[targetValue] = 1;
              
              // Backup old weights
              const oldWeights = {
                layer1: JSON.parse(JSON.stringify(nn.weights_input_hidden1.data)),
                layer2: JSON.parse(JSON.stringify(nn.weights_hidden1_hidden2.data)),
                output: JSON.parse(JSON.stringify(nn.weights_hidden2_output.data))
              };
              const oldBiases = {
                layer1: JSON.parse(JSON.stringify(nn.bias_hidden1.data)),
                layer2: JSON.parse(JSON.stringify(nn.bias_hidden2.data)),
                output: JSON.parse(JSON.stringify(nn.bias_output.data))
              };
              
              // Train to prepare backprop data
              nn.train(inputs, targetOneHot);
              const predictions = nn.lastOutput?.toArray() || [0, 0, 0];
              const currentLoss = nn.lastLoss;
              
              // Restore old weights for backprop visualization
              nn.weights_input_hidden1.data = oldWeights.layer1;
              nn.weights_hidden1_hidden2.data = oldWeights.layer2;
              nn.weights_hidden2_output.data = oldWeights.output;
              nn.bias_hidden1.data = oldBiases.layer1;
              nn.bias_hidden2.data = oldBiases.layer2;
              nn.bias_output.data = oldBiases.output;
              nn.feedforward(inputs);
              
              // Show loss modal
              animationMachine.forwardComplete();
              setLossModalData({ targetClass: targetValue, predictions, loss: currentLoss });
            }
          }
        } else if (state.type === 'backward_animating') {
          const nextStage = getNextBackpropStage(state.stage);
          if (nextStage) {
            // Apply update if completing 'update' stage
            if (state.stage === 'update' && state.neuronData) {
              const nn = nnRef.current;
              const neuronData = state.neuronData;
              if (state.layer === 'output') {
                nn.weights_hidden2_output.data[state.neuronIndex] = neuronData.newWeights;
                nn.bias_output.data[state.neuronIndex][0] = neuronData.newBias;
              } else if (state.layer === 'layer2') {
                nn.weights_hidden1_hidden2.data[state.neuronIndex] = neuronData.newWeights;
                nn.bias_hidden2.data[state.neuronIndex][0] = neuronData.newBias;
              } else if (state.layer === 'layer1') {
                nn.weights_input_hidden1.data[state.neuronIndex] = neuronData.newWeights;
                nn.bias_hidden1.data[state.neuronIndex][0] = neuronData.newBias;
              }
              nn.feedforward(nn.lastInput!.toArray());
            }
            animationMachine.backwardTick(state.layer, state.neuronIndex, nextStage, state.neuronData);
            updateVisualization();
          } else {
            // All stages done - move to next neuron while staying paused
            const nextNeuron = getNextBackwardNeuron(state.layer, state.neuronIndex);
            if (nextNeuron) {
              const backpropData = nn.lastBackpropSteps;
              if (backpropData) {
                const layerData = { layer1: backpropData.layer1, layer2: backpropData.layer2, output: backpropData.output };
                const nextNeuronData = layerData[nextNeuron.layer as 'layer1' | 'layer2' | 'output'][nextNeuron.index];
                // Jump to next neuron (stays paused with speed=0)
                animationMachine.jumpToNeuron(nextNeuron.layer, nextNeuron.index);
                animationMachine.backwardTick(nextNeuron.layer, nextNeuron.index, 'error', nextNeuronData);
                updateVisualization();
              }
            } else {
              // No more neurons - backward pass complete
              animationMachine.backwardComplete();
            }
          }
        }
      } else {
        animationMachine.nextStep();
      }
    }
  }, [animationMachine, animationSpeed, updateVisualization, continueFromJumpedPosition]);
  
  // =========================================================================
  // Next Step (manual mode button)
  // =========================================================================
  const nextStep = useCallback(() => {
    animationMachine.nextStep();
  }, [animationMachine]);
  
  // =========================================================================
  // Training Controls (non-animated)
  // =========================================================================
  const trainOneStep = useCallback(() => {
    const nn = nnRef.current;
    const inputs = [grade, attitude, response];
    const targetOneHot = [0, 0, 0];
    targetOneHot[targetValue] = 1;
    
    nn.train(inputs, targetOneHot);
    setLoss(nn.lastLoss);
    setEpoch(prev => prev + 1);
    updateVisualization();
  }, [grade, attitude, response, targetValue, updateVisualization]);
  
  const toggleTraining = useCallback(() => {
    if (isTraining) {
      setIsTraining(false);
      if (trainingIntervalRef.current) {
        clearInterval(trainingIntervalRef.current);
      }
    } else {
      setIsTraining(true);
      trainingIntervalRef.current = window.setInterval(() => {
        trainOneStep();
        if (nnRef.current.lastLoss < 0.001) {
          setIsTraining(false);
          if (trainingIntervalRef.current) {
            clearInterval(trainingIntervalRef.current);
          }
        }
      }, 50);
    }
  }, [isTraining, trainOneStep]);
  
  const reset = useCallback(() => {
    if (isTraining) {
      setIsTraining(false);
      if (trainingIntervalRef.current) {
        clearInterval(trainingIntervalRef.current);
      }
    }
    
    shouldStopRef.current = true;
    nnRef.current = new NeuralNetwork();
    setEpoch(0);
    setLoss(0);
    setOutput(null);
    setLossModalData(null);
    setBackpropSummaryData(null);
    
    animationMachine.reset();
    updateVisualization();
  }, [isTraining, updateVisualization, animationMachine]);
  
  // =========================================================================
  // Learning Rate Handler
  // =========================================================================
  const handleLearningRateChange = useCallback((v: number) => {
    setLearningRate(v);
    nnRef.current.learning_rate = v;
  }, []);
  
  // =========================================================================
  // Heatmap Toggles
  // =========================================================================
  const toggleCanvasHeatmap = useCallback(() => {
    const newValue = !showCanvasHeatmap;
    setShowCanvasHeatmap(newValue);
    if (visualizerRef.current) {
      visualizerRef.current.setHeatmapMode(newValue);
      visualizerRef.current.update(nnRef.current);
    }
  }, [showCanvasHeatmap]);

  const toggleGridHeatmap = useCallback(() => {
    setShowGridHeatmap(!showGridHeatmap);
  }, [showGridHeatmap]);
  
  // =========================================================================
  // Weight Comparison Modal
  // =========================================================================
  const openComparisonModal = useCallback(() => {
    setShowComparisonModal(true);
  }, []);

  const closeComparisonModal = useCallback(() => {
    setShowComparisonModal(false);
  }, []);
  
  // =========================================================================
  // Return
  // =========================================================================
  return {
    nn: nnRef.current,
    visualizer: visualizerRef.current,
    setVisualizer,
    grade,
    attitude,
    response,
    targetValue,
    learningRate,
    animationSpeed,
    isManualMode,
    setGrade,
    setAttitude,
    setResponse,
    setTargetValue,
    setLearningRate: handleLearningRateChange,
    setAnimationSpeed,
    setIsManualMode,
    nextStep,
    epoch,
    loss,
    output,
    steps,
    isTraining,
    isAnimating: animationMachine.isAnimating,
    showLossModal: animationMachine.state.type === 'showing_loss_modal',
    lossModalData,
    showBackpropModal: animationMachine.state.type === 'showing_backprop_modal',
    backpropSummaryData,
    trainOneStepWithAnimation,
    toggleTraining,
    reset,
    closeLossModal,
    closeBackpropModal,
    updateVisualization,
    handleCanvasClick,
    showCanvasHeatmap,
    showGridHeatmap,
    activations,
    toggleCanvasHeatmap,
    toggleGridHeatmap,
    showComparisonModal,
    weightComparisonData,
    openComparisonModal,
    closeComparisonModal,
  };
}
