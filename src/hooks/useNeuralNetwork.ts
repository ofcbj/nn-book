import { useState, useCallback, useRef } from 'react';
import { NeuralNetwork } from '../lib/network';
import { Visualizer } from '../lib/visualizer';
import type { CalculationSteps, NeuronCalculation } from '../lib/types';

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
  
  // Setters
  setGrade: (v: number) => void;
  setAttitude: (v: number) => void;
  setResponse: (v: number) => void;
  setTargetValue: (v: number) => void;
  setLearningRate: (v: number) => void;
  setAnimationSpeed: (v: number) => void;
  
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
  
  // Actions
  trainOneStepWithAnimation: () => Promise<void>;
  toggleTraining: () => void;
  reset: () => void;
  closeLossModal: () => void;
  updateVisualization: () => void;
}

export function useNeuralNetwork(): UseNeuralNetworkReturn {
  // Neural network ref (mutable, doesn't trigger re-renders)
  const nnRef = useRef(new NeuralNetwork());
  const visualizerRef = useRef<Visualizer | null>(null);
  const trainingIntervalRef = useRef<number | undefined>(undefined);
  
  // Input values
  const [grade, setGrade] = useState(0.7);
  const [attitude, setAttitude] = useState(0.5);
  const [response, setResponse] = useState(0.8);
  const [targetValue, setTargetValue] = useState(2);
  const [learningRate, setLearningRate] = useState(0.1);
  const [animationSpeed, setAnimationSpeed] = useState(1.0);
  
  // Stats
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(0);
  const [output, setOutput] = useState<number[] | null>(null);
  const [steps, setSteps] = useState<CalculationSteps | null>(null);
  
  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);
  const shouldStopAnimationRef = useRef(false);
  
  // Loss modal
  const [showLossModal, setShowLossModal] = useState(false);
  const [lossModalData, setLossModalData] = useState<{ targetClass: number; predictions: number[]; loss: number } | null>(null);

  const setVisualizer = useCallback((v: Visualizer) => {
    visualizerRef.current = v;
  }, []);

  const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
  
  const updateVisualization = useCallback(() => {
    const nn = nnRef.current;
    const inputs = [grade, attitude, response];
    nn.feedforward(inputs);
    
    if (nn.lastOutput) {
      setOutput(nn.lastOutput.toArray());
    }
    setSteps(nn.getCalculationSteps());
    
    if (visualizerRef.current) {
      visualizerRef.current.update(nn);
    }
  }, [grade, attitude, response]);
  
  const animateNeuronCalculation = async (
    layer: 'input' | 'layer1' | 'layer2' | 'output',
    index: number,
    neuronData: NeuronCalculation
  ) => {
    const visualizer = visualizerRef.current;
    if (!visualizer) return;
    
    const baseDelay = 400;
    const connectionDelay = 150;
    
    visualizer.currentNeuronData = neuronData;
    visualizer.highlightedNeuron = { layer, index };
    
    // Stage 1: Connections
    visualizer.calculationStage = 'connections';
    visualizer.intermediateValue = null;
    visualizer.update(nnRef.current);
    await sleep(connectionDelay / animationSpeed);
    
    // Stage 2: Dot product
    visualizer.calculationStage = 'dotProduct';
    visualizer.intermediateValue = neuronData.dotProduct;
    visualizer.update(nnRef.current);
    await sleep(baseDelay / animationSpeed);
    
    // Stage 3: Bias
    visualizer.calculationStage = 'bias';
    visualizer.intermediateValue = neuronData.withBias;
    visualizer.update(nnRef.current);
    await sleep(baseDelay / animationSpeed);
    
    // Stage 4: Activation
    visualizer.calculationStage = 'activation';
    visualizer.intermediateValue = neuronData.activated;
    visualizer.update(nnRef.current);
    await sleep(baseDelay / animationSpeed);
    
    // Clear
    visualizer.calculationStage = null;
    visualizer.intermediateValue = null;
    visualizer.currentNeuronData = null;
  };
  
  const animateForwardPropagation = async () => {
    const nn = nnRef.current;
    const calcSteps = nn.getCalculationSteps();
    if (!calcSteps) return;
    
    // Layer 1
    for (let i = 0; i < 5; i++) {
      if (shouldStopAnimationRef.current) break;
      await animateNeuronCalculation('layer1', i, calcSteps.layer1[i]);
    }
    
    // Layer 2
    if (!shouldStopAnimationRef.current) {
      for (let i = 0; i < 3; i++) {
        if (shouldStopAnimationRef.current) break;
        await animateNeuronCalculation('layer2', i, calcSteps.layer2[i]);
      }
    }
    
    // Output
    if (!shouldStopAnimationRef.current) {
      for (let i = 0; i < 3; i++) {
        if (shouldStopAnimationRef.current) break;
        await animateNeuronCalculation('output', i, calcSteps.output[i]);
      }
    }
    
    // Clear animation state
    if (visualizerRef.current) {
      visualizerRef.current.highlightedNeuron = null;
      visualizerRef.current.calculationStage = null;
      visualizerRef.current.intermediateValue = null;
      visualizerRef.current.update(nn);
    }
  };
  
  const animateBackwardPropagation = async () => {
    const visualizer = visualizerRef.current;
    if (!visualizer) return;
    
    const baseDelay = 250;
    
    // Output layer
    for (let i = 2; i >= 0; i--) {
      visualizer.backpropPhase = { layer: 'output', index: i };
      visualizer.update(nnRef.current);
      await sleep(baseDelay / animationSpeed);
    }
    
    // Layer 2
    for (let i = 2; i >= 0; i--) {
      visualizer.backpropPhase = { layer: 'layer2', index: i };
      visualizer.update(nnRef.current);
      await sleep(baseDelay / animationSpeed);
    }
    
    // Layer 1
    for (let i = 4; i >= 0; i--) {
      visualizer.backpropPhase = { layer: 'layer1', index: i };
      visualizer.update(nnRef.current);
      await sleep(baseDelay / animationSpeed);
    }
    
    visualizer.backpropPhase = null;
  };

  const trainOneStepWithAnimation = useCallback(async () => {
    if (isAnimating) {
      shouldStopAnimationRef.current = true;
      return;
    }
    
    setIsAnimating(true);
    shouldStopAnimationRef.current = false;
    
    const nn = nnRef.current;
    const inputs = [grade, attitude, response];
    const targetOneHot = [0, 0, 0];
    targetOneHot[targetValue] = 1;
    
    // Forward pass
    nn.feedforward(inputs);
    await animateForwardPropagation();
    
    // Train and show loss modal
    nn.train(inputs, targetOneHot);
    const predictions = nn.lastOutput?.toArray() || [0, 0, 0];
    const currentLoss = nn.lastLoss;
    
    setLossModalData({ targetClass: targetValue, predictions, loss: currentLoss });
    setShowLossModal(true);
  }, [grade, attitude, response, targetValue, isAnimating, animationSpeed]);
  
  const closeLossModal = useCallback(async () => {
    setShowLossModal(false);
    
    // Backward pass animation
    await animateBackwardPropagation();
    await sleep(500 / animationSpeed);
    
    // Update stats
    setEpoch(prev => prev + 1);
    setLoss(nnRef.current.lastLoss);
    
    // Final update
    updateVisualization();
    setIsAnimating(false);
    shouldStopAnimationRef.current = false;
  }, [animationSpeed, updateVisualization]);
  
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
    
    nnRef.current = new NeuralNetwork();
    setEpoch(0);
    setLoss(0);
    setOutput(null);
    
    if (visualizerRef.current) {
      visualizerRef.current.highlightedNeuron = null;
    }
    updateVisualization();
  }, [isTraining, updateVisualization]);

  // Update learning rate in NN
  const handleLearningRateChange = useCallback((v: number) => {
    setLearningRate(v);
    nnRef.current.learning_rate = v;
  }, []);

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
    setGrade,
    setAttitude,
    setResponse,
    setTargetValue,
    setLearningRate: handleLearningRateChange,
    setAnimationSpeed,
    epoch,
    loss,
    output,
    steps,
    isTraining,
    isAnimating,
    showLossModal,
    lossModalData,
    trainOneStepWithAnimation,
    toggleTraining,
    reset,
    closeLossModal,
    updateVisualization,
  };
}
