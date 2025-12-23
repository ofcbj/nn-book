import { useState, useCallback, useRef, useEffect } from 'react';
import { NeuralNetwork } from '../lib/network';
import { Visualizer } from '../lib/visualizer';
import type { CalculationSteps, NeuronCalculation, BackpropSummaryData } from '../lib/types';
import type { ActivationData } from '../components/ActivationHeatmap';

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

  // Actions
  trainOneStepWithAnimation: () => Promise<void>;
  toggleTraining: () => void;
  reset: () => void;
  closeLossModal: () => void;
  closeBackpropModal: () => void;
  updateVisualization: () => void;
  handleCanvasClick: () => void;
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
  const [isManualMode, setIsManualMode] = useState(false);

  // Use ref for animation speed so it's always current
  const animationSpeedRef = useRef(animationSpeed);
  const manualModeRef = useRef(isManualMode);
  const manualStepResolverRef = useRef<(() => void) | null>(null);
  const prevAnimationSpeedRef = useRef(animationSpeed);

  // Update ref whenever animationSpeed changes
  animationSpeedRef.current = animationSpeed;
  manualModeRef.current = isManualMode;

  // When animation speed changes from 0 to > 0, auto-resume animation
  useEffect(() => {
    if (prevAnimationSpeedRef.current === 0 && animationSpeed > 0) {
      // Speed was increased from 0, resume animation automatically
      if (manualStepResolverRef.current) {
        manualStepResolverRef.current();
        manualStepResolverRef.current = null;
      }
    }
    prevAnimationSpeedRef.current = animationSpeed;
  }, [animationSpeed]);

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

  // Backprop summary modal
  const [showBackpropModal, setShowBackpropModal] = useState(false);
  const [backpropSummaryData, setBackpropSummaryData] = useState<BackpropSummaryData | null>(null);

  // Heatmap visualization
  const [showCanvasHeatmap, setShowCanvasHeatmap] = useState(false);
  const [showGridHeatmap, setShowGridHeatmap] = useState(true);
  const [activations, setActivations] = useState<ActivationData | null>(null);

  const setVisualizer = useCallback((v: Visualizer) => {
    visualizerRef.current = v;
  }, []);

  const sleep = (ms: number) => {
    // Manual mode or speed is 0: wait for user to click "Next Step"
    if (manualModeRef.current || animationSpeedRef.current === 0) {
      return new Promise<void>(resolve => {
        manualStepResolverRef.current = resolve;
      });
    } else {
      return new Promise(resolve => setTimeout(resolve, ms));
    }
  };

  const nextStep = useCallback(() => {
    if (manualStepResolverRef.current) {
      manualStepResolverRef.current();
      manualStepResolverRef.current = null;
    }
  }, []);
  
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
    await sleep(connectionDelay / animationSpeedRef.current);

    // Stage 2: Dot product
    visualizer.calculationStage = 'dotProduct';
    visualizer.intermediateValue = neuronData.dotProduct;
    visualizer.update(nnRef.current);
    await sleep(baseDelay / animationSpeedRef.current);

    // Stage 3: Bias
    visualizer.calculationStage = 'bias';
    visualizer.intermediateValue = neuronData.withBias;
    visualizer.update(nnRef.current);
    await sleep(baseDelay / animationSpeedRef.current);

    // Stage 4: Activation
    visualizer.calculationStage = 'activation';
    visualizer.intermediateValue = neuronData.activated;
    visualizer.update(nnRef.current);
    await sleep(baseDelay / animationSpeedRef.current);

    // Clear - important to reset state before next neuron
    visualizer.calculationStage = null;
    visualizer.intermediateValue = null;
    visualizer.currentNeuronData = null;
    visualizer.highlightedNeuron = null;
    visualizer.update(nnRef.current);
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
  
  const animateBackpropNeuron = async (
    layer: 'layer1' | 'layer2' | 'output',
    index: number,
    neuronData: any
  ) => {
    const visualizer = visualizerRef.current;
    if (!visualizer) return;

    const stages: Array<'error' | 'derivative' | 'gradient' | 'weightDelta' | 'allWeightDeltas' | 'update'> =
      ['error', 'derivative', 'gradient', 'weightDelta', 'allWeightDeltas', 'update'];

    const stageDuration = [300, 350, 350, 350, 400, 300]; // Duration for each stage

    visualizer.backpropPhase = { layer, index };
    visualizer.currentBackpropData = neuronData;

    for (let i = 0; i < stages.length; i++) {
      if (shouldStopAnimationRef.current) break;

      visualizer.backpropStage = stages[i];
      visualizer.update(nnRef.current);
      await sleep(stageDuration[i] / animationSpeedRef.current);
      
      // After 'update' stage animation completes, apply the weight and bias updates
      if (stages[i] === 'update') {
        const nn = nnRef.current;
        if (layer === 'output') {
          nn.weights_hidden2_output.data[index] = neuronData.newWeights;
          nn.bias_output.data[index][0] = neuronData.newBias;
        } else if (layer === 'layer2') {
          nn.weights_hidden1_hidden2.data[index] = neuronData.newWeights;
          nn.bias_hidden2.data[index][0] = neuronData.newBias;
        } else if (layer === 'layer1') {
          nn.weights_input_hidden1.data[index] = neuronData.newWeights;
          nn.bias_hidden1.data[index][0] = neuronData.newBias;
        }
        
        // Re-run feedforward with new weights to update CalculationSteps
        nn.feedforward(nn.lastInput!.toArray());
        
        // Update visualization with new weights
        visualizer.update(nnRef.current);
      }
    }

    visualizer.backpropStage = null;
  };

  const animateBackwardPropagation = async () => {
    const visualizer = visualizerRef.current;
    const nn = nnRef.current;
    if (!visualizer || !nn.lastBackpropSteps) return;

    const backpropData = nn.lastBackpropSteps;

    // Output layer
    for (let i = 2; i >= 0; i--) {
      if (shouldStopAnimationRef.current) break;
      await animateBackpropNeuron('output', i, backpropData.output[i]);
    }

    // Layer 2
    if (!shouldStopAnimationRef.current) {
      for (let i = 2; i >= 0; i--) {
        if (shouldStopAnimationRef.current) break;
        await animateBackpropNeuron('layer2', i, backpropData.layer2[i]);
      }
    }

    // Layer 1
    if (!shouldStopAnimationRef.current) {
      for (let i = 4; i >= 0; i--) {
        if (shouldStopAnimationRef.current) break;
        await animateBackpropNeuron('layer1', i, backpropData.layer1[i]);
      }
    }

    visualizer.backpropPhase = null;
    visualizer.currentBackpropData = null;
    visualizer.backpropStage = null;

    // Collect backprop summary data and show modal
    if (!shouldStopAnimationRef.current) {
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
      setShowBackpropModal(true);
    }
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
    
    // Backup old weights before training
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
    
    // Train and show loss modal
    nn.train(inputs, targetOneHot);
    const predictions = nn.lastOutput?.toArray() || [0, 0, 0];
    const currentLoss = nn.lastLoss;
    
    // Restore old weights for backprop animation
    nn.weights_input_hidden1.data = oldWeights.layer1;
    nn.weights_hidden1_hidden2.data = oldWeights.layer2;
    nn.weights_hidden2_output.data = oldWeights.output;
    nn.bias_hidden1.data = oldBiases.layer1;
    nn.bias_hidden2.data = oldBiases.layer2;
    nn.bias_output.data = oldBiases.output;
    
    // Re-run feedforward with old weights to update CalculationSteps
    nn.feedforward(inputs);
    
    setLossModalData({ targetClass: targetValue, predictions, loss: currentLoss });
    setShowLossModal(true);
  }, [grade, attitude, response, targetValue, isAnimating, animationSpeed]);
  
  const closeLossModal = useCallback(async () => {
    setShowLossModal(false);

    // Backward pass animation
    await animateBackwardPropagation();
    await sleep(500 / animationSpeedRef.current);

    // Update stats
    setEpoch(prev => prev + 1);
    setLoss(nnRef.current.lastLoss);

    // Final update
    updateVisualization();
    setIsAnimating(false);
    shouldStopAnimationRef.current = false;
  }, [updateVisualization]);

  const closeBackpropModal = useCallback(() => {
    setShowBackpropModal(false);
  }, []);
  
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

  // Handle canvas click during animation
  const handleCanvasClick = useCallback(() => {
    if (!isAnimating) return;

    // If animation is playing (speed > 0), pause it
    if (animationSpeed > 0) {
      setAnimationSpeed(0);
    } else {
      // If paused (speed === 0), go to next step
      nextStep();
    }
  }, [isAnimating, animationSpeed, nextStep]);

  // Heatmap toggles
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
    isAnimating,
    showLossModal,
    lossModalData,
    showBackpropModal,
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
  };
}
