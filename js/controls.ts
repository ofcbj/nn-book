// UI Controls for Neural Network with TypeScript
// Handles input sliders, buttons, and training loop with step animation

import { NeuralNetwork } from './network';
import { Visualizer } from './visualizer';

// Global state
let nn: NeuralNetwork;
let visualizer: Visualizer;
let isTraining: boolean = false;
let trainingInterval: number | undefined;
let epoch: number = 0;
let currentLoss: number = 0;
let isAnimating: boolean = false;
let shouldStopAnimation: boolean = false;
let animationSpeed: number = 1.0;

function initControls(): void {
  // Initialize neural network
  nn = new NeuralNetwork();
  
  // Initialize visualizer
  visualizer = new Visualizer('networkCanvas', 'calculationDisplay');
  
  // Set up input sliders
  setupInputSliders();
  
  // Set up buttons
  const stepBtn = getElementById<HTMLButtonElement>('stepBtn');
  const trainBtn = getElementById<HTMLButtonElement>('trainBtn');
  const resetBtn = getElementById<HTMLButtonElement>('resetBtn');
  
  stepBtn.addEventListener('click', () => {
    if (isAnimating) {
      stopAnimation();
    } else {
      trainOneStepWithAnimation();
    }
  });
  
  trainBtn.addEventListener('click', toggleTraining);
  resetBtn.addEventListener('click', resetNetwork);
  
  // Set up learning rate
  const learningRateInput = getElementById<HTMLInputElement>('learningRate');
  learningRateInput.addEventListener('input', (e) => {
    const target = e.target as HTMLInputElement;
    nn.learning_rate = parseFloat(target.value);
    getElementById('learningRateValue').textContent = target.value;
  });
  
  // Set up animation speed control
  const animationSpeedInput = getElementById<HTMLInputElement>('animationSpeed');
  animationSpeedInput.addEventListener('input', (e) => {
    const target = e.target as HTMLInputElement;
    animationSpeed = parseFloat(target.value);
    getElementById('animationSpeedValue').textContent = animationSpeed.toFixed(1) + 'x';
  });
  
  // Initial forward pass
  updateVisualization();
  
  // Handle window resize
  window.addEventListener('resize', () => {
    visualizer.resizeCanvas();
    visualizer.update(nn);
  });
  
  // Target value slider
  const targetValueInput = getElementById<HTMLInputElement>('targetValue');
  targetValueInput.addEventListener('input', (e) => {
    const classNames = ['불합격', '보류', '합격'];
    const target = e.target as HTMLInputElement;
    const value = parseInt(target.value);
    getElementById('targetValueDisplay').textContent = classNames[value];
  });
}

function setupInputSliders(): void {
  const inputIds = ['grade', 'attitude', 'response'];
  
  inputIds.forEach((id) => {
    const slider = getElementById<HTMLInputElement>(id);
    const valueDisplay = getElementById(`${id}Value`);
    
    slider.addEventListener('input', (e) => {
      const target = e.target as HTMLInputElement;
      valueDisplay.textContent = target.value;
      updateVisualization();
    });
  });
}

function getInputValues(): number[] {
  return [
    parseFloat(getElementById<HTMLInputElement>('grade').value),
    parseFloat(getElementById<HTMLInputElement>('attitude').value),
    parseFloat(getElementById<HTMLInputElement>('response').value)
  ];
}

function getTargetValue(): number {
  return parseInt(getElementById<HTMLInputElement>('targetValue').value);
}

function updateVisualization(): void {
  const inputs = getInputValues();
  nn.feedforward(inputs);
  visualizer.update(nn);
  
  // Update output display (3 classes)
  if (nn.lastOutput) {
    const outputs = nn.lastOutput.toArray();
    const classNames = ['불합격', '보류', '합격'];
    const maxIndex = outputs.indexOf(Math.max(...outputs));
    
    const outputText = outputs.map((prob, i) => 
      `${classNames[i]}: ${(prob * 100).toFixed(1)}%`
    ).join(' | ');
    
    const outputElement = getElementById('currentOutput');
    outputElement.textContent = outputText;
    outputElement.style.color = 
      maxIndex === 0 ? '#ef4444' : maxIndex === 1 ? '#f59e0b' : '#22c55e';
  }
}

function stopAnimation(): void {
  shouldStopAnimation = true;
}

async function animateForwardPropagation(): Promise<void> {
  if (isAnimating) return;
  isAnimating = true;
  shouldStopAnimation = false;
  
  const stepBtn = getElementById<HTMLButtonElement>('stepBtn');
  stepBtn.textContent = '스탑';
  stepBtn.classList.add('stop-mode');
  
  const steps = nn.getCalculationSteps();
  if (!steps) {
    isAnimating = false;
    return;
  }
  
  // Animate Layer 1 neurons
  for (let i = 0; i < 5; i++) {
    if (shouldStopAnimation) break;
    await animateNeuronCalculation('layer1', i, steps.layer1[i]);
  }
  
  // Animate Layer 2 neurons
  if (!shouldStopAnimation) {
    for (let i = 0; i < 3; i++) {
      if (shouldStopAnimation) break;
      await animateNeuronCalculation('layer2', i, steps.layer2[i]);
    }
  }
  
  // Animate output neurons (3 classes)
  if (!shouldStopAnimation) {
    for (let i = 0; i < 3; i++) {
      if (shouldStopAnimation) break;
      await animateNeuronCalculation('output', i, steps.output[i]);
    }
  }
  
  // Clear all animation state
  visualizer.highlightedNeuron = null;
  visualizer.calculationStage = null;
  visualizer.intermediateValue = null;
  visualizer.update(nn);
  
  // Reset button
  stepBtn.textContent = '1 Step';
  stepBtn.classList.remove('stop-mode');
  
  isAnimating = false;
  shouldStopAnimation = false;
}

async function animateNeuronCalculation(
  layer: 'input' | 'layer1' | 'layer2' | 'output',
  index: number,
  neuronData: any
): Promise<void> {
  const baseDelay = 400;
  const connectionDelay = 150;
  
  visualizer.currentNeuronData = neuronData;
  visualizer.highlightedNeuron = { layer, index };
  
  // Stage 1: Highlight input connections
  visualizer.calculationStage = 'connections';
  visualizer.intermediateValue = null;
  visualizer.update(nn);
  await sleep(connectionDelay / animationSpeed);
  
  // Stage 2: Show dot product calculation
  visualizer.calculationStage = 'dotProduct';
  visualizer.intermediateValue = neuronData.dotProduct;
  visualizer.update(nn);
  await sleep(baseDelay / animationSpeed);
  
  // Stage 3: Show bias addition
  visualizer.calculationStage = 'bias';
  visualizer.intermediateValue = neuronData.withBias;
  visualizer.update(nn);
  await sleep(baseDelay / animationSpeed);
  
  // Stage 4: Show activation function
  visualizer.calculationStage = 'activation';
  visualizer.intermediateValue = neuronData.activated;
  visualizer.update(nn);
  await sleep(baseDelay / animationSpeed);
  
  // Clear calculation stage
  visualizer.calculationStage = null;
  visualizer.intermediateValue = null;
  visualizer.currentNeuronData = null;
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function trainOneStepWithAnimation(): Promise<void> {
  if (isAnimating) return;
  
  const inputs = getInputValues();
  const targetClass = getTargetValue();
  const targetOneHot = [0, 0, 0];
  targetOneHot[targetClass] = 1;
  
  // PHASE 1: FORWARD PASS
  nn.feedforward(inputs);
  await animateForwardPropagation();
  
  // PHASE 2: LOSS DISPLAY
  nn.train(inputs, targetOneHot);
  await showLossPhase(targetClass, nn.lastOutput!.toArray(), nn.lastLoss);
  
  // PHASE 3: BACKWARD PASS
  await animateBackwardPropagation();
  
  // PHASE 4: WEIGHT UPDATE
  await showUpdateComplete();
  
  // Calculate loss for display
  currentLoss = nn.lastLoss;
  
  // Update displays
  epoch++;
  getElementById('epochCount').textContent = epoch.toString();
  getElementById('lossValue').textContent = currentLoss.toFixed(6);
  
  // Final update
  updateVisualization();
}

async function showLossPhase(
  targetClass: number,
  output: number[],
  loss: number
): Promise<void> {
  const classNames = ['불합격', '보류', '합격'];
  
  const targetOneHot = [0, 0, 0];
  targetOneHot[targetClass] = 1;
  
  // Get modal elements
  const modal = getElementById('lossModal');
  const modalTargetClass = getElementById('modalTargetClass');
  const modalTargetVector = getElementById('modalTargetVector');
  const modalPredictions = getElementById('modalPredictions');
  const modalPredVector = getElementById('modalPredVector');
  const modalLossValue = getElementById('modalLossValue');
  const startBackwardBtn = getElementById<HTMLButtonElement>('startBackwardBtn');
  
  const explainTargetClass = getElementById('explainTargetClass');
  const explainLossValue = getElementById('explainLossValue');
  const lossInterpretation = getElementById('lossInterpretation');
  
  // Populate modal with data
  modalTargetClass.textContent = classNames[targetClass];
  modalTargetVector.textContent = `[${targetOneHot.join(', ')}]`;
  modalPredVector.textContent = `[${output.map(p => p.toFixed(3)).join(', ')}]`;
  modalLossValue.textContent = loss.toFixed(6);
  
  // Populate explanation section
  explainTargetClass.textContent = classNames[targetClass];
  explainLossValue.textContent = loss.toFixed(4);
  
  // Interpret loss value
  let interpretation: string;
  if (loss < 0.1) {
    interpretation = '매우 작은 값으로, 예측이 정답에 매우 가깝습니다';
  } else if (loss < 0.5) {
    interpretation = '작은 값으로, 예측이 정답에 가깝습니다';
  } else if (loss < 1.0) {
    interpretation = '중간 값으로, 예측과 정답 사이에 차이가 있습니다';
  } else {
    interpretation = '큰 값으로, 예측이 정답과 많이 다릅니다';
  }
  lossInterpretation.textContent = interpretation;
  
  // Populate loss calculation section
  const lossFormulaExpanded = getElementById('lossFormulaExpanded');
  const lossFormulaSimplified = getElementById('lossFormulaSimplified');
  const lossFormulaResult = getElementById('lossFormulaResult');
  const lossNote = getElementById('lossNote');
  
  // Build expanded formula
  const terms = output.map((prob, i) => 
    `${targetOneHot[i]}×log(${prob.toFixed(3)})`
  ).join(' + ');
  lossFormulaExpanded.textContent = `L = -(${terms})`;
  
  // Simplified formula
  const targetProb = output[targetClass];
  lossFormulaSimplified.textContent = `L = -log(${targetProb.toFixed(3)})`;
  
  // Result
  lossFormulaResult.textContent = `L = ${loss.toFixed(6)}`;
  
  // Dynamic note
  const targetProbPercent = (targetProb * 100).toFixed(1);
  const otherProbs = output.filter((_, i) => i !== targetClass);
  const otherProbsSum = otherProbs.reduce((sum, p) => sum + p, 0);
  const otherProbsPercent = (otherProbsSum * 100).toFixed(1);
  
  let noteText: string;
  
  if (targetProb > 0.9) {
    noteText = `💡 <strong>${classNames[targetClass]}</strong>의 확률이 <strong>${targetProbPercent}%</strong>로 매우 높아 Loss가 작습니다! 신경망이 정답을 확신하고 있습니다.<br><br>
    🎯 <strong>이상적인 경우:</strong> 확률 100% → Loss = -log(1) = <strong>0</strong>`;
  } else if (targetProb > 0.7) {
    noteText = `💡 <strong>${classNames[targetClass]}</strong>의 확률이 <strong>${targetProbPercent}%</strong>로 높아 Loss가 비교적 작습니다.<br><br>
    💭 다른 클래스들에게 <strong>${otherProbsPercent}%</strong>를 줬습니다. 이 확률을 더 낮추면 Loss가 줄어듭니다!`;
  } else if (targetProb > 0.5) {
    noteText = `💡 <strong>${classNames[targetClass]}</strong>을 가장 높게 예측했지만, 확률이 <strong>${targetProbPercent}%</strong>밖에 안 되어 Loss가 큽니다.<br><br>
    🔍 <strong>문제점:</strong> 다른 클래스들에게 <strong>${otherProbsPercent}%</strong>를 너무 많이 줬습니다!<br>
    📊 <strong>Softmax 특성:</strong> 모든 확률의 합 = 100% → 정답 클래스의 확률을 높이려면 다른 클래스에서 "빼앗아와야" 합니다.<br>
    ⚡ <strong>Backpropagation의 역할:</strong> 정답 확률 ↑, 오답 확률 ↓ 방향으로 가중치를 조정합니다.`;
  } else {
    const bestClass = output.indexOf(Math.max(...output));
    const bestProb = (Math.max(...output) * 100).toFixed(1);
    noteText = `⚠️ <strong>${classNames[targetClass]}</strong>의 확률이 <strong>${targetProbPercent}%</strong>로 낮아 Loss가 매우 큽니다!<br><br>
    ❌ 현재 가장 높게 예측한 클래스: <strong>${classNames[bestClass]}</strong> (${bestProb}%)<br>
    🎯 목표: <strong>${classNames[targetClass]}</strong>의 확률을 높이고, 나머지 클래스의 확률을 낮춰야 합니다.<br>
    📉 Loss = -log(${targetProb.toFixed(3)}) = ${(-Math.log(targetProb)).toFixed(3)} (매우 큼)`;
  }
  lossNote.innerHTML = noteText;
  
  // Create prediction bars
  modalPredictions.innerHTML = '';
  output.forEach((prob, i) => {
    const barDiv = document.createElement('div');
    barDiv.className = 'prediction-bar';
    
    const label = document.createElement('div');
    label.className = 'prediction-bar-label';
    label.textContent = classNames[i];
    
    const fillContainer = document.createElement('div');
    fillContainer.className = 'prediction-bar-fill';
    
    const fill = document.createElement('div');
    fill.className = 'prediction-bar-inner';
    fill.style.width = (prob * 100) + '%';
    if (i === targetClass) {
      fill.style.background = 'linear-gradient(90deg, #22c55e 0%, #34d399 100%)';
    }
    
    fillContainer.appendChild(fill);
    
    const value = document.createElement('div');
    value.className = 'prediction-bar-value';
    value.textContent = (prob * 100).toFixed(1) + '%';
    
    barDiv.appendChild(label);
    barDiv.appendChild(fillContainer);
    barDiv.appendChild(value);
    modalPredictions.appendChild(barDiv);
  });
  
  // Show modal
  modal.style.display = 'flex';
  
  // Return a Promise that resolves when user clicks the button
  return new Promise((resolve) => {
    const handleClick = () => {
      modal.style.display = 'none';
      startBackwardBtn.removeEventListener('click', handleClick);
      resolve();
    };
    
    startBackwardBtn.addEventListener('click', handleClick);
  });
}

async function animateBackwardPropagation(): Promise<void> {
  const baseDelay = 250;
  
  // Backward through output layer
  for (let i = 2; i >= 0; i--) {
    visualizer.backpropPhase = { layer: 'output', index: i };
    visualizer.update(nn);
    await sleep(baseDelay / animationSpeed);
  }
  
  // Backward through layer 2
  for (let i = 2; i >= 0; i--) {
    visualizer.backpropPhase = { layer: 'layer2',  index: i };
    visualizer.update(nn);
    await sleep(baseDelay / animationSpeed);
  }
  
  // Backward through layer 1
  for (let i = 4; i >= 0; i--) {
    visualizer.backpropPhase = { layer: 'layer1', index: i };
    visualizer.update(nn);
    await sleep(baseDelay / animationSpeed);
  }
  
  visualizer.backpropPhase = null;
}

async function showUpdateComplete(): Promise<void> {
  await sleep(500 / animationSpeed);
}

function trainOneStep(): void {
  const inputs = getInputValues();
  const targetClass = getTargetValue();
  const targetOneHot = [0, 0, 0];
  targetOneHot[targetClass] = 1;
  
  nn.train(inputs, targetOneHot);
  currentLoss = nn.lastLoss;
  
  epoch++;
  getElementById('epochCount').textContent = epoch.toString();
  getElementById('lossValue').textContent = currentLoss.toFixed(6);
  
  updateVisualization();
}

function toggleTraining(): void {
  const btn = getElementById<HTMLButtonElement>('trainBtn');
  
  if (isTraining) {
    isTraining = false;
    if (trainingInterval) {
      clearInterval(trainingInterval);
    }
    btn.textContent = '학습 시작';
    btn.classList.remove('training');
  } else {
    isTraining = true;
    btn.textContent = '학습 중지';
    btn.classList.add('training');
    
    trainingInterval = window.setInterval(() => {
      trainOneStep();
      
      if (currentLoss < 0.001) {
        toggleTraining();
      }
    }, 50);
  }
}

function resetNetwork(): void {
  if (isTraining) {
    toggleTraining();
  }
  
  nn = new NeuralNetwork();
  epoch = 0;
  currentLoss = 0;
  
  getElementById('epochCount').textContent = '0';
  getElementById('lossValue').textContent = '0.000000';
  getElementById('currentOutput').textContent = '0.000';
  
  visualizer.highlightedNeuron = null;
  updateVisualization();
}

// Helper function for type-safe getElementById
function getElementById<T extends HTMLElement = HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Element with id "${id}" not found`);
  }
  return element as T;
}

// Initialize when DOM is loaded
// For ES modules, check if DOM is already loaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initControls);
} else {
  // DOM is already loaded (common for ES modules)
  initControls();
}

// Export for debugging  
if (typeof window !== 'undefined') {
  (window as any).getNN = () => nn;
  (window as any).getVisualizer = () => visualizer;
}
