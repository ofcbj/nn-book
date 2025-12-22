// UI Controls for Neural Network
// Handles input sliders, buttons, and training loop with step animation

let nn;
let visualizer;
let isTraining = false;
let trainingInterval;
let epoch = 0;
let currentLoss = 0;
let isAnimating = false;
let shouldStopAnimation = false; // Flag to stop animation
let animationSpeed = 1.0; // Default speed (1.0x)

function initControls() {
  // Initialize neural network
  nn = new NeuralNetwork();
  
  // Initialize visualizer
  visualizer = new Visualizer('networkCanvas', 'calculationDisplay');
  
  // Set up input sliders
  setupInputSliders();
  
  // Set up buttons
  document.getElementById('stepBtn').addEventListener('click', () => {
    if (isAnimating) {
      // Stop the animation
      stopAnimation();
    } else {
      // Start animation
      trainOneStepWithAnimation();
    }
  });
  document.getElementById('trainBtn').addEventListener('click', toggleTraining);
  document.getElementById('resetBtn').addEventListener('click', resetNetwork);
  
  // Set up learning rate
  document.getElementById('learningRate').addEventListener('input', (e) => {
    nn.learning_rate = parseFloat(e.target.value);
    document.getElementById('learningRateValue').textContent = e.target.value;
  });
  
  // Set up animation speed control
  document.getElementById('animationSpeed').addEventListener('input', (e) => {
    animationSpeed = parseFloat(e.target.value);
    document.getElementById('animationSpeedValue').textContent = animationSpeed.toFixed(1) + 'x';
  });
  
  // Initial forward pass
  updateVisualization();
  
  // Handle window resize
  window.addEventListener('resize', () => {
    visualizer.resizeCanvas();
    visualizer.update(nn);
  });
  // Target value slider
  document.getElementById('targetValue').addEventListener('input', (e) => {
    const classNames = ['불합격', '보류', '합격'];
    const value = parseInt(e.target.value);
    document.getElementById('targetValueDisplay').textContent = classNames[value];
  });
}

function setupInputSliders() {
  const inputIds = ['grade', 'attitude', 'response'];
  const labels = ['성적', '태도', '응답수준'];
  
  inputIds.forEach((id, idx) => {
    const slider = document.getElementById(id);
    const valueDisplay = document.getElementById(`${id}Value`);
    
    slider.addEventListener('input', (e) => {
      valueDisplay.textContent = e.target.value;
      updateVisualization();
    });
  });
}

function getInputValues() {
  return [
    parseFloat(document.getElementById('grade').value),
    parseFloat(document.getElementById('attitude').value),
    parseFloat(document.getElementById('response').value)
  ];
}

function getTargetValue() {
  return parseInt(document.getElementById('targetValue').value);
}

function updateVisualization() {
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
    
    document.getElementById('currentOutput').textContent = outputText;
    document.getElementById('currentOutput').style.color = 
      maxIndex === 0 ? '#ef4444' : maxIndex === 1 ? '#f59e0b' : '#22c55e';
  }
}

function stopAnimation() {
  shouldStopAnimation = true;
  // Button will be reset when animation actually stops
}

// Animate forward propagation step-by-step with detailed calculations
async function animateForwardPropagation() {
  if (isAnimating) return;
  isAnimating = true;
  shouldStopAnimation = false;
  
  // Change button to "스탑"
  const stepBtn = document.getElementById('stepBtn');
  stepBtn.textContent = '스탑';
  stepBtn.classList.add('stop-mode');
  
  const steps = nn.getCalculationSteps();
  
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
  
  // Animate output neuron (3 classes)
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

async function animateNeuronCalculation(layer, index, neuronData) {
  const baseDelay = 400; // Base delay in ms for calculation stages
  const connectionDelay = 150; // Shorter delay for connection highlighting
  
  // Set neuron data for formula display
  visualizer.currentNeuronData = neuronData;
  visualizer.highlightedNeuron = { layer: layer, index: index };
  
  // Stage 1: Highlight input connections (shorter duration)
  visualizer.calculationStage = 'connections';
  visualizer.intermediateValue = null;
  visualizer.update(nn);
  await sleep(connectionDelay / animationSpeed);
  
  // Stage 2: Show dot product calculation (expanded formula)
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
  
  // Clear calculation stage but keep neuron highlighted briefly
  visualizer.calculationStage = null;
  visualizer.intermediateValue = null;
  visualizer.currentNeuronData = null;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function trainOneStepWithAnimation() {
  if (isAnimating) return;
  
  const inputs = getInputValues();
  const targetClass = getTargetValue();
  const targetOneHot = [0, 0, 0];
  targetOneHot[targetClass] = 1; // Convert to one-hot encoding
  
  // === PHASE 1: FORWARD PASS ===
  nn.feedforward(inputs);
  await animateForwardPropagation();
  
  // === PHASE 2: LOSS DISPLAY (with user control) ===
  nn.train(inputs, targetOneHot); // This computes gradients and loss
  await showLossPhase(targetClass, nn.lastOutput.toArray(), nn.lastLoss);
  
  // User has clicked "Start Backward Process" button, now continue
  
  // === PHASE 3: BACKWARD PASS ===
  await animateBackwardPropagation();
  
  // === PHASE 4: WEIGHT UPDATE ===
  // (Weight updates already applied in nn.train)
  // Just show completion message
  await showUpdateComplete();
  
  // Calculate loss for display
  const output = nn.lastOutput.toArray();
  currentLoss = nn.lastLoss;
  
  // Update displays
  epoch++;
  document.getElementById('epochCount').textContent = epoch;
  document.getElementById('lossValue').textContent = currentLoss.toFixed(6);
  
  // Final update
  updateVisualization();
}

async function showLossPhase(targetClass, output, loss) {
  const classNames = ['불합격', '보류', '합격'];
  
  // Create one-hot encoded target vector
  const targetOneHot = [0, 0, 0];
  targetOneHot[targetClass] = 1;
  
  // Get modal elements
  const modal = document.getElementById('lossModal');
  const modalTargetClass = document.getElementById('modalTargetClass');
  const modalTargetVector = document.getElementById('modalTargetVector');
  const modalPredictions = document.getElementById('modalPredictions');
  const modalPredVector = document.getElementById('modalPredVector');
  const modalLossValue = document.getElementById('modalLossValue');
  const startBackwardBtn = document.getElementById('startBackwardBtn');
  
  // Get explanation elements
  const explainTargetClass = document.getElementById('explainTargetClass');
  const explainLossValue = document.getElementById('explainLossValue');
  const lossInterpretation = document.getElementById('lossInterpretation');
  
  // Populate modal with data
  modalTargetClass.textContent = classNames[targetClass];
  modalTargetVector.textContent = `[${targetOneHot.join(', ')}]`;
  modalPredVector.textContent = `[${output.map(p => p.toFixed(3)).join(', ')}]`;
  modalLossValue.textContent = loss.toFixed(6);
  
  // Populate explanation section
  explainTargetClass.textContent = classNames[targetClass];
  explainLossValue.textContent = loss.toFixed(4);
  
  // Interpret loss value
  let interpretation;
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
  const lossFormulaExpanded = document.getElementById('lossFormulaExpanded');
  const lossFormulaSimplified = document.getElementById('lossFormulaSimplified');
  const lossFormulaResult = document.getElementById('lossFormulaResult');
  const lossNote = document.getElementById('lossNote');
  
  // Build expanded formula with actual values
  const terms = output.map((prob, i) => 
    `${targetOneHot[i]}×log(${prob.toFixed(3)})`
  ).join(' + ');
  lossFormulaExpanded.textContent = `L = -(${terms})`;
  
  // Simplified formula (only the target class term)
  const targetProb = output[targetClass];
  lossFormulaSimplified.textContent = `L = -log(${targetProb.toFixed(3)})`;
  
  // Result
  lossFormulaResult.textContent = `L = ${loss.toFixed(6)}`;
  
  // Dynamic note based on prediction
  const targetProbPercent = (targetProb * 100).toFixed(1);
  const otherProbs = output.filter((_, i) => i !== targetClass);
  const otherProbsSum = otherProbs.reduce((sum, p) => sum + p, 0);
  const otherProbsPercent = (otherProbsSum * 100).toFixed(1);
  
  let noteText;
  
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
      // Hide modal
      modal.style.display = 'none';
      // Remove event listener
      startBackwardBtn.removeEventListener('click', handleClick);
      // Resolve promise to continue with backward pass
      resolve();
    };
    
    startBackwardBtn.addEventListener('click', handleClick);
  });
}

async function animateBackwardPropagation() {
  const baseDelay = 250;
  
  // Backward through output layer (3 neurons)
  for (let i = 2; i >= 0; i--) {
    visualizer.backpropPhase = { layer: 'output', index: i };
    visualizer.update(nn);
    await sleep(baseDelay / animationSpeed);
  }
  
  // Backward through layer 2 (3 neurons)
  for (let i = 2; i >= 0; i--) {
    visualizer.backpropPhase = { layer: 'layer2', index: i };
    visualizer.update(nn);
    await sleep(baseDelay / animationSpeed);
  }
  
  // Backward through layer 1 (5 neurons)
  for (let i = 4; i >= 0; i--) {
    visualizer.backpropPhase = { layer: 'layer1', index: i };
    visualizer.update(nn);
    await sleep(baseDelay / animationSpeed);
  }
  
  visualizer.backpropPhase = null;
}

async function showUpdateComplete() {
  // Brief pause to show completion
  await sleep(500 / animationSpeed);
}

function trainOneStep() {
  const inputs = getInputValues();
  const targetClass = getTargetValue();
  const targetOneHot = [0, 0, 0];
  targetOneHot[targetClass] = 1;
  
  // Train
  nn.train(inputs, targetOneHot);
  
  // Use stored loss
  currentLoss = nn.lastLoss;
  
  // Update displays
  epoch++;
  document.getElementById('epochCount').textContent = epoch;
  document.getElementById('lossValue').textContent = currentLoss.toFixed(6);
  
  // Update visualization
  updateVisualization();
}

function toggleTraining() {
  const btn = document.getElementById('trainBtn');
  
  if (isTraining) {
    // Stop training
    isTraining = false;
    clearInterval(trainingInterval);
    btn.textContent = '학습 시작';
    btn.classList.remove('training');
  } else {
    // Start training
    isTraining = true;
    btn.textContent = '학습 중지';
    btn.classList.add('training');
    
    trainingInterval = setInterval(() => {
      trainOneStep();
      
      // Auto-stop if loss is very small
      if (currentLoss < 0.001) {
        toggleTraining();
      }
    }, 50); // Train every 50ms
  }
}

function resetNetwork() {
  // Stop training if running
  if (isTraining) {
    toggleTraining();
  }
  
  // Reset network
  nn = new NeuralNetwork();
  epoch = 0;
  currentLoss = 0;
  
  // Reset displays
  document.getElementById('epochCount').textContent = '0';
  document.getElementById('lossValue').textContent = '0.000000';
  document.getElementById('currentOutput').textContent = '0.000';
  
  // Reset visualization
  visualizer.highlightedNeuron = null;
  updateVisualization();
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initControls);

