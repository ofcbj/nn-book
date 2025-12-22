// Vector-Based Visualization for Neural Network
// Uses elliptical nodes to show vectors

class Visualizer {
  constructor(canvasId, textDisplayId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.textDisplay = document.getElementById(textDisplayId);
    
    this.inputLabels = ['성적', '태도', '응답수준'];
    this.highlightedNeuron = null; // {layer: 'layer1', index: 0}
    
    // Calculation animation properties
    this.calculationStage = null; // 'connections', 'dotProduct', 'bias', 'activation'
    this.intermediateValue = null; // Current intermediate calculation value
    this.activeConnections = []; // Array of active connection indices
    this.currentNeuronData = null; // Full neuron calculation data (inputs, weights, etc.)
    
    // Backpropagation visualization
    this.showLoss = null; // {targetClass, targetName, predictions, loss}
    this.backpropPhase = null; // {layer, index}
    
    this.resizeCanvas();
  }

  resizeCanvas() {
    this.canvas.width = this.canvas.offsetWidth;
    this.canvas.height = this.canvas.offsetHeight;
  }

  drawRoundedRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }

  drawInputVector(ctx, x, y, values) {
    const width = 140;
    const height = 100;
    const centerX = x - width / 2;
    const centerY = y - height / 2;
    
    // Draw ellipse background
    this.drawRoundedRect(ctx, centerX, centerY, width, height, 15);
    
    const gradient = ctx.createLinearGradient(centerX, centerY, centerX, centerY + height);
    gradient.addColorStop(0, 'rgba(59, 130, 246, 0.3)');
    gradient.addColorStop(1, 'rgba(37, 99, 235, 0.2)');
    ctx.fillStyle = gradient;
    ctx.fill();
    
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw title
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('면접자', x, centerY + 20);
    
    // Draw vector values
    ctx.font = '12px monospace';
    ctx.textAlign = 'left';
    const startY = centerY + 40;
    values.forEach((val, idx) => {
      ctx.fillStyle = '#cbd5e1';
      ctx.fillText(this.inputLabels[idx] + ':', centerX + 15, startY + idx * 18);
      ctx.fillStyle = '#60a5fa';
      ctx.font = 'bold 12px monospace';
      ctx.fillText(val.toFixed(2), centerX + 80, startY + idx * 18);
      ctx.font = '12px monospace';
    });
    
    return { x: centerX, y: centerY, width, height, centerX: x, centerY: y };
  }

  drawNeuronVector(ctx, x, y, weights, bias, activation, label, layerType, isHighlighted = false) {
    // Adjust width based on layer type to prevent text overflow
    const baseWidth = weights.length * 25;
    let width = Math.max(130, baseWidth + 40);
    if (layerType === 'layer1') {
      width += 30; // 1차 면접관: +30px
    } else if (layerType === 'layer2') {
      width += 60; // 2차 면접관: +60px
    } else if (layerType === 'output') {
      width += 20; // 최종 결정: +20px
    }
    const height = 90;
    const centerX = x - width / 2;
    const centerY = y - height / 2;
    
    // Draw ellipse background
    this.drawRoundedRect(ctx, centerX, centerY, width, height, 12);
    
    let gradient;
    let strokeColor;
    
    if (layerType === 'layer1') {
      gradient = ctx.createLinearGradient(centerX, centerY, centerX, centerY + height);
      if (isHighlighted) {
        gradient.addColorStop(0, 'rgba(34, 197, 94, 0.9)');
        gradient.addColorStop(1, 'rgba(22, 163, 74, 0.7)');
      } else {
        gradient.addColorStop(0, 'rgba(34, 197, 94, 0.3)');
        gradient.addColorStop(1, 'rgba(22, 163, 74, 0.2)');
      }
      ctx.fillStyle = gradient;
      strokeColor = isHighlighted ? '#4ade80' : '#22c55e';
    } else if (layerType === 'layer2') {
      gradient = ctx.createLinearGradient(centerX, centerY, centerX, centerY + height);
      if (isHighlighted) {
        gradient.addColorStop(0, 'rgba(249, 115, 22, 0.9)');
        gradient.addColorStop(1, 'rgba(234, 88, 12, 0.7)');
      } else {
        gradient.addColorStop(0, 'rgba(249, 115, 22, 0.3)');
        gradient.addColorStop(1, 'rgba(234, 88, 12, 0.2)');
      }
      ctx.fillStyle = gradient;
      strokeColor = isHighlighted ? '#fb923c' : '#f97316';
    } else {
      gradient = ctx.createLinearGradient(centerX, centerY, centerX, centerY + height);
      if (isHighlighted) {
        gradient.addColorStop(0, 'rgba(239, 68, 68, 0.9)');
        gradient.addColorStop(1, 'rgba(220, 38, 38, 0.7)');
      } else {
        gradient.addColorStop(0, 'rgba(239, 68, 68, 0.3)');
        gradient.addColorStop(1, 'rgba(220, 38, 38, 0.2)');
      } 
      ctx.fillStyle = gradient;
      strokeColor = isHighlighted ? '#f87171' : '#ef4444';
    }
    
    ctx.fill();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = isHighlighted ? 4 : 2;
    ctx.stroke();
    
    // Draw title
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(label, x, centerY + 16);
    
    // Draw weight vector with larger font size
    ctx.font = '12px monospace';
    ctx.fillStyle = '#cbd5e1';
    ctx.textAlign = 'left';
    ctx.fillText('W:', centerX + 8, centerY + 35);
    
    ctx.fillStyle = '#a5b4fc';
    const vectorStr = '[' + weights.map(w => w.toFixed(2)).join(', ') + ']';
    // Measure text to ensure it fits
    let textWidth = ctx.measureText(vectorStr).width;
    if (textWidth > width - 40) {
      // If too long, use slightly smaller font
      ctx.font = '11px monospace';
      textWidth = ctx.measureText(vectorStr).width;
    }
    ctx.fillText(vectorStr, centerX + 24, centerY + 35);
    
    // Draw bias
    ctx.font = '12px monospace';
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText('b:', centerX + 8, centerY + 50);
    ctx.fillStyle = '#fbbf24';
    ctx.fillText(bias.toFixed(2), centerX + 24, centerY + 50);
    
    // Draw activation
    ctx.fillStyle = '#34d399';
    ctx.font = 'bold 12px monospace';
    ctx.fillText('→ ' + activation.toFixed(3), centerX + 70, centerY + 68);
    
    // Draw calculation overlay if this neuron is being calculated
    if (this.highlightedNeuron && 
        this.highlightedNeuron.layer === layerType && 
        this.highlightedNeuron.index !== undefined) {
      const neuronIndex = label.includes('#') ? parseInt(label.split('#')[1]) - 1 : 0;
      if (this.highlightedNeuron.index === neuronIndex && this.calculationStage && this.intermediateValue !== null) {
        this.drawCalculationOverlay(ctx, x, centerY - 20, this.calculationStage, this.intermediateValue);
      }
    }
    
    return { x: centerX, y: centerY, width, height, centerX: x, centerY: y };
  }

  drawCalculationOverlay(ctx, x, y, stage, value) {
    if (!this.currentNeuronData) return;
    
    let text = '';
    let color = '';
    const data = this.currentNeuronData;
    
    switch(stage) {
      case 'connections':
        text = '입력 연결 계산 중...';
        color = '#60a5fa';
        break;
        
      case 'dotProduct':
        // Show expanded dot product formula: input1×weight1 + input2×weight2 + ...
        const terms = data.inputs.map((input, i) => 
          `${input.toFixed(2)}×${data.weights[i].toFixed(2)}`
        );
        text = terms.join(' + ') + ` = ${value.toFixed(3)}`;
        color = '#a5b4fc';
        break;
        
      case 'bias':
        // Show bias addition: dotProduct + bias = result
        text = `${data.dotProduct.toFixed(3)} + ${data.bias.toFixed(2)} = ${value.toFixed(3)}`;
        color = '#fbbf24';
        break;
        
      case 'activation':
        // Show activation function: σ(withBias) = activated
        text = `σ(${data.withBias.toFixed(3)}) = ${value.toFixed(3)}`;
        color = '#34d399';
        break;
    }
    
    if (!text) return;
    
    // Calculate text size for multi-line if needed
    ctx.font = 'bold 13px monospace';
    const metrics = ctx.measureText(text);
    const maxWidth = 600; // Maximum width before wrapping
    
    let lines = [text];
    if (metrics.width > maxWidth && stage === 'dotProduct') {
      // Split long dot product formulas into multiple lines
      const termsPerLine = Math.ceil(data.inputs.length / 2);
      const line1Terms = data.inputs.slice(0, termsPerLine).map((input, i) => 
        `${input.toFixed(2)}×${data.weights[i].toFixed(2)}`
      );
      const line2Terms = data.inputs.slice(termsPerLine).map((input, i) => 
        `${input.toFixed(2)}×${data.weights[i + termsPerLine].toFixed(2)}`
      );
      
      lines = [
        line1Terms.join(' + ') + ' +',
        line2Terms.join(' + ') + ` = ${value.toFixed(3)}`
      ];
    }
    
    // Draw background box
    const padding = 10;
    const lineHeight = 20;
    const maxLineWidth = Math.max(...lines.map(line => ctx.measureText(line).width));
    const boxWidth = maxLineWidth + padding * 2;
    const boxHeight = lineHeight * lines.length + padding * 2;
    
    ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
    ctx.beginPath();
    ctx.roundRect(x - boxWidth / 2, y - boxHeight / 2, boxWidth, boxHeight, 6);
    ctx.fill();
    
    // Draw border
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw text lines
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    lines.forEach((line, i) => {
      const lineY = y - boxHeight / 2 + padding + lineHeight / 2 + i * lineHeight;
      ctx.fillText(line, x, lineY);
    });
  }

  drawNetwork(nn, steps) {
    const ctx = this.ctx;
    const width = this.canvas.width;
    const height = this.canvas.height;
    
    // Clear canvas
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, height);
    
    if (!steps) return;
    
    const nodes = [];
    
    // Draw input vector (single ellipse)
    const inputNode = this.drawInputVector(ctx, 80, height / 2, steps.input);
    nodes.push([inputNode]);
    
    // Draw layer 1 neurons (1차 면접관)
    const layer1Nodes = [];
    const layer1StartY = (height - (5 - 1) * 100) / 2;
    for (let i = 0; i < 5; i++) {
      const neuron = steps.layer1[i];
      const isHighlighted = this.highlightedNeuron && 
                            this.highlightedNeuron.layer === 'layer1' && 
                            this.highlightedNeuron.index === i;
      const node = this.drawNeuronVector(
        ctx,
        340,
        layer1StartY + i * 100,
        neuron.weights,
        neuron.bias,
        neuron.activated,
        `1차 #${i + 1}`,
        'layer1',
        isHighlighted
      );
      layer1Nodes.push(node);
    }
    nodes.push(layer1Nodes);
    
    // Draw layer 2 neurons (2차 면접관)
    const layer2Nodes = [];
    const layer2StartY = (height - (3 - 1) * 130) / 2;
    for (let i = 0; i < 3; i++) {
      const neuron = steps.layer2[i];
      const isHighlighted = this.highlightedNeuron && 
                            this.highlightedNeuron.layer === 'layer2' && 
                            this.highlightedNeuron.index === i;
      const node = this.drawNeuronVector(
        ctx,
        660,
        layer2StartY + i * 130,
        neuron.weights,
        neuron.bias,
        neuron.activated,
        `2차 #${i + 1}`,
        'layer2',
        isHighlighted
      );
      layer2Nodes.push(node);
    }
    nodes.push(layer2Nodes);
    
    // Draw output neurons (3 classes: 불합격/보류/합격)
    const outputNodes = [];
    const classNames = ['불합격', '보류', '합격'];
    const outputStartY = (height - (3 - 1) * 100) / 2;
    
    for (let i = 0; i < 3; i++) {
      const output = steps.output[i];
      const isHighlighted = this.highlightedNeuron && 
                            this.highlightedNeuron.layer === 'output' &&
                            this.highlightedNeuron.index === i;
      const outputNode = this.drawNeuronVector(
        ctx,
        width - 100,
        outputStartY + i * 100,
        output.weights,
        output.bias,
        output.activated,
        classNames[i],
        'output',
        isHighlighted
      );
      outputNodes.push(outputNode);
    }
    nodes.push(outputNodes);
    
    // Draw connections
    this.drawConnectionsVector(ctx, nodes, nn);
    
    // Draw loss overlay if showing loss phase
    if (this.showLoss) {
      this.drawLossOverlay(ctx, width, height);
    }
    
    // Highlight neurons during backprop
    if (this.backpropPhase) {
      this.drawBackpropHighlight(ctx, nodes);
    }
  }

  drawConnectionsVector(ctx, nodes, nn) {
    // Input to Layer 1
    for (let i = 0; i < 5; i++) {
      const from = nodes[0][0];
      const to = nodes[1][i];
      
      const isActive = this.isConnectionActive('input', 0, 'layer1', i);
      
      ctx.beginPath();
      ctx.moveTo(from.centerX + from.width / 2, from.centerY);
      ctx.lineTo(to.centerX - to.width / 2, to.centerY);
      
      if (isActive) {
        ctx.strokeStyle = 'rgba(96, 165, 250, 0.9)';
        ctx.lineWidth = 3;
        ctx.shadowColor = 'rgba(96, 165, 250, 0.8)';
        ctx.shadowBlur = 10;
      } else {
        ctx.strokeStyle = 'rgba(100, 116, 139, 0.4)';
        ctx.lineWidth = 1;
        ctx.shadowBlur = 0;
      }
      ctx.stroke();
      ctx.shadowBlur = 0; // Reset shadow
    }
    
    // Layer 1 to Layer 2
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 5; j++) {
        const from = nodes[1][j];
        const to = nodes[2][i];
        
        const isActive = this.isConnectionActive('layer1', j, 'layer2', i);
        
        ctx.beginPath();
        ctx.moveTo(from.centerX + from.width / 2, from.centerY);
        ctx.lineTo(to.centerX - to.width / 2, to.centerY);
        
        if (isActive) {
          ctx.strokeStyle = 'rgba(52, 211, 153, 0.9)';
          ctx.lineWidth = 3;
          ctx.shadowColor = 'rgba(52, 211, 153, 0.8)';
          ctx.shadowBlur = 10;
        } else {
          ctx.strokeStyle = 'rgba(100, 116, 139, 0.3)';
          ctx.lineWidth = 1;
          ctx.shadowBlur = 0;
        }
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
    }
    
    // Layer 2 to Output (3 output neurons)
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const from = nodes[2][i];
        const to = nodes[3][j];
        
        const isActive = this.isConnectionActive('layer2', i, 'output', j);
        
        ctx.beginPath();
        ctx.moveTo(from.centerX + from.width / 2, from.centerY);
        ctx.lineTo(to.centerX - to.width / 2, to.centerY);
        
        if (isActive) {
          ctx.strokeStyle = 'rgba(251, 146, 60, 0.9)';
          ctx.lineWidth = 3;
          ctx.shadowColor = 'rgba(251, 146, 60, 0.8)';
          ctx.shadowBlur = 10;
        } else {
          ctx.strokeStyle = 'rgba(100, 116, 139, 0.4)';
          ctx.lineWidth = 1;
          ctx.shadowBlur = 0;
        }
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
    }
  }

  isConnectionActive(fromLayer, fromIndex, toLayer, toIndex) {
    if (this.calculationStage !== 'connections') return false;
    if (!this.highlightedNeuron) return false;
    
    // Check if this connection leads to the highlighted neuron
    return this.highlightedNeuron.layer === toLayer && 
           this.highlightedNeuron.index === toIndex;
  }

  drawLossOverlay(ctx, width, height) {
    const { targetClass, targetName, predictions, loss } = this.showLoss;
    const classNames = ['불합격', '보류', '합격'];
    
    // Semi-transparent overlay
    ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
    ctx.fillRect(width/2 - 250, height/2 - 150, 500, 300);
    
    // Border
    ctx.strokeStyle = '#60a5fa';
    ctx.lineWidth = 3;
    ctx.strokeRect(width/2 - 250, height/2 - 150, 500, 300);
    
    // Title
    ctx.font = 'bold 24px sans-serif';
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.fillText('📊 Loss Calculation', width/2, height/2 - 110);
    
    // Target
    ctx.font = 'bold 18px sans-serif';
    ctx.fillStyle = '#22c55e';
    ctx.fillText(`Target: ${targetName}`, width/2, height/2 - 70);
    
    // Predictions with bars
    ctx.font = '16px monospace';
    ctx.textAlign = 'left';
    predictions.forEach((prob, i) => {
      const y = height/2 - 30 + i * 30;
      const color = i === targetClass ? '#22c55e' : '#64748b';
      ctx.fillStyle = color;
      const barWidth = prob * 200;
      ctx.fillRect(width/2 - 100, y, barWidth, 20);
      
      ctx.fillStyle = '#ffffff';
      ctx.fillText(`${classNames[i]}: ${(prob * 100).toFixed(1)}%`, width/2 + 110, y + 15);
    });
    
    // Loss value
    ctx.font = 'bold 20px sans-serif';
    ctx.fillStyle = '#fbbf24';
    ctx.textAlign = 'center';
    ctx.fillText(`Cross-Entropy Loss: ${loss.toFixed(4)}`, width/2, height/2 + 80);
  }

  drawBackpropHighlight(ctx, nodes) {
    const { layer, index } = this.backpropPhase;
    let nodeInfo = null;
    
    // Find the node to highlight
    if (layer === 'layer1' && nodes[1]) {
      nodeInfo = nodes[1][index];
    } else if (layer === 'layer2' && nodes[2]) {
      nodeInfo = nodes[2][index];
    } else if (layer === 'output' && nodes[3]) {
      nodeInfo = nodes[3][index];
    }
    
    if (!nodeInfo) return;
    
    // Draw pulsing gradient glow (red for backprop)
    ctx.save();
    const gradient = ctx.createRadialGradient(
      nodeInfo.centerX, nodeInfo.centerY, 0,
      nodeInfo.centerX, nodeInfo.centerY, nodeInfo.width / 2 + 20
    );
    gradient.addColorStop(0, 'rgba(239, 68, 68, 0.6)');
    gradient.addColorStop(1, 'rgba(239, 68, 68, 0)');
    
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(nodeInfo.centerX, nodeInfo.centerY, nodeInfo.width / 2 + 25, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    
    // Draw "BACKPROP" label with arrow
    ctx.font = 'bold 14px sans-serif';
    ctx.fillStyle = '#ef4444';
    ctx.textAlign = 'center';
    ctx.fillText('◄ BACKPROP', nodeInfo.centerX, nodeInfo.y - 35);
  }

  updateTextDisplay(steps) {
    if (!steps) {
      this.textDisplay.innerHTML = '<div class="step-info">No calculation data available</div>';
      return;
    }
    
    let html = '<div class="calculation-display">';
    
    // Input vector
    html += '<div class="layer-section input-section">';
    html += '<h3>📥 입력 벡터</h3>';
    html += '<div class="vector-display">';
    html += '<div class="vector-label">면접자 = </div>';
    html += '<div class="vector-values">[';
    steps.input.forEach((val, idx) => {
      if (idx > 0) html += ', ';
      html += `<span class="vector-component">${val.toFixed(2)}</span>`;
    });
    html += ']</div>';
    html += '<div class="vector-labels">(성적, 태도, 응답수준)</div>';
    html += '</div></div>';
    
    // Layer 1: 1차 면접관
    html += '<div class="layer-section layer1-section">';
    html += '<h3>👥 1차 면접관 (5명)</h3>';
    steps.layer1.forEach((neuron, idx) => {
      html += `<div class="neuron-calc">`;
      html += `<div class="neuron-header">1차 면접관 #${idx + 1}</div>`;
      
      // Weight vector
      html += '<div class="calc-step">';
      html += '<div class="vector-display">';
      html += '<div class="vector-label">가중치 벡터 w = </div>';
      html += '<div class="vector-values">[';
      neuron.weights.forEach((w, i) => {
        if (i > 0) html += ', ';
        html += `<span class="vector-component">${w.toFixed(2)}</span>`;
      });
      html += ']</div>';
      html += '</div></div>';
      
      // Dot product
      html += '<div class="calc-step">';
      html += '<div class="calc-label">벡터 내적 (x · w):</div>';
      html += '<div class="calc-detail">';
      neuron.inputs.forEach((input, i) => {
        if (i > 0) html += ' + ';
        html += `<span class="term">${input.toFixed(2)}×${neuron.weights[i].toFixed(2)}</span>`;
      });
      html += ` = <span class="result">${neuron.dotProduct.toFixed(3)}</span>`;
      html += '</div></div>';
      
      // Add bias
      html += '<div class="calc-step">';
      html += '<div class="calc-label">바이아스 추가:</div>';
      html += `<div class="calc-detail">${neuron.dotProduct.toFixed(3)} + ${neuron.bias.toFixed(2)} = <span class="result">${neuron.withBias.toFixed(3)}</span></div>`;
      html += '</div>';
      
      // Activation
      html += '<div class="calc-step">';
      html += '<div class="calc-label">활성화 (Sigmoid):</div>';
      html += `<div class="calc-detail">σ(${neuron.withBias.toFixed(3)}) = <span class="result final">${neuron.activated.toFixed(3)}</span></div>`;
      html += '</div>';
      
      html += '</div>';
    });
    html += '</div>';
    
    // Layer 2: 2차 면접관
    html += '<div class="layer-section layer2-section">';
    html += '<h3>👔 2차 면접관 (3명)</h3>';
    steps.layer2.forEach((neuron, idx) => {
      html += `<div class="neuron-calc">`;
      html += `<div class="neuron-header">2차 면접관 #${idx + 1}</div>`;
      
      // Weight vector
      html += '<div class="calc-step">';
      html += '<div class="vector-display">';
      html += '<div class="vector-label">가중치 벡터 w = </div>';
      html += '<div class="vector-values">[';
      neuron.weights.forEach((w, i) => {
        if (i > 0) html += ', ';
        html += `<span class="vector-component">${w.toFixed(2)}</span>`;
      });
      html += ']</div>';
      html += '</div></div>';
      
      // Dot product
      html += '<div class="calc-step">';
      html += '<div class="calc-label">벡터 내적:</div>';
      html += '<div class="calc-detail">';
      neuron.inputs.forEach((input, i) => {
        if (i > 0) html += ' + ';
        html += `<span class="term">${input.toFixed(2)}×${neuron.weights[i].toFixed(2)}</span>`;
      });
      html += ` = <span class="result">${neuron.dotProduct.toFixed(3)}</span>`;
      html += '</div></div>';
      
      // Add bias
      html += '<div class="calc-step">';
      html += '<div class="calc-label">바이아스 추가:</div>';
      html += `<div class="calc-detail">${neuron.dotProduct.toFixed(3)} + ${neuron.bias.toFixed(2)} = <span class="result">${neuron.withBias.toFixed(3)}</span></div>`;
      html += '</div>';
      
      // Activation
      html += '<div class="calc-step">';
      html += '<div class="calc-label">활성화 (Sigmoid):</div>';
      html += `<div class="calc-detail">σ(${neuron.withBias.toFixed(3)}) = <span class="result final">${neuron.activated.toFixed(3)}</span></div>`;
      html += '</div>';
      
      html += '</div>';
    });
    html += '</div>';
    
    // Output
    html += '<div class="layer-section output-section">';
    html += '<h3>⚖️ 최종 결정</h3>';
    const output = steps.output[0];
    html += `<div class="neuron-calc">`;
    
    // Weight vector
    html += '<div class="calc-step">';
    html += '<div class="vector-display">';
    html += '<div class="vector-label">가중치 벡터 w = </div>';
    html += '<div class="vector-values">[';
    output.weights.forEach((w, i) => {
      if (i > 0) html += ', ';
      html += `<span class="vector-component">${w.toFixed(2)}</span>`;
    });
    html += ']</div>';
    html += '</div></div>';
    
    // Dot product
    html += '<div class="calc-step">';
    html += '<div class="calc-label">벡터 내적:</div>';
    html += '<div class="calc-detail">';
    output.inputs.forEach((input, i) => {
      if (i > 0) html += ' + ';
      html += `<span class="term">${input.toFixed(2)}×${output.weights[i].toFixed(2)}</span>`;
    });
    html += ` = <span class="result">${output.dotProduct.toFixed(3)}</span>`;
    html += '</div></div>';
    
    // Add bias
    html += '<div class="calc-step">';
    html += '<div class="calc-label">바이아스 추가:</div>';
    html += `<div class="calc-detail">${output.dotProduct.toFixed(3)} + ${output.bias.toFixed(2)} = <span class="result">${output.withBias.toFixed(3)}</span></div>`;
    html += '</div>';
    
    // Activation
    html += '<div class="calc-step">';
    html += '<div class="calc-label">활성화 (Sigmoid):</div>';
    html += `<div class="calc-detail">σ(${output.withBias.toFixed(3)}) = <span class="result final">${output.activated.toFixed(3)}</span></div>`;
    html += '</div>';
    
    // Final decision
    const decision = output.activated > 0.5 ? '합격' : '불합격';
    const confidence = (Math.abs(output.activated - 0.5) * 200).toFixed(1);
    html += `<div class="final-decision">결정: <strong>${decision}</strong> (확신도: ${confidence}%)</div>`;
    
    html += '</div>';
    html += '</div>';
    
    html += '</div>';
    
    this.textDisplay.innerHTML = html;
  }

  update(nn) {
    const steps = nn.getCalculationSteps();
    this.drawNetwork(nn, steps);
    this.updateTextDisplay(steps);
  }
}
