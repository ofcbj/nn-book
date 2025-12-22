// Helper functions for loss and backprop visualization
// These will be added to visualizer.js

function drawLossOverlay(ctx, width, height) {
  if (!this.showLoss) return;
  
  const { targetClass, targetName, predictions, loss } = this.showLoss;
  const classNames = ['불합격', '보류', '합격'];
  
  // Semi-transparent overlay
  ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
  ctx.fillRect(width/2 - 250, height/2 - 150, 500, 300);
  
  // Border
  ctx.strokeStyle = '#60a5fa';
  ctx.lineWidth = 3;
  ctx.strokeRect(width/2 - 250, height/2 - 150, 500, 300);
  
  // Title
  ctx.font = 'bold 24px sans-serif';
  ctx.fillStyle = '#ffffff';
  ctx.textAlign = 'center';
  ctx.fillText('Loss Calculation', width/2, height/2 - 110);
  
  // Target
  ctx.font = 'bold 18px sans-serif';
  ctx.fillStyle = '#22c55e';
  ctx.fillText(`Target: ${targetName}`, width/2, height/2 - 70);
  
  // Predictions
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
  ctx.fillText(`Loss: ${loss.toFixed(4)}`, width/2, height/2 + 80);
}

function drawBackpropHighlight(ctx, nodes) {
  if (!this.backpropPhase) return;
  
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
  
  // Draw pulsing gradient glow
  ctx.save();
  const gradient = ctx.createRadialGradient(
    nodeInfo.centerX, nodeInfo.centerY, 0,
    nodeInfo.centerX, nodeInfo.centerY, nodeInfo.width / 2 + 20
  );
  gradient.addColorStop(0, 'rgba(239, 68, 68, 0.4)');
  gradient.addColorStop(1, 'rgba(239, 68, 68, 0)');
  
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(nodeInfo.centerX, nodeInfo.centerY, nodeInfo.width / 2 + 20, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
  
  // Draw "BACKPROP" label
  ctx.font = 'bold 12px sans-serif';
  ctx.fillStyle = '#ef4444';
  ctx.textAlign = 'center';
  ctx.fillText('◄ BACKPROP', nodeInfo.centerX, nodeInfo.y - 30);
}
