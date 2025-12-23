// Calculation overlay renderer
import type { CalculationStage, NeuronCalculation } from '../types';
import i18n from '../../i18n';

export function drawCalculationOverlay(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  x: number,
  y: number,
  stage: CalculationStage,
  value: number,
  currentNeuronData: NeuronCalculation | null
): void {
  if (!currentNeuronData) return;

  let text = '';
  let color = '';
  const data = currentNeuronData;

  switch(stage) {
    case 'connections':
      text = i18n.t('calculation.connectionsCalc');
      color = '#60a5fa';
      break;

    case 'dotProduct':
      const terms = data.inputs.map((input, i) =>
        `${input.toFixed(2)}×${data.weights[i].toFixed(2)}`
      );
      text = terms.join(' + ') + ` = ${value.toFixed(3)}`;
      color = '#a5b4fc';
      break;

    case 'bias':
      text = `${data.dotProduct.toFixed(3)} + ${data.bias.toFixed(2)} = ${value.toFixed(3)}`;
      color = '#fbbf24';
      break;

    case 'activation':
      text = `σ(${data.withBias.toFixed(3)}) = ${value.toFixed(3)}`;
      color = '#34d399';
      break;
  }

  if (!text) return;

  ctx.font = 'bold 13px monospace';
  const metrics = ctx.measureText(text);
  const maxWidth = 600;

  let lines = [text];
  if (metrics.width > maxWidth && stage === 'dotProduct') {
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

  const padding = 10;
  const lineHeight = 20;
  const maxLineWidth = Math.max(...lines.map(line => ctx.measureText(line).width));
  const boxWidth = maxLineWidth + padding * 2;
  const boxHeight = lineHeight * lines.length + padding * 2;

  // Calculate initial box position (centered on neuron)
  let boxX = x - boxWidth / 2;
  let boxY = y - boxHeight / 2;

  // Adjust position to stay within canvas boundaries
  const canvasWidth = canvas.width;
  const canvasHeight = canvas.height;
  const margin = 10;

  // Adjust horizontal position
  if (boxX < margin) {
    boxX = margin;
  } else if (boxX + boxWidth > canvasWidth - margin) {
    boxX = canvasWidth - margin - boxWidth;
  }

  // Adjust vertical position
  if (boxY < margin) {
    boxY = margin;
  } else if (boxY + boxHeight > canvasHeight - margin) {
    boxY = canvasHeight - margin - boxHeight;
  }

  ctx.fillStyle = 'rgba(0, 0, 0, 0.65)';
  ctx.beginPath();
  ctx.roundRect(boxX, boxY, boxWidth, boxHeight, 6);
  ctx.fill();

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.fillStyle = color;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  const textCenterX = boxX + boxWidth / 2;
  lines.forEach((line, i) => {
    const lineY = boxY + padding + lineHeight / 2 + i * lineHeight;
    ctx.fillText(line, textCenterX, lineY);
  });
}
