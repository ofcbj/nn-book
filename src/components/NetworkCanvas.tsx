import { useRef, useEffect, useCallback } from 'react';
import { Box, Paper, Typography, Stack } from '@mui/material';
import { useTranslation } from 'react-i18next';
import { Visualizer } from '../lib/visualizer';

interface NetworkCanvasProps {
  /** Called once with the created visualizer */
  onVisualizerReady: (visualizer: Visualizer) => void;
  /** Redraw the canvas from the current network / animation state */
  onRedraw: () => void;
  /** Click position in logical (CSS pixel) canvas coordinates */
  onCanvasClick?: (x: number, y: number) => void;
}

/** Explains the connection-line encoding: colour = sign of the weight, thickness = |w| */
function WeightLegend() {
  const { t } = useTranslation();
  const sample = (color: string, width: number) => (
    <Box component="span" sx={{ display: 'inline-block', width: 28, height: width, bgcolor: color, borderRadius: 1, mr: 0.75, verticalAlign: 'middle' }} />
  );
  return (
    <Stack direction="row" spacing={2.5} justifyContent="center" sx={{ mt: 1.5, flexWrap: 'wrap' }}>
      <Typography variant="caption" color="text.secondary">
        {sample('rgba(96, 165, 250, 0.9)', 3)}{t('network.legendPositive')}
      </Typography>
      <Typography variant="caption" color="text.secondary">
        {sample('rgba(251, 113, 133, 0.9)', 3)}{t('network.legendNegative')}
      </Typography>
      <Typography variant="caption" color="text.secondary">
        {sample('rgba(148, 163, 184, 0.8)', 1)}{sample('rgba(148, 163, 184, 0.8)', 4)}{t('network.legendThickness')}
      </Typography>
      <Typography variant="caption" color="text.secondary">
        {sample('rgba(251, 113, 133, 0.9)', 6)}{sample('rgba(96, 165, 250, 0.9)', 6)}{t('network.legendDelta')}
      </Typography>
    </Stack>
  );
}

export default function NetworkCanvas({ onVisualizerReady, onRedraw, onCanvasClick }: NetworkCanvasProps) {
  const { i18n } = useTranslation();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const visualizerRef = useRef<Visualizer | null>(null);

  // Create the visualizer once
  useEffect(() => {
    if (!canvasRef.current || visualizerRef.current) return;
    visualizerRef.current = new Visualizer(canvasRef.current);
    onVisualizerReady(visualizerRef.current);
    onRedraw();
  }, [onVisualizerReady, onRedraw]);

  // Keep the canvas bitmap in sync with its container size (also fires once on mount)
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => {
      visualizerRef.current?.resizeCanvas();
      onRedraw();
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [onRedraw]);

  // Canvas labels are translated, so redraw on language change
  useEffect(() => {
    i18n.on('languageChanged', onRedraw);
    return () => {
      i18n.off('languageChanged', onRedraw);
    };
  }, [i18n, onRedraw]);

  const handleClick = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onCanvasClick || !canvasRef.current) return;
    // The visualizer draws in CSS pixels, so client-relative coordinates map directly
    const rect = canvasRef.current.getBoundingClientRect();
    onCanvasClick(event.clientX - rect.left, event.clientY - rect.top);
  }, [onCanvasClick]);

  return (
    <Paper sx={{ p: 1.5 }}>
      <Box
        ref={containerRef}
        sx={{
          width: '100%',
          // Fill the viewport below the header; the canvas follows via ResizeObserver
          height: 'calc(100vh - 300px)',
          minHeight: 520,
          borderRadius: 2,
          bgcolor: '#0a0a0a',
          border: '1px solid #334155',
          overflow: 'hidden',
        }}
      >
        <canvas
          ref={canvasRef}
          onClick={handleClick}
          style={{ display: 'block', width: '100%', height: '100%', cursor: onCanvasClick ? 'pointer' : 'default' }}
        />
      </Box>
      <WeightLegend />
    </Paper>
  );
}
