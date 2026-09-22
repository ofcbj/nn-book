import { useRef, useEffect, useCallback } from 'react';
import { Box, Paper, Typography } from '@mui/material';
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

export default function NetworkCanvas({ onVisualizerReady, onRedraw, onCanvasClick }: NetworkCanvasProps) {
  const { t, i18n } = useTranslation();
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
    <Paper sx={{ p: 2.5, height: '100%', minHeight: 700 }}>
      <Typography variant="h2" sx={{ mb: 2, textAlign: 'center' }}>
        {t('network.title')}
      </Typography>
      <Box
        ref={containerRef}
        sx={{
          width: '100%',
          height: 650,
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
    </Paper>
  );
}
