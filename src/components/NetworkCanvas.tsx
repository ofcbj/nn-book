import { useRef, useEffect, useCallback } from 'react';
import { Box, Paper, Typography } from '@mui/material';
import { useTranslation } from 'react-i18next';
import { Visualizer } from '../lib/visualizer';
import type { NeuralNetwork } from '../lib/core';

interface NetworkCanvasProps {
  nn: NeuralNetwork;
  onVisualizerReady: (visualizer: Visualizer) => void;
  onCanvasClick?: (x?: number, y?: number) => void;
}

export default function NetworkCanvas({ nn, onVisualizerReady, onCanvasClick }: NetworkCanvasProps) {
  const { t, i18n } = useTranslation();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const visualizerRef = useRef<Visualizer | null>(null);

  const resizeCanvas = useCallback(() => {
    if (canvasRef.current && containerRef.current) {
      const container = containerRef.current;
      const canvas = canvasRef.current;
      
      // Set canvas size to match container
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
      
      if (visualizerRef.current) {
        visualizerRef.current.update(nn);
      }
    }
  }, [nn]);

  useEffect(() => {
    if (canvasRef.current && !visualizerRef.current) {
      // Initial resize before creating visualizer
      if (containerRef.current) {
        canvasRef.current.width = containerRef.current.clientWidth;
        canvasRef.current.height = containerRef.current.clientHeight;
      }
      
      visualizerRef.current = new Visualizer(canvasRef.current);
      onVisualizerReady(visualizerRef.current);
      
      // Trigger initial draw after a short delay to ensure layout is complete
      setTimeout(() => {
        resizeCanvas();
      }, 100);
    }
  }, [onVisualizerReady, resizeCanvas]);

  useEffect(() => {
    if (visualizerRef.current) {
      visualizerRef.current.update(nn);
    }
  }, [nn]);

  // Listen for language changes and update canvas
  useEffect(() => {
    const handleLanguageChange = () => {
      if (visualizerRef.current) {
        visualizerRef.current.update(nn);
      }
    };

    i18n.on('languageChanged', handleLanguageChange);
    
    return () => {
      i18n.off('languageChanged', handleLanguageChange);
    };
  }, [i18n, nn]);

  useEffect(() => {
    const handleResize = () => {
      resizeCanvas();
    };

    window.addEventListener('resize', handleResize);
    
    // Also resize when component mounts
    resizeCanvas();
    
    return () => window.removeEventListener('resize', handleResize);
  }, [resizeCanvas]);

  const handleClick = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onCanvasClick || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    // Get click position relative to canvas
    const x = event.clientX-rect.left;
    const y = event.clientY-rect.top;
    
    // Scale for high DPI
    const scaleX = canvas.width/rect.width;
    const scaleY = canvas.height/rect.height;
    const canvasX = x*scaleX;
    const canvasY = y*scaleY;
    
    onCanvasClick(canvasX, canvasY);
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
          style={{ display: 'block', cursor: onCanvasClick ? 'pointer' : 'default' }}
        />
      </Box>
    </Paper>
  );
}
