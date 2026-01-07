import { Box, Paper, Typography } from '@mui/material';
import { useTranslation } from 'react-i18next';
import { activationToColor, getColorStops } from '../lib/visualizer/activationColors';

export interface ActivationData {
  input: number[];
  layer1: number[];
  layer2: number[];
  output: number[];
}

interface ActivationHeatmapProps {
  activations: ActivationData | null;
}

export default function ActivationHeatmap({ activations }: ActivationHeatmapProps) {
  const { t } = useTranslation();

  if (!activations) {
    return null;
  }

  const colorStops = getColorStops(10);

  const renderLayer = (layerName: string, values: number[], label: string) => {
    return (
      <Box sx={{ flex: 1, minWidth: 150 }}>
        <Typography variant="subtitle2" sx={{ mb: 1, textAlign: 'center', fontWeight: 'bold' }}>
          {label}
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, justifyContent: 'center' }}>
          {values.map((value, idx) => {
            const color = activationToColor(value);
            return (
              <Box
                key={`${layerName}-${idx}`}
                sx={{
                  width: 50,
                  height: 50,
                  bgcolor: color,
                  borderRadius: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: '1px solid rgba(255,255,255,0.2)',
                  cursor: 'pointer',
                  transition: 'transform 0.2s',
                  '&:hover': {
                    transform: 'scale(1.1)',
                    zIndex: 10,
                  },
                }}
                title={`${label} #${idx + 1}: ${value.toFixed(3)}`}
              >
                <Typography variant="caption" sx={{ color: 'white', fontWeight: 'bold', fontSize: '0.65rem' }}>
                  #{idx + 1}
                </Typography>
                <Typography variant="caption" sx={{ color: 'white', fontSize: '0.7rem', fontWeight: 'bold' }}>
                  {value.toFixed(2)}
                </Typography>
              </Box>
            );
          })}
        </Box>
      </Box>
    );
  };

  return (
    <Paper sx={{ p: 2.5, mt: 2 }}>
      <Typography variant="h2" sx={{ mb: 2, textAlign: 'center' }}>
        {t('visualizer.activationTitle')}
      </Typography>

      <Box sx={{ display: 'flex', gap: 3, mb: 2, flexWrap: 'wrap', justifyContent: 'center' }}>
        {renderLayer('input', activations.input, t('layers.input'))}
        {renderLayer('layer1', activations.layer1, `${t('layers.layer1Prefix')} ${t('layers.interviewer')}`)}
        {renderLayer('layer2', activations.layer2, `${t('layers.layer2Prefix')} ${t('layers.interviewer')}`)}
        {renderLayer('output', activations.output, t('layers.output'))}
      </Box>

      {/* Color Legend */}
      <Box sx={{ mt: 3, pt: 2, borderTop: '1px solid rgba(255,255,255,0.1)' }}>
        <Typography variant="subtitle2" sx={{ mb: 1, textAlign: 'center', color: 'text.secondary' }}>
          {t('visualizer.colorLegend')}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            0.0
          </Typography>
          <Box sx={{ display: 'flex', height: 20, width: 300, borderRadius: 1, overflow: 'hidden' }}>
            {colorStops.map((stop, idx) => (
              <Box
                key={idx}
                sx={{
                  flex: 1,
                  bgcolor: stop.color,
                }}
              />
            ))}
          </Box>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            1.0
          </Typography>
        </Box>
      </Box>
    </Paper>
  );
}
