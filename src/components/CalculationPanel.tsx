import { Box, Paper, Typography, Button } from '@mui/material';
import { useTranslation } from 'react-i18next';
import type { ForwardSteps } from '../lib/types';

interface CalculationPanelProps {
  steps: ForwardSteps | null;
  hasComparisonData: boolean;
  onViewComparison: () => void;
}

export default function CalculationPanel({ steps, hasComparisonData, onViewComparison }: CalculationPanelProps) {
  const { t } = useTranslation();
  const classNames = [t('classes.fail'), t('classes.pending'), t('classes.pending')];
  if (!steps) {
    return (
      <Paper sx={{ p: 2.5, height: '100%' }}>
        <Typography variant="h3" sx={{ mb: 2, pb: 1, borderBottom: '2px solid #334155' }}>
          🔢 {t('calculation.title')}
        </Typography>
        
        {/* Weight Comparison Button */}
        <Box sx={{ mb: 2 }}>
          <Button
            variant="outlined"
            fullWidth
            size="small"
            onClick={onViewComparison}
            disabled={!hasComparisonData}
            sx={{
              borderColor: 'primary.main',
              color: 'primary.main',
              fontSize: '0.85rem',
              py: 0.75,
              '&:hover': {
                borderColor: 'primary.dark',
                bgcolor: 'rgba(59, 130, 246, 0.1)',
              },
              '&:disabled': {
                borderColor: 'action.disabled',
                color: 'text.disabled',
              },
            }}
          >
            📊 {t('comparison.viewComparison')}
          </Button>
        </Box>
        
        <Typography color="text.secondary">{t('calculation.noData')}</Typography>
      </Paper>
    );
  }

  return (
    <Paper sx={{ p: 2.5, height: '100%', overflow: 'hidden' }}>
      <Typography variant="h3" sx={{ mb: 2, pb: 1, borderBottom: '2px solid #334155' }}>
        🔢 {t('calculation.title')}
      </Typography>
      
      {/* Weight Comparison Button */}
      <Box sx={{ mb: 2 }}>
        <Button
          variant="outlined"
          fullWidth
          size="small"
          onClick={onViewComparison}
          disabled={!hasComparisonData}
          sx={{
            borderColor: 'primary.main',
            color: 'primary.main',
            fontSize: '0.85rem',
            py: 0.75,
            '&:hover': {
              borderColor: 'primary.dark',
              bgcolor: 'rgba(59, 130, 246, 0.1)',
            },
            '&:disabled': {
              borderColor: 'action.disabled',
              color: 'text.disabled',
            },
          }}
        >
          📊 {t('comparison.viewComparison')}
        </Button>
      </Box>
      
      <Box sx={{ maxHeight: 650, overflowY: 'auto', pr: 1 }}>
        {/* Input Vector */}
        <Box sx={{ mb: 3, p: 2, borderRadius: 2, bgcolor: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
          <Typography variant="h3" sx={{ mb: 1.5 }}>📥 {t('layers.input')}</Typography>
          <Box sx={{ p: 1, bgcolor: 'rgba(0,0,0,0.3)', borderRadius: 1 }}>
            <Typography component="span" color="text.secondary">{t('layers.input')} = </Typography>
            <Typography component="span" sx={{ fontFamily: 'monospace', color: 'primary.light', fontWeight: 600 }}>
              [{steps.input.map(v => v.toFixed(2)).join(', ')}]
            </Typography>
            <Typography variant="caption" display="block" color="text.disabled" sx={{ mt: 0.5 }}>
              ({t('controls.grade')}, {t('controls.attitude')}, {t('controls.response')})
            </Typography>
          </Box>
        </Box>

        {/* Layer 1 */}
        <Box sx={{ mb: 3, p: 2, borderRadius: 2, bgcolor: 'rgba(34, 197, 94, 0.1)', border: '1px solid rgba(34, 197, 94, 0.3)' }}>
          <Typography variant="h3" sx={{ mb: 1.5 }}>👥 {t('help.structure.layer1.title')}</Typography>
          {steps.layer1.map((neuron, idx) => (
            <NeuronCalcDisplay key={idx} neuron={neuron} label={`${t('layers.layer1Prefix')}${t('layers.interviewer')} #${idx + 1}`} />
          ))}
        </Box>

        {/* Layer 2 */}
        <Box sx={{ mb: 3, p: 2, borderRadius: 2, bgcolor: 'rgba(249, 115, 22, 0.1)', border: '1px solid rgba(249, 115, 22, 0.3)' }}>
          <Typography variant="h3" sx={{ mb: 1.5 }}>👔 {t('help.structure.layer2.title')}</Typography>
          {steps.layer2.map((neuron, idx) => (
            <NeuronCalcDisplay key={idx} neuron={neuron} label={`${t('layers.layer2Prefix')}${t('layers.interviewer')} #${idx + 1}`} />
          ))}
        </Box>

        {/* Output */}
        <Box sx={{ mb: 3, p: 2, borderRadius: 2, bgcolor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)' }}>
          <Typography variant="h3" sx={{ mb: 1.5 }}>⚖️ {t('layers.output')}</Typography>
          {steps.output.map((neuron, idx) => (
            <NeuronCalcDisplay key={idx} neuron={neuron} label={classNames[idx]} isOutput />
          ))}
        </Box>
      </Box>
    </Paper>
  );
}

interface NeuronCalcDisplayProps {
  neuron: {
    weights: number[];
    bias: number;
    inputs: number[];
    dotProduct: number;
    withBias: number;
    activated: number;
  };
  label: string;
  isOutput?: boolean;
}

function NeuronCalcDisplay({ neuron, label, isOutput }: NeuronCalcDisplayProps) {
  return (
    <Box sx={{ mb: 2, p: 1.5, bgcolor: 'rgba(15, 23, 42, 0.6)', borderRadius: 1, border: '1px solid rgba(51, 65, 85, 0.5)' }}>
      <Typography sx={{ fontWeight: 600, mb: 1, fontSize: '0.95rem' }}>{label}</Typography>
      
      {/* Weights */}
      <Box sx={{ mb: 1 }}>
        <Typography component="span" color="text.secondary" fontSize="0.85rem">
          w = 
        </Typography>
        <Typography component="span" sx={{ fontFamily: 'monospace', color: '#a5b4fc', fontSize: '0.85rem' }}>
          [{neuron.weights.map(w => w.toFixed(2)).join(', ')}]
        </Typography>
      </Box>

      {/* Dot Product */}
      <Box sx={{ mb: 0.5, p: 0.75, bgcolor: 'rgba(0,0,0,0.3)', borderRadius: 1 }}>
        <Typography component="span" color="text.secondary" fontSize="0.8rem">
          x·w = 
        </Typography>
        <Typography component="span" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
          {neuron.inputs.map((inp, i) => `${inp.toFixed(2)}×${neuron.weights[i].toFixed(2)}`).join(' + ')}
          {' = '}
          <Box component="span" sx={{ color: 'secondary.light', fontWeight: 700 }}>
            {neuron.dotProduct.toFixed(3)}
          </Box>
        </Typography>
      </Box>

      {/* Bias */}
      <Box sx={{ mb: 0.5, p: 0.75, bgcolor: 'rgba(0,0,0,0.3)', borderRadius: 1 }}>
        <Typography component="span" color="text.secondary" fontSize="0.8rem">
          +b = 
        </Typography>
        <Typography component="span" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
          {neuron.dotProduct.toFixed(3)} + {neuron.bias.toFixed(2)} = 
          <Box component="span" sx={{ color: 'secondary.light', fontWeight: 700 }}>
            {neuron.withBias.toFixed(3)}
          </Box>
        </Typography>
      </Box>

      {/* Activation */}
      <Box sx={{ p: 0.75, bgcolor: 'rgba(0,0,0,0.3)', borderRadius: 1 }}>
        <Typography component="span" color="text.secondary" fontSize="0.8rem">
          {isOutput ? 'softmax' : 'σ'} = 
        </Typography>
        <Typography component="span" sx={{ fontFamily: 'monospace', color: 'warning.main', fontWeight: 700, fontSize: '0.9rem' }}>
          {isOutput ? `${(neuron.activated * 100).toFixed(1)}%` : neuron.activated.toFixed(3)}
        </Typography>
      </Box>
    </Box>
  );
}
