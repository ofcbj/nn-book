import { Box, Paper, Typography, Stack, Button } from '@mui/material';
import { useTranslation } from 'react-i18next';

interface StatsDisplayProps {
  epoch: number;
  loss: number;
  output: number[] | null;
  isTraining: boolean;
  onTrainOnce: () => void;
  onTrainToggle: () => void;
}

export default function StatsDisplay({ epoch, loss, output, isTraining, onTrainOnce, onTrainToggle }: StatsDisplayProps) {
  const { t } = useTranslation();
  const classNames = [t('classes.fail'), t('classes.pending'), t('classes.pass')];
  const getOutputText = () => {
    if (!output) return '0.000';
    return output.map((prob, i) => 
      `${classNames[i]}: ${(prob * 100).toFixed(1)}%`
    ).join(' | ');
  };

  const getOutputColor = () => {
    if (!output) return 'primary.light';
    const maxIndex = output.indexOf(Math.max(...output));
    return maxIndex === 0 ? 'error.main' : maxIndex === 1 ? 'warning.main' : 'secondary.main';
  };

  return (
    <Paper 
      sx={{ 
        p: 2, 
        bgcolor: '#0f172a', 
        borderRadius: 2 
      }}
    >
      <Typography variant="h3" sx={{ mb: 2, pb: 1, borderBottom: '2px solid #334155' }}>
        📈 {t('stats.title')}
      </Typography>
      
      <Stack spacing={1.5}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 1, borderBottom: '1px solid #1e293b' }}>
          <Typography variant="body2" color="text.secondary">
            {t('stats.epoch')}:
          </Typography>
          <Typography 
            sx={{ 
              fontFamily: 'monospace', 
              fontWeight: 700,
              color: 'primary.light'
            }}
          >
            {epoch}
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 1, borderBottom: '1px solid #1e293b' }}>
          <Typography variant="body2" color="text.secondary">
            {t('stats.loss')}:
          </Typography>
          <Typography 
            sx={{ 
              fontFamily: 'monospace', 
              fontWeight: 700,
              color: 'primary.light'
            }}
          >
            {loss.toFixed(6)}
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 1, borderBottom: '1px solid #1e293b' }}>
          <Typography variant="body2" color="text.secondary">
            {t('stats.prediction')}:
          </Typography>
          <Typography 
            sx={{ 
              fontFamily: 'monospace', 
              fontWeight: 700,
              color: getOutputColor(),
              fontSize: '0.85rem'
            }}
          >
            {getOutputText()}
          </Typography>
        </Box>
        
        {/* Training Buttons */}
        <Box sx={{ pt: 1 }}>
          <Stack spacing={1}>
            {/* Train Once Button */}
            <Button 
              variant="contained" 
              fullWidth
              color="primary"
              onClick={onTrainOnce}
              disabled={isTraining}
            >
              {t('controls.oneStep')}
            </Button>
            
            {/* Auto Train Button */}
            <Button 
              variant="contained" 
              fullWidth
              color={isTraining ? 'error' : 'success'}
              onClick={onTrainToggle}
              sx={{
                animation: isTraining ? 'pulse 1.5s infinite' : 'none',
                '@keyframes pulse': {
                  '0%, 100%': { opacity: 1 },
                  '50%': { opacity: 0.7 },
                },
              }}
            >
              {isTraining ? t('controls.stop') : t('controls.autoTrain')}
            </Button>
          </Stack>
        </Box>
      </Stack>
    </Paper>
  );
}
