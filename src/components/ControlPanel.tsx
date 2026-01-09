import { Box, Paper, Typography, Slider, Button, Stack } from '@mui/material';
import { useTranslation } from 'react-i18next';

interface ControlPanelProps {
  // Input values
  grade: number;
  attitude: number;
  response: number;
  targetValue: number;
  animationSpeed: number;
  // Event handlers
  onGradeChange: (value: number) => void;
  onAttitudeChange: (value: number) => void;
  onResponseChange: (value: number) => void;
  onTargetChange: (value: number) => void;
  onAnimationSpeedChange: (value: number) => void;
  // Button handlers
  onStep: () => void;
  onReset: () => void;
  // State
  isAnimating: boolean;
  isJumped: boolean;
}

export default function ControlPanel({
  grade,
  attitude,
  response,
  targetValue,
  animationSpeed,
  onGradeChange,
  onAttitudeChange,
  onResponseChange,
  onTargetChange,
  onAnimationSpeedChange,
  onStep,
  onReset,
  isAnimating,
  isJumped,
}: ControlPanelProps) {
  const { t } = useTranslation();
  const classNames = [t('classes.fail'), t('classes.pending'), t('classes.pass')];

  return (
    <Paper sx={{ p: 2, height: '100%' }}>
      {/* Input Controls */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h3" sx={{ mb: 1.5, pb: 0.5, borderBottom: '2px solid #334155', fontSize: '0.95rem' }}>
          📊 {t('controls.inputSection')}
        </Typography>
        
        <Box sx={{ mb: 2.5 }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            {t('controls.grade')}
          </Typography>
          <Stack direction="row" alignItems="center" spacing={1}>
            <Slider
              value={grade}
              onChange={(_, v) => onGradeChange(v as number)}
              min={0}
              max={1}
              step={0.01}
              sx={{ flex: 1 }}
            />
            <Typography 
              sx={{ 
                minWidth: 40, 
                fontFamily: 'monospace', 
                fontWeight: 600,
                fontSize: '0.85rem',
                color: 'primary.light'
              }}
            >
              {grade.toFixed(2)}
            </Typography>
          </Stack>
        </Box>

        <Box sx={{ mb: 2.5 }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            {t('controls.attitude')}
          </Typography>
          <Stack direction="row" alignItems="center" spacing={2}>
            <Slider
              value={attitude}
              onChange={(_, v) => onAttitudeChange(v as number)}
              min={0}
              max={1}
              step={0.01}
              sx={{ flex: 1 }}
            />
            <Typography 
              sx={{ 
                minWidth: 50, 
                fontFamily: 'monospace', 
                fontWeight: 600,
                color: 'primary.light'
              }}
            >
              {attitude.toFixed(2)}
            </Typography>
          </Stack>
        </Box>

        <Box sx={{ mb: 2.5 }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            {t('controls.response')}
          </Typography>
          <Stack direction="row" alignItems="center" spacing={2}>
            <Slider
              value={response}
              onChange={(_, v) => onResponseChange(v as number)}
              min={0}
              max={1}
              step={0.01}
              sx={{ flex: 1 }}
            />
            <Typography 
              sx={{ 
                minWidth: 50, 
                fontFamily: 'monospace', 
                fontWeight: 600,
                color: 'primary.light'
              }}
            >
              {response.toFixed(2)}
            </Typography>
          </Stack>
        </Box>

        <Box sx={{ mb: 2.5 }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            {t('controls.target')}
          </Typography>
          <Stack direction="row" alignItems="center" spacing={2}>
            <Slider
              value={targetValue}
              onChange={(_, v) => onTargetChange(v as number)}
              min={0}
              max={2}
              step={1}
              marks
              sx={{ flex: 1 }}
            />
            <Typography 
              sx={{ 
                minWidth: 50, 
                fontFamily: 'monospace', 
                fontWeight: 600,
                color: targetValue === 0 ? 'error.main' : targetValue === 1 ? 'warning.main' : 'secondary.main'
              }}
            >
              {classNames[targetValue]}
            </Typography>
          </Stack>
        </Box>
        
        {/* Reset Button */}
        <Box sx={{ mb: 3 }}>
          <Button 
            variant="contained" 
            fullWidth
            sx={{ 
              bgcolor: '#64748b',
              '&:hover': { bgcolor: 'error.main' }
            }}
            onClick={onReset}
          >
            {t('controls.reset')}
          </Button>
        </Box>
      </Box>

      {/* Training Controls */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h3" sx={{ mb: 2, pb: 1, borderBottom: '2px solid #334155' }}>
          ⚙️ {t('controls.trainingSection')}
        </Typography>

        <Box sx={{ mb: 2.5 }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            {t('controls.animationSpeed')}
          </Typography>
          <Stack direction="row" alignItems="center" spacing={2}>
            <Slider
              value={animationSpeed}
              onChange={(_, v) => onAnimationSpeedChange(v as number)}
              min={0.1}
              max={2}
              step={0.1}
              sx={{ flex: 1 }}
            />
            <Typography
              sx={{
                minWidth: 50,
                fontFamily: 'monospace',
                fontWeight: 600,
                color: 'primary.light'
              }}
            >
              {`${animationSpeed.toFixed(1)}x`}
            </Typography>
          </Stack>
        </Box>

        <Button 
          variant="contained"
          fullWidth
          onClick={onStep}
          sx={{ 
            bgcolor: isJumped ? 'success.main' : (isAnimating ? 'warning.main' : 'primary.main'),
            '&:hover': {
              bgcolor: isJumped ? 'success.dark' : (isAnimating ? 'warning.dark' : 'primary.dark'),
            }
          }}
        >
          {isJumped ? t('controls.resume') : (isAnimating ? t('controls.pause') : t('controls.start'))}
        </Button>
      </Box>
    </Paper>
  );
}
