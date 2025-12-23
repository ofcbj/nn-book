import { Box, Paper, Typography, Slider, Button, Stack, Switch, FormControlLabel } from '@mui/material';
import { useTranslation } from 'react-i18next';

interface ControlPanelProps {
  // Input values
  grade: number;
  attitude: number;
  response: number;
  targetValue: number;
  learningRate: number;
  animationSpeed: number;
  isManualMode: boolean;
  // Event handlers
  onGradeChange: (value: number) => void;
  onAttitudeChange: (value: number) => void;
  onResponseChange: (value: number) => void;
  onTargetChange: (value: number) => void;
  onLearningRateChange: (value: number) => void;
  onAnimationSpeedChange: (value: number) => void;
  onManualModeChange: (value: boolean) => void;
  onNextStep: () => void;
  // Button handlers
  onStep: () => void;
  onTrainToggle: () => void;
  onReset: () => void;
  // State
  isTraining: boolean;
  isAnimating: boolean;
  // Heatmap toggles
  showCanvasHeatmap: boolean;
  showGridHeatmap: boolean;
  onToggleCanvasHeatmap: () => void;
  onToggleGridHeatmap: () => void;
  // Weight comparison
  hasComparisonData: boolean;
  onViewComparison: () => void;
}

export default function ControlPanel({
  grade,
  attitude,
  response,
  targetValue,
  learningRate,
  animationSpeed,
  isManualMode,
  onGradeChange,
  onAttitudeChange,
  onResponseChange,
  onTargetChange,
  onLearningRateChange,
  onAnimationSpeedChange,
  onManualModeChange,
  onNextStep,
  onStep,
  onTrainToggle,
  onReset,
  isTraining,
  isAnimating,
  showCanvasHeatmap,
  showGridHeatmap,
  onToggleCanvasHeatmap,
  onToggleGridHeatmap,
  hasComparisonData,
  onViewComparison,
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
      </Box>

      {/* Training Controls */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h3" sx={{ mb: 2, pb: 1, borderBottom: '2px solid #334155' }}>
          ⚙️ {t('controls.trainingSection')}
        </Typography>
        
        <Box sx={{ mb: 2.5 }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            {t('controls.learningRate')}
          </Typography>
          <Stack direction="row" alignItems="center" spacing={2}>
            <Slider
              value={learningRate}
              onChange={(_, v) => onLearningRateChange(v as number)}
              min={0.01}
              max={0.5}
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
              {learningRate.toFixed(2)}
            </Typography>
          </Stack>
        </Box>

        <Box sx={{ mb: 2.5 }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            {t('controls.animationSpeed')}
          </Typography>
          <Stack direction="row" alignItems="center" spacing={2}>
            <Slider
              value={animationSpeed}
              onChange={(_, v) => onAnimationSpeedChange(v as number)}
              min={0}
              max={2}
              step={0.1}
              disabled={isManualMode}
              sx={{ flex: 1 }}
            />
            <Typography
              sx={{
                minWidth: 50,
                fontFamily: 'monospace',
                fontWeight: 600,
                color: isManualMode ? 'text.disabled' : 'primary.light'
              }}
            >
              {isManualMode ? 'Manual' : `${animationSpeed.toFixed(1)}x`}
            </Typography>
          </Stack>
        </Box>

        <Box sx={{ mb: 2.5 }}>
          <FormControlLabel
            control={
              <Switch
                checked={isManualMode}
                onChange={(e) => onManualModeChange(e.target.checked)}
                disabled={isAnimating}
              />
            }
            label={
              <Typography variant="body2" color="text.secondary">
                {t('controls.manualMode')}
              </Typography>
            }
          />
        </Box>

        {(isManualMode || animationSpeed === 0) && isAnimating && (
          <Box sx={{ mb: 2.5 }}>
            <Button
              variant="contained"
              fullWidth
              onClick={onNextStep}
              sx={{
                bgcolor: 'secondary.main',
                '&:hover': { bgcolor: 'secondary.dark' },
                fontWeight: 700,
              }}
            >
              ▶ {t('controls.nextStep')}
            </Button>
          </Box>
        )}

        <Stack spacing={1.5}>
          <Button 
            variant="contained" 
            onClick={onStep}
            sx={{ 
              bgcolor: isAnimating ? 'warning.main' : 'primary.main',
              '&:hover': {
                bgcolor: isAnimating ? 'warning.dark' : 'primary.dark',
              }
            }}
          >
            {isAnimating ? t('controls.stop') : t('controls.oneStep')}
          </Button>
          <Button 
            variant="contained" 
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
          <Button 
            variant="contained" 
            sx={{ 
              bgcolor: '#64748b',
              '&:hover': { bgcolor: 'error.main' }
            }}
            onClick={onReset}
          >
            {t('controls.reset')}
          </Button>
        </Stack>
      </Box>

      {/* Visualization Options */}
      <Box>
        <Typography variant="h3" sx={{ mb: 2, pb: 1, borderBottom: '2px solid #334155' }}>
          👁️ {t('visualization.activationTitle')}
        </Typography>
        
        <Box sx={{ mb: 1.5 }}>
          <FormControlLabel
            control={
              <Switch
                checked={showCanvasHeatmap}
                onChange={onToggleCanvasHeatmap}
                size="small"
              />
            }
            label={
              <Typography variant="body2" color="text.secondary">
                {t('visualization.canvasHeatmap')}
              </Typography>
            }
          />
        </Box>

        <Box>
          <FormControlLabel
            control={
              <Switch
                checked={showGridHeatmap}
                onChange={onToggleGridHeatmap}
                size="small"
              />
            }
            label={
              <Typography variant="body2" color="text.secondary">
                {t('visualization.gridHeatmap')}
              </Typography>
            }
          />
        </Box>

        <Box sx={{ mt: 2 }}>
          <Button
            variant="outlined"
            fullWidth
            onClick={onViewComparison}
            disabled={!hasComparisonData}
            sx={{
              borderColor: 'primary.main',
              color: 'primary.main',
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
      </Box>
    </Paper>
  );
}
