import { Box, Paper, Typography, Slider, Button, Stack } from '@mui/material';
import { useTranslation } from 'react-i18next';

interface ControlPanelProps {
  // Input values
  grade           : number;
  attitude        : number;
  response        : number;
  targetValue     : number;
  animationSpeed  : number;
  // Event handlers
  onGradeChange   : (value: number) => void;
  onAttitudeChange: (value: number) => void;
  onResponseChange: (value: number) => void;
  onTargetChange  : (value: number) => void;
  onAnimationSpeedChange: (value: number) => void;
  // Button handlers
  onStep          : () => void;
  onReset         : () => void;
  // State
  isAnimating     : boolean;
  isPaused        : boolean;
  /** Data training mode: sliders are a test candidate, training happens in the right panel */
  dataMode        : boolean;
}

interface LabeledSliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  marks?: boolean;
  /** Text shown next to the slider (defaults to the value with 2 decimals) */
  display?: string;
  /** Color of the display text */
  color?: string;
}

function LabeledSlider({ label, value, onChange, min, max, step, marks, display, color = 'primary.light' }: LabeledSliderProps) {
  return (
    <Box sx={{ mb: 2.5 }}>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
        {label}
      </Typography>
      <Stack direction="row" alignItems="center" spacing={1.5}>
        <Slider
          value={value}
          onChange={(_, v) => onChange(v as number)}
          min={min}
          max={max}
          step={step}
          marks={marks}
          sx={{ flex: 1 }}
        />
        <Typography
          sx={{
            minWidth: 50,
            fontFamily: 'monospace',
            fontWeight: 600,
            fontSize: '0.85rem',
            color,
          }}
        >
          {display ?? value.toFixed(2)}
        </Typography>
      </Stack>
    </Box>
  );
}

const TARGET_COLORS = ['error.main', 'warning.main', 'secondary.main'];

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
  isPaused,
  dataMode,
}: ControlPanelProps) {
  const { t } = useTranslation();
  const classNames = [t('classes.fail'), t('classes.pending'), t('classes.pass')];

  const stepLabel = isPaused ? t('controls.resume') : (isAnimating ? t('controls.pause') : t('controls.start'));
  const stepColor = isPaused ? 'success' : (isAnimating ? 'warning' : 'primary');

  return (
    <Paper sx={{ p: 2, height: '100%' }}>
      {/* Input Controls */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h3" sx={{ mb: 1.5, pb: 0.5, borderBottom: '2px solid #334155', fontSize: '0.95rem' }}>
          📊 {dataMode ? t('controls.testCandidateSection') : t('controls.inputSection')}
        </Typography>
        {dataMode && (
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1.5 }}>
            {t('controls.testCandidateHint')}
          </Typography>
        )}

        <LabeledSlider label={t('controls.grade')}    value={grade}    onChange={onGradeChange}    min={0} max={1} step={0.01} />
        <LabeledSlider label={t('controls.attitude')} value={attitude} onChange={onAttitudeChange} min={0} max={1} step={0.01} />
        <LabeledSlider label={t('controls.response')} value={response} onChange={onResponseChange} min={0} max={1} step={0.01} />
        <LabeledSlider
          label={t('controls.target')}
          value={targetValue}
          onChange={onTargetChange}
          min={0}
          max={classNames.length - 1}
          step={1}
          marks
          display={classNames[targetValue]}
          color={TARGET_COLORS[targetValue]}
        />

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

      {/* Training Controls (single-candidate mode only) */}
      {!dataMode && <Box sx={{ mb: 3 }}>
        <Typography variant="h3" sx={{ mb: 2, pb: 1, borderBottom: '2px solid #334155' }}>
          ⚙️ {t('controls.trainingSection')}
        </Typography>

        <LabeledSlider
          label={t('controls.animationSpeed')}
          value={animationSpeed}
          onChange={onAnimationSpeedChange}
          min={0.1}
          max={2}
          step={0.1}
          display={`${animationSpeed.toFixed(1)}x`}
        />

        <Button
          variant="contained"
          fullWidth
          onClick={onStep}
          sx={{
            bgcolor: `${stepColor}.main`,
            '&:hover': { bgcolor: `${stepColor}.dark` },
          }}
        >
          {stepLabel}
        </Button>
      </Box>}
    </Paper>
  );
}
