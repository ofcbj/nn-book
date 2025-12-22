import { Box, Paper, Typography, Slider, Button, Stack } from '@mui/material';

interface ControlPanelProps {
  // Input values
  grade: number;
  attitude: number;
  response: number;
  targetValue: number;
  learningRate: number;
  animationSpeed: number;
  // Event handlers
  onGradeChange: (value: number) => void;
  onAttitudeChange: (value: number) => void;
  onResponseChange: (value: number) => void;
  onTargetChange: (value: number) => void;
  onLearningRateChange: (value: number) => void;
  onAnimationSpeedChange: (value: number) => void;
  // Button handlers
  onStep: () => void;
  onTrainToggle: () => void;
  onReset: () => void;
  // State
  isTraining: boolean;
  isAnimating: boolean;
}

const classNames = ['불합격', '보류', '합격'];

export default function ControlPanel({
  grade,
  attitude,
  response,
  targetValue,
  learningRate,
  animationSpeed,
  onGradeChange,
  onAttitudeChange,
  onResponseChange,
  onTargetChange,
  onLearningRateChange,
  onAnimationSpeedChange,
  onStep,
  onTrainToggle,
  onReset,
  isTraining,
  isAnimating,
}: ControlPanelProps) {
  return (
    <Paper sx={{ p: 2, height: '100%' }}>
      {/* Input Controls */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h3" sx={{ mb: 1.5, pb: 0.5, borderBottom: '2px solid #334155', fontSize: '0.95rem' }}>
          📊 입력 제어
        </Typography>
        
        <Box sx={{ mb: 2.5 }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            성적 (Grade)
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
            태도 (Attitude)
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
            응답수준 (Response)
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
            목표 결정 (Target)
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
          ⚙️ 학습 제어
        </Typography>
        
        <Box sx={{ mb: 2.5 }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            학습률 (Learning Rate)
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
            애니메이션 속도 (Speed)
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
              {animationSpeed.toFixed(1)}x
            </Typography>
          </Stack>
        </Box>

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
            {isAnimating ? '스탑' : '1 Step'}
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
            {isTraining ? '학습 중지' : '학습 시작'}
          </Button>
          <Button 
            variant="contained" 
            sx={{ 
              bgcolor: '#64748b',
              '&:hover': { bgcolor: 'error.main' }
            }}
            onClick={onReset}
          >
            리셋
          </Button>
        </Stack>
      </Box>
    </Paper>
  );
}
