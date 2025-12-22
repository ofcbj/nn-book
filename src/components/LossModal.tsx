import {
  Dialog,
  DialogTitle,
  DialogContent,
  Button,
  Box,
  Typography,
  LinearProgress,
  Stack,
} from '@mui/material';

interface LossModalProps {
  open: boolean;
  targetClass: number;
  predictions: number[];
  loss: number;
  onClose: () => void;
}

const classNames = ['불합격', '보류', '합격'];

export default function LossModal({
  open,
  targetClass,
  predictions,
  loss,
  onClose,
}: LossModalProps) {
  const targetOneHot = [0, 0, 0];
  targetOneHot[targetClass] = 1;
  const targetProb = predictions[targetClass] || 0;
  const targetProbPercent = (targetProb * 100).toFixed(1);

  const getLossInterpretation = () => {
    if (loss < 0.1) return '매우 작은 값으로, 예측이 정답에 매우 가깝습니다';
    if (loss < 0.5) return '작은 값으로, 예측이 정답에 가깝습니다';
    if (loss < 1.0) return '중간 값으로, 예측과 정답 사이에 차이가 있습니다';
    return '큰 값으로, 예측이 정답과 많이 다릅니다';
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: {
          bgcolor: 'background.paper',
          backgroundImage: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)',
          border: '2px solid #60a5fa',
        },
      }}
    >
      <DialogTitle sx={{ textAlign: 'center', borderBottom: '2px solid #475569', pb: 2 }}>
        <Typography variant="h5" fontWeight={700}>
          📊 Forward Pass 결과 및 Loss 계산
        </Typography>
      </DialogTitle>
      
      <DialogContent sx={{ pt: 3 }}>
        <Box sx={{ display: 'flex', gap: 3, flexDirection: { xs: 'column', md: 'row' } }}>
          {/* Left Column */}
          <Box sx={{ flex: 1 }}>
            {/* Expected & Actual */}
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ mb: 3 }}>
              <Box sx={{ flex: 1, p: 2, bgcolor: 'rgba(15, 23, 42, 0.6)', borderRadius: 2, border: '1px solid #475569' }}>
                <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 1.5 }}>
                  🎯 기대 결과 (Expected)
                </Typography>
                <Typography color="text.secondary" fontSize="0.85rem">Target Class:</Typography>
                <Typography 
                  sx={{ 
                    color: 'secondary.main', 
                    fontWeight: 700, 
                    fontSize: '1.3rem',
                    p: 1,
                    bgcolor: 'rgba(34, 197, 94, 0.15)',
                    borderRadius: 1,
                    mb: 1
                  }}
                >
                  {classNames[targetClass]}
                </Typography>
                <Typography color="text.secondary" fontSize="0.85rem">One-Hot:</Typography>
                <Typography sx={{ fontFamily: 'monospace', color: 'primary.light', fontWeight: 600 }}>
                  [{targetOneHot.join(', ')}]
                </Typography>
              </Box>
              
              <Box sx={{ flex: 1, p: 2, bgcolor: 'rgba(15, 23, 42, 0.6)', borderRadius: 2, border: '1px solid #475569' }}>
                <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 1.5 }}>
                  🔮 Forward 결과 (Actual)
                </Typography>
                {predictions.map((prob, i) => (
                  <Box key={i} sx={{ mb: 1 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography fontSize="0.8rem">{classNames[i]}</Typography>
                      <Typography fontSize="0.8rem" fontWeight={600}>{(prob * 100).toFixed(1)}%</Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={prob * 100}
                      sx={{
                        height: 8,
                        borderRadius: 1,
                        bgcolor: 'rgba(59, 130, 246, 0.2)',
                        '& .MuiLinearProgress-bar': {
                          bgcolor: i === targetClass ? 'secondary.main' : 'primary.main',
                        },
                      }}
                    />
                  </Box>
                ))}
              </Box>
            </Stack>

            {/* Loss Value */}
            <Box sx={{ p: 2, bgcolor: 'rgba(251, 191, 36, 0.1)', border: '2px solid rgba(251, 191, 36, 0.4)', borderRadius: 2, textAlign: 'center', mb: 3 }}>
              <Typography sx={{ color: 'warning.main', fontWeight: 600, textTransform: 'uppercase', letterSpacing: 1, fontSize: '0.9rem' }}>
                Cross-Entropy Loss
              </Typography>
              <Typography sx={{ fontFamily: 'monospace', fontSize: '2rem', color: 'warning.main', fontWeight: 700, my: 1 }}>
                {loss.toFixed(6)}
              </Typography>
              <Typography sx={{ fontFamily: 'monospace', color: 'text.secondary', fontSize: '0.9rem' }}>
                L = -Σ(target × log(predicted))
              </Typography>
            </Box>

            {/* Loss Calculation Steps */}
            <Box sx={{ p: 2, bgcolor: 'rgba(251, 191, 36, 0.08)', border: '1px solid rgba(251, 191, 36, 0.3)', borderRadius: 2 }}>
              <Typography variant="subtitle1" sx={{ color: 'warning.main', fontWeight: 600, mb: 1.5 }}>
                📐 Loss 계산 과정
              </Typography>
              
              <Box sx={{ mb: 1.5 }}>
                <Typography color="text.secondary" fontSize="0.85rem">1. Cross-Entropy 수식 전개:</Typography>
                <Typography sx={{ fontFamily: 'monospace', fontSize: '0.8rem', p: 1, bgcolor: 'rgba(0,0,0,0.3)', borderRadius: 1 }}>
                  L = -({predictions.map((p, i) => `${targetOneHot[i]}×log(${p.toFixed(3)})`).join(' + ')})
                </Typography>
              </Box>
              
              <Box sx={{ mb: 1.5 }}>
                <Typography color="text.secondary" fontSize="0.85rem">2. 정답 클래스만 계산:</Typography>
                <Typography sx={{ fontFamily: 'monospace', fontSize: '0.8rem', p: 1, bgcolor: 'rgba(0,0,0,0.3)', borderRadius: 1 }}>
                  L = -log({targetProb.toFixed(3)})
                </Typography>
              </Box>
              
              <Box>
                <Typography color="text.secondary" fontSize="0.85rem">3. 최종 결과:</Typography>
                <Typography sx={{ fontFamily: 'monospace', fontSize: '0.9rem', fontWeight: 700, color: 'warning.main', p: 1, bgcolor: 'rgba(0,0,0,0.3)', borderRadius: 1 }}>
                  L = {loss.toFixed(6)}
                </Typography>
              </Box>
            </Box>
          </Box>

          {/* Right Column */}
          <Box sx={{ flex: 1 }}>
            <Box sx={{ p: 2, bgcolor: 'rgba(15, 23, 42, 0.6)', borderRadius: 2, border: '1px solid #475569', mb: 3 }}>
              <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 2 }}>
                📚 결과 해석
              </Typography>

              <Box sx={{ mb: 2 }}>
                <Typography color="text.secondary" fontSize="0.85rem" sx={{ mb: 0.5 }}>
                  🎯 Expected (기대 결과):
                </Typography>
                <Typography fontSize="0.9rem">
                  신경망이 <strong>{classNames[targetClass]}</strong>을 예측하도록 학습해야 합니다.
                </Typography>
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography color="text.secondary" fontSize="0.85rem" sx={{ mb: 0.5 }}>
                  📊 Loss 값의 의미:
                </Typography>
                <Typography fontSize="0.9rem">
                  Loss 값 <strong>{loss.toFixed(4)}</strong>는 {getLossInterpretation()}.
                </Typography>
              </Box>

              <Box sx={{ mb: 2, p: 1.5, bgcolor: 'rgba(0,0,0,0.3)', borderRadius: 1 }}>
                <Typography fontSize="0.85rem" sx={{ mb: 1 }}>
                  💡 <strong>{classNames[targetClass]}</strong>의 확률: <strong>{targetProbPercent}%</strong>
                </Typography>
                <Typography fontSize="0.8rem" color="text.secondary">
                  {targetProb > 0.7 
                    ? '확률이 높아 Loss가 작습니다!' 
                    : '확률이 낮아 Loss가 큽니다. 학습이 필요합니다.'}
                </Typography>
              </Box>

              <Box>
                <Typography color="text.secondary" fontSize="0.85rem" sx={{ mb: 0.5 }}>
                  ⚡ Backpropagation 과정:
                </Typography>
                <Typography fontSize="0.9rem" component="div">
                  <ol style={{ margin: 0, paddingLeft: 20 }}>
                    <li><strong>오차 계산</strong>: 각 출력 뉴런의 오차를 계산</li>
                    <li><strong>역전파</strong>: 출력층 → 은닉층 → 입력층 순으로 전달</li>
                    <li><strong>가중치 업데이트</strong>: 오차를 줄이는 방향으로 조정</li>
                  </ol>
                </Typography>
              </Box>
            </Box>

            <Button
              variant="contained"
              fullWidth
              size="large"
              onClick={onClose}
              sx={{
                py: 1.5,
                fontSize: '1.1rem',
                fontWeight: 700,
                background: 'linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%)',
              }}
            >
              ▶ Backward Process 시작
            </Button>
          </Box>
        </Box>
      </DialogContent>
    </Dialog>
  );
}
