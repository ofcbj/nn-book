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
import { useTranslation } from 'react-i18next';

interface LossModalProps {
  open: boolean;
  targetClass: number;
  predictions: number[];
  loss: number;
  onClose: () => void;
}

export default function LossModal({
  open,
  targetClass,
  predictions,
  loss,
  onClose,
}: LossModalProps) {
  const { t } = useTranslation();
  const classNames = [t('classes.fail'), t('classes.pending'), t('classes.pass')];
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

              <Box sx={{ mb: 2 }}>
                <Typography color="text.secondary" fontSize="0.85rem" sx={{ mb: 1 }}>
                  🔍 이번에 틀린 이유:
                </Typography>
                <Box sx={{ p: 1.5, bgcolor: 'rgba(239, 68, 68, 0.15)', borderRadius: 1, border: '1px solid rgba(239, 68, 68, 0.3)' }}>
                  {predictions.map((prob, i) => {
                    const expected = targetOneHot[i];
                    const error = expected - prob;
                    if (Math.abs(error) < 0.01) return null;
                    return (
                      <Typography key={i} fontSize="0.8rem" sx={{ mb: 0.5 }}>
                        <strong>{classNames[i]}</strong>:
                        {error > 0
                          ? ` 너무 낮음 (${(error * 100).toFixed(1)}% 더 높여야 함)`
                          : ` 너무 높음 (${(Math.abs(error) * 100).toFixed(1)}% 낮춰야 함)`}
                      </Typography>
                    );
                  })}
                </Box>
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography color="text.secondary" fontSize="0.85rem" sx={{ mb: 1 }}>
                  🎓 역전파를 쉽게 이해하기:
                </Typography>
                <Box sx={{ p: 1.5, bgcolor: 'rgba(59, 130, 246, 0.1)', borderRadius: 1, border: '1px solid rgba(59, 130, 246, 0.3)' }}>
                  <Typography fontSize="0.85rem" sx={{ mb: 1 }}>
                    <strong>비유:</strong> 팀 프로젝트에서 실수했을 때
                  </Typography>
                  <Typography fontSize="0.8rem" color="text.secondary" component="div">
                    • <strong>오류 발견</strong>: "최종 결과가 틀렸네!"<br/>
                    • <strong>영향력 파악</strong>: 각 팀원이 결과에 미친 영향도 측정<br/>
                    • <strong>조정량 계산</strong>: 영향이 큰 사람일수록 더 많이 개선<br/>
                    • <strong>개선 적용</strong>: 다음번엔 더 나은 판단을 하도록
                  </Typography>
                </Box>
              </Box>

              <Box>
                <Typography color="text.secondary" fontSize="0.85rem" sx={{ mb: 0.5 }}>
                  ⚡ 역전파 4단계 과정:
                </Typography>
                <Typography fontSize="0.9rem" component="div">
                  <ol style={{ margin: 0, paddingLeft: 20, fontSize: '0.85rem' }}>
                    <li><strong>받은 오류</strong>: 이 뉴런이 받은 실수 신호</li>
                    <li><strong>조정 방향 계산</strong>: 오류 × 민감도(기울기)</li>
                    <li><strong>가중치 변화량</strong>: 조정방향 × 입력 × 학습률</li>
                    <li><strong>업데이트</strong>: 가중치와 bias를 실제로 수정</li>
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
