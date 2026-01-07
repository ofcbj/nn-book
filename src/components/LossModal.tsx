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
    if (loss < 0.1) return t('lossModal.verySmall');
    if (loss < 0.5) return t('lossModal.small');
    if (loss < 1.0) return t('lossModal.medium');
    return t('lossModal.large');
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
        <Typography variant="h5" component="div" fontWeight={700}>
          {t('lossModal.title')}
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
                  {t('lossModal.expected')}
                </Typography>
                <Typography color="text.secondary" fontSize="0.85rem">{t('lossModal.targetClass')}:</Typography>
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
                <Typography color="text.secondary" fontSize="0.85rem">{t('lossModal.oneHot')}:</Typography>
                <Typography sx={{ fontFamily: 'monospace', color: 'primary.light', fontWeight: 600 }}>
                  [{targetOneHot.join(', ')}]
                </Typography>
              </Box>
              
              <Box sx={{ flex: 1, p: 2, bgcolor: 'rgba(15, 23, 42, 0.6)', borderRadius: 2, border: '1px solid #475569' }}>
                <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 1.5 }}>
                  {t('lossModal.actual')}
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
                {t('lossModal.lossValue')}
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
              <Typography variant="subtitle1" sx={{ color: 'warning.main', fontWeight:  600, mb: 1.5 }}>
                📐 {t('lossModal.lossCalculation')}
              </Typography>
              
              <Box sx={{ mb: 1.5 }}>
                <Typography color="text.secondary" fontSize="0.85rem">{t('lossModal.step1')}</Typography>
                <Typography sx={{ fontFamily: 'monospace', fontSize: '0.8rem', p: 1, bgcolor: 'rgba(0,0,0,0.3)', borderRadius: 1 }}>
                  L = -({predictions.map((p, i) => `${targetOneHot[i]}×log(${p.toFixed(3)})`).join(' + ')})
                </Typography>
              </Box>
              
              <Box sx={{ mb: 1.5 }}>
                <Typography color="text.secondary" fontSize="0.85rem">{t('lossModal.step2')}</Typography>
                <Typography sx={{ fontFamily: 'monospace', fontSize: '0.8rem', p: 1, bgcolor: 'rgba(0,0,0,0.3)', borderRadius: 1 }}>
                  L = -log({targetProb.toFixed(3)})
                </Typography>
              </Box>
              
              <Box>
                <Typography color="text.secondary" fontSize="0.85rem">{t('lossModal.step3')}</Typography>
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
                📚 {t('lossModal.interpretation')}
              </Typography>

              <Box sx={{ mb: 2 }}>
                <Typography color="text.secondary" fontSize="0.85rem" sx={{ mb: 0.5 }}>
                  🎯 {t('lossModal.expectedDesc')}
                </Typography>
                <Typography fontSize="0.9rem">
                  {t('lossModal.expectedText', { class: classNames[targetClass] })}
                </Typography>
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography color="text.secondary" fontSize="0.85rem" sx={{ mb: 0.5 }}>
                  📊 {t('lossModal.lossValueMeaning')}
                </Typography>
                <Typography fontSize="0.9rem">
                  {t('lossModal.lossValueText', { value: loss.toFixed(4), interpretation: getLossInterpretation() })}
                </Typography>
              </Box>

              <Box sx={{ mb: 2, p: 1.5, bgcolor: 'rgba(0,0,0,0.3)', borderRadius: 1 }}>
                <Typography fontSize="0.85rem" sx={{ mb: 1 }}>
                  💡 {t('lossModal.probText', { class: classNames[targetClass], prob: targetProbPercent })}
                </Typography>
                <Typography fontSize="0.8rem" color="text.secondary">
                  {targetProb > 0.7 ? t('lossModal.probHigh') : t('lossModal.probLow')}
                </Typography>
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography color="text.secondary" fontSize="0.85rem" sx={{ mb: 1 }}>
                  🔍 {t('lossModal.errorReason')}
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
                          ? ` ${t('lossModal.tooLow', { value: (error * 100).toFixed(1) })}`
                          : ` ${t('lossModal.tooHigh', { value: (Math.abs(error) * 100).toFixed(1) })}`}
                      </Typography>
                    );
                  })}
                </Box>
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography color="text.secondary" fontSize="0.85rem" sx={{ mb: 1 }}>
                  🎓 {t('lossModal.backpropAnalogy')}
                </Typography>
                <Box sx={{ p: 1.5, bgcolor: 'rgba(59, 130, 246, 0.1)', borderRadius: 1, border: '1px solid rgba(59, 130, 246, 0.3)' }}>
                  <Typography fontSize="0.85rem" sx={{ mb: 1 }}>
                    <strong>{t('lossModal.analogyTitle')}</strong>
                  </Typography>
                  <Typography fontSize="0.8rem" color="text.secondary" component="div">
                    • <strong>{t('lossModal.analogyStep1')}</strong>: {t('lossModal.analogyStep1Desc')}<br/>
                    • <strong>{t('lossModal.analogyStep2')}</strong>: {t('lossModal.analogyStep2Desc')}<br/>
                    • <strong>{t('lossModal.analogyStep3')}</strong>: {t('lossModal.analogyStep3Desc')}<br/>
                    • <strong>{t('lossModal.analogyStep4')}</strong>: {t('lossModal.analogyStep4Desc')}
                  </Typography>
                </Box>
              </Box>

              <Box>
                <Typography color="text.secondary" fontSize="0.85rem" sx={{ mb: 0.5 }}>
                  ⚡ {t('lossModal.backpropSteps')}
                </Typography>
                <Typography fontSize="0.9rem" component="div">
                  <ol style={{ margin: 0, paddingLeft: 20, fontSize: '0.85rem' }}>
                    <li><strong>{t('lossModal.backpropStep1')}</strong></li>
                    <li><strong>{t('lossModal.backpropStep2')}</strong></li>
                    <li><strong>{t('lossModal.backpropStep3')}</strong></li>
                    <li><strong>{t('lossModal.backpropStep4')}</strong></li>
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
              {t('lossModal.startBackward')}
            </Button>
          </Box>
        </Box>
      </DialogContent>
    </Dialog>
  );
}
