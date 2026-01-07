import {
  Dialog,
  DialogTitle,
  DialogContent,
  Button,
  Box,
  Typography,
  Stack,
} from '@mui/material';
import { useTranslation } from 'react-i18next';
import type { BackpropSummaryData } from '../lib/types';

interface BackpropModalProps {
  open: boolean;
  data: BackpropSummaryData | null;
  onClose: () => void;
}

export default function BackpropModal({ open, data, onClose }: BackpropModalProps) {
  const { t } = useTranslation();

  if (!data) return null;

  const totalWeights = 
    data.oldWeights.layer1.reduce((sum, w) => sum + w.length, 0) +
    data.oldWeights.layer2.reduce((sum, w) => sum + w.length, 0) +
    data.oldWeights.output.reduce((sum, w) => sum + w.length, 0);

  const totalBiases = 
    data.oldBiases.layer1.length +
    data.oldBiases.layer2.length +
    data.oldBiases.output.length;

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: {
          bgcolor: '#0f172a',
          color: '#e2e8f0',
        }
      }}
    >
      <DialogTitle sx={{ textAlign: 'center', borderBottom: '2px solid #475569', pb: 2 }}>
        <Typography variant="h5" component="div" fontWeight={700}>
          {t('backpropSummary.title')}
        </Typography>
      </DialogTitle>

      <DialogContent sx={{ pt: 3 }}>
        <Box sx={{ display: 'flex', gap: 3, flexDirection: { xs: 'column', lg: 'row' } }}>
          {/* Left Column - Overview & Detailed Process */}
          <Box sx={{ flex: 1 }}>
            {/* Overview */}
            <Box sx={{ mb: 3, p: 2, bgcolor: 'rgba(59, 130, 246, 0.1)', borderRadius: 2, border: '1px solid rgba(59, 130, 246, 0.3)' }}>
              <Typography variant="h6" sx={{ mb: 1 }}>
                ✅ {t('backpropSummary.overview')}
              </Typography>
              <Typography color="text.secondary" fontSize="0.9rem">
                {t('backpropSummary.totalUpdated')}: <strong>{totalWeights}</strong> {t('backpropSummary.weightChanges')} + <strong>{totalBiases}</strong> {t('backpropSummary.biasChanges')}
              </Typography>
              <Typography color="text.secondary" fontSize="0.9rem">
                {t('backpropSummary.learningRate')}: <strong>{data.learningRate}</strong>
              </Typography>
            </Box>

            {/* Detailed Process */}
            <Box sx={{ mb: 3, p: 2, bgcolor: 'rgba(139, 92, 246, 0.1)', borderRadius: 2, border: '1px solid rgba(139, 92, 246, 0.3)' }}>
              <Typography variant="h6" sx={{ mb: 2 }}>
                🔄 {t('backpropSummary.detailedProcess')}
              </Typography>
              
              <Stack spacing={2}>
                {/* Step 1 */}
                <Box>
                  <Typography fontWeight={600} color="primary.light" sx={{ mb: 0.5 }}>
                    {t('backpropSummary.step1Title')}
                  </Typography>
                  <Typography fontSize="0.85rem" color="text.secondary" sx={{ mb: 1 }}>
                    {t('backpropSummary.step1Desc')}
                  </Typography>
                  <Box sx={{ pl: 2, borderLeft: '2px solid rgba(139, 92, 246, 0.5)' }}>
                    <Typography sx={{ fontFamily: 'monospace', fontSize: '0.8rem', color: 'primary.light' }}>
                      {t('backpropSummary.outputLayer')}: error = target - output
                    </Typography>
                    <Typography sx={{ fontFamily: 'monospace', fontSize: '0.8rem', color: 'primary.light' }}>
                      {t('backpropSummary.hiddenLayer')}: error = Σ(next_error × weight)
                    </Typography>
                  </Box>
                </Box>

                {/* Step 2 */}
                <Box>
                  <Typography fontWeight={600} color="primary.light" sx={{ mb: 0.5 }}>
                    {t('backpropSummary.step2Title')}
                  </Typography>
                  <Typography fontSize="0.85rem" color="text.secondary" sx={{ mb: 1 }}>
                    {t('backpropSummary.step2Desc')}
                  </Typography>
                  <Box sx={{ pl: 2, borderLeft: '2px solid rgba(139, 92, 246, 0.5)' }}>
                    <Typography sx={{ fontFamily: 'monospace', fontSize: '0.8rem', color: 'primary.light' }}>
                      δ = error × σ'(activation)
                    </Typography>
                    <Typography sx={{ fontFamily: 'monospace', fontSize: '0.8rem', color: 'primary.light' }}>
                      σ'(y) = y × (1 - y)
                    </Typography>
                  </Box>
                </Box>

                {/* Step 3 */}
                <Box>
                  <Typography fontWeight={600} color="primary.light" sx={{ mb: 0.5 }}>
                    {t('backpropSummary.step3Title')}
                  </Typography>
                  <Typography fontSize="0.85rem" color="text.secondary" sx={{ mb: 1 }}>
                    {t('backpropSummary.step3Desc')}
                  </Typography>
                  <Box sx={{ pl: 2, borderLeft: '2px solid rgba(139, 92, 246, 0.5)' }}>
                    <Typography sx={{ fontFamily: 'monospace', fontSize: '0.8rem', color: 'primary.light' }}>
                      ΔW = learning_rate × δ × input
                    </Typography>
                    <Typography sx={{ fontFamily: 'monospace', fontSize: '0.8rem', color: 'primary.light' }}>
                      Δb = learning_rate × δ
                    </Typography>
                  </Box>
                </Box>

                {/* Step 4 */}
                <Box>
                  <Typography fontWeight={600} color="primary.light" sx={{ mb: 0.5 }}>
                    {t('backpropSummary.step4Title')}
                  </Typography>
                  <Typography fontSize="0.85rem" color="text.secondary" sx={{ mb: 1 }}>
                    {t('backpropSummary.step4Desc')}
                  </Typography>
                  <Box sx={{ pl: 2, borderLeft: '2px solid rgba(139, 92, 246, 0.5)' }}>
                    <Typography sx={{ fontFamily: 'monospace', fontSize: '0.8rem', color: 'primary.light' }}>
                      W_new = W_old - ΔW
                    </Typography>
                    <Typography sx={{ fontFamily: 'monospace', fontSize: '0.8rem', color: 'primary.light' }}>
                      b_new = b_old - Δb
                    </Typography>
                  </Box>
                </Box>
              </Stack>
            </Box>
          </Box>

          {/* Right Column - Weight Changes */}
          <Box sx={{ flex: 1 }}>
            {/* Weight Changes Summary */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" sx={{ mb: 2 }}>
                🔄 {t('backpropSummary.weightChanges')}
              </Typography>
              <Stack spacing={2}>
                {/* Layer 1 */}
                <Box sx={{ p: 2, bgcolor: 'rgba(34, 197, 94, 0.1)', border: '1px solid rgba(34, 197, 94, 0.3)', borderRadius: 2 }}>
                  <Typography fontWeight={600} color="success.light">
                    {t('backpropSummary.layer1Weights')} ({data.oldWeights.layer1.reduce((sum, w) => sum + w.length, 0)} {t('common.count')})
                  </Typography>
                  <Typography fontSize="0.85rem" color="text.secondary" sx={{ mt: 0.5 }}>
                    {t('backpropSummary.averageChange')} {calculateAverageChange(data.oldWeights.layer1, data.newWeights.layer1).toFixed(6)}
                  </Typography>
                </Box>

                {/* Layer 2 */}
                <Box sx={{ p: 2, bgcolor: 'rgba(249, 115, 22, 0.1)', border: '1px solid rgba(249, 115, 22, 0.3)', borderRadius: 2 }}>
                  <Typography fontWeight={600} color="warning.light">
                    {t('backpropSummary.layer2Weights')} ({data.oldWeights.layer2.reduce((sum, w) => sum + w.length, 0)} {t('common.count')})
                  </Typography>
                  <Typography fontSize="0.85rem" color="text.secondary" sx={{ mt: 0.5 }}>
                    {t('backpropSummary.averageChange')} {calculateAverageChange(data.oldWeights.layer2, data.newWeights.layer2).toFixed(6)}
                  </Typography>
                </Box>

                {/* Output */}
                <Box sx={{ p: 2, bgcolor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: 2 }}>
                  <Typography fontWeight={600} color="error.light">
                    {t('backpropSummary.outputWeights')} ({data.oldWeights.output.reduce((sum, w) => sum + w.length, 0)} {t('common.count')})
                  </Typography>
                  <Typography fontSize="0.85rem" color="text.secondary" sx={{ mt: 0.5 }}>
                    {t('backpropSummary.averageChange')} {calculateAverageChange(data.oldWeights.output, data.newWeights.output).toFixed(6)}
                  </Typography>
                </Box>
              </Stack>
            </Box>

            {/* Bias Changes */}
            <Box sx={{ mb: 3, p: 2, bgcolor: 'rgba(139, 92, 246, 0.1)', border: '1px solid rgba(139, 92, 246, 0.3)', borderRadius: 2 }}>
              <Typography variant="h6" sx={{ mb: 1.5 }}>
                ⚖️ {t('backpropSummary.biasChanges')}
              </Typography>
              <Typography fontSize="0.85rem" color="text.secondary">
                Layer 1: {data.oldBiases.layer1.length} {t('common.countUpdates')}
              </Typography>
              <Typography fontSize="0.85rem" color="text.secondary">
                Layer 2: {data.oldBiases.layer2.length} {t('common.countUpdates')}
              </Typography>
              <Typography fontSize="0.85rem" color="text.secondary">
                Output: {data.oldBiases.output.length} {t('common.countUpdates')}
              </Typography>
            </Box>
          </Box>
        </Box>

        {/* Close Button */}
        <Button
          variant="contained"
          fullWidth
          size="large"
          onClick={onClose}
          sx={{
            py: 1.5,
            fontSize: '1.1rem',
            fontWeight: 700,
            background: 'linear-gradient(90deg, #34d399 0%, #10b981 100%)',
          }}
        >
          {t('backpropSummary.close')}
        </Button>
      </DialogContent>
    </Dialog>
  );
}

function calculateAverageChange(oldWeights: number[][], newWeights: number[][]): number {
  let totalChange = 0;
  let count = 0;

  for (let i = 0; i < oldWeights.length; i++) {
    for (let j = 0; j < oldWeights[i].length; j++) {
      totalChange += Math.abs(newWeights[i][j] - oldWeights[i][j]);
      count++;
    }
  }

  return count > 0 ? totalChange / count : 0;
}
