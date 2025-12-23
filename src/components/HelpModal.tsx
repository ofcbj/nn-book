import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
} from '@mui/material';
import { useTranslation } from 'react-i18next';

interface HelpModalProps {
  open: boolean;
  onClose: () => void;
}

export default function HelpModal({ open, onClose }: HelpModalProps) {
  const { t } = useTranslation();
  
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
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
          {t('help.title')}
        </Typography>
      </DialogTitle>
      
      <DialogContent sx={{ pt: 3 }}>
        {/* Introduction */}
        <Box sx={{ mb: 3, p: 2, bgcolor: 'rgba(59, 130, 246, 0.1)', borderRadius: 2, border: '1px solid rgba(59, 130, 246, 0.3)' }}>
          <Typography variant="h6" fontWeight={600} sx={{ mb: 1.5 }}>
            🎯 {t('help.whatIs.title')}
          </Typography>
          <Typography fontSize="0.95rem" sx={{ lineHeight: 1.7 }}>
            {t('help.whatIs.description')}
          </Typography>
        </Box>

        {/* Network Structure */}
        <Box sx={{ mb: 3, p: 2, bgcolor: 'rgba(34, 197, 94, 0.1)', borderRadius: 2, border: '1px solid rgba(34, 197, 94, 0.3)' }}>
          <Typography variant="h6" fontWeight={600} sx={{ mb: 1.5 }}>
            🏗️ {t('help.structure.title')}
          </Typography>
          
          <Box sx={{ mb: 2 }}>
            <Typography fontWeight={600} fontSize="0.95rem" sx={{ mb: 0.5 }}>
              📥 {t('help.structure.input.title')}
            </Typography>
            <Typography fontSize="0.9rem" color="text.secondary" sx={{ ml: 2, whiteSpace: 'pre-line' }}>
              {t('help.structure.input.items')}
            </Typography>
          </Box>

          <Box sx={{ mb: 2 }}>
            <Typography fontWeight={600} fontSize="0.95rem" sx={{ mb: 0.5 }}>
              👥 {t('help.structure.layer1.title')}
            </Typography>
            <Typography fontSize="0.9rem" color="text.secondary" sx={{ ml: 2 }}>
              {t('help.structure.layer1.description')}
            </Typography>
          </Box>

          <Box sx={{ mb: 2 }}>
            <Typography fontWeight={600} fontSize="0.95rem" sx={{ mb: 0.5 }}>
              👔 {t('help.structure.layer2.title')}
            </Typography>
            <Typography fontSize="0.9rem" color="text.secondary" sx={{ ml: 2 }}>
              {t('help.structure.layer2.description')}
            </Typography>
          </Box>

          <Box>
            <Typography fontWeight={600} fontSize="0.95rem" sx={{ mb: 0.5 }}>
              ⚖️ {t('help.structure.output.title')}
            </Typography>
            <Typography fontSize="0.9rem" color="text.secondary" sx={{ ml: 2, whiteSpace: 'pre-line' }}>
              {t('help.structure.output.items')}
            </Typography>
          </Box>
        </Box>

        {/* How it works */}
        <Box sx={{ mb: 3, p: 2, bgcolor: 'rgba(249, 115, 22, 0.1)', borderRadius: 2, border: '1px solid rgba(249, 115, 22, 0.3)' }}>
          <Typography variant="h6" fontWeight={600} sx={{ mb: 1.5 }}>
            ⚙️ {t('help.howItWorks.title')}
          </Typography>
          
          <Box sx={{ mb: 2 }}>
            <Typography fontWeight={600} fontSize="0.95rem" sx={{ mb: 0.5 }}>
              1️⃣ {t('help.howItWorks.forward.title')}
            </Typography>
            <Typography fontSize="0.9rem" color="text.secondary" sx={{ ml: 2, whiteSpace: 'pre-line' }}>
              {t('help.howItWorks.forward.description')}
            </Typography>
          </Box>

          <Box>
            <Typography fontWeight={600} fontSize="0.95rem" sx={{ mb: 0.5 }}>
              2️⃣ {t('help.howItWorks.backward.title')}
            </Typography>
            <Typography fontSize="0.9rem" color="text.secondary" sx={{ ml: 2, whiteSpace: 'pre-line' }}>
              {t('help.howItWorks.backward.description')}
            </Typography>
          </Box>
        </Box>

        {/* Why it learns this way */}
        <Box sx={{ mb: 3, p: 2, bgcolor: 'rgba(168, 85, 247, 0.1)', borderRadius: 2, border: '1px solid rgba(168, 85, 247, 0.3)' }}>
          <Typography variant="h6" fontWeight={600} sx={{ mb: 1.5 }}>
            💡 {t('help.whyItLearns.title')}
          </Typography>
          
          <Box sx={{ mb: 2 }}>
            <Typography fontWeight={600} fontSize="0.95rem" sx={{ mb: 0.5 }}>
              📉 {t('help.whyItLearns.gradient.title')}
            </Typography>
            <Typography fontSize="0.9rem" color="text.secondary" sx={{ ml: 2, whiteSpace: 'pre-line' }}>
              {t('help.whyItLearns.gradient.description')}
            </Typography>
          </Box>

          <Box sx={{ mb: 2 }}>
            <Typography fontWeight={600} fontSize="0.95rem" sx={{ mb: 0.5 }}>
              🎯 {t('help.whyItLearns.lossMinimization.title')}
            </Typography>
            <Typography fontSize="0.9rem" color="text.secondary" sx={{ ml: 2, whiteSpace: 'pre-line' }}>
              {t('help.whyItLearns.lossMinimization.description')}
            </Typography>
          </Box>

          <Box>
            <Typography fontWeight={600} fontSize="0.95rem" sx={{ mb: 0.5 }}>
              🎲 {t('help.whyItLearns.analogy.title')}
            </Typography>
            <Typography fontSize="0.9rem" color="text.secondary" sx={{ ml: 2, whiteSpace: 'pre-line' }}>
              {t('help.whyItLearns.analogy.description')}
            </Typography>
          </Box>
        </Box>

        {/* Controls */}
        <Box sx={{ p: 2, bgcolor: 'rgba(239, 68, 68, 0.1)', borderRadius: 2, border: '1px solid rgba(239, 68, 68, 0.3)' }}>
          <Typography variant="h6" fontWeight={600} sx={{ mb: 1.5 }}>
            🎮 {t('help.features.title')}
          </Typography>
          
          <Typography fontSize="0.9rem" color="text.secondary" component="div" sx={{ whiteSpace: 'pre-line' }} dangerouslySetInnerHTML={{ __html: t('help.features.list') }} />
        </Box>
      </DialogContent>

      <DialogActions sx={{ p: 2, borderTop: '1px solid #475569' }}>
        <Button 
          onClick={onClose} 
          variant="contained"
          sx={{
            background: 'linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%)',
            fontWeight: 600,
            px: 4
          }}
        >
          {t('help.close')}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
