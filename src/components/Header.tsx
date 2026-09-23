import { Typography, Paper, IconButton, Box, Button, Switch, FormControlLabel, Tooltip } from '@mui/material';
import { useTranslation } from 'react-i18next';
import type { TrainingMode } from '../hooks/useNetworkState';
import { DATASET_SIZE } from '../lib/core';

interface HeaderProps {
  mode: TrainingMode;
  onModeChange: (mode: TrainingMode) => void;
  /** Disable the mode switch while an animation or training is running */
  modeDisabled?: boolean;
  onHelpClick?: () => void;
}

const LANGUAGES = [
  { code: 'en', flag: '🇺🇸', label: 'English' },
  { code: 'ko', flag: '🇰🇷', label: '한국어' },
  { code: 'ja', flag: '🇯🇵', label: '日本語' },
] as const;

export default function Header({ mode, onModeChange, modeDisabled = false, onHelpClick }: HeaderProps) {
  const { t, i18n } = useTranslation();
  const dataMode = mode === 'dataset';

  return (
    <Paper
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        px: 2,
        py: 1,
        mb: 1.5,
        background: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)',
      }}
    >
      {/* Language Switcher */}
      <Box sx={{ display: 'flex', gap: 0.5, flexShrink: 0 }}>
        {LANGUAGES.map(({ code, flag, label }) => {
          const active = i18n.language === code;
          return (
            <Button
              key={code}
              onClick={() => i18n.changeLanguage(code)}
              variant={active ? 'contained' : 'outlined'}
              size="small"
              aria-label={label}
              aria-pressed={active}
              sx={{
                minWidth: 40,
                px: 1,
                py: 0.25,
                color: active ? 'white' : 'rgba(255,255,255,0.7)',
                bgcolor: active ? 'rgba(59, 130, 246, 0.8)' : 'transparent',
                borderColor: 'rgba(255,255,255,0.3)',
                '&:hover': {
                  bgcolor: active ? 'rgba(59, 130, 246, 1)' : 'rgba(255,255,255,0.1)',
                },
              }}
            >
              {flag}
            </Button>
          );
        })}
      </Box>

      {/* Title */}
      <Box sx={{ flex: 1, textAlign: 'center', minWidth: 0 }}>
        <Typography variant="h1" sx={{ color: 'white', fontSize: { xs: '1.05rem', md: '1.35rem' }, lineHeight: 1.3 }}>
          {t('header.title')}
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ display: { xs: 'none', md: 'block' } }}>
          {t('header.subtitle')}
        </Typography>
      </Box>

      {/* Data training mode */}
      <Tooltip title={modeDisabled ? t('header.dataModeBusy') : t('header.dataModeHint', { count: DATASET_SIZE })}>
        <FormControlLabel
          sx={{ mr: 0, flexShrink: 0, '& .MuiFormControlLabel-label': { fontSize: '0.85rem', fontWeight: 600, color: dataMode ? 'primary.light' : 'text.secondary' } }}
          control={
            <Switch
              checked={dataMode}
              disabled={modeDisabled}
              onChange={(_, checked) => onModeChange(checked ? 'dataset' : 'single')}
              color="primary"
            />
          }
          label={t('header.dataMode')}
        />
      </Tooltip>

      {/* Help Button */}
      {onHelpClick && (
        <IconButton
          onClick={onHelpClick}
          aria-label={t('help.title')}
          size="small"
          sx={{
            flexShrink: 0,
            color: 'white',
            bgcolor: 'rgba(59, 130, 246, 0.2)',
            '&:hover': { bgcolor: 'rgba(59, 130, 246, 0.4)' },
          }}
        >
          <Typography fontSize="1.2rem">❓</Typography>
        </IconButton>
      )}
    </Paper>
  );
}

export function Footer() {
  const { t } = useTranslation();
  return (
    <Paper sx={{ textAlign: 'center', py: 1.5, px: 3, mt: 1.5 }}>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
        Architecture: <strong>3 inputs</strong> → <strong>5 neurons({t('layers.layer1Prefix')})</strong> →
        <strong> 3 neurons({t('layers.layer2Prefix')})</strong> → <strong>3 outputs</strong>
      </Typography>
      <Typography variant="caption" color="text.disabled" sx={{ maxWidth: 800, mx: 'auto', display: 'block' }}>
        {t('footer.description')}
      </Typography>
    </Paper>
  );
}
