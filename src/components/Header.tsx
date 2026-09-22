import { Typography, Paper, IconButton, Box, Button } from '@mui/material';
import { useTranslation } from 'react-i18next';

interface HeaderProps {
  onHelpClick?: () => void;
}

const LANGUAGES = [
  { code: 'en', flag: '🇺🇸', label: 'English' },
  { code: 'ko', flag: '🇰🇷', label: '한국어' },
  { code: 'ja', flag: '🇯🇵', label: '日本語' },
] as const;

export default function Header({ onHelpClick }: HeaderProps) {
  const { t, i18n } = useTranslation();

  return (
    <Paper
      sx={{
        position: 'relative',
        textAlign: 'center',
        py: 4,
        px: 3,
        mb: 2.5,
        background: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)',
      }}
    >
      {/* Language Switcher */}
      <Box sx={{ position: 'absolute', top: 16, left: 16, display: 'flex', gap: 1 }}>
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
                minWidth: '50px',
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

      {/* Help Button */}
      {onHelpClick && (
        <IconButton
          onClick={onHelpClick}
          aria-label={t('help.title')}
          sx={{
            position: 'absolute',
            top: 16,
            right: 16,
            color: 'white',
            bgcolor: 'rgba(59, 130, 246, 0.2)',
            '&:hover': {
              bgcolor: 'rgba(59, 130, 246, 0.4)',
            },
          }}
        >
          <Typography fontSize="1.5rem">❓</Typography>
        </IconButton>
      )}

      <Typography variant="h1" sx={{ color: 'white', mb: 1.5 }}>
        {t('header.title')}
      </Typography>
      <Typography variant="body1" color="text.secondary">
        {t('header.subtitle')}
      </Typography>
    </Paper>
  );
}

export function Footer() {
  const { t } = useTranslation();
  return (
    <Paper sx={{ textAlign: 'center', py: 2.5, px: 3, mt: 2.5 }}>
      <Typography color="text.secondary" sx={{ mb: 1 }}>
        Architecture: <strong>3 inputs</strong> → <strong>5 neurons({t('layers.layer1Prefix')})</strong> →
        <strong> 3 neurons({t('layers.layer2Prefix')})</strong> → <strong>3 outputs</strong>
      </Typography>
      <Typography variant="body2" color="text.disabled" sx={{ maxWidth: 800, mx: 'auto' }}>
        {t('footer.description')}
      </Typography>
    </Paper>
  );
}
