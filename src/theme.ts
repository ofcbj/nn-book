import { createTheme } from '@mui/material/styles';

// Dark theme matching current design
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#3b82f6',
      light: '#60a5fa',
      dark: '#2563eb',
    },
    secondary: {
      main: '#22c55e',
      light: '#34d399',
      dark: '#16a34a',
    },
    warning: {
      main: '#f97316',
      light: '#fb923c',
      dark: '#ea580c',
    },
    error: {
      main: '#ef4444',
      light: '#f87171',
      dark: '#dc2626',
    },
    background: {
      default: '#0f0f1e',
      paper: '#1e293b',
    },
    text: {
      primary: '#e0e0e0',
      secondary: '#94a3b8',
    },
  },
  typography: {
    fontFamily: "'Inter', 'Noto Sans KR', sans-serif",
    h1: {
      fontSize: '2.2rem',
      fontWeight: 700,
    },
    h2: {
      fontSize: '1.2rem',
      fontWeight: 600,
    },
    h3: {
      fontSize: '1.1rem',
      fontWeight: 600,
    },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          border: '1px solid #334155',
          borderRadius: 12,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 600,
          padding: '12px 20px',
        },
      },
    },
    MuiSlider: {
      styleOverrides: {
        root: {
          height: 6,
        },
        thumb: {
          width: 18,
          height: 18,
        },
        track: {
          borderRadius: 3,
        },
        rail: {
          borderRadius: 3,
          backgroundColor: '#334155',
        },
      },
    },
  },
});

export default theme;
