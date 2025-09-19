import { createTheme } from '@mui/material/styles';

// Modern, professional color palette
const theme = (mode = 'light') =>
  createTheme({
    cssVariables: true,
    palette: {
      mode,
      primary: {
        main: mode === 'light' ? '#2563eb' : '#3b82f6', // Modern blue
        light: mode === 'light' ? '#60a5fa' : '#7c3aed',
        dark: mode === 'light' ? '#1d4ed8' : '#1e40af',
        contrastText: '#ffffff',
      },
      secondary: {
        main: mode === 'light' ? '#64748b' : '#94a3b8', // Slate gray
        light: mode === 'light' ? '#94a3b8' : '#cbd5e1',
        dark: mode === 'light' ? '#475569' : '#334155',
        contrastText: mode === 'light' ? '#ffffff' : '#0f172a',
      },
      background: {
        default: mode === 'light' ? '#f8fafc' : '#0f172a', // Light slate for light mode, dark slate for dark mode
        paper: mode === 'light' ? '#ffffff' : '#1e293b', // Pure white for light mode, dark slate for dark mode
      },
      surface: {
        main: mode === 'light' ? '#f1f5f9' : '#334155',
        contrastText: mode === 'light' ? '#334155' : '#f1f5f9',
      },
      text: {
        primary: mode === 'light' ? '#0f172a' : '#f8fafc', // Almost black for light mode, almost white for dark mode
        secondary: mode === 'light' ? '#64748b' : '#94a3b8', // Muted slate gray
      },
      divider: mode === 'light' ? '#e2e8f0' : '#334155',
      error: {
        main: '#ef4444',
        light: '#f87171',
        dark: '#dc2626',
      },
      warning: {
        main: '#f59e0b',
        light: '#fbbf24',
        dark: '#d97706',
      },
      success: {
        main: '#10b981',
        light: '#34d399',
        dark: '#059669',
      },
    },
    typography: {
      fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
      fontWeightLight: 300,
      fontWeightRegular: 400,
      fontWeightMedium: 500,
      fontWeightBold: 600,
      h1: {
        fontWeight: 700,
        fontSize: '2.5rem',
        lineHeight: 1.2,
        letterSpacing: '-0.025em',
      },
      h2: {
        fontWeight: 600,
        fontSize: '2rem',
        lineHeight: 1.3,
        letterSpacing: '-0.025em',
      },
      h3: {
        fontWeight: 600,
        fontSize: '1.5rem',
        lineHeight: 1.4,
        letterSpacing: '-0.025em',
      },
      h4: {
        fontWeight: 600,
        fontSize: '1.25rem',
        lineHeight: 1.4,
      },
      h5: {
        fontWeight: 600,
        fontSize: '1.125rem',
        lineHeight: 1.5,
      },
      h6: {
        fontWeight: 600,
        fontSize: '1rem',
        lineHeight: 1.5,
      },
      body1: {
        fontSize: '1rem',
        lineHeight: 1.6,
      },
      body2: {
        fontSize: '0.875rem',
        lineHeight: 1.6,
      },
      button: {
        fontWeight: 500,
        textTransform: 'none' as const,
        letterSpacing: '0.025em',
      },
    },
    shape: {
      borderRadius: 8,
    },
    shadows: mode === 'light' 
      ? [
          'none',
          '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
          '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
          '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
          '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
          '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
        ]
      : [
          'none',
          '0 1px 2px 0 rgba(0, 0, 0, 0.3)',
          '0 1px 3px 0 rgba(0, 0, 0, 0.4), 0 1px 2px 0 rgba(0, 0, 0, 0.25)',
          '0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -1px rgba(0, 0, 0, 0.25)',
          '0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -2px rgba(0, 0, 0, 0.25)',
          '0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.2)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.6)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.6)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.6)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.6)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.6)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.6)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.6)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.6)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.6)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.6)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.6)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.6)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.6)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.6)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.6)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.6)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.6)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.6)',
          '0 25px 50px -12px rgba(0, 0, 0, 0.6)',
        ],
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            padding: '10px 20px',
            fontSize: '0.875rem',
            fontWeight: 500,
            textTransform: 'none',
            boxShadow: 'none',
            '&:hover': {
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            },
          },
          contained: {
            '&:hover': {
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            },
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 12,
            border: mode === 'light' ? '1px solid #e2e8f0' : '1px solid #334155',
            boxShadow: mode === 'light' 
              ? '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)'
              : '0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -1px rgba(0, 0, 0, 0.25)',
          },
        },
      },
      MuiTextField: {
        styleOverrides: {
          root: {
            '& .MuiOutlinedInput-root': {
              borderRadius: 8,
            },
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: 'none',
          },
        },
      },
      MuiAppBar: {
        styleOverrides: {
          root: {
            boxShadow: 'none',
            borderBottom: mode === 'light' ? '1px solid #e2e8f0' : '1px solid #334155',
          },
        },
      },
      MuiListItemButton: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            margin: '2px 8px',
            '&.Mui-selected': {
              backgroundColor: mode === 'light' ? '#2563eb' : '#3b82f6',
              '&:hover': {
                backgroundColor: mode === 'light' ? '#1d4ed8' : '#2563eb',
              },
            },
          },
        },
      },
    },
  });

export default theme;
