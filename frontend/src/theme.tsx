import { createTheme } from '@mui/material/styles';
import { blue, grey, brown } from '@mui/material/colors';

// Create a theme instance with light and dark modes
const theme = (mode = 'light') =>
  createTheme({
    cssVariables: true,
    palette: {
      // mode, // 'light' or 'dark'
      primary: {
        main: blue[500], // Vibrant blue
        contrastText: '#ffffff', // White text for contrast
      },
      secondary: {
        main: grey[500], // Neutral grey for secondary elements
        contrastText: '#000000', // Black text for contrast
      },
      background: {
        default: mode === 'light' ? '#ffffff' : brown[200], // White for light mode, beige for dark mode
        paper: mode === 'light' ? '#f5f5dc' : brown[300], // Beige for paper in light mode, darker beige in dark mode
      },
      text: {
        primary: mode === 'light' ? '#1a237e' : '#ffffff', // Deep blue for light mode, white for dark mode
        secondary: mode === 'light' ? '#283593' : '#cccccc', // Muted blue for light mode, grey for dark mode
      },
      error: {
        main: '#f44336', // Default red for errors
      },
    },
    typography: {
      fontFamily: `'Roboto', 'Arial', sans-serif`,
    },
  });

export default theme;
