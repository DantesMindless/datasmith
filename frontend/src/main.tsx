import App from "./App.tsx";
import AlertBanner from "./components/helpers/AlertBanner.tsx";
import { ContextProvider } from "./providers/Providers.tsx";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import CssBaseline from '@mui/material/CssBaseline';
import { ThemeProvider } from '@mui/material/styles';
import theme from './theme.tsx';

const rootElement = document.getElementById('root') as HTMLElement;
const root = createRoot(rootElement);

root.render(
  <StrictMode>
    <ContextProvider>
      <AlertBanner />
      <ThemeProvider theme={theme()}>
        <CssBaseline />
        <App />
      </ThemeProvider>
    </ContextProvider>
  </StrictMode>,
);
