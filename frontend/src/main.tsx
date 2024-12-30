import App from "./App.tsx";
import AlertBanner from "./components/helpers/AlertBanner.tsx";
import { ContextProvider } from "./providers/Providers.tsx";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ContextProvider>
      <AlertBanner/>
      <App />
    </ContextProvider>
  </StrictMode>
);
