import React from "react";
import Box from "@mui/material/Box";
import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import Tabs from "./components/Tabs";
import { useAppContext } from "./providers/useAppContext";

import { pageComponents } from "./utils/constants";

export default function JoyOrderDashboardTemplate() {
  const { activeTab, activePage } = useAppContext();

  const renderActivePage = () => {
    const activePageConfig = pageComponents[activePage];
    if (activePageConfig && activePageConfig.component) {
      const ActiveComponent = activePageConfig.component;
      return <ActiveComponent />;
    }
    return null;
  };

  return (
    <Box sx={{ display: "flex", minHeight: "100dvh" }}>
      {/* <Header /> */}
      <Sidebar />
      <Box
        component="main"
        className="MainContent"
        sx={{
          px: { xs: 2, md: 6 },
          pt: {
            xs: "calc(12px + var(--Header-height))",
            sm: "calc(12px + var(--Header-height))",
            md: 3,
          },
          pb: { xs: 2, sm: 2, md: 3 },
          flex: 1,
          display: "flex",
          flexDirection: "column",
          minWidth: 0,
          height: "100dvh",
          gap: 1,
        }}
      >
        <Tabs />
        {renderActivePage()}
      </Box>
    </Box>
  );
}
