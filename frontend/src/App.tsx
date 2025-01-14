import Box from "@mui/material/Box";
import Sidebar from "./components/Sidebar";
import Tabs from "./components/Tabs";
import { useAppContext } from "./providers/useAppContext";

import { pageComponents } from "./utils/constants";

export default function JoyOrderDashboardTemplate() {
  const { activePage } = useAppContext();

  const renderActivePage = () => {
    const activePageConfig = pageComponents[activePage];
    if (activePageConfig && activePageConfig.component) {
      const ActiveComponent = activePageConfig.component;
      return <ActiveComponent />;
    }
    return null;
  };

  return (
    <Box sx={{ display: "flex", minHeight: "100dvh", width: '100dvw'}}>
      {/* <Header /> */}
      <Sidebar />
      <Box
        component="main"
        className="MainContent"
        sx={{
          // px: { xs: 1, md: 1 },
          pt: {
            xs: "calc(4px + var(--Header-height))",
            sm: "calc(4px + var(--Header-height))",
            md: 1,
          },
          pb: { xs: 1, sm: 1, md: 1 },
          flex: 1,
          display: "flex",
          flexDirection: "column",
          minWidth: 0,
          height: "100dvh",
          gap: 1,
          overflowY: 'scroll'
        }}
      >
        <Tabs />
        {renderActivePage()}
      </Box>
    </Box>
  );
}
