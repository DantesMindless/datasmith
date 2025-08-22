import Box from "@mui/material/Box";
import MainNavigation from "./components/MainNavigation";
import DataManagementPage from "./components/pages/DataManagementPage";
import MLManagementPage from "./components/pages/MLManagementPage";
import Login from "./components/Login";
import { useState } from "react";
import { useAppContext } from "./providers/useAppContext";

export default function JoyOrderDashboardTemplate() {
  const [currentPage, setCurrentPage] = useState('dataManagement');
  const { isAuthenticated } = useAppContext();

  const handlePageChange = (page: string) => {
    setCurrentPage(page);
  };

  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'dataManagement':
        return <DataManagementPage />;
      case 'mlManagement':
        return <MLManagementPage />;
      default:
        return <DataManagementPage />;
    }
  };

  if (!isAuthenticated) {
    return <Login />;
  }

  return (
    <Box sx={{ display: "flex", minHeight: "100dvh", width: '100dvw'}}>
      <MainNavigation 
        activePage={currentPage} 
        onPageChange={handlePageChange} 
      />
      <Box
        component="main"
        className="MainContent"
        sx={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          minWidth: 0,
          height: "100dvh",
        }}
      >
        {renderCurrentPage()}
      </Box>
    </Box>
  );
}
