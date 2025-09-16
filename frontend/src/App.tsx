import Box from "@mui/material/Box";
import MainNavigation from "./components/MainNavigation";
import DataManagementPage from "./components/pages/DataManagementPage";
import MLManagementPage from "./components/pages/MLManagementPage";
import ModelAnalysisPage from "./components/pages/ModelAnalysisPage";
import Login from "./components/Login";
import { useState } from "react";
import { useAppContext } from "./providers/useAppContext";
import { Fade } from "@mui/material";
import "./App.css";

export default function DataSmithApp() {
  const [currentPage, setCurrentPage] = useState('dataManagement');
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);
  const { isAuthenticated } = useAppContext();

  const handlePageChange = (page: string) => {
    setCurrentPage(page);
    if (page !== 'modelAnalysis') {
      setSelectedModelId(null);
    }
  };

  const handleNavigateToAnalysis = (modelId: number) => {
    setSelectedModelId(modelId);
    setCurrentPage('modelAnalysis');
  };

  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'dataManagement':
        return <DataManagementPage />;
      case 'mlManagement':
        return <MLManagementPage onNavigateToAnalysis={handleNavigateToAnalysis} />;
      case 'modelAnalysis':
        return (
          <ModelAnalysisPage
            modelId={selectedModelId}
            onBack={() => handlePageChange('mlManagement')}
          />
        );
      default:
        return <DataManagementPage />;
    }
  };

  if (!isAuthenticated) {
    return (
      <Fade in timeout={500}>
        <Box className="app-container">
          <Login />
        </Box>
      </Fade>
    );
  }

  return (
    <Box className="app-container">
      <Fade in timeout={300}>
        <Box sx={{ display: "flex", minHeight: "100vh", width: '100vw', bgcolor: 'background.default' }}>
          <MainNavigation 
            activePage={currentPage} 
            onPageChange={handlePageChange} 
          />
          <Box
            component="main"
            className="main-content fade-in"
            sx={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              minWidth: 0,
              height: "100vh",
              overflow: "hidden",
              bgcolor: "background.default",
            }}
          >
            <Fade 
              in 
              timeout={400}
              key={currentPage} // This ensures fade transition when switching pages
            >
              <Box sx={{ width: '100%', height: '100%', overflow: 'auto' }}>
                {renderCurrentPage()}
              </Box>
            </Fade>
          </Box>
        </Box>
      </Fade>
    </Box>
  );
}
