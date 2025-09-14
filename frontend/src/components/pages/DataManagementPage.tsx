import React from 'react';
import { Box, Container, Typography, Fade } from '@mui/material';
import Sidebar from '../Sidebar';
import Tabs from '../Tabs';
import { useAppContext } from '../../providers/useAppContext';
import { pageComponents } from '../../utils/constants';

export default function DataManagementPage() {
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
    <Container maxWidth="xl" sx={{ py: 3, height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Page Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" fontWeight={700} gutterBottom>
          Data Management
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Connect to databases, explore data, and manage your data sources
        </Typography>
      </Box>

      {/* Content Area */}
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <Tabs />
        
        <Box sx={{ flex: 1, overflowY: 'auto' }}>
          <Fade in timeout={300}>
            <Box>
              {renderActivePage()}
            </Box>
          </Fade>
        </Box>
      </Box>
    </Container>
  );
}