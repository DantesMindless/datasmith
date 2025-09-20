import * as React from "react";
import { useState } from "react";

import Avatar from "@mui/material/Avatar";
import Box from "@mui/material/Box";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemText from "@mui/material/ListItemText";
import ListItemIcon from "@mui/material/ListItemIcon";
import Typography from "@mui/material/Typography";
import Tooltip from "@mui/material/Tooltip";
import Badge from "@mui/material/Badge";

import BrightnessAutoRoundedIcon from "@mui/icons-material/BrightnessAutoRounded";
import LogoutRoundedIcon from "@mui/icons-material/LogoutRounded";
import StorageIcon from "@mui/icons-material/Storage";
import ModelTrainingIcon from "@mui/icons-material/ModelTraining";
import DataObjectIcon from "@mui/icons-material/DataObject";
import AnalyticsIcon from "@mui/icons-material/Analytics";

import { useAppContext } from "../providers/useAppContext";

interface MainNavigationProps {
  activePage: string;
  onPageChange: (page: string) => void;
}

export default function MainNavigation({ activePage, onPageChange }: MainNavigationProps) {
  const { user, logout } = useAppContext();
  
  const navigationItems = [
    {
      key: 'dataManagement',
      label: 'Data Management',
      icon: <StorageIcon />,
      description: 'Manage database connections and query data'
    },
    {
      key: 'mlManagement',
      label: 'ML Management',
      icon: <ModelTrainingIcon />,
      description: 'Create and train ML models'
    }
  ];

  const handleNavigationClick = (pageKey: string) => {
    onPageChange(pageKey);
  };

  return (
    <Box
      sx={{
        position: { xs: "fixed", md: "sticky" },
        zIndex: 10000,
        height: "100vh",
        width: "280px",
        top: 0,
        flexShrink: 0,
        display: "flex",
        flexDirection: "column",
        bgcolor: "background.paper",
        borderRight: "1px solid",
        borderColor: "divider",
        backdropFilter: "blur(10px)",
        boxShadow: { xs: 4, md: 'none' },
      }}
    >
      {/* Brand Header */}
      <Box
        sx={{
          p: 3,
          display: "flex",
          alignItems: "center",
          gap: 2,
          minHeight: 72,
        }}
      >
        <Box
          sx={{
            width: 40,
            height: 40,
            borderRadius: 2,
            bgcolor: "primary.main",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: 2,
          }}
        >
          <DataObjectIcon sx={{ color: "white", fontSize: 24 }} />
        </Box>
        <Box>
          <Typography variant="h6" fontWeight={700} color="text.primary">
            DataSmith
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Data Analytics Platform
          </Typography>
        </Box>
      </Box>

      <Divider />

      {/* Navigation Items */}
      <Box
        sx={{
          flex: 1,
          overflowY: "auto",
          px: 2,
          py: 3,
        }}
      >
        <Typography 
          variant="caption" 
          color="text.secondary" 
          sx={{ 
            px: 2, 
            mb: 1.5, 
            display: "block",
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: "0.1em"
          }}
        >
          Workspace
        </Typography>
        
        <List sx={{ p: 0 }}>
          {navigationItems.map((item) => (
            <ListItem key={item.key} disablePadding sx={{ mb: 0.5 }}>
              <ListItemButton
                selected={activePage === item.key}
                onClick={() => handleNavigationClick(item.key)}
                sx={{
                  borderRadius: 2,
                  py: 1.5,
                  px: 2,
                  transition: "all 0.2s ease-in-out",
                  '&.Mui-selected': {
                    bgcolor: 'primary.main',
                    color: 'white',
                    boxShadow: '0 4px 12px rgba(37, 99, 235, 0.3)',
                    '&:hover': {
                      bgcolor: 'primary.dark',
                    },
                    '& .MuiListItemIcon-root': {
                      color: 'white',
                    },
                  },
                  '&:not(.Mui-selected):hover': {
                    bgcolor: 'action.hover',
                    transform: 'translateX(4px)',
                  },
                }}
              >
                <ListItemIcon sx={{ minWidth: 40 }}>
                  {item.icon}
                </ListItemIcon>
                <ListItemText 
                  primary={
                    <Typography variant="body2" fontWeight={500}>
                      {item.label}
                    </Typography>
                  }
                  secondary={
                    !activePage || activePage !== item.key ? (
                      <Typography 
                        variant="caption" 
                        color={activePage === item.key ? 'inherit' : 'text.secondary'}
                        sx={{ opacity: 0.8 }}
                      >
                        {item.description}
                      </Typography>
                    ) : undefined
                  }
                />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Box>

      {/* User Profile Section */}
      <Box sx={{ p: 2.5, borderTop: "1px solid", borderColor: "divider" }}>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 2,
            p: 1.5,
            borderRadius: 2,
            bgcolor: "background.default",
          }}
        >
          <Avatar 
            sx={{
              width: 36,
              height: 36,
              bgcolor: "primary.main",
              fontSize: "0.875rem",
              fontWeight: 600,
            }}
          >
            {user?.username?.charAt(0).toUpperCase() || 'U'}
          </Avatar>
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Typography variant="body2" fontWeight={500} noWrap>
              {user?.username || 'User'}
            </Typography>
            <Typography variant="caption" color="text.secondary" noWrap>
              {user?.email || 'user@example.com'}
            </Typography>
          </Box>
          <Tooltip title="Sign out" arrow>
            <IconButton 
              onClick={logout}
              size="small"
              sx={{
                color: "text.secondary",
                "&:hover": {
                  color: "error.main",
                  bgcolor: "error.lighter",
                },
              }}
            >
              <LogoutRoundedIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
    </Box>
  );
}