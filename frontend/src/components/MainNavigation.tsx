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

import BrightnessAutoRoundedIcon from "@mui/icons-material/BrightnessAutoRounded";
import LogoutRoundedIcon from "@mui/icons-material/LogoutRounded";
import StorageIcon from "@mui/icons-material/Storage";
import ModelTrainingIcon from "@mui/icons-material/ModelTraining";

interface MainNavigationProps {
  activePage: string;
  onPageChange: (page: string) => void;
}

export default function MainNavigation({ activePage, onPageChange }: MainNavigationProps) {
  
  const navigationItems = [
    {
      key: 'dataManagement',
      label: 'Data Management',
      icon: <StorageIcon />,
      description: 'Manage database connections and query data'
    },
    {
      key: 'mlManagement', 
      label: 'Manage ML models',
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
        p: 2,
        flexShrink: 0,
        display: "flex",
        flexDirection: "column",
        gap: 2,
        borderRight: "1px solid",
        borderColor: "divider",
        bgcolor: "background.default",
      }}
    >
      <Box
        sx={{
          display: "flex",
          gap: 1,
          alignItems: "center",
        }}
      >
        <IconButton color="primary">
          <BrightnessAutoRoundedIcon />
        </IconButton>
        <Typography variant="h6">DataSmith</Typography>
      </Box>

      <Divider />

      <Box
        sx={{
          overflowY: "auto",
          flexGrow: 1,
          display: "flex",
          flexDirection: "column",
        }}
      >
        <Typography variant="subtitle2" color="textSecondary" sx={{ px: 2, py: 1 }}>
          Main Sections
        </Typography>
        
        <List>
          {navigationItems.map((item) => (
            <ListItem key={item.key} disablePadding>
              <ListItemButton
                selected={activePage === item.key}
                onClick={() => handleNavigationClick(item.key)}
                sx={{
                  borderRadius: 1,
                  mx: 1,
                  '&.Mui-selected': {
                    backgroundColor: 'primary.main',
                    color: 'white',
                    '&:hover': {
                      backgroundColor: 'primary.dark',
                    },
                    '& .MuiListItemIcon-root': {
                      color: 'white',
                    },
                  },
                }}
              >
                <ListItemIcon>
                  {item.icon}
                </ListItemIcon>
                <ListItemText 
                  primary={item.label}
                  secondary={activePage !== item.key ? item.description : undefined}
                  secondaryTypographyProps={{
                    variant: 'caption',
                    color: activePage === item.key ? 'inherit' : 'textSecondary'
                  }}
                />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Box>

      <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
        <Avatar
          src="https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?auto=format&fit=crop&w=286"
          alt="Avatar"
        />
        <Box>
          <Typography variant="body1">Siriwat K.</Typography>
          <Typography variant="body2" color="textSecondary">
            siriwatk@test.com
          </Typography>
        </Box>
        <IconButton>
          <LogoutRoundedIcon />
        </IconButton>
      </Box>
    </Box>
  );
}