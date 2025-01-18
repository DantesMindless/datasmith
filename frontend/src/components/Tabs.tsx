import React from "react";
import Box from "@mui/material/Box";
import Grid from "@mui/material/Grid2";
import { useAppContext } from "../providers/useAppContext";
import { pageComponents } from "../utils/constants";
import { Button, ButtonGroup } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";

const getDataSourceOptionsButtonStyle = (isActive: boolean) => ({
  ...(isActive
    ? { color: "text.primary", bgcolor: "orange" }
    : { color: "text.secondary", bgcolor: "grey.200" }),
});

const getTabButtonStyle = (isActive: boolean) => ({
  ...(isActive
    ? { color: "text.primary", bgcolor: "primary.main" }
    : { color: "text.secondary", bgcolor: "secondary.main" }),
});

export default function TabsSegmentedControls() {
  const { tabs, removeTab, updateActiveTab, setActivePage, activeTab, setActiveTab, activePage } = useAppContext();

  return (
    <Box component={"div"} sx={{ bgcolor: "background.default" }}>
      <Grid container spacing={0.5}>
        {Object.keys(pageComponents).filter((key) => !pageComponents[key].skip).map((key, index) =>
        (
          <Grid key={index}>
            <Box
            >
              <Button
                size="small"
                sx={getDataSourceOptionsButtonStyle(key === activePage)}
                onClick={() => {setActiveTab(null); setActivePage(key)}}
              >
                {pageComponents[key].name}
              </Button>
            </Box>
          </Grid>
        )
        )}
      {tabs &&
          tabs.map((tab, index) => (
            <Grid key={index}>
            <Box
              sx={{
                color: "text.primary",
              }}
            >
              <ButtonGroup variant="outlined" aria-label="Basic button group">
              <Button
              size="small"
              sx={getTabButtonStyle(activeTab === index)}
              onClick={() => {
                updateActiveTab(index)
                setActivePage("queryTab")
              }}
              >
                {tab.name}
              </Button>
              <Button sx={{...getTabButtonStyle(activeTab === index), with: 2, px:1}} aria-label="clise-tab" size="small"
                onClick={() => removeTab(index)}
                >
                <CloseIcon fontSize="inherit" />
              </Button>
                </ButtonGroup>
            </Box>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}
