import React from "react";
import Box from "@mui/material/Box";
import Grid from "@mui/material/Grid2";
import { useAppContext } from "../providers/useAppContext";
import { pageComponents } from "../utils/constants";
import { Button, ButtonGroup } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";

const tabButtonStyle = {color: "text.primary", bgcolor: "secondary.main",}

export default function TabsSegmentedControls() {
  const { tabs, removeTab, updateActiveTab, setActivePage } = useAppContext();

  return (
    <Box component={"div"} sx={{ width: "100%", bgcolor: "background.default" }}>
      <Grid container spacing={0.5}>
        {Object.keys(pageComponents).filter((key) => !pageComponents[key].skip).map((key, index) =>
        (
          <Grid key={index}>
            <Box
            >
              <Button
                size="small"
                sx={tabButtonStyle}
                onClick={() => setActivePage(key)}
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
              sx={tabButtonStyle}
              onClick={() => {
                updateActiveTab(index)
                setActivePage("queryTab")
              }}
              >
                {tab.name}
              </Button>
              <Button sx={{...tabButtonStyle, with: 2, px:1}} aria-label="clise-tab" size="small"
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
