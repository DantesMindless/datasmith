import React from "react";
import Box from "@mui/material/Box";
import Grid from "@mui/material/Grid2";
import { useAppContext } from "../providers/useAppContext";
import { pageComponents } from "../utils/constants";
import { Button, ButtonGroup } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";

const tabButtonStyle = {
  color: "text.primary",
  bgcolor: "background.paper",
  height: "36px",
  fontSize: "0.875rem",
  fontWeight: 500,
  textTransform: "none",
  borderRadius: "8px",
  border: "1px solid",
  borderColor: "divider",
  px: 2,
  py: 1,
  minHeight: "auto",
  '&:hover': {
    bgcolor: "secondary.light",
    borderColor: "secondary.main",
    boxShadow: "0 2px 4px -1px rgba(0, 0, 0, 0.06)"
  }
}

export default function TabsSegmentedControls() {
  const { tabs, removeTab, updateActiveTab, setActivePage } = useAppContext();

  return (
    <Box
      component={"div"}
      sx={{
        bgcolor: "background.default",
        p: 2,
        borderBottom: "1px solid",
        borderColor: "divider",
        mb: 2
      }}
    >
      <Grid container spacing={1}>
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
              <ButtonGroup
                variant="outlined"
                aria-label="Query tab actions"
                sx={{
                  borderRadius: "8px",
                  overflow: "hidden",
                  '& .MuiButton-root': {
                    borderColor: "divider"
                  }
                }}
              >
                <Button
                  size="small"
                  sx={{
                    ...tabButtonStyle,
                    borderRadius: "0",
                    border: "none"
                  }}
                  onClick={() => {
                    updateActiveTab(index)
                    setActivePage("queryTab")
                  }}
                >
                  {tab.name}
                </Button>
                <Button
                  sx={{
                    ...tabButtonStyle,
                    width: "32px",
                    minWidth: "32px",
                    px: 1,
                    borderRadius: "0",
                    border: "none",
                    borderLeft: "1px solid",
                    borderColor: "divider"
                  }}
                  aria-label="Close tab"
                  size="small"
                  onClick={() => removeTab(index)}
                >
                  <CloseIcon fontSize="small" />
                </Button>
              </ButtonGroup>
            </Box>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}
