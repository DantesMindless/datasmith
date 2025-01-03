import React from 'react';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import { useAppContext } from '../providers/useAppContext';
import { pageComponents } from "../utils/constants";

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

export default function TabsSegmentedControls() {
  const { tabs } = useAppContext();
  const [value, setValue] = React.useState(0);

  const handleChange = (event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
  };

  return (
    <Box component={'div'} sx={{ width: '100%', bgcolor: 'background.paper' }}>
      <Tabs
        value={value}
        onChange={handleChange}
        aria-label="tabs"
        variant="scrollable"
        scrollButtons="auto"
        sx={{ minWidth: '100%' }}
      >
        {Object.keys(pageComponents).map((key, index) => (
          <Tab key={index} label={pageComponents[key].name} />
        ))}
        {tabs &&
          tabs.map((tab, index) => (
            <Tab key={`dynamic-${index}`} label="Specifications" />
          ))}
      </Tabs>
      {Object.keys(pageComponents).map((key, index) => (
        <TabPanel key={index} value={value} index={index}>
          <div>{pageComponents ? React.createElement(pageComponents[key].component) : null}</div>
        </TabPanel>
      ))}
    </Box>
  );
}
