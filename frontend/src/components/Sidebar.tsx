import * as React from "react";
import { useEffect, useState } from 'react';

import Avatar from "@mui/joy/Avatar";
import Box from "@mui/joy/Box";
import Divider from "@mui/joy/Divider";
import GlobalStyles from "@mui/joy/GlobalStyles";
import IconButton from "@mui/joy/IconButton";
import List from "@mui/joy/List";
import ListItem from "@mui/joy/ListItem";
import ListItemButton, { listItemButtonClasses } from "@mui/joy/ListItemButton";
import ListItemContent from "@mui/joy/ListItemContent";
import Sheet from "@mui/joy/Sheet";
import Typography from "@mui/joy/Typography";

import BrightnessAutoRoundedIcon from "@mui/icons-material/BrightnessAutoRounded";
import LogoutRoundedIcon from "@mui/icons-material/LogoutRounded";

import { useAppContext } from '../providers/useAppContext';
import ColorSchemeToggle from "./ColorSchemeToggle";
import { closeSidebar } from "../utils";
import { pageComponents } from "../utils/constants";
import Link from "@mui/joy/Link";
import { getDatabasesList, getDatabaseTablesList } from "../utils/requests";

function Toggler({
  defaultExpanded = false,
  renderToggle,
  children,
}: {
  defaultExpanded?: boolean;
  children: React.ReactNode;
  renderToggle: (params: {
    open: boolean;
    setOpen: React.Dispatch<React.SetStateAction<boolean>>;
  }) => React.ReactNode;
}) {
  const [open, setOpen] = React.useState(defaultExpanded);
  return (
    <React.Fragment>
      {renderToggle({ open, setOpen })}
      <Box
        sx={[
          {
            display: "grid",
            transition: "0.2s ease",
            "& > *": {
              overflow: "hidden",
            },
          },
          open ? { gridTemplateRows: "1fr" } : { gridTemplateRows: "0fr" },
        ]}
      >
        {children}
      </Box>
    </React.Fragment>
  );
}

export default function Sidebar({activePageComponent, setPageComponent }) {

  const { connections, updateConnections, activeConnection, setActiveConnection } = useAppContext();
  const [ tablesList, setTablesList ] = useState([]);
  const [ activeDatabase, setActiveDatabase ] = useState('');
  const [ databasesList, setDatabasesList ] = useState([]);

  useEffect(()=>{
    if (connections === null){
      updateConnections();
    }
  }, [connections, updateConnections])

  useEffect(()=>{
    if(activeConnection != null){
      (async () =>{
        const databases = await getDatabasesList(activeConnection)
        if (databases.length > 0){
          setDatabasesList(databases)
        }
      })()
    }
  }, [activeConnection, setDatabasesList])

  useEffect(() => {
    if(activeConnection != null && activeDatabase != ''){
        (async () =>{
          const databases = await getDatabaseTablesList(activeConnection, activeDatabase)
          if (databases.length > 0){
            setTablesList(databases)
          }
        })()
    }},[activeDatabase, activeConnection]
  )

  function RenderTables(){
    return (
      <Box>
      <Divider/>
        <h4>Active Connection</h4>
        <List>
        {tablesList.map((row)=>
          <ListItem >
              {row.table_name}
            </ListItem>
        )}
        </List>
        <Divider/>
      </Box>
    )
  }

  function RenderDatabases(){
    if (databasesList.length > 0){
    return (
      <Box>
      <Divider/>
        <h4>Active Connection</h4>
        <List>
        {databasesList.map((row)=>
          <ListItem >
            <Link overlay onClick={ () => setActiveDatabase(row)}>
              {row}
            </Link>
            </ListItem>
        )}
        </List>
        <Divider/>
      </Box>
    )
  }
  }

   return (
    <Sheet
      className="Sidebar"
      sx={{
        position: { xs: "fixed", md: "sticky" },
        transform: {
          xs: "translateX(calc(100% * (var(--SideNavigation-slideIn, 0) - 1)))",
          md: "none",
        },
        transition: "transform 0.4s, width 0.4s",
        zIndex: 10000,
        height: "100dvh",
        width: "var(--Sidebar-width)",
        top: 0,
        p: 2,
        flexShrink: 0,
        display: "flex",
        flexDirection: "column",
        gap: 2,
        borderRight: "1px solid",
        borderColor: "divider",
      }}
    >
      <GlobalStyles
        styles={(theme) => ({
          ":root": {
            "--Sidebar-width": "220px",
            [theme.breakpoints.up("lg")]: {
              "--Sidebar-width": "240px",
            },
          },
        })}
      />

      <Box
        className="Sidebar-overlay"
        sx={{
          position: "fixed",
          zIndex: 9998,
          top: 0,
          left: 0,
          width: "100vw",
          height: "100vh",
          opacity: "var(--SideNavigation-slideIn)",
          backgroundColor: "var(--joy-palette-background-backdrop)",
          transition: "opacity 0.4s",
          transform: {
            xs: "translateX(calc(100% * (var(--SideNavigation-slideIn, 0) - 1) + var(--SideNavigation-slideIn, 0) * var(--Sidebar-width, 0px)))",
            lg: "translateX(-100%)",
          },
        }}
        onClick={() => closeSidebar()}
      />

      <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
        <IconButton variant="soft" color="primary" size="sm">
          <BrightnessAutoRoundedIcon />
        </IconButton>
        <Typography level="title-lg">DataSmith</Typography>
        <ColorSchemeToggle sx={{ ml: "auto" }} />
      </Box>
      <Box
        sx={{
          minHeight: 0,
          overflow: "hidden auto",
          flexGrow: 1,
          display: "flex",
          flexDirection: "column",
          [`& .${listItemButtonClasses.root}`]: {
            gap: 1.5,
          },
        }}
      >
        <List
          size="sm"
          sx={{
            gap: 1,
            "--List-nestedInsetStart": "30px",
            "--ListItem-radius": (theme) => theme.vars.radius.sm,
          }}
        >
          <Divider/>
          {connections ? connections.map((row)=> (
          <ListItem>
            <Link overlay onClick={() => {setActiveConnection(row.id)}} underline="none">
              { row.name }
            </Link>
          </ListItem>
          )) : ""}
          <Divider/>
          {RenderDatabases()}
          <h4>Tables list for active database {activeDatabase} </h4>
          {RenderTables()}
          {Object.keys(pageComponents).map((key, index) => (
            <ListItem key={index}>
              <ListItemButton
                onClick={() => setPageComponent(pageComponents[key])}
              >
                <ListItemContent>
                  <Typography level="title-sm">
                    {pageComponents[key].name}
                  </Typography>
                </ListItemContent>
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Box>
      <Divider />
      <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
        <Avatar
          variant="outlined"
          size="sm"
          src="https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?auto=format&fit=crop&w=286"
        />
        <Box sx={{ minWidth: 0, flex: 1 }}>
          <Typography level="title-sm">Siriwat K.</Typography>
          <Typography level="body-xs">siriwatk@test.com</Typography>
        </Box>
        <IconButton size="sm" variant="plain" color="neutral">
          <LogoutRoundedIcon />
        </IconButton>
      </Box>
    </Sheet>
  );
}
