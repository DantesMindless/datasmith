import * as React from "react";
import { useEffect, useState } from "react";

import Avatar from "@mui/material/Avatar";
import Box from "@mui/material/Box";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemText from "@mui/material/ListItemText";
import Typography from "@mui/material/Typography";

import BrightnessAutoRoundedIcon from "@mui/icons-material/BrightnessAutoRounded";
import LogoutRoundedIcon from "@mui/icons-material/LogoutRounded";
import PowerIcon from "@mui/icons-material/Power";
import MemoryIcon from "@mui/icons-material/Memory";
import Switch from "@mui/material/Switch";
import Tooltip from "@mui/material/Tooltip";

import { useAppContext } from "../providers/useAppContext";
import { getDatabasesList, getSchemaTablesList } from "../utils/requests";

import AddBoxIcon from '@mui/icons-material/AddBox';
import IndeterminateCheckBoxIcon from '@mui/icons-material/IndeterminateCheckBox';
import { SimpleTreeView } from '@mui/x-tree-view/SimpleTreeView';
import { TreeItem } from '@mui/x-tree-view/TreeItem';
import { Connection } from "../providers/constants";
import TableRowsIcon from '@mui/icons-material/TableRows';

interface ActiveConnection extends Connection {
  id: string;
  schemas: Record<string, string[]>;
}


export default function Sidebar() {
  const { connections, activeConnections, updateActiveConnections, updateConnections, addTableViewTab, cudaEnabled, setCudaEnabled, user, logout } = useAppContext();
  const [expandedItems, setExpandedItems] = useState([])

  useEffect(() => {
    if (connections === null) {
      updateConnections();
    }
  }, [connections, updateConnections]);

  const handleItemClick = (event, item) => {
    if(!expandedItems.includes(item)){
      expandedItems.push(item)
      setExpandedItems([...expandedItems])
    }else{
      setExpandedItems(expandedItems.filter((element)=>element != item))
    }
  }

  async function addActiveConnection(connection : Connection) {
    const databasesList = await getDatabasesList(connection.id);
    const connectionCopy: ActiveConnection = {
      ...connection,
      schemas: {}
    }
    connectionCopy["schemas"] = databasesList.reduce((acc: Record<string, []>, value: string) => {
      acc[value] = [];
      return acc;
    }, {});
    activeConnections[connection.name] = connectionCopy;
    updateActiveConnections(activeConnections);

  }

  async function addSchemaTables(connection: Connection, schema: string) {
    if (schema in activeConnections[connection.name].schemas && activeConnections[connection.name].schemas[schema].length === 0){
      const schemaTables = await getSchemaTablesList(connection.id, schema);
      activeConnections[connection.name].schemas[schema] = schemaTables;
      updateActiveConnections(activeConnections);
    }
  }

  function RenderSchemas() {
    if (activeConnections) {
      return (
        <SimpleTreeView
        // checkboxSelection
          expandedItems={expandedItems}
          onItemClick = {handleItemClick}
          slots={{
            expandIcon: AddBoxIcon,
            collapseIcon: IndeterminateCheckBoxIcon,
            endIcon: TableRowsIcon,
          }}
        >
          {Object.keys(activeConnections).map((key) =>
            Object.keys(activeConnections[key].schemas).map((schema) => (
              <TreeItem
                key={`schema_${key}_${schema}`}
                onClick={() => addSchemaTables(activeConnections[key], schema)}
                itemId={`id_${key}_${schema}`}
                label={schema}
              >
                {activeConnections[key].schemas[schema].map((row) => (
                  <TreeItem
                    onClick={()=>{addTableViewTab(activeConnections[key], schema, row.table_name)}}
                    key={`table_${key}_${schema}_${row.table_name}`}
                    itemId={`id_${key}_${schema}_${row.table_name}`}
                    label={row.table_name}
                  />
                ))}
              </TreeItem>
            ))
          )}
        </SimpleTreeView>
      );
    } else {
      return null;
    }
  }

  return (
    <Box
      sx={{
        position: { xs: "fixed", md: "sticky" },
        zIndex: 10000,
        height: "100vh",
        width: "240px",
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

      <Box
        sx={{
          overflowY: "auto",
          flexGrow: 1,
          display: "flex",
          flexDirection: "column",
        }}
      >
        <List>
          <Divider />
          {
          connections &&
            connections.map((row) => (
              <ListItem key={`connection_${row.id}`}>
                <PowerIcon />
                <ListItemButton onClick={() => addActiveConnection(row)}>
                  <ListItemText primary={row.name} />
                </ListItemButton>
              </ListItem>
            ))}
        </List>
        {RenderSchemas()}
      </Box>

      {/* CUDA Toggle */}
      <Tooltip
        title={cudaEnabled ? "CUDA GPU acceleration enabled for training" : "Enable CUDA GPU acceleration for faster training"}
        placement="right"
      >
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            px: 1,
            py: 0.5,
            borderRadius: 1,
            bgcolor: cudaEnabled ? "rgba(76, 175, 80, 0.1)" : "grey.100",
            border: "1px solid",
            borderColor: cudaEnabled ? "success.light" : "grey.300",
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <MemoryIcon
              fontSize="small"
              sx={{ color: cudaEnabled ? "success.main" : "grey.500" }}
            />
            <Typography
              variant="body2"
              sx={{
                fontWeight: 500,
                color: cudaEnabled ? "success.main" : "text.secondary",
              }}
            >
              CUDA
            </Typography>
          </Box>
          <Switch
            size="small"
            checked={cudaEnabled}
            onChange={(e) => setCudaEnabled(e.target.checked)}
            color="success"
          />
        </Box>
      </Tooltip>

      <Divider />

      {/* User Info */}
      <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
        <Avatar
          sx={{ bgcolor: "primary.main", width: 36, height: 36 }}
        >
          {user?.username?.[0]?.toUpperCase() || "U"}
        </Avatar>
        <Box sx={{ flexGrow: 1, minWidth: 0 }}>
          <Typography variant="body2" noWrap fontWeight={500}>
            {user?.username || "User"}
          </Typography>
          <Typography variant="caption" color="textSecondary" noWrap sx={{ display: "block" }}>
            {user?.email || ""}
          </Typography>
        </Box>
        <IconButton size="small" onClick={logout}>
          <LogoutRoundedIcon fontSize="small" />
        </IconButton>
      </Box>
    </Box>
  );
}
