import * as React from "react";
import { useEffect } from "react";

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

import { useAppContext } from "../providers/useAppContext";
import { getDatabasesList, getSchemaTablesList } from "../utils/requests";

import AddBoxIcon from '@mui/icons-material/AddBox';
import IndeterminateCheckBoxIcon from '@mui/icons-material/IndeterminateCheckBox';
import SvgIcon, { SvgIconProps } from '@mui/material/SvgIcon';
import { styled } from '@mui/material/styles';
import { SimpleTreeView } from '@mui/x-tree-view/SimpleTreeView';
import { TreeItem, treeItemClasses } from '@mui/x-tree-view/TreeItem';
import { Connection } from "../providers/constants";

const CustomTreeItem = styled(TreeItem)({
  [`& .${treeItemClasses.iconContainer}`]: {
    '& .close': {
      opacity: 0.3,
    },
  },
});

function CloseSquare(props: SvgIconProps) {
  return (
    <SvgIcon
      className="close"
      fontSize="inherit"
      style={{ width: 14, height: 14 }}
      {...props}
    >
      {/* tslint:disable-next-line: max-line-length */}
      <path d="M13.5 2c-5.629 0-10.212 4.436-10.475 10h-3.025l4.537 5.917 4.463-5.917h-2.975c.26-3.902 3.508-7 7.475-7 4.136 0 7.5 3.364 7.5 7.5s-3.364 7.5-7.5 7.5c-2.381 0-4.502-1.119-5.876-2.854l-1.847 2.449c1.919 2.088 4.664 3.405 7.723 3.405 5.798 0 10.5-4.702 10.5-10.5s-4.702-10.5-10.5-10.5z"/>
    </SvgIcon>
  );
}

interface ActiveConnection extends Connection {
  id: string;
  schemas: Record<string, string[]>;
}


export default function Sidebar() {
  const { connections, activeConnections, updateActiveConnections, updateConnections, addTableViewTab } = useAppContext();

  useEffect(() => {
    if (connections?.length === 0) {
      updateConnections();
    }
  }, [connections, updateConnections]);

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
          defaultExpandedItems={['grid']}
          slots={{
            expandIcon: AddBoxIcon,
            collapseIcon: IndeterminateCheckBoxIcon,
            endIcon: CloseSquare,
          }}
        >
          {Object.keys(activeConnections).map((key) =>
            Object.keys(activeConnections[key].schemas).map((schema) => (
              <CustomTreeItem
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
              </CustomTreeItem>
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
