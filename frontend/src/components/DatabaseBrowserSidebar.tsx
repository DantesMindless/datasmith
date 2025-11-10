import React, { useEffect, useState } from "react";
import {
  Box,
  Typography,
  CircularProgress,
  IconButton,
  Divider,
  Collapse,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemIcon,
} from "@mui/material";
import {
  ExpandMore,
  ChevronRight,
  Storage,
  Folder,
  TableChart,
  Refresh,
} from "@mui/icons-material";
import { useAppContext } from "../providers/useAppContext";
import { getDatabasesList, getSchemaTablesList } from "../utils/requests";
import { Connection } from "../providers/constants";

interface DatabaseBrowserSidebarProps {
  open: boolean;
}

export default function DatabaseBrowserSidebar({ open }: DatabaseBrowserSidebarProps) {
  const { connections, updateConnections, addTableViewTab, showAlert } = useAppContext();
  const [expandedConnections, setExpandedConnections] = useState<Set<string>>(new Set());
  const [expandedSchemas, setExpandedSchemas] = useState<Set<string>>(new Set());
  const [connectionSchemas, setConnectionSchemas] = useState<Record<string, string[]>>({});
  const [schemaTables, setSchemaTables] = useState<Record<string, any[]>>({});
  const [loading, setLoading] = useState<Record<string, boolean>>({});

  useEffect(() => {
    if (connections === null) {
      updateConnections();
    }
  }, [connections, updateConnections]);

  const handleConnectionClick = async (connection: Connection) => {
    const connectionKey = connection.id;

    if (expandedConnections.has(connectionKey)) {
      // Collapse
      const newExpanded = new Set(expandedConnections);
      newExpanded.delete(connectionKey);
      setExpandedConnections(newExpanded);
    } else {
      // Expand and fetch schemas if not already loaded
      const newExpanded = new Set(expandedConnections);
      newExpanded.add(connectionKey);
      setExpandedConnections(newExpanded);

      if (!connectionSchemas[connectionKey]) {
        setLoading({ ...loading, [connectionKey]: true });
        try {
          const schemas = await getDatabasesList(connection.id);
          setConnectionSchemas({ ...connectionSchemas, [connectionKey]: schemas });
        } catch (error) {
          console.error("Error fetching schemas:", error);
          showAlert("Failed to fetch database schemas");
        } finally {
          setLoading({ ...loading, [connectionKey]: false });
        }
      }
    }
  };

  const handleSchemaClick = async (connection: Connection, schema: string) => {
    const schemaKey = `${connection.id}_${schema}`;

    if (expandedSchemas.has(schemaKey)) {
      // Collapse
      const newExpanded = new Set(expandedSchemas);
      newExpanded.delete(schemaKey);
      setExpandedSchemas(newExpanded);
    } else {
      // Expand and fetch tables if not already loaded
      const newExpanded = new Set(expandedSchemas);
      newExpanded.add(schemaKey);
      setExpandedSchemas(newExpanded);

      if (!schemaTables[schemaKey]) {
        setLoading({ ...loading, [schemaKey]: true });
        try {
          const tables = await getSchemaTablesList(connection.id, schema);
          setSchemaTables({ ...schemaTables, [schemaKey]: tables });
        } catch (error) {
          console.error("Error fetching tables:", error);
          showAlert("Failed to fetch schema tables");
        } finally {
          setLoading({ ...loading, [schemaKey]: false });
        }
      }
    }
  };

  const handleTableClick = (connection: Connection, schema: string, tableName: string) => {
    addTableViewTab(connection, schema, tableName);
  };

  const handleRefresh = () => {
    updateConnections();
    setExpandedConnections(new Set());
    setExpandedSchemas(new Set());
    setConnectionSchemas({});
    setSchemaTables({});
  };

  if (!open) return null;

  return (
    <Box
      sx={{
        width: 280,
        height: "100%",
        borderRight: "1px solid",
        borderColor: "divider",
        bgcolor: "background.paper",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <Box
        sx={{
          p: 2,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          borderBottom: "1px solid",
          borderColor: "divider",
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <Storage color="primary" />
          <Typography variant="subtitle1" fontWeight={600}>
            Database Browser
          </Typography>
        </Box>
        <IconButton size="small" onClick={handleRefresh} title="Refresh">
          <Refresh fontSize="small" />
        </IconButton>
      </Box>

      {/* Connections List */}
      <Box sx={{ flex: 1, overflowY: "auto", overflowX: "hidden" }}>
        {!connections || connections.length === 0 ? (
          <Box sx={{ p: 3, textAlign: "center" }}>
            <Typography variant="body2" color="text.secondary">
              No database connections found.
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Create a connection to get started.
            </Typography>
          </Box>
        ) : (
          <List disablePadding sx={{ py: 1 }}>
            {connections.map((connection) => {
              const connectionKey = connection.id;
              const isExpanded = expandedConnections.has(connectionKey);
              const isLoading = loading[connectionKey];
              const schemas = connectionSchemas[connectionKey] || [];

              return (
                <React.Fragment key={connectionKey}>
                  {/* Connection Item */}
                  <ListItem disablePadding>
                    <ListItemButton
                      onClick={() => handleConnectionClick(connection)}
                      sx={{
                        pl: 2,
                        "&:hover": { bgcolor: "action.hover" },
                      }}
                    >
                      <ListItemIcon sx={{ minWidth: 36 }}>
                        {isExpanded ? <ExpandMore /> : <ChevronRight />}
                      </ListItemIcon>
                      <ListItemIcon sx={{ minWidth: 36 }}>
                        <Storage fontSize="small" color="primary" />
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Typography variant="body2" fontWeight={500}>
                            {connection.name}
                          </Typography>
                        }
                        secondary={
                          <Typography variant="caption" color="text.secondary">
                            {connection.type}
                          </Typography>
                        }
                      />
                    </ListItemButton>
                  </ListItem>

                  {/* Schemas */}
                  <Collapse in={isExpanded} timeout="auto" unmountOnExit>
                    {isLoading ? (
                      <Box sx={{ display: "flex", justifyContent: "center", py: 2 }}>
                        <CircularProgress size={20} />
                      </Box>
                    ) : (
                      <List disablePadding>
                        {schemas.map((schema) => {
                          const schemaKey = `${connectionKey}_${schema}`;
                          const isSchemaExpanded = expandedSchemas.has(schemaKey);
                          const isSchemaLoading = loading[schemaKey];
                          const tables = schemaTables[schemaKey] || [];

                          return (
                            <React.Fragment key={schemaKey}>
                              {/* Schema Item */}
                              <ListItem disablePadding>
                                <ListItemButton
                                  onClick={() => handleSchemaClick(connection, schema)}
                                  sx={{
                                    pl: 6,
                                    "&:hover": { bgcolor: "action.hover" },
                                  }}
                                >
                                  <ListItemIcon sx={{ minWidth: 36 }}>
                                    {isSchemaExpanded ? <ExpandMore /> : <ChevronRight />}
                                  </ListItemIcon>
                                  <ListItemIcon sx={{ minWidth: 36 }}>
                                    <Folder fontSize="small" color="secondary" />
                                  </ListItemIcon>
                                  <ListItemText
                                    primary={
                                      <Typography variant="body2">
                                        {schema}
                                      </Typography>
                                    }
                                  />
                                </ListItemButton>
                              </ListItem>

                              {/* Tables */}
                              <Collapse in={isSchemaExpanded} timeout="auto" unmountOnExit>
                                {isSchemaLoading ? (
                                  <Box sx={{ display: "flex", justifyContent: "center", py: 2 }}>
                                    <CircularProgress size={16} />
                                  </Box>
                                ) : tables.length === 0 ? (
                                  <Box sx={{ pl: 12, py: 1 }}>
                                    <Typography variant="caption" color="text.secondary">
                                      No tables found
                                    </Typography>
                                  </Box>
                                ) : (
                                  <List disablePadding>
                                    {tables.map((table) => (
                                      <ListItem key={table.table_name} disablePadding>
                                        <ListItemButton
                                          onClick={() =>
                                            handleTableClick(connection, schema, table.table_name)
                                          }
                                          sx={{
                                            pl: 10,
                                            "&:hover": { bgcolor: "action.hover" },
                                          }}
                                        >
                                          <ListItemIcon sx={{ minWidth: 36 }}>
                                            <TableChart fontSize="small" />
                                          </ListItemIcon>
                                          <ListItemText
                                            primary={
                                              <Typography variant="body2">
                                                {table.table_name}
                                              </Typography>
                                            }
                                          />
                                        </ListItemButton>
                                      </ListItem>
                                    ))}
                                  </List>
                                )}
                              </Collapse>
                            </React.Fragment>
                          );
                        })}
                      </List>
                    )}
                  </Collapse>
                </React.Fragment>
              );
            })}
          </List>
        )}
      </Box>
    </Box>
  );
}
