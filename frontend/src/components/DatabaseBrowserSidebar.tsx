import React, { useEffect, useState } from "react";
import {
  Box,
  Typography,
  CircularProgress,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemIcon,
  Collapse,
  Paper,
} from "@mui/material";
import {
  ExpandMore,
  ChevronRight,
  Storage,
  Folder,
  TableChart,
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
      const newExpanded = new Set(expandedConnections);
      newExpanded.delete(connectionKey);
      setExpandedConnections(newExpanded);
    } else {
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
      const newExpanded = new Set(expandedSchemas);
      newExpanded.delete(schemaKey);
      setExpandedSchemas(newExpanded);
    } else {
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

  if (!open) return null;

  return (
    <Box
      sx={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
        bgcolor: "background.paper",
      }}
    >
      {/* Sidebar Header */}
      <Box
        sx={{
          p: 3,
          borderBottom: 1,
          borderColor: "divider",
          bgcolor: "grey.50",
        }}
      >
        <Typography variant="h6" fontWeight={600} color="text.primary">
          Connections
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Browse your databases
        </Typography>
      </Box>

      {/* Connections List */}
      <Box sx={{ flex: 1, overflowY: "auto", p: 2 }}>
        {!connections || connections.length === 0 ? (
          <Box sx={{ p: 4, textAlign: "center" }}>
            <Storage sx={{ fontSize: 64, color: "text.disabled", mb: 2 }} />
            <Typography variant="body2" color="text.secondary" gutterBottom>
              No database connections found.
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Create a connection to get started.
            </Typography>
          </Box>
        ) : (
          <List disablePadding>
            {connections.map((connection) => {
              const connectionKey = connection.id;
              const isExpanded = expandedConnections.has(connectionKey);
              const isLoading = loading[connectionKey];
              const schemas = connectionSchemas[connectionKey] || [];

              return (
                <Box key={connectionKey} sx={{ mb: 2 }}>
                  {/* Connection Item */}
                  <Paper elevation={0} variant="outlined" sx={{ mb: 1 }}>
                    <ListItemButton
                      onClick={() => handleConnectionClick(connection)}
                      sx={{
                        py: 2,
                        px: 2,
                        borderRadius: 1,
                        "&:hover": {
                          bgcolor: "action.hover",
                        },
                      }}
                    >
                      <ListItemIcon sx={{ minWidth: 40 }}>
                        {isExpanded ? (
                          <ExpandMore fontSize="medium" />
                        ) : (
                          <ChevronRight fontSize="medium" />
                        )}
                      </ListItemIcon>
                      <ListItemIcon sx={{ minWidth: 40 }}>
                        <Storage fontSize="medium" color="primary" />
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Typography variant="body1" fontWeight={600}>
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
                  </Paper>

                  {/* Schemas */}
                  <Collapse in={isExpanded} timeout="auto" unmountOnExit>
                    {isLoading ? (
                      <Box sx={{ display: "flex", justifyContent: "center", py: 3 }}>
                        <CircularProgress size={24} />
                      </Box>
                    ) : (
                      <Box sx={{ pl: 3 }}>
                        <List disablePadding>
                          {schemas.map((schema) => {
                            const schemaKey = `${connectionKey}_${schema}`;
                            const isSchemaExpanded = expandedSchemas.has(schemaKey);
                            const isSchemaLoading = loading[schemaKey];
                            const tables = schemaTables[schemaKey] || [];

                            return (
                              <Box key={schemaKey} sx={{ mb: 1.5 }}>
                                {/* Schema Item */}
                                <ListItemButton
                                  onClick={() => handleSchemaClick(connection, schema)}
                                  sx={{
                                    py: 1.5,
                                    px: 2,
                                    borderRadius: 1,
                                    bgcolor: "grey.50",
                                    "&:hover": {
                                      bgcolor: "action.hover",
                                    },
                                  }}
                                >
                                  <ListItemIcon sx={{ minWidth: 36 }}>
                                    {isSchemaExpanded ? (
                                      <ExpandMore fontSize="small" />
                                    ) : (
                                      <ChevronRight fontSize="small" />
                                    )}
                                  </ListItemIcon>
                                  <ListItemIcon sx={{ minWidth: 36 }}>
                                    <Folder fontSize="small" color="secondary" />
                                  </ListItemIcon>
                                  <ListItemText
                                    primary={
                                      <Typography variant="body2" fontWeight={500}>
                                        {schema}
                                      </Typography>
                                    }
                                  />
                                </ListItemButton>

                                {/* Tables */}
                                <Collapse in={isSchemaExpanded} timeout="auto" unmountOnExit>
                                  {isSchemaLoading ? (
                                    <Box sx={{ display: "flex", justifyContent: "center", py: 2 }}>
                                      <CircularProgress size={20} />
                                    </Box>
                                  ) : tables.length === 0 ? (
                                    <Box sx={{ pl: 6, py: 2 }}>
                                      <Typography variant="caption" color="text.secondary" fontStyle="italic">
                                        No tables found
                                      </Typography>
                                    </Box>
                                  ) : (
                                    <Box sx={{ pl: 4, mt: 1 }}>
                                      <List disablePadding>
                                        {tables.map((table) => (
                                          <ListItem key={table.table_name} disablePadding sx={{ mb: 0.5 }}>
                                            <ListItemButton
                                              onClick={() =>
                                                handleTableClick(connection, schema, table.table_name)
                                              }
                                              sx={{
                                                py: 1,
                                                px: 2,
                                                borderRadius: 1,
                                                "&:hover": {
                                                  bgcolor: "action.hover",
                                                },
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
                                    </Box>
                                  )}
                                </Collapse>
                              </Box>
                            );
                          })}
                        </List>
                      </Box>
                    )}
                  </Collapse>
                </Box>
              );
            })}
          </List>
        )}
      </Box>
    </Box>
  );
}
