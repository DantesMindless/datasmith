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
    <>
      {/* Sidebar Header */}
      <Box sx={{
        px: 2.5,
        py: 2,
        borderBottom: 1,
        borderColor: 'divider',
        bgcolor: 'grey.50'
      }}>
        <Typography variant="overline" fontWeight={700} color="text.secondary" letterSpacing={1}>
          Connections
        </Typography>
      </Box>

      {/* Connections List */}
      <Box sx={{ flex: 1, overflowY: "auto", overflowX: "hidden", pt: 1.5 }}>
        {!connections || connections.length === 0 ? (
          <Box sx={{ p: 4, textAlign: "center" }}>
            <Storage sx={{ fontSize: 48, color: "text.disabled", mb: 2 }} />
            <Typography variant="body2" color="text.secondary" gutterBottom>
              No database connections found.
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Create a connection to get started.
            </Typography>
          </Box>
        ) : (
          <List disablePadding sx={{ py: 0.5 }}>
            {connections.map((connection) => {
              const connectionKey = connection.id;
              const isExpanded = expandedConnections.has(connectionKey);
              const isLoading = loading[connectionKey];
              const schemas = connectionSchemas[connectionKey] || [];

              return (
                <React.Fragment key={connectionKey}>
                  {/* Connection Item */}
                  <ListItem disablePadding sx={{ mb: 0.5 }}>
                    <ListItemButton
                      onClick={() => handleConnectionClick(connection)}
                      sx={{
                        pl: 1.5,
                        py: 1,
                        borderRadius: 1,
                        mx: 1,
                        "&:hover": {
                          bgcolor: "action.hover",
                        },
                        ...(isExpanded && {
                          bgcolor: "action.selected",
                        })
                      }}
                    >
                      <ListItemIcon sx={{ minWidth: 32 }}>
                        {isExpanded ? (
                          <ExpandMore fontSize="small" />
                        ) : (
                          <ChevronRight fontSize="small" />
                        )}
                      </ListItemIcon>
                      <ListItemIcon sx={{ minWidth: 32 }}>
                        <Storage fontSize="small" color="primary" />
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Typography variant="body2" fontWeight={500} noWrap>
                            {connection.name}
                          </Typography>
                        }
                        secondary={
                          <Typography variant="caption" color="text.secondary" noWrap>
                            {connection.type}
                          </Typography>
                        }
                      />
                    </ListItemButton>
                  </ListItem>

                  {/* Schemas */}
                  <Collapse in={isExpanded} timeout="auto" unmountOnExit>
                    {isLoading ? (
                      <Box sx={{ display: "flex", justifyContent: "center", py: 2, pl: 4 }}>
                        <CircularProgress size={20} />
                      </Box>
                    ) : (
                      <List disablePadding sx={{ pl: 1 }}>
                        {schemas.map((schema) => {
                          const schemaKey = `${connectionKey}_${schema}`;
                          const isSchemaExpanded = expandedSchemas.has(schemaKey);
                          const isSchemaLoading = loading[schemaKey];
                          const tables = schemaTables[schemaKey] || [];

                          return (
                            <React.Fragment key={schemaKey}>
                              {/* Schema Item */}
                              <ListItem disablePadding sx={{ mb: 0.5 }}>
                                <ListItemButton
                                  onClick={() => handleSchemaClick(connection, schema)}
                                  sx={{
                                    pl: 3,
                                    py: 0.75,
                                    borderRadius: 1,
                                    mx: 1,
                                    "&:hover": { bgcolor: "action.hover" },
                                    ...(isSchemaExpanded && {
                                      bgcolor: "action.selected",
                                    })
                                  }}
                                >
                                  <ListItemIcon sx={{ minWidth: 28 }}>
                                    {isSchemaExpanded ? (
                                      <ExpandMore fontSize="small" />
                                    ) : (
                                      <ChevronRight fontSize="small" />
                                    )}
                                  </ListItemIcon>
                                  <ListItemIcon sx={{ minWidth: 28 }}>
                                    <Folder fontSize="small" color="secondary" />
                                  </ListItemIcon>
                                  <ListItemText
                                    primary={
                                      <Typography variant="body2" fontSize="0.875rem" noWrap>
                                        {schema}
                                      </Typography>
                                    }
                                  />
                                </ListItemButton>
                              </ListItem>

                              {/* Tables */}
                              <Collapse in={isSchemaExpanded} timeout="auto" unmountOnExit>
                                {isSchemaLoading ? (
                                  <Box sx={{ display: "flex", justifyContent: "center", py: 2, pl: 6 }}>
                                    <CircularProgress size={16} />
                                  </Box>
                                ) : tables.length === 0 ? (
                                  <Box sx={{ pl: 8, py: 1.5 }}>
                                    <Typography variant="caption" color="text.secondary" fontStyle="italic">
                                      No tables found
                                    </Typography>
                                  </Box>
                                ) : (
                                  <List disablePadding sx={{ pl: 2 }}>
                                    {tables.map((table) => (
                                      <ListItem key={table.table_name} disablePadding sx={{ mb: 0.25 }}>
                                        <ListItemButton
                                          onClick={() =>
                                            handleTableClick(connection, schema, table.table_name)
                                          }
                                          sx={{
                                            pl: 5,
                                            py: 0.5,
                                            borderRadius: 1,
                                            mx: 1,
                                            "&:hover": {
                                              bgcolor: "action.hover",
                                            },
                                          }}
                                        >
                                          <ListItemIcon sx={{ minWidth: 28 }}>
                                            <TableChart fontSize="small" sx={{ fontSize: '1.1rem' }} />
                                          </ListItemIcon>
                                          <ListItemText
                                            primary={
                                              <Typography variant="body2" fontSize="0.8125rem" noWrap>
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
    </>
  );
}
