import React, { useEffect, useState } from "react";
import {
  Box,
  Typography,
  FormControl,
  Select,
  MenuItem,
  Chip,
  Button,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Paper,
  Toolbar,
  Tabs,
  Tab,
  CircularProgress,
  Alert,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Checkbox,
  Collapse,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
} from "@mui/material";
import {
  Refresh,
  Close,
  FilterList,
  Download,
  Search,
  ViewColumn,
  ExpandMore,
  ChevronRight,
  FileDownload,
} from "@mui/icons-material";
import { useAppContext } from "../providers/useAppContext";
import { getDatabasesList, getSchemaTablesList, queryTab, getJoins, exportTableToCSV } from "../utils/requests";
import { Connection } from "../providers/constants";

interface TableTab {
  id: string;
  connection: Connection;
  schema: string;
  table: string;
  label: string;
  data: any[];
  columns: string[];
  allColumns: string[];
  activeColumns: string[];
  joins: any;
  page: number;
  rowsPerPage: number;
  columnSelectorOpen: boolean;
}

export default function DatabaseBrowser() {
  const { connections, updateConnections, showAlert } = useAppContext();

  // Selection state
  const [selectedConnection, setSelectedConnection] = useState<Connection | null>(null);
  const [selectedSchema, setSelectedSchema] = useState<string>("");
  const [selectedTable, setSelectedTable] = useState<string>("");

  // Data state
  const [schemas, setSchemas] = useState<string[]>([]);
  const [tables, setTables] = useState<any[]>([]);
  const [openTabs, setOpenTabs] = useState<TableTab[]>([]);
  const [activeTabIndex, setActiveTabIndex] = useState<number>(0);

  // Loading state
  const [loadingSchemas, setLoadingSchemas] = useState(false);
  const [loadingTables, setLoadingTables] = useState(false);
  const [loadingData, setLoadingData] = useState(false);

  // Column selector state
  const [expandedTables, setExpandedTables] = useState<Set<string>>(new Set(['parent_level_1']));

  // Export dialog state
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [exportLimit, setExportLimit] = useState(10000);
  const [exportDatasetName, setExportDatasetName] = useState("");
  const [exportDatasetDescription, setExportDatasetDescription] = useState("");
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    if (connections === null) {
      updateConnections();
    }
  }, [connections, updateConnections]);

  // Load schemas when connection changes
  useEffect(() => {
    if (selectedConnection) {
      loadSchemas(selectedConnection.id);
    } else {
      setSchemas([]);
      setSelectedSchema("");
    }
  }, [selectedConnection]);

  // Load tables when schema changes
  useEffect(() => {
    if (selectedConnection && selectedSchema) {
      loadTables(selectedConnection.id, selectedSchema);
    } else {
      setTables([]);
      setSelectedTable("");
    }
  }, [selectedSchema]);

  const loadSchemas = async (connectionId: string) => {
    setLoadingSchemas(true);
    try {
      const schemaList = await getDatabasesList(connectionId);
      setSchemas(schemaList);
      if (schemaList.length > 0) {
        setSelectedSchema(schemaList[0]);
      }
    } catch (error) {
      console.error("Error loading schemas:", error);
      showAlert("Failed to load schemas");
    } finally {
      setLoadingSchemas(false);
    }
  };

  const loadTables = async (connectionId: string, schema: string) => {
    setLoadingTables(true);
    try {
      const tableList = await getSchemaTablesList(connectionId, schema);
      setTables(tableList);
    } catch (error) {
      console.error("Error loading tables:", error);
      showAlert("Failed to load tables");
    } finally {
      setLoadingTables(false);
    }
  };

  const openTableTab = async () => {
    if (!selectedConnection || !selectedSchema || !selectedTable) return;

    // Check if tab already exists
    const existingTabIndex = openTabs.findIndex(
      (tab) =>
        tab.connection.id === selectedConnection.id &&
        tab.schema === selectedSchema &&
        tab.table === selectedTable
    );

    if (existingTabIndex !== -1) {
      setActiveTabIndex(existingTabIndex);
      return;
    }

    // Create new tab with proper structure
    const newTab: TableTab = {
      id: `${selectedConnection.id}_${selectedSchema}_${selectedTable}`,
      connection: selectedConnection,
      schema: selectedSchema,
      table: selectedTable,
      label: `${selectedTable}`,
      data: [],
      columns: [],
      allColumns: [],
      activeColumns: [],
      joins: {},
      page: 0,
      rowsPerPage: 25,
      columnSelectorOpen: false,
    };

    setLoadingData(true);
    try {
      // Create tab object that matches the old API format
      const tabObj = {
        ID: selectedConnection.id,
        schema: selectedSchema,
        table: selectedTable,
        page: 1,
        perPage: 25,
        joins: {},
        columns: [],
        activeColumns: [],
      };

      // Fetch joins/metadata first
      const joins = await getJoins(tabObj);
      tabObj.joins = joins;

      // Build column list from joins
      const buildColumns = (table: any, level: number, collector: string[]) => {
        const schema_name = Object.keys(table)[0];
        const table_fields = table[schema_name];
        const rootId = `parent_level_${level}^-^${schema_name}`;
        collector.push(rootId);

        if (table_fields?.fields && table_fields.fields.length > 0) {
          table_fields.fields.forEach((row: any) => {
            collector.push(`level_${level}^-^${schema_name}.${row.column_name}`);
          });
          if (table_fields.relations) {
            table_fields.relations.forEach((rel: any) => buildColumns(rel, level + 1, collector));
          }
        }
      };

      buildColumns({ [selectedTable]: joins }, 1, tabObj.columns);
      // Select all level 1 columns initially
      tabObj.activeColumns = tabObj.columns.filter((item: string) => item.includes("level_1"));

      // Store metadata in new tab
      newTab.allColumns = [...tabObj.columns];
      newTab.activeColumns = [...tabObj.activeColumns];
      newTab.joins = joins;

      // Fetch actual data
      const result = await queryTab(tabObj);

      if (result && result.length > 0) {
        newTab.data = result;
        newTab.columns = Object.keys(result[0]).filter((col) => col !== "total_rows_number");
      }

      setOpenTabs([...openTabs, newTab]);
      setActiveTabIndex(openTabs.length);
    } catch (error) {
      console.error("Error loading table data:", error);
      showAlert("Failed to load table data");
    } finally {
      setLoadingData(false);
    }
  };

  const closeTab = (index: number) => {
    const newTabs = openTabs.filter((_, i) => i !== index);
    setOpenTabs(newTabs);
    if (activeTabIndex >= newTabs.length) {
      setActiveTabIndex(Math.max(0, newTabs.length - 1));
    }
  };

  const handleChangePage = async (tabIndex: number, newPage: number) => {
    const currentTab = openTabs[tabIndex];
    setLoadingData(true);
    try {
      const tabObj = {
        ID: currentTab.connection.id,
        schema: currentTab.schema,
        table: currentTab.table,
        page: newPage + 1, // Backend uses 1-based pagination
        perPage: currentTab.rowsPerPage,
        joins: currentTab.joins,
        columns: currentTab.allColumns,
        activeColumns: currentTab.activeColumns,
      };

      const result = await queryTab(tabObj);

      if (result && result.length > 0) {
        const newTabs = [...openTabs];
        newTabs[tabIndex].data = result;
        newTabs[tabIndex].page = newPage;
        setOpenTabs(newTabs);
      }
    } catch (error) {
      console.error("Error fetching page:", error);
      showAlert("Failed to fetch page data");
    } finally {
      setLoadingData(false);
    }
  };

  const handleChangeRowsPerPage = async (tabIndex: number, newRowsPerPage: number) => {
    const currentTab = openTabs[tabIndex];
    setLoadingData(true);
    try {
      const tabObj = {
        ID: currentTab.connection.id,
        schema: currentTab.schema,
        table: currentTab.table,
        page: 1, // Reset to first page
        perPage: newRowsPerPage,
        joins: currentTab.joins,
        columns: currentTab.allColumns,
        activeColumns: currentTab.activeColumns,
      };

      const result = await queryTab(tabObj);

      if (result && result.length > 0) {
        const newTabs = [...openTabs];
        newTabs[tabIndex].data = result;
        newTabs[tabIndex].rowsPerPage = newRowsPerPage;
        newTabs[tabIndex].page = 0;
        setOpenTabs(newTabs);
      }
    } catch (error) {
      console.error("Error changing rows per page:", error);
      showAlert("Failed to fetch data");
    } finally {
      setLoadingData(false);
    }
  };

  const handleExport = async () => {
    if (!activeTab) return;

    setExporting(true);
    try {
      const result = await exportTableToCSV(activeTab.connection.id, {
        schema: activeTab.schema,
        table: activeTab.table,
        columns: activeTab.activeColumns.length > 0
          ? activeTab.activeColumns
              .filter((col: string) => !col.startsWith("parent_"))
              .map((col: string) => col.split(".").pop() || col)
          : undefined,
        limit: exportLimit,
        dataset_name: exportDatasetName || `${activeTab.table} Export`,
        dataset_description: exportDatasetDescription || `Exported from ${activeTab.connection.name}/${activeTab.schema}/${activeTab.table}`,
      });

      showAlert(`Export successful! Dataset "${result.dataset.name}" created with ${result.rows_exported} rows.`, "success");
      setExportDialogOpen(false);

      // Reset export form
      setExportDatasetName("");
      setExportDatasetDescription("");
      setExportLimit(10000);
    } catch (error: any) {
      console.error("Export error:", error);
      showAlert(error.response?.data || "Failed to export table", "error");
    } finally {
      setExporting(false);
    }
  };

  const activeTab = openTabs[activeTabIndex];

  return (
    <Box sx={{ display: "flex", flexDirection: "column", height: "100%", bgcolor: "background.default" }}>
      {/* Top Toolbar - Connection/Schema/Table Selectors */}
      <Paper elevation={0} sx={{ borderBottom: 1, borderColor: "divider" }}>
        <Toolbar sx={{ gap: 2, py: 2 }}>
          <Typography variant="h6" sx={{ minWidth: 120 }}>
            Database Browser
          </Typography>

          {/* Connection Selector */}
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <Select
              value={selectedConnection?.id || ""}
              onChange={(e) => {
                const conn = connections?.find((c) => c.id === e.target.value);
                setSelectedConnection(conn || null);
              }}
              displayEmpty
              renderValue={(value) => {
                if (!value) return <em>Select Connection</em>;
                return selectedConnection?.name || "";
              }}
            >
              <MenuItem value="" disabled>
                <em>Select Connection</em>
              </MenuItem>
              {connections?.map((conn) => (
                <MenuItem key={conn.id} value={conn.id}>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <Typography>{conn.name}</Typography>
                    <Chip label={conn.type} size="small" />
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Schema Selector */}
          <FormControl size="small" sx={{ minWidth: 180 }} disabled={!selectedConnection}>
            <Select
              value={selectedSchema}
              onChange={(e) => setSelectedSchema(e.target.value)}
              displayEmpty
            >
              <MenuItem value="" disabled>
                <em>Select Schema</em>
              </MenuItem>
              {schemas.map((schema) => (
                <MenuItem key={schema} value={schema}>
                  {schema}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Table Selector */}
          <FormControl size="small" sx={{ minWidth: 180 }} disabled={!selectedSchema}>
            <Select
              value={selectedTable}
              onChange={(e) => setSelectedTable(e.target.value)}
              displayEmpty
            >
              <MenuItem value="" disabled>
                <em>Select Table</em>
              </MenuItem>
              {tables.map((table) => (
                <MenuItem key={table.table_name} value={table.table_name}>
                  {table.table_name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Open Table Button */}
          <Button
            variant="contained"
            onClick={openTableTab}
            disabled={!selectedTable || loadingData}
          >
            {loadingData ? <CircularProgress size={20} /> : "Open Table"}
          </Button>

          <Box sx={{ flex: 1 }} />

          {/* Refresh Button */}
          <IconButton onClick={() => updateConnections()}>
            <Refresh />
          </IconButton>
        </Toolbar>
      </Paper>

      {/* Tabs for open tables */}
      {openTabs.length > 0 && (
        <Paper elevation={0} sx={{ borderBottom: 1, borderColor: "divider" }}>
          <Tabs
            value={activeTabIndex}
            onChange={(e, newValue) => setActiveTabIndex(newValue)}
            variant="scrollable"
            scrollButtons="auto"
          >
            {openTabs.map((tab, index) => (
              <Tab
                key={tab.id}
                label={
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <Typography variant="body2">{tab.label}</Typography>
                    <IconButton
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        closeTab(index);
                      }}
                      sx={{ ml: 1, p: 0.5 }}
                    >
                      <Close fontSize="small" />
                    </IconButton>
                  </Box>
                }
              />
            ))}
          </Tabs>
        </Paper>
      )}

      {/* Main Content Area */}
      <Box sx={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
        {openTabs.length === 0 ? (
          // Empty State
          <Box
            sx={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              gap: 2,
            }}
          >
            <Search sx={{ fontSize: 80, color: "text.disabled" }} />
            <Typography variant="h5" color="text.secondary">
              No tables open
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Select a connection, schema, and table to get started
            </Typography>
          </Box>
        ) : (
          // Active Tab Content
          activeTab && (
            <Box sx={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
              {/* Table Toolbar */}
              <Paper elevation={0} sx={{ borderBottom: 1, borderColor: "divider" }}>
                <Toolbar variant="dense" sx={{ gap: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    {activeTab.connection.name} / {activeTab.schema} / {activeTab.table}
                  </Typography>
                  <Box sx={{ flex: 1 }} />
                  <Button
                    size="small"
                    startIcon={<FileDownload />}
                    onClick={() => setExportDialogOpen(true)}
                    variant="outlined"
                  >
                    Export to CSV
                  </Button>
                  <IconButton
                    size="small"
                    onClick={() => {
                      const newTabs = [...openTabs];
                      newTabs[activeTabIndex].columnSelectorOpen = !newTabs[activeTabIndex].columnSelectorOpen;
                      setOpenTabs(newTabs);
                    }}
                    color={activeTab.columnSelectorOpen ? "primary" : "default"}
                  >
                    <ViewColumn />
                  </IconButton>
                </Toolbar>
              </Paper>

              {/* Data Table */}
              <Box sx={{ flex: 1, overflow: "auto", p: 3 }}>
                <TableContainer component={Paper}>
                  <Table size="small" stickyHeader>
                    <TableHead>
                      <TableRow>
                        {activeTab.columns.map((col) => (
                          <TableCell key={col} sx={{ fontWeight: 600, bgcolor: "grey.50" }}>
                            {col}
                          </TableCell>
                        ))}
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {activeTab.data.map((row, rowIndex) => (
                        <TableRow key={rowIndex} hover>
                          {activeTab.columns.map((col) => (
                            <TableCell key={col}>
                              {row[col] !== null && row[col] !== undefined ? String(row[col]) : (
                                <Typography variant="body2" color="text.disabled" fontStyle="italic">
                                  NULL
                                </Typography>
                              )}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>

                {/* Pagination */}
                <TablePagination
                  component="div"
                  count={activeTab.data[0]?.total_rows_number || activeTab.data.length}
                  page={activeTab.page}
                  onPageChange={(e, newPage) => handleChangePage(activeTabIndex, newPage)}
                  rowsPerPage={activeTab.rowsPerPage}
                  onRowsPerPageChange={(e) =>
                    handleChangeRowsPerPage(activeTabIndex, parseInt(e.target.value, 10))
                  }
                  rowsPerPageOptions={[10, 25, 50, 100]}
                />
              </Box>

              {/* Column Selector Drawer */}
              <Drawer
                anchor="right"
                open={activeTab.columnSelectorOpen}
                onClose={() => {
                  const newTabs = [...openTabs];
                  newTabs[activeTabIndex].columnSelectorOpen = false;
                  setOpenTabs(newTabs);
                }}
                variant="persistent"
                sx={{
                  '& .MuiDrawer-paper': {
                    width: 400,
                    top: 240,
                    height: 'calc(100% - 240px)',
                    boxShadow: '-4px 0 8px rgba(0,0,0,0.1)',
                  },
                }}
              >
                <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                  {/* Header with close button */}
                  <Box sx={{ p: 3, display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', borderBottom: 1, borderColor: 'divider' }}>
                    <Box>
                      <Typography variant="h6" gutterBottom>
                        Select Columns
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Choose which columns to display
                      </Typography>
                    </Box>
                    <IconButton
                      size="small"
                      onClick={() => {
                        const newTabs = [...openTabs];
                        newTabs[activeTabIndex].columnSelectorOpen = false;
                        setOpenTabs(newTabs);
                      }}
                    >
                      <Close />
                    </IconButton>
                  </Box>

                  {/* Content */}
                  <Box sx={{ flex: 1, overflowY: 'auto', p: 3 }}>
                    {/* Select All Toggle */}
                    <ListItem disablePadding sx={{ mb: 2 }}>
                      <ListItemButton
                        onClick={() => {
                          const newTabs = [...openTabs];
                          const allSelected = newTabs[activeTabIndex].activeColumns.length === newTabs[activeTabIndex].allColumns.length;
                          if (allSelected) {
                            newTabs[activeTabIndex].activeColumns = [];
                          } else {
                            newTabs[activeTabIndex].activeColumns = [...newTabs[activeTabIndex].allColumns];
                          }
                          setOpenTabs(newTabs);
                        }}
                        sx={{
                          borderRadius: 1,
                          py: 1.5,
                          bgcolor: 'primary.50',
                          '&:hover': {
                            bgcolor: 'primary.100',
                          }
                        }}
                      >
                        <ListItemText
                          primary={
                            <Typography variant="body2" fontWeight={600}>
                              Select All
                            </Typography>
                          }
                        />
                        <Checkbox
                          edge="end"
                          checked={activeTab.activeColumns.length === activeTab.allColumns.length}
                          indeterminate={activeTab.activeColumns.length > 0 && activeTab.activeColumns.length < activeTab.allColumns.length}
                          tabIndex={-1}
                          disableRipple
                        />
                      </ListItemButton>
                    </ListItem>

                    <Divider sx={{ mb: 2 }} />

                    {renderColumnSelector(activeTab)}
                  </Box>

                  {/* Footer with Apply button */}
                  <Box sx={{ p: 3, borderTop: 1, borderColor: 'divider', bgcolor: 'grey.50' }}>
                    <Button
                      fullWidth
                      variant="contained"
                      size="large"
                      onClick={async () => {
                        const currentTab = openTabs[activeTabIndex];
                        setLoadingData(true);
                        try {
                          // Refetch data with selected columns
                          const tabObj = {
                            ID: currentTab.connection.id,
                            schema: currentTab.schema,
                            table: currentTab.table,
                            page: 1,
                            perPage: currentTab.rowsPerPage,
                            joins: currentTab.joins,
                            columns: currentTab.allColumns,
                            activeColumns: currentTab.activeColumns,
                          };

                          const result = await queryTab(tabObj);

                          if (result && result.length > 0) {
                            const newTabs = [...openTabs];
                            newTabs[activeTabIndex].data = result;
                            newTabs[activeTabIndex].columns = Object.keys(result[0]).filter((col) => col !== "total_rows_number");
                            newTabs[activeTabIndex].page = 0;
                            newTabs[activeTabIndex].columnSelectorOpen = false;
                            setOpenTabs(newTabs);
                          }
                        } catch (error) {
                          console.error("Error refetching data:", error);
                          showAlert("Failed to apply column selection");
                        } finally {
                          setLoadingData(false);
                        }
                      }}
                      disabled={loadingData || activeTab.activeColumns.length === 0}
                    >
                      {loadingData ? <CircularProgress size={24} /> : 'Apply'}
                    </Button>
                    {activeTab.activeColumns.length === 0 && (
                      <Typography variant="caption" color="error" sx={{ mt: 1, display: 'block', textAlign: 'center' }}>
                        Please select at least one column
                      </Typography>
                    )}
                  </Box>
                </Box>
              </Drawer>
            </Box>
          )
        )}
      </Box>

      {/* Export Dialog */}
      <Dialog
        open={exportDialogOpen}
        onClose={() => !exporting && setExportDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          Export Table to CSV Dataset
        </DialogTitle>
        <DialogContent>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 3, pt: 2 }}>
            <Alert severity="info">
              {activeTab
                ? `Exporting from: ${activeTab.connection.name} / ${activeTab.schema} / ${activeTab.table}`
                : "No table selected"}
            </Alert>

            <TextField
              label="Dataset Name"
              value={exportDatasetName}
              onChange={(e) => setExportDatasetName(e.target.value)}
              placeholder={activeTab ? `${activeTab.table} Export` : ""}
              fullWidth
              helperText="Leave empty to use default name"
            />

            <TextField
              label="Dataset Description"
              value={exportDatasetDescription}
              onChange={(e) => setExportDatasetDescription(e.target.value)}
              placeholder={activeTab ? `Exported from ${activeTab.connection.name}` : ""}
              fullWidth
              multiline
              rows={3}
              helperText="Optional description for the dataset"
            />

            <TextField
              label="Row Limit"
              type="number"
              value={exportLimit}
              onChange={(e) => setExportLimit(Math.min(parseInt(e.target.value) || 10000, 1000000))}
              fullWidth
              helperText="Maximum rows to export (up to 1,000,000)"
              InputProps={{
                inputProps: { min: 1, max: 1000000 }
              }}
            />

            {activeTab && activeTab.activeColumns.length > 0 && (
              <Alert severity="success">
                {activeTab.activeColumns.filter((col: string) => !col.startsWith("parent_")).length} columns selected for export
              </Alert>
            )}
          </Box>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 3 }}>
          <Button onClick={() => setExportDialogOpen(false)} disabled={exporting}>
            Cancel
          </Button>
          <Button
            onClick={handleExport}
            variant="contained"
            disabled={exporting || !activeTab}
            startIcon={exporting ? <CircularProgress size={20} /> : <FileDownload />}
          >
            {exporting ? "Exporting..." : "Export & Create Dataset"}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );

  function renderColumnSelector(tab: TableTab) {
    const toggleTable = (tableId: string) => {
      const newExpanded = new Set(expandedTables);
      if (newExpanded.has(tableId)) {
        newExpanded.delete(tableId);
      } else {
        newExpanded.add(tableId);
      }
      setExpandedTables(newExpanded);
    };

    const toggleColumn = (columnId: string) => {
      const newTabs = [...openTabs];
      const currentTab = newTabs[activeTabIndex];

      if (currentTab.activeColumns.includes(columnId)) {
        currentTab.activeColumns = currentTab.activeColumns.filter(c => c !== columnId);
      } else {
        currentTab.activeColumns = [...currentTab.activeColumns, columnId];
      }

      setOpenTabs(newTabs);
      // TODO: Refetch data with new columns
    };

    const renderTable = (table: any, level: number): React.ReactNode => {
      const tableName = Object.keys(table)[0];
      const tableData = table[tableName];
      const tableId = `parent_level_${level}^-^${tableName}`;
      const isExpanded = expandedTables.has(tableId);

      if (!tableData?.fields || tableData.fields.length === 0) return null;

      return (
        <Box key={tableId} sx={{ mb: 2 }}>
          <ListItemButton
            onClick={() => toggleTable(tableId)}
            sx={{
              bgcolor: 'grey.50',
              borderRadius: 1,
              mb: 1,
            }}
          >
            <ListItemIcon sx={{ minWidth: 36 }}>
              {isExpanded ? <ExpandMore /> : <ChevronRight />}
            </ListItemIcon>
            <ListItemText
              primary={tableName}
              secondary={`${tableData.fields.length} columns`}
            />
          </ListItemButton>

          <Collapse in={isExpanded}>
            <List disablePadding sx={{ pl: 4 }}>
              {tableData.fields.map((field: any) => {
                const columnId = `level_${level}^-^${tableName}.${field.column_name}`;
                const isChecked = tab.activeColumns.includes(columnId);

                return (
                  <ListItem key={columnId} disablePadding sx={{ mb: 0.5 }}>
                    <ListItemButton
                      onClick={() => toggleColumn(columnId)}
                      sx={{
                        borderRadius: 1,
                        py: 1,
                      }}
                    >
                      <ListItemText primary={field.column_name} />
                      <Checkbox
                        edge="end"
                        checked={isChecked}
                        tabIndex={-1}
                        disableRipple
                      />
                    </ListItemButton>
                  </ListItem>
                );
              })}
            </List>

            {tableData.relations && tableData.relations.length > 0 && (
              <Box sx={{ pl: 2, mt: 1 }}>
                <Divider sx={{ mb: 1 }} />
                {tableData.relations.map((relation: any) => renderTable(relation, level + 1))}
              </Box>
            )}
          </Collapse>
        </Box>
      );
    };

    return (
      <List disablePadding>
        {renderTable({ [tab.table]: tab.joins }, 1)}
      </List>
    );
  }
}
