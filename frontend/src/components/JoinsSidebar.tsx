import { Box, Typography, Checkbox, List, ListItem, ListItemButton, ListItemIcon, ListItemText, Collapse, Divider } from "@mui/material";
import { ExpandMore, ChevronRight, TableChart, ViewColumn } from "@mui/icons-material";
import { useState } from "react";

export default function JoinsSidebar({
  tab,
  tabs,
  setTabs,
  openedColumns: boolean
}) {
  const [expandedTables, setExpandedTables] = useState<Set<string>>(new Set());

  const handleChecks = (event: any, checkedItems: string[]) => {
    if (tab && tabs) {
      const newSelectedParent = checkedItems.filter(
        (item: string) =>
          tab?.activeColumns.length === 0 ||
          (!tab.activeColumns.includes(item) && item.includes("parent"))
      );
      const newDeselectedParent = tab.activeColumns.filter(
        (item: string) =>
          !checkedItems.includes(item) && item.includes("parent")
      );
      if (newSelectedParent.length > 0) {
        const selectedItemsPattern = newSelectedParent[0].replace(
          "parent_",
          ""
        );
        const itemsToCheck = tab.columns.filter((item: string) =>
          item.includes(selectedItemsPattern)
        );
        const updatedSelectedItems = [...tab.activeColumns, ...itemsToCheck];
        tab.activeColumns = updatedSelectedItems;
        setTabs([...tabs]);
      } else if (newDeselectedParent.length > 0) {
        const deselectedItemsPattern = newDeselectedParent[0].replace(
          "parent_",
          ""
        );
        const itemsToUncheck: string[] = tab.activeColumns.filter(
          (item: string) => !item.includes(deselectedItemsPattern)
        );
        tab.activeColumns = [...itemsToUncheck];
        setTabs([...tabs]);
      } else {
        tab.activeColumns = [...checkedItems];
        setTabs([...tabs]);
      }
    }
  };

  const toggleTableExpansion = (tableId: string) => {
    const newExpanded = new Set(expandedTables);
    if (newExpanded.has(tableId)) {
      newExpanded.delete(tableId);
    } else {
      newExpanded.add(tableId);
    }
    setExpandedTables(newExpanded);
  };

  const handleTableCheck = (tableId: string) => {
    if (!tab || !tabs) return;

    const pattern = tableId.replace("parent_", "");
    const relatedColumns = tab.columns.filter((item: string) => item.includes(pattern));

    const allSelected = relatedColumns.every((col: string) => tab.activeColumns.includes(col));

    if (allSelected) {
      // Deselect all columns from this table
      tab.activeColumns = tab.activeColumns.filter((col: string) => !relatedColumns.includes(col));
    } else {
      // Select all columns from this table
      const newColumns = [...new Set([...tab.activeColumns, ...relatedColumns])];
      tab.activeColumns = newColumns;
    }
    setTabs([...tabs]);
  };

  const handleColumnCheck = (columnId: string) => {
    if (!tab || !tabs) return;

    if (tab.activeColumns.includes(columnId)) {
      tab.activeColumns = tab.activeColumns.filter((col: string) => col !== columnId);
    } else {
      tab.activeColumns = [...tab.activeColumns, columnId];
    }
    setTabs([...tabs]);
  };

  const renderTable = (table: any, level: number) => {
    const schema_name = Object.keys(table)[0];
    const table_fields = table[schema_name];
    const tableId = `parent_level_${level}^-^${schema_name}`;
    const isExpanded = expandedTables.has(tableId);

    if (!table_fields?.fields || table_fields.fields.length === 0) return null;

    const pattern = tableId.replace("parent_", "");
    const relatedColumns = tab?.columns.filter((item: string) => item.includes(pattern)) || [];
    const allSelected = relatedColumns.length > 0 && relatedColumns.every((col: string) => tab?.activeColumns.includes(col));
    const someSelected = relatedColumns.some((col: string) => tab?.activeColumns.includes(col));

    return (
      <Box key={tableId} sx={{ mb: 1 }}>
        {/* Table Header */}
        <ListItemButton
          sx={{
            py: 2,
            px: 2,
            borderRadius: 1,
            bgcolor: level === 1 ? 'primary.50' : 'grey.50',
            mb: 0.5,
            '&:hover': {
              bgcolor: level === 1 ? 'primary.100' : 'grey.100',
            }
          }}
        >
          <ListItemIcon
            sx={{ minWidth: 36 }}
            onClick={(e) => {
              e.stopPropagation();
              toggleTableExpansion(tableId);
            }}
          >
            {isExpanded ? <ExpandMore /> : <ChevronRight />}
          </ListItemIcon>

          <ListItemIcon sx={{ minWidth: 36 }}>
            <TableChart color={level === 1 ? "primary" : "action"} />
          </ListItemIcon>

          <ListItemText
            primary={
              <Typography variant="body1" fontWeight={600}>
                {schema_name}
              </Typography>
            }
            secondary={
              <Typography variant="caption" color="text.secondary">
                {table_fields.fields.length} columns
              </Typography>
            }
          />

          <Checkbox
            edge="end"
            checked={allSelected}
            indeterminate={someSelected && !allSelected}
            onChange={(e) => {
              e.stopPropagation();
              handleTableCheck(tableId);
            }}
            onClick={(e) => e.stopPropagation()}
          />
        </ListItemButton>

        {/* Columns List */}
        <Collapse in={isExpanded} timeout="auto" unmountOnExit>
          <Box sx={{ pl: level === 1 ? 6 : 4, pr: 2, py: 1 }}>
            <List disablePadding>
              {table_fields.fields.map((field: any) => {
                const columnId = `level_${level}^-^${schema_name}.${field.column_name}`;
                const isChecked = tab?.activeColumns.includes(columnId);

                return (
                  <ListItem
                    key={columnId}
                    disablePadding
                    sx={{ mb: 0.5 }}
                  >
                    <ListItemButton
                      onClick={() => handleColumnCheck(columnId)}
                      sx={{
                        py: 1.5,
                        px: 2,
                        borderRadius: 1,
                        '&:hover': {
                          bgcolor: 'action.hover',
                        }
                      }}
                    >
                      <ListItemIcon sx={{ minWidth: 36 }}>
                        <ViewColumn fontSize="small" color="action" />
                      </ListItemIcon>

                      <ListItemText
                        primary={
                          <Typography variant="body2">
                            {field.column_name}
                          </Typography>
                        }
                      />

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

            {/* Related Tables */}
            {table_fields?.relations && table_fields.relations.length > 0 && (
              <Box sx={{ mt: 2 }}>
                <Divider sx={{ mb: 2 }} />
                <Typography variant="caption" color="text.secondary" sx={{ pl: 2, display: 'block', mb: 1 }}>
                  Related Tables
                </Typography>
                {table_fields.relations.map((relation: any) => renderTable(relation, level + 1))}
              </Box>
            )}
          </Box>
        </Collapse>
      </Box>
    );
  };

  return tab ? (
    <Box
      sx={{
        width: tab.openedColumns ? 380 : 0,
        minHeight: "100%",
        transition: "width 0.3s ease-in-out",
        borderRight: tab.openedColumns ? 1 : 0,
        borderColor: "divider",
        bgcolor: "background.paper",
        overflow: "hidden",
      }}
    >
      {tab.openedColumns && (
        <Box sx={{ height: "100%", display: 'flex', flexDirection: 'column' }}>
          {/* Header */}
          <Box sx={{ p: 3, borderBottom: 1, borderColor: 'divider', bgcolor: 'grey.50' }}>
            <Typography variant="h6" fontWeight={600} gutterBottom>
              Select Columns
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Choose columns to display in the table
            </Typography>
          </Box>

          {/* Content */}
          <Box sx={{ flex: 1, overflowY: "auto", p: 3 }}>
            <List disablePadding>
              {renderTable({ [tab.table]: tab.joins }, 1)}
            </List>
          </Box>
        </Box>
      )}
    </Box>
  ) : null;
}
