import { Box } from "@mui/material";

import { useAppContext } from "../providers/useAppContext";

import { SimpleTreeView } from "@mui/x-tree-view/SimpleTreeView";
import { TreeItem } from "@mui/x-tree-view/TreeItem";
import { useMemo } from "react";

function renderTreeItems(table, level: number) {
  const schema_name = Object.keys(table)[0];
  const table_fields = table[schema_name];
  const rootId = `parent_level_${level}^-^${schema_name}`;

  return table_fields?.fields && table_fields.fields.length > 0 ? (
    <TreeItem itemId={rootId} label={schema_name}>
      {table_fields.fields.map((row) => (
        <TreeItem
          key={`level_${level}^-^${schema_name}.${row.column_name}`}
          itemId={`level_${level}^-^${schema_name}.${row.column_name}`}
          label={row.column_name}
        />
      ))}
      {table_fields?.relations && table_fields.relations.length > 0
        ? table_fields.relations.map((row) => renderTreeItems(row, level + 1))
        : null}
    </TreeItem>
  ) : null;
}

export default function JoinsSidebar({
  tab,
  tabs,
  setTabs,
  openedColumns: boolean
}) {
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

  return tab ? (
    <Box
      sx={{
        maxWidth: tab.openedColumns ? "15%" : 0,
        minHeight: "100%",
        transition: "max-width 0.2s ease-in-out",
        borderTop: "1px solid gray",
        borderLeft: "1px solid gray",
        borderBottom: "1px solid gray"
      }}
    >
      <Box sx={{ overflow: "hidden", minHeight: "100%", overflowX: "scroll" }}>
        <SimpleTreeView
          multiSelect
          checkboxSelection
          onSelectedItemsChange={handleChecks}
          selectedItems={tab.activeColumns}
        >
          {renderTreeItems({ [tab.table]: tab.joins }, 1)}
        </SimpleTreeView>
      </Box>
    </Box>
  ) : null;
}
