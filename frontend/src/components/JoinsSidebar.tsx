import { Box } from "@mui/material";

import { useAppContext } from "../providers/useAppContext";

import { SimpleTreeView } from '@mui/x-tree-view/SimpleTreeView';
import { TreeItem } from '@mui/x-tree-view/TreeItem';
import { v4 as uuid } from "uuid";

function renderTreeItems(table){
    const schema_name = Object.keys(table)[0]
    const table_fields = table[schema_name]
    return table_fields?.fields && table_fields.fields.length > 0 ?
    (
        <TreeItem itemId={uuid() + schema_name} label={schema_name}>
        { table_fields.fields.map((row) => (
            <TreeItem key={schema_name + row.column_name} itemId={uuid() + schema_name + row.column_name} label={row.column_name}/>
        ))}
            { table_fields?.relations && table_fields.relations.length > 0 ? table_fields.relations.map((row) => (
                renderTreeItems(row)
            )) : null }
        </TreeItem>
    ) : null
}

export default function JoinsSidebar() {
    const { activeTab, tabs } = useAppContext();
    const tab = tabs ? tabs[activeTab] : null;

    return tab ? (
        <Box sx={{ border: "1px solid gray", maxWidth: tab.openedColumns ? 0 : "20%", minHeight: "100%" }}>
            <Box sx={{ overflow: "hidden", minHeight: "100%", transition: "max-width 0.5s ease-in-out" }}>
                <SimpleTreeView
                    multiSelect
                    checkboxSelection // Handle selection changes
                >
                    {
                        renderTreeItems(
                            { [tab.table]: tab.joins }
                        )
                    }
                </SimpleTreeView>
            </Box>
        </Box>
    ) : null;
}
