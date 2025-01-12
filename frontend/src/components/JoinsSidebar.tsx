import { Box } from "@mui/material";

import { useAppContext } from "../providers/useAppContext";

import { SimpleTreeView } from '@mui/x-tree-view/SimpleTreeView';
import { TreeItem } from '@mui/x-tree-view/TreeItem';
import { useState } from "react";
import { v4 as uuid } from "uuid";

function renderTreeItems(table, level: number, itemsCollector: string[]){
    const schema_name = Object.keys(table)[0]
    const table_fields = table[schema_name]
    const rootId = `parent_level_${level}^-^${schema_name}`
    const childIds: string[] = []
    itemsCollector.push(rootId)
    if(table_fields?.fields && table_fields.fields.length > 0){
        table_fields.fields.forEach((row)=>{
            childIds.push(`level_${level}^-^${schema_name}^-^${row.column_name}`)
        })
    }
    itemsCollector.push(...childIds)

    return table_fields?.fields && table_fields.fields.length > 0 ?
    (
        <TreeItem itemId={rootId} label={schema_name} >
        { table_fields.fields.map((row, index: number) => (
            <TreeItem key={childIds[index]} itemId={childIds[index]} label={row.column_name}/>
        ))}
            { table_fields?.relations && table_fields.relations.length > 0 ? table_fields.relations.map((row) => (
                renderTreeItems(row, level+1, itemsCollector)
            )) : null }
        </TreeItem>
    ) : null
}

export default function JoinsSidebar() {
    const { activeTab, tabs } = useAppContext();
    const tab = tabs ? tabs[activeTab] : null;
    const [selectedItems, setSelectedItems] = useState<string[]>([])
    const treeItemsCollector: string[] = []

    const handleChecks = (event: any, checkedItems: string[]) => {
        const newSelectedParent = checkedItems.filter((item: string)=> selectedItems.length === 0 || (!selectedItems.includes(item) && item.includes("parent")))
        const newDeselectedParent = selectedItems.filter((item: string)=> (!checkedItems.includes(item) && item.includes("parent")))

        if (newSelectedParent.length > 0){
            const selectedItemsPattern = newSelectedParent[0].replace("parent_", "")
            const itemsToCheck = treeItemsCollector.filter((item: string)=> item.includes(selectedItemsPattern))
            const updatedSelectedItems = [...selectedItems,...itemsToCheck]
            setSelectedItems(updatedSelectedItems)
        }
        else if (newDeselectedParent.length > 0) {
            const deselectedItemsPattern = newDeselectedParent[0].replace("parent_", "")
            const itemsToUncheck = selectedItems.filter((item: string)=> !item.includes(deselectedItemsPattern))
            setSelectedItems([...itemsToUncheck])
        }
        else {
            setSelectedItems([...checkedItems])
        }
    }

    return tab ? (
        <Box sx={{ border: "1px solid gray", maxWidth: tab.openedColumns ? 0 : "20%", minHeight: "100%" }}>
            <Box sx={{ overflow: "hidden", minHeight: "100%", transition: "max-width 0.5s ease-in-out" }}>
                <SimpleTreeView
                    multiSelect
                    checkboxSelection
                    onSelectedItemsChange={handleChecks}
                    selectedItems={selectedItems}
                >
                    {
                        renderTreeItems(
                            { [tab.table]: tab.joins }, 1, treeItemsCollector
                        )
                    }
                </SimpleTreeView>
            </Box>
        </Box>
    ) : null;
}
