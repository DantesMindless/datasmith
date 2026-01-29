import * as React from "react";
import { useState, useEffect } from "react";
import {
  Box,
  Button,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  Menu,
  MenuList,
  MenuItem as MenuItemMaterial,
  Dialog,
  DialogTitle,
  DialogContent,
  CircularProgress,
} from "@mui/material";
import MoreHorizRoundedIcon from "@mui/icons-material/MoreHorizRounded";
import EditIcon from "@mui/icons-material/Edit";
import { v4 as uuidv4 } from "uuid";

const excludeFields = ["id", "schemas"];

import { useAppContext } from "../providers/useAppContext";
import httpfetch from "../utils/axios";
import { AxiosResponse } from "axios";
import CreateConnection, { EditConnectionData } from "./CreateConnection";

function descendingComparator<T>(a: T, b: T, orderBy: keyof T) {
  if (b[orderBy] < a[orderBy]) {
    return -1;
  }
  if (b[orderBy] > a[orderBy]) {
    return 1;
  }
  return 0;
}

type Order = "asc" | "desc";

function getComparator<Key extends keyof any>(
  order: Order,
  orderBy: Key
): (a: { [key in Key]: number | string }, b: { [key in Key]: number | string }) => number {
  return order === "desc"
    ? (a, b) => descendingComparator(a, b, orderBy)
    : (a, b) => -descendingComparator(a, b, orderBy);
}

function RowMenu() {
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const open = Boolean(anchorEl);

  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  return (
    <>
      <IconButton
        onClick={handleClick}
        size="small"
        color="default"
        aria-label="more options"
      >
        <MoreHorizRoundedIcon />
      </IconButton>
      <Menu anchorEl={anchorEl} open={open} onClose={handleClose}>
        <MenuList>
          <MenuItemMaterial onClick={handleClose}>Edit</MenuItemMaterial>
          <MenuItemMaterial onClick={handleClose}>Rename</MenuItemMaterial>
          <MenuItemMaterial onClick={handleClose}>Move</MenuItemMaterial>
          <Divider />
          <MenuItemMaterial onClick={handleClose} style={{ color: "red" }}>
            Delete
          </MenuItemMaterial>
        </MenuList>
      </Menu>
    </>
  );
}

export default function OrderTable() {
  const [open, setOpen] = React.useState(false);
  const [skipIndexes, setSkipIndexes] = useState<number[]>([]);
  const [headers, setHeaders] = useState<{ title: string }[]>([]);
  const [loading, setLoading] = useState(false);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editData, setEditData] = useState<EditConnectionData | null>(null);
  const [loadingEdit, setLoadingEdit] = useState(false);
  const { connections, showAlert, showInfo, updateConnections } = useAppContext();

  const renderFilters = () => (
    <React.Fragment>
      <FormControl fullWidth size="small">
        <InputLabel>Type</InputLabel>
        <Select size="small" defaultValue="all">
          <MenuItem value="all">All</MenuItem>
          <MenuItem value="olivia">Olivia Rhye</MenuItem>
          <MenuItem value="steve">Steve Hampton</MenuItem>
          <MenuItem value="ciaran">Ciaran Murray</MenuItem>
          <MenuItem value="marina">Marina Macdonald</MenuItem>
          <MenuItem value="charles">Charles Fulton</MenuItem>
          <MenuItem value="jay">Jay Hoper</MenuItem>
        </Select>
      </FormControl>
    </React.Fragment>
  );

  const handleDelete = async (id: string) => {
    try {
      setLoading(true);
      const response: AxiosResponse = await httpfetch.delete(
        `datasource/detail/${id}/`
      );
      if (response.status === 204 || response.status === 200) {
        await updateConnections();
        showInfo("DataSource deleted successfully");
      };
    } catch (error: any) {
      console.error("Error details:", error.response?.data || error.message);
      showAlert(`Failed to delete DataSource: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleEdit = async (id: string) => {
    try {
      setLoadingEdit(true);
      const response: AxiosResponse = await httpfetch.get(
        `datasource/detail/${id}/?edit=true`
      );
      if (response.status === 200) {
        setEditData(response.data);
        setEditDialogOpen(true);
      }
    } catch (error: any) {
      console.error("Error fetching connection details:", error.response?.data || error.message);
      showAlert(`Failed to fetch connection details: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoadingEdit(false);
    }
  };

  const handleEditClose = () => {
    setEditDialogOpen(false);
    setEditData(null);
    updateConnections();
  };

  useEffect(() => {
    if (connections && connections.length > 0) {
      const tableHeaders: Record<string, string>[] = [];
      const indexesToSkip: number[] = [];
      Object.keys(connections[0]).forEach((key, index: number) => {
        if (excludeFields.includes(key)) {
          indexesToSkip.push(index);
        }
        const transformedKey = key
          .split("_")
          .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
          .join(" ");
        tableHeaders.push({ title: transformedKey });
      });
      setSkipIndexes(indexesToSkip);
      setHeaders(tableHeaders);
    }
  }, [connections]);

  return (
    <>
      <Box
        sx={{
          border: "1px solid",
          borderColor: "divider",
          borderRadius: 2,
          overflow: "hidden",
          bgcolor: "background.paper",
          boxShadow: "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)"
        }}
      >
        <Table stickyHeader>
          <TableHead>
            <TableRow>
              {headers.map((header, index) =>
                !skipIndexes.includes(index) ? (
                  <TableCell key={`table_cell${index}`} align="center">
                    {header.title}
                  </TableCell>
                ) : null
              )}
              <TableCell key={"actions"} align="center">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {connections &&
              connections.map((row, rowIndex) => {
                const rowData = Object.values(row).map((element, colIndex) => {
                  if (skipIndexes.includes(colIndex)) {
                    return null;
                  }
                  // Check if element is an ISO date string (e.g., 2024-11-16T07:50:27Z)
                  let value = element;
                  if (typeof element === "string" && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/.test(element)) {
                    const date = new Date(element);
                    value = date.toLocaleString();
                  }
                  return <TableCell key={uuidv4()} align="center">{value}</TableCell>;
                });
                return (
                  <TableRow key={uuidv4()}>
                    {rowData}
                    <TableCell key={uuidv4()}>
                      <Box display="flex" gap={1} justifyContent="center">
                        <Button
                          size="small"
                          variant="outlined"
                          color="primary"
                          disabled={loading || loadingEdit}
                          onClick={() => handleEdit(row.id)}
                          startIcon={loadingEdit ? <CircularProgress size={16} /> : <EditIcon />}
                          sx={{
                            borderRadius: 2,
                            textTransform: "none",
                            fontSize: "0.75rem"
                          }}
                        >
                          Edit
                        </Button>
                        <Button
                          size="small"
                          variant="outlined"
                          color="error"
                          disabled={loading || loadingEdit}
                          onClick={() => handleDelete(row.id)}
                          sx={{
                            borderRadius: 2,
                            textTransform: "none",
                            fontSize: "0.75rem"
                          }}
                        >
                          Delete
                        </Button>
                      </Box>
                    </TableCell>
                  </TableRow>
                );
              })}
          </TableBody>
        </Table>
      </Box>

      {/* Edit Connection Dialog */}
      <Dialog
        open={editDialogOpen}
        onClose={handleEditClose}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Edit Connection</DialogTitle>
        <DialogContent>
          {editData && (
            <CreateConnection
              editData={editData}
              onClose={handleEditClose}
            />
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
