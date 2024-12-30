import * as React from "react";
import { useState, useEffect } from "react";
import Box from "@mui/joy/Box";
import Button from "@mui/joy/Button";
import Divider from "@mui/joy/Divider";
import FormControl from "@mui/joy/FormControl";
import FormLabel from "@mui/joy/FormLabel";
import Input from "@mui/joy/Input";
import Modal from "@mui/joy/Modal";
import ModalDialog from "@mui/joy/ModalDialog";
import ModalClose from "@mui/joy/ModalClose";
import Select from "@mui/joy/Select";
import Option from "@mui/joy/Option";
import Table from "@mui/joy/Table";
import Sheet from "@mui/joy/Sheet";
import IconButton from "@mui/joy/IconButton";
import Typography from "@mui/joy/Typography";
import Menu from "@mui/joy/Menu";
import MenuButton from "@mui/joy/MenuButton";
import MenuItem from "@mui/joy/MenuItem";
import Dropdown from "@mui/joy/Dropdown";

import FilterAltIcon from "@mui/icons-material/FilterAlt";
import SearchIcon from "@mui/icons-material/Search";
import MoreHorizRoundedIcon from "@mui/icons-material/MoreHorizRounded";
import { v4 as uuidv4 } from "uuid";

const exludeFields = ["id"];

import { useAppContext } from "../providers/useAppContext";

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
  orderBy: Key,
): (
  a: { [key in Key]: number | string },
  b: { [key in Key]: number | string },
) => number {
  return order === "desc"
    ? (a, b) => descendingComparator(a, b, orderBy)
    : (a, b) => -descendingComparator(a, b, orderBy);
}

function RowMenu() {
  return (
    <Dropdown>
      <MenuButton
        slots={{ root: IconButton }}
        slotProps={{ root: { variant: "plain", color: "neutral", size: "sm" } }}
      >
        <MoreHorizRoundedIcon />
      </MenuButton>
      <Menu size="sm" sx={{ minWidth: 140 }}>
        <MenuItem>Edit</MenuItem>
        <MenuItem>Rename</MenuItem>
        <MenuItem>Move</MenuItem>
        <Divider />
        <MenuItem color="danger">Delete</MenuItem>
      </Menu>
    </Dropdown>
  );
}
export default function OrderTable() {
  const [open, setOpen] = React.useState(false);
  const renderFilters = () => (
    <React.Fragment>
      <FormControl size="sm">
        <FormLabel>Type</FormLabel>
        <Select size="sm" placeholder="All">
          <Option value="all">All</Option>
          <Option value="olivia">Olivia Rhye</Option>
          <Option value="steve">Steve Hampton</Option>
          <Option value="ciaran">Ciaran Murray</Option>
          <Option value="marina">Marina Macdonald</Option>
          <Option value="charles">Charles Fulton</Option>
          <Option value="jay">Jay Hoper</Option>
        </Select>
      </FormControl>
    </React.Fragment>
  );
  const [skipIndexes, setSkipIndexes] = useState([]);
  const [headers, setHeaders] = useState([]);
  const { connections } = useAppContext();

  useEffect(() => {
    let isMounted = true;
    if (isMounted) {
      const tableHeaders:Record<string, string>[] = [];
      const indexesToSkip:number[] = [];
      if (connections && connections.length > 0){
        console.log(connections[0])
      Object.keys(connections[0]).forEach((key, index: number) => {
        if (exludeFields.includes(key)) {
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
    return () => {
      isMounted = false;
    };
  }}, [connections]);

  return (
    <React.Fragment>
      <Sheet
        className="SearchAndFilters-mobile"
        sx={{ display: { xs: "flex", sm: "none" }, my: 1, gap: 1 }}
      >
        <Input
          size="sm"
          placeholder="Search"
          startDecorator={<SearchIcon />}
          sx={{ flexGrow: 1 }}
        />
        <IconButton
          size="sm"
          variant="outlined"
          color="neutral"
          onClick={() => setOpen(true)}
        >
          <FilterAltIcon />
        </IconButton>
        <Modal open={open} onClose={() => setOpen(false)}>
          <ModalDialog aria-labelledby="filter-modal" layout="fullscreen">
            <ModalClose />
            <Typography id="filter-modal" level="h2">
              Filters
            </Typography>
            <Divider sx={{ my: 2 }} />
            <Sheet sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
              {renderFilters()}
              <Button color="primary" onClick={() => setOpen(false)}>
                Submit
              </Button>
            </Sheet>
          </ModalDialog>
        </Modal>
      </Sheet>
      <Box
        className="SearchAndFilters-tabletUp"
        sx={{
          borderRadius: "sm",
          py: 2,
          display: { xs: "none", sm: "flex" },
          flexWrap: "wrap",
          gap: 1.5,
          "& > *": {
            minWidth: { xs: "120px", md: "160px" },
          },
        }}
      >
        <FormControl sx={{ flex: 1 }} size="sm">
          <FormLabel>Search for order</FormLabel>
          <Input
            size="sm"
            placeholder="Search"
            startDecorator={<SearchIcon />}
          />
        </FormControl>
        {renderFilters()}
      </Box>
      <Sheet
        className="OrderTableContainer"
        variant="outlined"
        sx={{
          display: { xs: "none", sm: "initial" },
          width: "100%",
          borderRadius: "sm",
          flexShrink: 1,
          overflow: "auto",
          minHeight: 0,
        }}
      >
        <Table
          aria-labelledby="tableTitle"
          stickyHeader
          hoverRow
          sx={{
            "--TableCell-headBackground":
              "var(--joy-palette-background-level1)",
            "--Table-headerUnderlineThickness": "1px",
            "--TableRow-hoverBackground":
              "var(--joy-palette-background-level1)",
            "--TableCell-paddingY": "4px",
            "--TableCell-paddingX": "8px",
          }}
        >
          <thead>
            <tr key={uuidv4()}>
              {headers.map((header, index) =>
                !skipIndexes.includes(index) ? (
                  <th
                    key={uuidv4()}
                    style={{
                      width: 40,
                      textAlign: "center",
                      padding: "12px 6px",
                    }}
                  >
                    {header.title}
                  </th>
                ) : (
                  ""
                ),
              )}
              <th
                key={uuidv4()}
                style={{
                  width: 40,
                  textAlign: "center",
                  padding: "12px 6px",
                }}
              >
                Actions
              </th>
            </tr>
          </thead>
          <tbody>
            {connections ? connections.map((row, rowIndex) => {
              const rowData = Object.values(row).map((element, colIndex) => {
                if (skipIndexes.includes(colIndex)) {
                  return;
                }
                let value;
                if (typeof element === "string") {
                  // Check if string can be parsed as a valid date
                  if (
                    element.includes("2024-") === true &&
                    element.includes(":") === true
                  ) {
                    const parsedDate = Date.parse(element);
                    if (!isNaN(parsedDate)) {
                      // Format as a human-readable date
                      value = new Date(parsedDate).toLocaleDateString();
                    } else {
                      value = element; // Return the string as-is
                    }
                  } else {
                    value = element;
                  }
                } else if (typeof element === "boolean") {
                  value = element ? "Yes" : "No"; // Handle booleans
                } else if (typeof element === "object") {
                  value = " "; // Handle objects (return space)
                } else {
                  value = element; // Default case for other types (e.g., numbers)
                }
                return (
                  <td key={colIndex} style={{ textAlign: "center" }}>
                    {value}
                  </td>
                ); // Render the value in a table cell
              });
              return (
                <tr key={uuidv4()}>
                  {rowData}
                  <td>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        width: "100%",
                      }}
                    >
                      <Button size="sm" variant="solid" color="primary">
                        View
                      </Button>
                      <Button size="sm" variant="solid" color="danger">
                        Delete
                      </Button>
                    </div>
                  </td>
                </tr>
              ); // Wrap the rowData in a <tr>
            }): ""}
          </tbody>
        </Table>
      </Sheet>
    </React.Fragment>
  );
}
