import React, { useEffect, useState } from "react";
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TablePagination,
  TableRow,
  TableSortLabel,
  Paper,
} from "@mui/material";
import { queryTab } from "../utils/requests";
import { useAppContext } from "../providers/useAppContext";

type Order = "asc" | "desc";

export default function DynamicTable() {
  const [order, setOrder] = useState<Order>("asc");
  const [orderBy, setOrderBy] = useState<string>("");
  const [data, setData] = useState<any[]>([]);
  const [headers, setHeaders] = useState<string[]>([]);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  const { activeTab, tabs } = useAppContext();

  // Fetch table data
  useEffect(() => {
    const fetchData = async () => {
      const result = await queryTab(tabs[activeTab]);
      if (result && result.length > 0) {
        setHeaders(Object.keys(result[0])); // Extract headers dynamically from the first row
        setData(result);
      }
    };

    fetchData();
  }, [activeTab, tabs]);

  const handleRequestSort = (property: string) => {
    const isAsc = orderBy === property && order === "asc";
    setOrder(isAsc ? "desc" : "asc");
    setOrderBy(property);
    setData((prevData) =>
      [...prevData].sort((a, b) =>
        isAsc
          ? a[property] > b[property]
            ? -1
            : 1
          : a[property] < b[property]
          ? -1
          : 1
      )
    );
  };

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const visibleRows = data.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);

  return (
    <Box sx={{ width: "100%" }}>
      <Paper sx={{ width: "100%", mb: 2 }}>
        <TableContainer>
          <Table sx={{ minWidth: 750 }} aria-labelledby="tableTitle" size="medium">
            {/* Dynamic Table Header */}
            <TableHead>
              <TableRow>
                {headers.map((header) => (
                  <TableCell
                    key={header}
                    sortDirection={orderBy === header ? order : false}
                  >
                    <TableSortLabel
                      active={orderBy === header}
                      direction={orderBy === header ? order : "asc"}
                      onClick={() => handleRequestSort(header)}
                    >
                      {header}
                    </TableSortLabel>
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>

            {/* Dynamic Table Body */}
            <TableBody>
              {visibleRows.map((row, index) => (
                <TableRow hover tabIndex={-1} key={index}>
                  {headers.map((header) => (
                    <TableCell key={header}>{row[header]}</TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          rowsPerPageOptions={[5, 10, 25]}
          component="div"
          count={data.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
        />
      </Paper>
    </Box>
  );
}
