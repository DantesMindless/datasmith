import React, { useEffect, useState, useMemo } from "react";
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
  Button
} from "@mui/material";
import { queryTab, getJoins } from "../utils/requests";
import { useAppContext } from "../providers/useAppContext";
import JoinsSidebar from "./JoinsSidebar";
import KeyboardDoubleArrowRightIcon from '@mui/icons-material/KeyboardDoubleArrowRight';


type Order = "asc" | "desc";

export default function DynamicTable() {

  const { activeTab, tabs, setTabs } = useAppContext();
  const [order, setOrder] = useState<Order>("asc");
  const [orderBy, setOrderBy] = useState<string>("");
  const [data, setData] = useState<any[]>([]);
  const [headers, setHeaders] = useState<string[]>([]);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [expandedColumns, setExpandedColumns] = useState<Set<string>>(new Set());
  const tab = tabs && activeTab != null ? tabs[activeTab] : null

  const getQueryTableCellStyles = (header: string) => ({
    maxWidth: expandedColumns.has(header) ? 'none' : '150px',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    border: "1px solid gray"
  });

  function setColumnsIds(table, level: number, itemsCollector: string[]) {
    console.log("table")
    console.log(table)

    const schema_name = Object.keys(table)[0]
    console.log("schema_name")
    console.log(schema_name)
    const table_fields = table[schema_name]
    console.log("table_fields")
    console.log(table_fields)

    console.log("table")
    console.log(table)

    const rootId = `parent_level_${level}^-^${schema_name}`
    const childIds: string[] = []
    itemsCollector.push(rootId)
    console.log(table_fields?.fields)

    if (table_fields?.fields && table_fields.fields.length > 0) {
      table_fields.fields.forEach((row) => {
        childIds.push(`level_${level}^-^${schema_name}.${row.column_name}`)
      })
      itemsCollector.push(...childIds)
      // table_fields.relations.forEach((row) => {
      //   setColumnsIds(row, level + 1, itemsCollector)
      // })
    }
  }

  useEffect(() => {
    if (tabs && tab) {
      const fetchData = async () => {
        const result = await queryTab(tab);
        if (result) {
          if (result?.length > 0){
            setHeaders(Object.keys(result[0]));
            setData(data.length === 0 || (data[data.length - 1].toString() === result[result.length - 1].toString()) ? [...result] : [...data, ...result]);
          } else {
            setData([])
          }
        }
      };
      const fetchJoins = async () => {
        const result = await getJoins(tab);
        tab.joins = result
        setColumnsIds({[tab.table]: tab.joins}, 1, tab.columns)
        tab.activeColumns = tab.columns.filter((item)=>item.includes("level_1"))
      };
      fetchData();
      fetchJoins();
      setTabs([...tabs])

    }
  }, [activeTab])

  const handleOpenColumns = () => {
    if (tabs && tab && activeTab !=null ) {
      tab.openedColumns = tab.openedColumns ? false : true
      setTabs([...tabs])
    }
  }

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

  const handleColumnClick = async (header: string) => {
    setExpandedColumns(prev => {
      const newSet = new Set(prev);
      if (newSet.has(header)) {
        newSet.delete(header);
      } else {
        newSet.add(header);
      }
      return newSet;
    });
  };

  const visibleRows = data

  const renderIndexes = () => {
    const indexes = []
    for (let i = 1; i <= data.length; i++) {
      indexes.push(
        <TableRow>
          <TableCell align="center">
            {i}
          </TableCell>
        </TableRow>
      )
    }
    return (
      <TableContainer>
        <Table size="small">
          <TableBody>
            {indexes}
          </TableBody>
        </Table>
      </TableContainer>
    )
  }

  const renderIndexesMemo = useMemo(() => renderIndexes(), [data]);
  return (
    <Box sx={{ display: 'flex', flexDirection: 'row', width: "100%" }}>
      {tabs != null && tab ? (
        <>
          <JoinsSidebar tab={tab} tabs={tabs} setTabs={setTabs} openedColumns={tab.openedColumns}/>
          <Box sx={{ display: 'flex', flexDirection: "column", justifyContent: 'flex-start', border: "1px solid gray" }}>
            <Button onClick={handleOpenColumns} sx={{ height: "37px" }}>
              <KeyboardDoubleArrowRightIcon
                sx={{ rotate: tab?.openedColumns ? "180deg" : "0deg", transition: "rotate 0.1s ease-in-out" }}
              />
            </Button>
            {renderIndexesMemo}
          </Box>
        </>
      ) : activeTab}
      {data.length > 0 ?
      (<Box sx={{ overflowX: 'scroll' }}>
        <TableContainer>
          <Table aria-labelledby="tableTitle" size="small">
            {/* Dynamic Table Header */}
            <TableHead>
              <TableRow>
                {headers.map((header) => (
                  <TableCell
                    key={header}
                    sortDirection={orderBy === header ? order : false}
                    sx={getQueryTableCellStyles(header)}
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
                    <TableCell
                      key={header}
                      sx={getQueryTableCellStyles(header)}
                      title={row[header]}
                      onClick={(e) => {
                        handleColumnClick(header);
                      }}
                    >
                      {row[header]}
                    </TableCell>
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
      </Box>):
      <Box>
      "No Data Found"
      </Box>
      }
    </Box>
  );
}
