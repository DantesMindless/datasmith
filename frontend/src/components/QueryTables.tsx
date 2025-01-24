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

  const [tableHeight, setTableHeight] = useState('700px');

  const getQueryTableCellStyles = (header: string, isHeader: boolean = false) => ({
    ...(expandedColumns.has(header) 
      ? { width: 'auto', maxWidth: 'none' }
      : { width: '150px', maxWidth: '150px' }
    ),
    height: '53px',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    border: "1px solid gray",
    backgroundColor: isHeader ? "#f5f5f5" : "inherit",
    position: isHeader ? "sticky" : "inherit",
    top: isHeader ? 0 : "inherit",
    zIndex: isHeader ? 1 : "inherit",
    cursor: 'pointer'
  });

  // const getIndexCellStyles = (isHeader: boolean = false) => ({
  //   width: '50px',
  //   minWidth: '50px',
  //   maxWidth: '50px',
  //   padding: '6px',
  //   textAlign: 'center',
  //   border: "1px solid gray",
  //   backgroundColor: isHeader ? "#f5f5f5" : "#f5f5f5",
  //   position: isHeader ? "sticky" : "inherit",
  //   top: isHeader ? 0 : "inherit",
  //   zIndex: isHeader ? 1 : "inherit",
  // });

  const getIndexCellStyles = (isHeader: boolean = false) => ({
    width: '50px',
    minWidth: '50px', 
    maxWidth: '50px',
    padding: '6px',
    height: '53px', // Добавьте фиксированную высоту, совпадающую с основной таблицей
    textAlign: 'center',
    border: "1px solid gray",
    backgroundColor: isHeader ? "#f5f5f5" : "#f5f5f5",
    position: isHeader ? "sticky" : "inherit",
    top: isHeader ? 0 : "inherit",
    zIndex: isHeader ? 1 : "inherit",
  });

  function setColumnsIds(table, level: number, itemsCollector: string[]) {
    const schema_name = Object.keys(table)[0]
    const table_fields = table[schema_name]
    const rootId = `parent_level_${level}^-^${schema_name}`
    const childIds: string[] = []
    itemsCollector.push(rootId)

    if (table_fields?.fields && table_fields.fields.length > 0) {
      table_fields.fields.forEach((row) => {
        childIds.push(`level_${level}^-^${schema_name}.${row.column_name}`)
      })
      itemsCollector.push(...childIds)
      table_fields.relations.forEach((row) => {
        setColumnsIds(row, level + 1, itemsCollector)
      })
    }
  }

  const fetchData = async (tabData) => {
    const result = await queryTab(tabData);
    if (result) {
      if (result?.length > 0) {
        const tableHeaders = Object.keys(result[0]).filter((item) => !item.includes("total_rows_number"))
        setHeaders(tableHeaders);
        setData(data.length === 0 || (data[data.length - 1].toString() === result[result.length - 1].toString()) ? [...result] : [...data, ...result]);
      } else {
        setData([])
      }
      return
    }
  };

  const fetchJoins = async (tabData) => {
    const result = await getJoins(tabData);
    tabData.joins = result
    setColumnsIds({ [tabData.table]: tabData.joins }, 1, tabData.columns)
    tabData.activeColumns = tabData.columns.filter((item) => item.includes("level_1"))
    return
  };

  useEffect(() => {

    if (tabs && tab) {
      (async () => {
        await fetchJoins(tab);
        await fetchData(tab);
        setTabs([...tabs]);
      })();
    }
  }, [activeTab])

  useEffect(() => {
    console.log(tab?.activeColumns)
    fetchData(tab);
  }, [tab?.activeColumns])

  const calculateTableHeight = () => {
    const windowHeight = window.innerHeight;
    const heightWithMargin = windowHeight * 0.85;
    setTableHeight(`${heightWithMargin}px`);
  };

  useEffect(() => {//calculate table height
    calculateTableHeight();
    const handleResize = () => {
      calculateTableHeight();
    };
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  const handleOpenColumns = () => {
    if (tabs && tab && activeTab != null) {
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

  // const renderIndexes = () => {
  //   const indexes = headers.map((_, index) => (
  //     <TableRow key={index}>
  //       <TableCell align="center">
  //         {index + 1}
  //       </TableCell>
  //     </TableRow>
  //   ));
  
  //   return (
  //     <TableContainer>
  //       <Table size="small">
  //         <TableBody>
  //           {indexes}
  //         </TableBody>
  //       </Table>
  //     </TableContainer>
  //   );
  // };

  const renderIndexesMemo = useMemo(() => renderIndexes(), [data]);
  return (
    <Box sx={{ display: 'flex', flexDirection: 'row', width: "100%"}}>
      {tabs != null && tab ? (
        <>
           <JoinsSidebar tab={tab} tabs={tabs} setTabs={setTabs} openedColumns={tab.openedColumns} />

         </>
      ) : activeTab}

      {data.length > 0 ? (
          <Box
            className="table-container"
            sx={{
              overflowX: 'auto',
              overflowY: 'auto',
              maxHeight: tableHeight,
              position: 'relative',
              width: '99%',
              display: 'flex',
              border: "1px solid gray"
            }}
            //onScroll={handleScroll}
          >
          {/* <TableContainer> */}
            {/*<Table aria-labelledby="tableTitle" size="small">*/}



            {/*<TableRow sx={{ height: '53px' }}>*/}



              {/* The table with indexes */}
              <Table size="small" sx={{ width: 'auto', flexShrink: 0 }}>
                <TableHead>
                  <TableRow>
                    <TableCell sx={getIndexCellStyles(true)}>
                      <Button 
                        onClick={handleOpenColumns}
                        sx={{
                          minWidth: 'unset',
                          padding: 0,
                          width: '100%',
                          height: '100%',
                        
                        }}
                      >
                        <KeyboardDoubleArrowRightIcon
                          sx={{
                            rotate: tabs[activeTab].openedColumns ? "0deg" : "180deg",
                            transition: "rotate 0.1s ease-in-out"
                          }}
                        />
                      </Button>
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {data.map((_, index) => (
                    <TableRow key={index}>
                      <TableCell sx={getIndexCellStyles()}>
                        {index + 1}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
   {/*             <TableHead>
   <TableRow>
    <TableCell sx={getIndexCellStyles(true)}>
      <Button onClick={handleOpenColumns}>
        <KeyboardDoubleArrowRightIcon />
      </Button>
    </TableCell>
  </TableRow>
</TableHead>
<TableBody>
  {data.map((_, index) => (
    <TableRow key={index}>
      <TableCell sx={getIndexCellStyles()}>
        {index + 1}
      </TableCell>
    </TableRow>
  ))}
</TableBody> */}
              </Table>











              <Table size="small">
                {/* Static Table Header */}
                <TableHead>
                  <TableRow>
                    {headers.map((header) => (
                      <TableCell
                        key={header}
                        sortDirection={orderBy === header ? order : false}
                        sx={{...getQueryTableCellStyles(header, true), fontWeight: "bold"}}
                        onClick={() => handleColumnClick(header)}
                      >
                        <TableSortLabel
                          active={orderBy === header}
                          direction={orderBy === header ? order : "asc"}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRequestSort(header);
                          }}
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
                      {/*{headers.map((header) => (
                        <TableCell
                          key={header}
                          sx={getQueryTableCellStyles(header)}
                          title={row[header]}
                          onClick={(e) => {
                            handleColumnClick(header);
                          }}
                        >
                          {row[header] != null ? row[header] : "NULL"}
                        </TableCell>
                      ))}*/}
                      {headers.map((header) => {
                      const cellContent = row[header]?.toString() || '';
                      return (
                        <TableCell
                          //key={`${index}-${header}`}
                          key={header}
                          sx={getQueryTableCellStyles(header)}
                          title={cellContent || undefined}
                          //title={row[header]}
                          onClick={() => handleColumnClick(header)}
                        >
                          {cellContent}
                        </TableCell>
                      );
                    })}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>

















              
            {/*</Table>*/}
          {/* </TableContainer> */}
          </Box>
        ) :
        <Box sx={{display: "flex", width: "100%", border: "1px solid gray"}}>
          {tab && tab.activeColumns.length == 0 ? (
            <h3 style={{ textAlign: 'center', width: "100%" }}>Select Columns</h3>
          ) : (
            <h3 style={{ textAlign: 'center', width: "100%" }}>No Data Found</h3>
          )}
        </Box>
      }
    </Box>
  );
}
