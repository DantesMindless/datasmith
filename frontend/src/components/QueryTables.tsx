import React, { useEffect, useState, useMemo, useCallback } from "react";
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TableSortLabel,
  Button
} from "@mui/material";
import { queryTab, getJoins } from "../utils/requests";
import { useAppContext } from "../providers/useAppContext";
import JoinsSidebar from "./JoinsSidebar";
import KeyboardDoubleArrowRightIcon from '@mui/icons-material/KeyboardDoubleArrowRight';
import { debounce } from '../utils/debounce';
import CircularProgress from '@mui/material/CircularProgress';
type Order = "asc" | "desc";

export default function DynamicTable() {

  const { activeTab, tabs, setTabs } = useAppContext();
  const [order, setOrder] = useState<Order>("asc");
  const [orderBy, setOrderBy] = useState<string>("");
  const [data, setData] = useState<any[]>([]);
  const [headers, setHeaders] = useState<string[]>([]);
  const [isNewTab, setIsNewTab] = useState(false);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [expandedColumns, setExpandedColumns] = useState<Set<string>>(new Set());
  //const tab = tabs && activeTab != null ? tabs[activeTab] : null
  //const [hasMore, setHasMore] = useState(true);

  const [isScrollLoading, setIsScrollLoading] = useState(false);
  const [cnt, setCnt] = useState(0);

  const [tableHeight, setTableHeight] = useState('700px');

  const getQueryTableCellStyles = (header: string, isHeader: boolean = false) => ({
    ...(expandedColumns.has(header) 
      ? { width: 'auto', maxWidth: 'none' }
      : { width: '150px', maxWidth: '150px' }
    ),
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

  const getIndexCellStyles = (isHeader: boolean = false) => ({
    width: '50px',
    minWidth: '50px', 
    maxWidth: '50px',
    padding: '6px',
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

  const fetchData = async (tabData, updateHeaderOnly: boolean = false) => {
    if (tabs[activeTab].scrollState.newTab) {
      setData([]);
    }
    const result = await queryTab(tabData);
    if (result) {
      if (result?.length > 0) {
        const tableHeaders = Object.keys(result[0]).filter((item) => !item.includes("total_rows_number"))
        setHeaders(tableHeaders);
        console.log("* (1.1) * HeaderUpdated");
        if (!updateHeaderOnly) {
          setData(data.length === 0 || (data[data.length - 1].toString() === result[result.length - 1].toString()) ? [...result] : [...data, ...result]);
          console.log("* (1.1) * Data Updated data_size=", data.length);
          if (tabs[activeTab].scrollState.newTab) {
            setIsNewTab(true);
          }
        } else {
          console.log("* (1.1) * Data Bypassed");
        }

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
    if (isNewTab && tabs[activeTab].scrollState.newTab) {
      console.log("* (2) * copy data to new tab");
      tabs[activeTab].data = [...data];
      console.log("* (2) * data_size=", tabs[activeTab].data.length);
      console.log("* (2) * tab.data_size=", tabs[activeTab].data.length);
      
      console.log("* (2) * tscrollState.newTab=true");
      tabs[activeTab].scrollState.newTab = false;
      console.log("* (2) * tscrollState.newTab=false");
      setScrollPosition(0);
    }
    setIsNewTab(false);
  }, [isNewTab===true])

  const setScrollPosition = (position: number) => {
    const tableContainer = document.querySelector('.table-container');
    if (tableContainer) {
      tableContainer.scrollTop = position;
    }
  }

  useEffect(() => {
    console.log("* (1) * ACTIVE TAB = ", activeTab);
    const tab = tabs[activeTab];
    if (! tab.scrollState.newTab) {
      console.log("* (1) * oldTab");
      const tableHeaders = Object.keys(tab.data[0]).filter((item) => !item.includes("total_rows_number"))
      setScrollPosition(tab.scrollState.scrollTop);
      setHeaders(tableHeaders);
      setData(tab.data);
      
      
    } else {
      console.log("* (1) * newTab");
      setData([]);
      if (tabs && tab) {
        (async () => {
          await fetchJoins(tab);
          await fetchData(tab);
          setTabs([...tabs]);
        })();
      }
      setData([]);
      console.log("* (1) * fill new tab with data, size=", data.length);
    }
    setCnt(0);
  }, [activeTab])

  useEffect(() => {
    const tab = tabs[activeTab];
    console.log("tab?.activeColumns=", tab?.activeColumns)
    fetchData(tab, true);
  //}, [tab?.activeColumns])
}, [tabs[activeTab]?.activeColumns])

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
    const tab = tabs[activeTab];
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
        <TableRow key={i}>
          <TableCell sx={getIndexCellStyles()}>
            {i}
          </TableCell>
        </TableRow>
      )
    }
    return (
          <TableBody>
            {indexes}
          </TableBody>
    )
  }

  //const [isFetching, setIsFetching] = useState(false);
  const handleScroll = async (event: React.UIEvent<HTMLDivElement>) => {
    if (isScrollLoading) {
     return;
    }
    //setIsFetching(true);
    setIsScrollLoading(true);
    const { scrollTop, scrollHeight, clientHeight } = event.currentTarget;
    const tab = tabs[activeTab];
    tab.scrollState.scrollTop = scrollTop;
    if (tabs && activeTab !== null) {
      

      console.log("SCROLL: tab.data.length=", tab.data.length, "tab.allDataLoaded=", tab.scrollState.allDataLoaded);
      console.log("SCROLL: data.length=", data.length);
      // load more data
      setCnt(cnt + 1);
      if (!tab.scrollState.allDataLoaded && scrollTop + clientHeight >= scrollHeight * 0.8) {

        //console.log("expanding page=yes");
        tab.page = Math.floor(tab.data.length / tab.perPage) + 1;
        //console.log("expanding page=", tab.page)
        const newData = await queryTab(tab);
        //console.log("sliced data=", newData)
        if (newData.length > 0) {
          tab.data = [...tab.data, ...newData];
          setData(tab.data);
          setTabs([...tabs]);
          if (newData.length < tab.perPage) {
            tab.scrollState.allDataLoaded = true;
          }
        } else {
          tab.scrollState.allDataLoaded = true;
        }
      } else {
        //console.log("expanding page=no");
      }

    }
    setIsScrollLoading(false);
    //setIsFetching(false);
  };

  const renderIndexesMemo = useMemo(() => renderIndexes(), [data]);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'row', width: "100%"}}>
      {tabs != null && tabs[activeTab] ? (
        <>
          <JoinsSidebar tab={tabs[activeTab]} tabs={tabs} setTabs={setTabs} openedColumns={tabs[activeTab].openedColumns} />
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
              border: "1px solid gray",
              paddingBottom: '50px'
            }}
            onScroll={handleScroll}
          >
            {/* The table with indexes */}
            <Table size="small" sx={{ width: 'auto', flexShrink: 0 }}>
              <TableHead> {/*Arrow button*/}
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
              {renderIndexesMemo}
            </Table>

            {/* Query Table */}
            <Table aria-labelledby="tableTitle" size="small">
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
                    {headers.map((header) => (
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
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        ) :
        <Box sx={{display: "flex", width: "100%", border: "1px solid gray"}}>
          {tabs[activeTab] && tabs[activeTab].activeColumns.length == 0 ? (
            <h3 style={{ textAlign: 'center', width: "100%" }}>Select Columns</h3>
          ) : (
            <h3 style={{ textAlign: 'center', width: "100%" }}>No Data Found</h3>
          )}
        </Box>
      }
    </Box>
  );
}