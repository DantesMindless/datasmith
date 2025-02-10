import React, { useEffect, useState, useMemo} from "react";
import {
  Box,
  Table,
  TableBody,
  TableCell,
  //TableContainer,//unused
  TableHead,
  //TablePagination,//unused
  TableRow,
  TableSortLabel,
  Button,
  TextField,
  Select,
  MenuItem
} from "@mui/material";
import { queryTab, getJoins } from "../utils/requests";
import { useAppContext } from "../providers/useAppContext";
import JoinsSidebar from "./JoinsSidebar";
import KeyboardDoubleArrowRightIcon from '@mui/icons-material/KeyboardDoubleArrowRight';
import { TableViewTab, FilterOperator, Filter } from "../providers/constants";
import { combineFilters } from "./Filters";
type Order = "asc" | "desc";

export default function DynamicTable() {

  const { activeTab, tabs, setTabs } = useAppContext();
  const [order, setOrder] = useState<Order>("asc");
  const [orderBy, setOrderBy] = useState<string>("");
  const [data, setData] = useState<any[]>([]);
  const [headers, setHeaders] = useState<string[]>([]);
  const [dataToTabCopyRequest, setDataToTabCopyRequest] = useState(false);
  // const [page, setPage] = useState(0);//unused
  // const [rowsPerPage, setRowsPerPage] = useState(10);//unused
  const [expandedColumns, setExpandedColumns] = useState<Set<string>>(new Set());

  const [isScrollLoading, setIsScrollLoading] = useState(false);
  const [postponedScrollUpdate, setPostponedScrollUpdate] = useState(false);

  const [tableHeight, setTableHeight] = useState('700px');
  const [searchTerms, setSearchTerms] = useState<{ [key: string]: Filter }>({});

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

  const fetchData = async (tabData : TableViewTab, updateHeaderOnly: boolean = false) => {
    const result = await queryTab(tabData);
    if (result) {
      if (result?.length > 0) {
        const tableHeaders = Object.keys(result[0]).filter((item) => !item.includes("total_rows_number"))
        setHeaders(tableHeaders);
        if (!updateHeaderOnly) {
          //setData(data.length === 0 || (data[data.length - 1].toString() === result[result.length - 1].toString()) ? [...result] : [...data, ...result]);
          setData([...result]);
          if (tabs[activeTab].scrollState.newTab) {
            setDataToTabCopyRequest(true);
          }
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
    if (dataToTabCopyRequest && tabs[activeTab].scrollState.newTab) {
      tabs[activeTab].data = [...data];
      tabs[activeTab].scrollState.newTab = false;
      setScrollPosition(0);
    }
    setDataToTabCopyRequest(false);
  }, [dataToTabCopyRequest===true])

  let bypassHandleScroll = false;

  const setScrollPosition = (position: number) => {
    bypassHandleScroll = true;
    const tableContainer = document.querySelector('.table-container');
    if (tableContainer) {
      tableContainer.scrollTop = position;
    }
  }

  useEffect(() => {
    const tab = tabs[activeTab];
    if (tab.scrollState.newTab) {
      if (tabs && tab) {
        (async () => {
          await fetchJoins(tab);
          await fetchData(tab);
          setTabs([...tabs]);
        })();
      }
    } else {
      setScrollPosition(0);
      const tableHeaders = Object.keys(tab.data[0]).filter((item) => !item.includes("total_rows_number"))
      setHeaders(tableHeaders);
      setData(tab.data);
      setPostponedScrollUpdate(true);
    }
  }, [activeTab])

  useEffect(() => {
    const tab = tabs[activeTab];
    if (postponedScrollUpdate===true) {
      setScrollPosition(tab.scrollState.scrollTop);
    }
    setPostponedScrollUpdate(false);
  }, [postponedScrollUpdate===true])

  useEffect(() => {
    const tab = tabs[activeTab];
    fetchData(tab, true);
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

  const handleScroll = async (event: React.UIEvent<HTMLDivElement>) => {
    if (isScrollLoading) {//to prevent double loading
     return;
    }
    if (bypassHandleScroll) {//in case of manual set scrollPosition
      bypassHandleScroll = false;
      return;
    }
    setIsScrollLoading(true);
    const { scrollTop, scrollHeight, clientHeight } = event.currentTarget;
    const tab = tabs[activeTab];
    tab.scrollState.scrollTop = scrollTop;
    if (tabs && activeTab !== null) {
      if (!tab.scrollState.allDataLoaded && scrollTop + clientHeight >= scrollHeight * 0.8) {// load more data
        tab.page = Math.floor(tab.data.length / tab.perPage) + 1;
        const newData = await queryTab(tab);
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
      } 
    }
    setIsScrollLoading(false);
  };

  const handleSearchChange = (header: string, value: string) => {
    const columnType = getColumnType(data[0][header]);
    let typeDefaultOperator: FilterOperator = FilterOperator.EQUALS;
    switch (columnType) {
      case "number":
        typeDefaultOperator = FilterOperator.EQUALS;
        break;
      case "boolean":
        typeDefaultOperator = FilterOperator.EQUALS;
        break;
      default:
        typeDefaultOperator = FilterOperator.CONTAINS;
    }
    const filter: Filter = {
      value: value,
      operator: typeDefaultOperator,
      field: header,
      type: columnType
    };

    const newFilters = {
      ...searchTerms,
      [header]: filter
    };
  
    const whereClause = combineFilters(Object.values(newFilters));
    
    tabs[activeTab].filter = whereClause;

    fetchData(tabs[activeTab]);

    // if (tabs && activeTab !== null) {
    //   const newTabs = [...tabs];
    //   newTabs[activeTab] = {
    //     ...newTabs[activeTab],
    //     filter: whereClause
    //   };
    //   setTabs(newTabs);
    // }
  
    setSearchTerms(newFilters);

  };

  const getColumnType = (value: string | number | boolean | Date) => {
    if (value === null || value === undefined) return "string";
    if (typeof value === "number") return "number";
    if (typeof value === "boolean") return "boolean";
    const dateRegex = /^\d{4}[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])$/;
    if (typeof value === "string" && dateRegex.test(value)) {
      return "date";
    }
    return "string";
  };
  
  const renderSearchField = (header: string) => {
    const type = getColumnType(data[0][header]);
    const currentValue = searchTerms[header]?.value || "";
    switch (type) {
      case "number":
        return (
          <TextField
            type="number"
            size="small"
            variant="outlined"
            fullWidth
            placeholder="Search..."
            value={currentValue}
            onChange={(e) => handleSearchChange(header, e.target.value)}
          />
        );
      case "boolean":
        return (
          <Select
            size="small"
            variant="outlined"
            fullWidth
            value={currentValue}
            onChange={(e) => handleSearchChange(header, e.target.value)}
          >
            <MenuItem value="">All</MenuItem>
            <MenuItem value="true">True</MenuItem>
            <MenuItem value="false">False</MenuItem>
          </Select>
        );
      case "date":
        return (
          <TextField
            type="date"
            size="small"
            variant="outlined"
            fullWidth
            value={currentValue}
            onChange={(e) => handleSearchChange(header, e.target.value)}
          />
        );
      default:
        return (
          <TextField
            size="small"
            variant="outlined"
            fullWidth
            placeholder="Search..."
            value={currentValue}
            onChange={(e) => handleSearchChange(header, e.target.value)}
          />
        );
    }
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

                <TableRow>
                  {headers.map((header) => (
                    <TableCell key={`search-${header}`}
                      sx={{'& .MuiInputBase-root': {
                          height: '30px', 
                          fontSize: '12px',
                        },
                        '& .MuiInputBase-input': {padding: '5px',},
                        border: "1px solid gray",
                        backgroundColor: "#f5f5f5",
                        position: "sticky",
                        top: 0,
                        zIndex: 1,
                      }}>
                        {renderSearchField(header)}
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
