import React, { useEffect, useState, useMemo} from "react";
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TableSortLabel,
  Button,
} from "@mui/material";
import { queryTab, getJoins } from "../utils/requests";
import { useAppContext } from "../providers/useAppContext";
import JoinsSidebar from "./JoinsSidebar";
import KeyboardDoubleArrowRightIcon from '@mui/icons-material/KeyboardDoubleArrowRight';
import { TableViewTab, FilterOperator, Filter } from "../providers/constants";
import { combineFilters, getColumnType } from "./FilterForm";
import SQLFilterForm from "./FilterForm";
type Order = "asc" | "desc";

let detectedColumnTypes: { [key: string]: string } = {};

export default function DynamicTable() {

  const { activeTab, tabs, setTabs } = useAppContext();
  const [order, setOrder] = useState<Order>("asc");
  const [orderBy, setOrderBy] = useState<string>("");
  const [data, setData] = useState<any[]>([]);
  const [headers, setHeaders] = useState<string[]>([]);
  const [dataToTabCopyRequest, setDataToTabCopyRequest] = useState(false);
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
    if (data.length > 0) {
      const keys = Object.keys(data[0]);
      keys.forEach(key => {
        detectedColumnTypes[key] = getColumnType(data[0][key]);
      });
    }
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
      detectedColumnTypes = {};//reset column types
      setSearchTerms({});//reset search terms
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

  



  const handleFilterChange = (header: string, clause: string) => {
    const newFilters = { ...searchTerms };
    if (clause) {
      newFilters[header] = {
        clause: clause,
        column: header,
      };
      console.log("header =", header, "clause =", clause);
    } else {
      delete newFilters[header];
    }
    console.log("newFilters =", newFilters);
    tabs[activeTab].filter = combineFilters(Object.values(newFilters));
    console.log("fetching data");
    fetchData(tabs[activeTab]);
    setSearchTerms(newFilters);
  };

  const renderIndexesMemo = useMemo(() => renderIndexes(), [data]);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'row', width: "100%"}}>
      {tabs != null && tabs[activeTab] ? (
        <>
          <JoinsSidebar tab={tabs[activeTab]} tabs={tabs} setTabs={setTabs} openedColumns={tabs[activeTab].openedColumns} />
        </>
      ) : activeTab}

      {headers.length > 0 ? (
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
                  <TableCell sx={{...getIndexCellStyles(true), height: '118px'}}>
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
                        <SQLFilterForm
                          fieldName={typeof header !== 'object' ? header : ''}
                          fieldType={detectedColumnTypes[header] || 'string'}
                          onFilterChange={(clause) => handleFilterChange(header, clause)}
                        />
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              {/* Dynamic Table Body */}
              { data.length > 0 ? (
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
              )
              : (
                <TableBody>
                  <TableRow>
                    <TableCell colSpan={headers.length} sx={{textAlign: "center"}}>No Data Found</TableCell>
                  </TableRow>
                </TableBody>
              )}








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
