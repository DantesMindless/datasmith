import React, { useEffect, useState} from "react";
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

type Order = "asc" | "desc";

export default function DynamicTable() {
  const [order, setOrder] = useState<Order>("asc");
  const [orderBy, setOrderBy] = useState<string>("");
  const [data, setData] = useState<any[]>([]);
  const [headers, setHeaders] = useState<string[]>([]);
  const { activeTab, tabs, setTabs } = useAppContext();
  const [expandedColumns, setExpandedColumns] = useState<Set<string>>(new Set());
  const [hasMore, setHasMore] = useState(true);
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

  const calculateTableHeight = () => {
    const windowHeight = window.innerHeight;
    const heightWithMargin = windowHeight * 0.85;
    setTableHeight(`${heightWithMargin}px`);
  };
  
  useEffect(() => {
    if (tabs && activeTab !== null) {
      const tab = tabs[activeTab];
  
      const fetchData = async () => {
        // first load
        if (tab.new) {
          const initialData = await queryTab(tab);
          setHeaders(Object.keys(initialData[0] || {}));
          setData(initialData);
          tab.new = false;
          tab.loadedRows = initialData.length;
          tab.data = initialData;
          setTabs([...tabs]);
        } else {
          // if data is partially loaded
          setHeaders(Object.keys(tab.data[0] || {}));
          setData(tab.data);
  
          // load more data if less than before
          if (tab.data.length < tab.loadedRows) {
            const additionalData = await queryTab(tab, tab.data.length);
            tab.data = [...tab.data, ...additionalData];
            tab.loadedRows = tab.data.length;
            setData(tab.data);
            setTabs([...tabs]);
          }
        }
      };
  
      fetchData();
    }
  }, [activeTab, tabs]); // load data after tab change

  useEffect(() => {
    if (tabs && activeTab !== null) {
      const tab = tabs[activeTab];
      const fetchJoins = async () => {
        if (tab.joins.length === 0) {
          const result = await getJoins(tab);
          tab.joins = result;
          setTabs([...tabs]);
        }
      };
      fetchJoins();
    }
  }, [activeTab]);//, tabs]); // load joins after tab change

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

  useEffect(() => {
    if (tabs && activeTab !== null) {
      const tab = tabs[activeTab];
      const container = document.querySelector(".table-container");
  
      if (container) {
        container.scrollTop = tab.scrollPosition || 0;
      }
    }
  }, [activeTab, tabs]); // scroll restore after tab change

  const handleOpenColumns = () => {
    if(tabs){
      const tab = tabs[activeTab]
      tab.openedColumns = tab.openedColumns ? false : true
      tabs[activeTab] = tab
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

  const handleScroll = async (event: React.UIEvent<HTMLDivElement>) => {
    const { scrollTop, scrollHeight, clientHeight } = event.currentTarget;
  
    if (tabs && activeTab !== null) {
      const tab = tabs[activeTab];
      tab.scrollPosition = scrollTop;
    
      // load more data
      if (hasMore && scrollTop + clientHeight >= scrollHeight - 100) {
        const newData = await queryTab(tab, tab.data.length);
        if (newData.length > 0) {
          tab.data = [...tab.data, ...newData];
          tab.loadedRows = tab.data.length;
          setData(tab.data);
          setTabs([...tabs]);
        } else {
          setHasMore(false);
        }
      }
    }
  };
    
  return (
      <Box sx={{ display: 'flex', flexDirection: 'row', width: "100%"}}>
      {tabs != null && tabs.length > 0 && activeTab != null && (
        <>
          <JoinsSidebar />
          <Box
            className="table-container"
            sx={{
              overflowX: 'auto',
              overflowY: 'auto',
              maxHeight: tableHeight,
              position: 'relative',
              width: '100%',
              display: 'flex',
              lexDirection: 'column',
              border: "1px solid gray"
            }}
            onScroll={handleScroll}
          >
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
            </Table>

            {/* Static Table Header */}
            <Table size="small">
              <TableHead>
                <TableRow>
                  {headers.map((header) => (
                    <TableCell
                      key={header}
                      sortDirection={orderBy === header ? order : false}
                      sx={getQueryTableCellStyles(header, true)}
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
                {data.map((row, index) => (
                  <TableRow hover tabIndex={-1} key={index}>
{headers.map((header) => {
                      const cellContent = row[header]?.toString() || '';
                      return (
                        <TableCell
                          key={`${index}-${header}`}
                          sx={getQueryTableCellStyles(header)}
                          title={cellContent || undefined}
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
          </Box>
        </>
      )}
    </Box>
  );
}
