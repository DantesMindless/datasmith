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
import { CircularProgress } from "@mui/material";


type Order = "asc" | "desc";

export default function DynamicTable() {
  const [order, setOrder] = useState<Order>("asc");
  const [orderBy, setOrderBy] = useState<string>("");
  const [data, setData] = useState<any[]>([]);
  const [headers, setHeaders] = useState<string[]>([]);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const { activeTab, tabs, setTabs } = useAppContext();
  const [expandedColumns, setExpandedColumns] = useState<Set<string>>(new Set());
//
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true); // Флаг, есть ли ещё данные

//
  const getQueryTableCellStyles = (header: string) => ({
    maxWidth: expandedColumns.has(header) ? 'none' : '150px',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    border: "1px solid gray"
  });

  // Fetch table data
  useEffect(() => {
    if (tabs){
      const fetchData = async () => {
        const result = await queryTab(tabs[activeTab]);
        if (result && result.length > 0) {
          setHeaders(Object.keys(result[0])); // Extract headers dynamically from the first row
          setData(data.length === 0 || (data[data.length - 1].toString() === result[result.length - 1].toString()) ? [...result]: [...data, ...result]);
        }
      };
      fetchData();
    }
  }, [activeTab, tabs]);

  useEffect(()=>{
    if(tabs){
    const tab = tabs[activeTab]
    const fetchData = async () => {
      const result = await getJoins(tab);
      tabs[activeTab].joins = result
      setTabs([...tabs])
    };
    if(tab.joins.length == 0){
      fetchData();
    }
    }
  }, [activeTab, tabs, setTabs])

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
//
  const handleScroll = async (event: React.UIEvent<HTMLDivElement>) => {
    const { scrollTop, scrollHeight, clientHeight } = event.currentTarget;
  
    if (!loading && hasMore && scrollTop + clientHeight >= scrollHeight - 10) {
      setLoading(true);
      const result = await queryTab(tabs[activeTab], data.length); // OFFSET равен текущему количеству строк
      if (result.length > 0) {
        setData((prevData) => [...prevData, ...result]);
      } else {
        setHasMore(false); // Нет больше данных
      }
      setLoading(false);
    }
  };
//
  const visibleRows = data

  const renderIndexes = () => {
    const indexes = []
    for (let i = 1; i <= data.length; i++){
      indexes.push(
        <TableRow key={i}>
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

  const MAX_ROWS = 700;
  return (
  //   <Box sx={{ display: 'flex', flexDirection: 'row', width: "100%"}}>
  //     {tabs != null && tabs.length > 0 && activeTab != null && (
  //       <>
  //         <JoinsSidebar />
  //         <Box sx={{ display: 'flex', flexDirection: "column", justifyContent: 'flex-start', border: "1px solid gray" }}>
  //           <Button onClick={handleOpenColumns} sx={{height:"37px"}}>
  //             <KeyboardDoubleArrowRightIcon
  //               sx={{rotate: tabs[activeTab].openedColumns ? "0deg" : "180deg", transition: "rotate 0.1s ease-in-out" }}
  //             />
  //           </Button>
  //           {renderIndexesMemo}
  //         </Box>
  //       </>
  //     )}
  //     <Box sx={{overflowX:'scroll'}}>
  //       <TableContainer>
  //         <Table aria-labelledby="tableTitle" size="small">
  //           {/* Dynamic Table Header */}
  //           <TableHead>
  //             <TableRow>
  //               {headers.map((header) => (
  //                 <TableCell
  //                   key={header}
  //                   sortDirection={orderBy === header ? order : false}
  //                   sx={getQueryTableCellStyles(header)}
  //                 >
  //                   <TableSortLabel
  //                     active={orderBy === header}
  //                     direction={orderBy === header ? order : "asc"}
  //                     onClick={() => handleRequestSort(header)}
  //                     >
  //                     {header}
  //                   </TableSortLabel>
  //                 </TableCell>
  //               ))}
  //             </TableRow>
  //           </TableHead>

  //           {/* Dynamic Table Body */}
  //           <TableBody>
  //             {visibleRows.map((row, index) => (
  //               <TableRow hover tabIndex={-1} key={index}>
  //                 {headers.map((header) => (
  //                 <TableCell
  //                   key={header}
  //                   sx={getQueryTableCellStyles(header)}
  //                   title={row[header]}
  //                   onClick={(e) => {
  //                     handleColumnClick(header);
  //                   }}
  //                 >
  //                   {row[header]}
  //                 </TableCell>
  //                 ))}
  //               </TableRow>
  //             ))}
  //           </TableBody>
  //         </Table>
  //       </TableContainer>
  //       </Box>
  //       <TablePagination
  //         rowsPerPageOptions={[5, 10, 25]}
  //         component="div"
  //         count={data.length}
  //         rowsPerPage={rowsPerPage}
  //         page={page}
  //         onPageChange={handleChangePage}
  //           onRowsPerPageChange={handleChangeRowsPerPage}
  //           />
  //   </Box>
  // 


/// NO PAGINATION
//   <Box sx={{ display: 'flex', flexDirection: 'row', width: "100%"}}>
//   {tabs != null && tabs.length > 0 && activeTab != null && (
//     <>
//       <JoinsSidebar />
//       <Box sx={{ display: 'flex', flexDirection: "column", justifyContent: 'flex-start', border: "1px solid gray" }}>
//         <Button onClick={handleOpenColumns} sx={{height:"37px"}}>
//           <KeyboardDoubleArrowRightIcon
//             sx={{rotate: tabs[activeTab].openedColumns ? "0deg" : "180deg", transition: "rotate 0.1s ease-in-out" }}
//           />
//         </Button>
//         {renderIndexesMemo}
//       </Box>
//     </>
//   )}
//   <Box sx={{ overflowX: 'scroll', overflowY: 'auto', maxHeight: `${MAX_ROWS}px`, width: '100%' }}>
//     <TableContainer>
//       <Table aria-labelledby="tableTitle" size="small">
//         {/* Dynamic Table Header */}
//         <TableHead>
//           <TableRow>
//             {headers.map((header) => (
//               <TableCell
//                 key={header}
//                 sortDirection={orderBy === header ? order : false}
//                 sx={getQueryTableCellStyles(header)}
//               >
//                 <TableSortLabel
//                   active={orderBy === header}
//                   direction={orderBy === header ? order : "asc"}
//                   onClick={() => handleRequestSort(header)}
//                 >
//                   {header}
//                 </TableSortLabel>
//               </TableCell>
//             ))}
//           </TableRow>
//         </TableHead>

//         {/* Dynamic Table Body */}
//         <TableBody>
//           {data.map((row, index) => (
//             <TableRow hover tabIndex={-1} key={index}>
//               {headers.map((header) => (
//                 <TableCell
//                   key={header}
//                   sx={getQueryTableCellStyles(header)}
//                   title={row[header]}
//                   onClick={() => handleColumnClick(header)}
//                 >
//                   {row[header]}
//                 </TableCell>
//               ))}
//             </TableRow>
//           ))}
//         </TableBody>
//       </Table>
//     </TableContainer>
//   </Box>
// </Box>


<Box sx={{ display: 'flex', flexDirection: 'row', width: "100%"}}>
{tabs != null && tabs.length > 0 && activeTab != null && (
  <>
    <JoinsSidebar />
    <Box sx={{ display: 'flex', flexDirection: "column", justifyContent: 'flex-start', border: "1px solid gray" }}>
      <Button onClick={handleOpenColumns} sx={{height:"37px"}}>
        <KeyboardDoubleArrowRightIcon
          sx={{rotate: tabs[activeTab].openedColumns ? "0deg" : "180deg", transition: "rotate 0.1s ease-in-out" }}
        />
      </Button>
      {renderIndexesMemo}
    </Box>
  </>
)}


<Box
  sx={{overflowX: 'scroll', overflowY: 'auto', maxHeight: '400px', width: '100%' }}
  onScroll={handleScroll}
>
  <TableContainer>
    <Table aria-labelledby="tableTitle" size="small">
      {/* Dynamic Table Header */}
      <TableHead>
        <TableRow>
          {headers.map((header) => (
            <TableCell
              key={header}
              sortDirection={orderBy === header ? order : false}
              sx={{
                ...getQueryTableCellStyles(header),
                position: "sticky",
                top: 0,
                backgroundColor: "gray", // Чтобы заголовок был виден на фоне
                zIndex: 1,
              }}
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
        {data.map((row, index) => (
          <TableRow hover tabIndex={-1} key={index}>
            {headers.map((header) => (
              <TableCell
                //key={header}
                key={`${index}-${header}`} // Уникальный ключ для каждой ячейки
                sx={getQueryTableCellStyles(header)}
                title={row[header]}
                onClick={() => handleColumnClick(header)}
              >
                {row[header]}
              </TableCell>
            ))}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  </TableContainer>
  {loading && (
    <Box sx={{ textAlign: 'center', padding: '10px' }}>
      <CircularProgress size={24} />
    </Box>
  )}
</Box>

</Box>
  );
}
