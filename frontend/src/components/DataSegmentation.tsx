import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  IconButton,
  Chip,
  Grid,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Alert,
  CircularProgress,
  Stack,
  Divider,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Checkbox,
  TablePagination,
  Tooltip,
  Menu,
  MenuItem,
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Label as LabelIcon,
  FileDownload as ExportIcon,
  BarChart as StatsIcon,
  CheckCircle,
  RadioButtonUnchecked,
  MoreVert,
  AutoFixHigh as AutoIcon,
  BubbleChart as ClusterIcon,
} from '@mui/icons-material';
import {
  FormControl,
  InputLabel,
  Select,
  Slider,
  LinearProgress,
} from '@mui/material';
import { useAppContext } from '../providers/useAppContext';
import httpfetch from '../utils/axios';
import {
  getDatasetSegmentation,
  getSegments,
  createSegment,
  updateSegment,
  deleteSegment,
  getSegmentLabels,
  bulkAssignRows,
  bulkDeleteLabels,
  exportSegment,
  autoClusterDataset,
  getOptimalClusters,
} from '../utils/requests';

interface Segment {
  id: string;
  name: string;
  description: string;
  color: string;
  row_count: number;
  label_count: number;
  dataset: string;
  created_at: string;
}

interface SegmentLabel {
  id: string;
  segment: string;
  segment_name: string;
  segment_color: string;
  row_index: number;
  row_data: any;
  assigned_by_username: string;
  assignment_method: string;
  confidence: number;
}

interface DataSegmentationProps {
  datasetId: string;
  previewData: any[];
  totalRows: number;
}

const PRESET_COLORS = [
  '#3b82f6', // blue
  '#10b981', // green
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // purple
  '#ec4899', // pink
  '#14b8a6', // teal
  '#f97316', // orange
];

export default function DataSegmentation({ datasetId, previewData, totalRows }: DataSegmentationProps) {
  const { showAlert } = useAppContext();
  const [segments, setSegments] = useState<Segment[]>([]);
  const [labels, setLabels] = useState<SegmentLabel[]>([]);
  const [loading, setLoading] = useState(true);
  const [overview, setOverview] = useState<any>(null);

  // Dialog states
  const [segmentDialogOpen, setSegmentDialogOpen] = useState(false);
  const [editingSegment, setEditingSegment] = useState<Segment | null>(null);
  const [segmentName, setSegmentName] = useState('');
  const [segmentDescription, setSegmentDescription] = useState('');
  const [segmentColor, setSegmentColor] = useState(PRESET_COLORS[0]);

  // Row browser states
  const [selectedRows, setSelectedRows] = useState<Set<number>>(new Set());
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);
  const [assignMenuAnchor, setAssignMenuAnchor] = useState<null | HTMLElement>(null);
  const [activeSegment, setActiveSegment] = useState<Segment | null>(null);
  const [currentPageData, setCurrentPageData] = useState<any[]>([]);
  const [loadingRows, setLoadingRows] = useState(false);

  // Auto-cluster dialog states
  const [autoClusterDialogOpen, setAutoClusterDialogOpen] = useState(false);
  const [clusterAlgorithm, setClusterAlgorithm] = useState<'kmeans' | 'dbscan' | 'hierarchical' | 'gaussian_mixture' | 'mean_shift'>('kmeans');
  const [numClusters, setNumClusters] = useState(3);
  const [dbscanEps, setDbscanEps] = useState(0.5);
  const [dbscanMinSamples, setDbscanMinSamples] = useState(5);
  const [segmentPrefix, setSegmentPrefix] = useState('Cluster');
  const [clustering, setClustering] = useState(false);
  const [clusterResult, setClusterResult] = useState<any>(null);
  const [findingOptimal, setFindingOptimal] = useState(false);
  const [optimalK, setOptimalK] = useState<number | null>(null);

  useEffect(() => {
    loadSegmentationData();
  }, [datasetId]);

  useEffect(() => {
    fetchPageData();
  }, [datasetId, page, rowsPerPage]);

  // Fetch labels for current page when page data changes
  useEffect(() => {
    if (currentPageData.length > 0) {
      fetchLabelsForCurrentPage();
    }
  }, [currentPageData, page, rowsPerPage]);

  const fetchLabelsForCurrentPage = async () => {
    try {
      const startIdx = page * rowsPerPage;
      const rowIndices = Array.from({ length: currentPageData.length }, (_, i) => startIdx + i);
      const labelsData = await getSegmentLabels(undefined, datasetId, rowIndices);
      setLabels(labelsData || []);
    } catch (error: any) {
      console.error('Error fetching labels for page:', error);
      setLabels([]);
    }
  };

  const loadSegmentationData = async () => {
    try {
      setLoading(true);
      const [overviewData, segmentsData] = await Promise.all([
        getDatasetSegmentation(datasetId),
        getSegments(datasetId),
      ]);

      setOverview(overviewData);
      setSegments(segmentsData || []);
      // Labels will be fetched per-page by fetchLabelsForCurrentPage
    } catch (error: any) {
      console.error('Error loading segmentation data:', error);
      showAlert(error.response?.data?.error || 'Failed to load segmentation data', 'error');
      setSegments([]);
    } finally {
      setLoading(false);
    }
  };

  const fetchPageData = async () => {
    try {
      setLoadingRows(true);
      // Server-side pagination - load only the current page
      const offset = page * rowsPerPage;
      const response = await httpfetch.get(`datasets/${datasetId}/preview/`, {
        params: {
          offset: offset,
          limit: rowsPerPage,
        }
      });
      setCurrentPageData(response.data.preview_data || []);
    } catch (error: any) {
      console.error('Error fetching page data:', error);
      showAlert(error.response?.data?.error || 'Failed to load page data', 'error');
      setCurrentPageData([]);
    } finally {
      setLoadingRows(false);
    }
  };

  const handleCreateSegment = () => {
    setEditingSegment(null);
    setSegmentName('');
    setSegmentDescription('');
    setSegmentColor(PRESET_COLORS[segments.length % PRESET_COLORS.length]);
    setSegmentDialogOpen(true);
  };

  const handleEditSegment = (segment: Segment) => {
    setEditingSegment(segment);
    setSegmentName(segment.name);
    setSegmentDescription(segment.description);
    setSegmentColor(segment.color);
    setSegmentDialogOpen(true);
  };

  const handleSaveSegment = async () => {
    if (!segmentName.trim()) {
      showAlert('Segment name is required', 'error');
      return;
    }

    try {
      if (editingSegment) {
        await updateSegment(editingSegment.id, {
          name: segmentName,
          description: segmentDescription,
          color: segmentColor,
        });
        showAlert('Segment updated successfully', 'success');
      } else {
        await createSegment({
          dataset: datasetId,
          name: segmentName,
          description: segmentDescription,
          color: segmentColor,
        });
        showAlert('Segment created successfully', 'success');
      }
      setSegmentDialogOpen(false);
      loadSegmentationData();
    } catch (error: any) {
      showAlert(error.response?.data?.error || 'Failed to save segment', 'error');
    }
  };

  const handleDeleteSegment = async (segmentId: string) => {
    if (!window.confirm('Are you sure you want to delete this segment? All labels will be removed.')) {
      return;
    }

    try {
      await deleteSegment(segmentId);
      showAlert('Segment deleted successfully', 'success');
      // Refresh both segmentation data and labels for current page
      await loadSegmentationData();
      await fetchLabelsForCurrentPage();
    } catch (error: any) {
      showAlert(error.response?.data?.error || 'Failed to delete segment', 'error');
    }
  };

  const handleSelectRow = (rowIndex: number) => {
    const newSelected = new Set(selectedRows);
    if (newSelected.has(rowIndex)) {
      newSelected.delete(rowIndex);
    } else {
      newSelected.add(rowIndex);
    }
    setSelectedRows(newSelected);
  };

  const handleSelectAllRows = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.checked) {
      const start = page * rowsPerPage;
      const newSelected = new Set(selectedRows);
      // Select all rows on the current page
      for (let i = 0; i < currentPageData.length; i++) {
        newSelected.add(start + i);
      }
      setSelectedRows(newSelected);
    } else {
      setSelectedRows(new Set());
    }
  };

  const handleAssignToSegment = async (segment: Segment) => {
    if (selectedRows.size === 0) {
      showAlert('Please select at least one row', 'warning');
      return;
    }

    const rowIndices = Array.from(selectedRows);

    try {
      const result = await bulkAssignRows({
        segment_id: segment.id,
        row_indices: rowIndices,
        assignment_method: 'manual',
      });

      // Build a more informative message
      let message = `Successfully assigned ${result.total} rows to ${segment.name}`;
      if (result.moved && result.moved > 0) {
        message += ` (${result.moved} moved from other segments)`;
      }
      if (result.created > 0) {
        message += ` • ${result.created} newly lasbeled`;
      }
      if (result.updated > 0) {
        message += ` • ${result.updated} updated`;
      }

      showAlert(message, 'success');
      setSelectedRows(new Set());
      setAssignMenuAnchor(null);
      // Refresh both segmentation data and labels for current page
      await loadSegmentationData();
      await fetchLabelsForCurrentPage();
    } catch (error: any) {
      console.error('Assignment error:', error);
      showAlert(error.response?.data?.error || 'Failed to assign rows', 'error');
    }
  };

  const handleRemoveFromSegment = async (segment: Segment) => {
    if (selectedRows.size === 0) {
      showAlert('Please select at least one row', 'warning');
      return;
    }

    try {
      const result = await bulkDeleteLabels(segment.id, Array.from(selectedRows));
      showAlert(`Removed ${result.deleted} rows from ${segment.name}`, 'success');
      setSelectedRows(new Set());
      loadSegmentationData();
    } catch (error: any) {
      showAlert(error.response?.data?.error || 'Failed to remove rows', 'error');
    }
  };

  const handleExportSegment = async (segmentId: string) => {
    try {
      const result = await exportSegment(segmentId);
      showAlert(`Exported ${result.rows_exported} rows to dataset "${result.dataset_name}"`, 'success');
    } catch (error: any) {
      showAlert(error.response?.data?.error || 'Failed to export segment', 'error');
    }
  };

  const getRowSegment = (rowIndex: number): SegmentLabel | undefined => {
    return labels.find(label => label.row_index === rowIndex);
  };

  const handleOpenAutoCluster = () => {
    setClusterAlgorithm('kmeans');
    setNumClusters(3);
    setDbscanEps(0.5);
    setDbscanMinSamples(5);
    setSegmentPrefix('Cluster');
    setClusterResult(null);
    setOptimalK(null);
    setAutoClusterDialogOpen(true);
  };

  const handleFindOptimalClusters = async () => {
    setFindingOptimal(true);
    try {
      const result = await getOptimalClusters(datasetId, {
        max_clusters: 10,
        method: 'silhouette',
      });
      setOptimalK(result.optimal_k);
      setNumClusters(result.optimal_k);
      showAlert(`Optimal number of clusters: ${result.optimal_k}`, 'success');
    } catch (error: any) {
      console.error('Error finding optimal clusters:', error);
      showAlert(error.response?.data?.error || 'Failed to find optimal clusters', 'error');
    } finally {
      setFindingOptimal(false);
    }
  };

  const handleRunAutoClustering = async () => {
    setClustering(true);
    setClusterResult(null);

    try {
      const config: any = {};

      if (clusterAlgorithm === 'kmeans' || clusterAlgorithm === 'hierarchical' || clusterAlgorithm === 'gaussian_mixture') {
        config.n_clusters = numClusters;
      }
      if (clusterAlgorithm === 'dbscan') {
        config.eps = dbscanEps;
        config.min_samples = dbscanMinSamples;
      }

      const result = await autoClusterDataset(datasetId, {
        algorithm: clusterAlgorithm,
        config,
        create_segments: true,
        segment_prefix: segmentPrefix,
      });

      setClusterResult(result);
      showAlert(
        `Created ${result.n_clusters} segments with ${result.segments_created?.length || 0} clusters`,
        'success'
      );

      // Reload segmentation data and labels for current page
      await loadSegmentationData();
      await fetchLabelsForCurrentPage();
    } catch (error: any) {
      console.error('Error running auto-clustering:', error);
      showAlert(error.response?.data?.error || 'Failed to run auto-clustering', 'error');
    } finally {
      setClustering(false);
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 8 }}>
        <CircularProgress />
      </Box>
    );
  }

  // Server-side pagination - currentPageData already contains only the current page
  const paginatedData = currentPageData;

  return (
    <Box>
      {/* Overview Stats */}
      {overview && (
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom variant="body2">
                  Total Rows
                </Typography>
                <Typography variant="h4" fontWeight={600}>
                  {overview.total_rows?.toLocaleString()}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom variant="body2">
                  Labeled Rows
                </Typography>
                <Typography variant="h4" fontWeight={600} color="primary">
                  {overview.labeled_rows?.toLocaleString()}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom variant="body2">
                  Coverage
                </Typography>
                <Typography variant="h4" fontWeight={600} color="success.main">
                  {overview.coverage_percentage?.toFixed(1)}%
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom variant="body2">
                  Segments
                </Typography>
                <Typography variant="h4" fontWeight={600}>
                  {segments.length}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      <Grid container spacing={3}>
        {/* Segments Panel */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" fontWeight={600}>
                  Segments
                </Typography>
                <Stack direction="row" spacing={1}>
                  <Button
                    variant="outlined"
                    startIcon={<AutoIcon />}
                    onClick={handleOpenAutoCluster}
                    size="small"
                    color="secondary"
                  >
                    Auto-Segment
                  </Button>
                  <Button
                    variant="contained"
                    startIcon={<AddIcon />}
                    onClick={handleCreateSegment}
                    size="small"
                  >
                    New Segment
                  </Button>
                </Stack>
              </Box>

              {segments.length === 0 ? (
                <Alert severity="info">
                  No segments yet. Create your first segment to start organizing your data.
                </Alert>
              ) : (
                <List>
                  {segments.map((segment) => (
                    <ListItem
                      key={segment.id}
                      sx={{
                        border: 1,
                        borderColor: 'divider',
                        borderRadius: 1,
                        mb: 1,
                        bgcolor: activeSegment?.id === segment.id ? 'action.selected' : 'transparent',
                      }}
                    >
                      <Box
                        sx={{
                          width: 4,
                          height: 40,
                          bgcolor: segment.color,
                          borderRadius: 1,
                          mr: 2,
                        }}
                      />
                      <ListItemText
                        primary={
                          <Typography variant="body1" fontWeight={600}>
                            {segment.name}
                          </Typography>
                        }
                        secondary={
                          <Stack direction="row" spacing={1} sx={{ mt: 0.5 }}>
                            <Chip
                              label={`${segment.row_count} rows`}
                              size="small"
                              sx={{ height: 20 }}
                            />
                          </Stack>
                        }
                      />
                      <ListItemSecondaryAction>
                        <Tooltip title="Edit">
                          <IconButton size="small" onClick={() => handleEditSegment(segment)}>
                            <EditIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Export">
                          <IconButton size="small" onClick={() => handleExportSegment(segment.id)}>
                            <ExportIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete">
                          <IconButton size="small" onClick={() => handleDeleteSegment(segment.id)}>
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Row Browser Panel */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" fontWeight={600}>
                  Dataset Rows
                </Typography>
                {selectedRows.size > 0 && (
                  <Stack direction="row" spacing={1}>
                    <Chip label={`${selectedRows.size} selected`} color="primary" />
                    <Button
                      variant="outlined"
                      size="small"
                      onClick={(e) => setAssignMenuAnchor(e.currentTarget)}
                    >
                      Assign to Segment
                    </Button>
                    <Menu
                      anchorEl={assignMenuAnchor}
                      open={Boolean(assignMenuAnchor)}
                      onClose={() => setAssignMenuAnchor(null)}
                    >
                      {segments.map((segment) => (
                        <MenuItem key={segment.id} onClick={() => handleAssignToSegment(segment)}>
                          <Box sx={{ width: 12, height: 12, bgcolor: segment.color, borderRadius: '50%', mr: 1 }} />
                          {segment.name}
                        </MenuItem>
                      ))}
                    </Menu>
                  </Stack>
                )}
              </Box>

              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell padding="checkbox">
                        <Checkbox
                          indeterminate={
                            (() => {
                              const start = page * rowsPerPage;
                              const pageRowIndices = Array.from({ length: currentPageData.length }, (_, i) => start + i);
                              const selectedOnPage = pageRowIndices.filter(idx => selectedRows.has(idx)).length;
                              return selectedOnPage > 0 && selectedOnPage < currentPageData.length;
                            })()
                          }
                          checked={
                            (() => {
                              const start = page * rowsPerPage;
                              const pageRowIndices = Array.from({ length: currentPageData.length }, (_, i) => start + i);
                              return currentPageData.length > 0 && pageRowIndices.every(idx => selectedRows.has(idx));
                            })()
                          }
                          onChange={handleSelectAllRows}
                        />
                      </TableCell>
                      <TableCell>Row #</TableCell>
                      <TableCell>Segment</TableCell>
                      {currentPageData[0] && Object.keys(currentPageData[0]).map((key) => (
                        <TableCell key={key}>{key}</TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {loadingRows ? (
                      <TableRow>
                        <TableCell colSpan={100} align="center" sx={{ py: 4 }}>
                          <CircularProgress size={24} />
                        </TableCell>
                      </TableRow>
                    ) : paginatedData.map((row, idx) => {
                      const rowIndex = page * rowsPerPage + idx;
                      const rowLabel = getRowSegment(rowIndex);
                      const isSelected = selectedRows.has(rowIndex);

                      return (
                        <TableRow key={rowIndex} hover selected={isSelected}>
                          <TableCell padding="checkbox">
                            <Checkbox
                              checked={isSelected}
                              onChange={() => handleSelectRow(rowIndex)}
                            />
                          </TableCell>
                          <TableCell>{rowIndex + 1}</TableCell>
                          <TableCell>
                            {rowLabel ? (
                              <Chip
                                label={rowLabel.segment_name}
                                size="small"
                                sx={{
                                  bgcolor: rowLabel.segment_color + '20',
                                  color: rowLabel.segment_color,
                                  borderLeft: 3,
                                  borderColor: rowLabel.segment_color,
                                }}
                              />
                            ) : (
                              <Typography variant="caption" color="text.secondary">
                                Unlabeled
                              </Typography>
                            )}
                          </TableCell>
                          {Object.values(row).map((value: any, i) => (
                            <TableCell key={i}>
                              {value !== null && value !== undefined ? String(value) : '-'}
                            </TableCell>
                          ))}
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </TableContainer>

              <TablePagination
                component="div"
                count={totalRows}
                page={page}
                onPageChange={(e, newPage) => setPage(newPage)}
                rowsPerPage={rowsPerPage}
                onRowsPerPageChange={(e) => {
                  setRowsPerPage(parseInt(e.target.value, 10));
                  setPage(0);
                }}
                rowsPerPageOptions={[5, 10, 25, 50, 100]}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Segment Dialog */}
      <Dialog open={segmentDialogOpen} onClose={() => setSegmentDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>{editingSegment ? 'Edit Segment' : 'Create New Segment'}</DialogTitle>
        <DialogContent>
          <Stack spacing={3} sx={{ mt: 2 }}>
            <TextField
              label="Segment Name"
              value={segmentName}
              onChange={(e) => setSegmentName(e.target.value)}
              fullWidth
              required
            />
            <TextField
              label="Description"
              value={segmentDescription}
              onChange={(e) => setSegmentDescription(e.target.value)}
              fullWidth
              multiline
              rows={3}
            />
            <Box>
              <Typography variant="body2" sx={{ mb: 1 }}>
                Color
              </Typography>
              <Stack direction="row" spacing={1}>
                {PRESET_COLORS.map((color) => (
                  <Box
                    key={color}
                    sx={{
                      width: 40,
                      height: 40,
                      bgcolor: color,
                      borderRadius: 1,
                      cursor: 'pointer',
                      border: segmentColor === color ? 3 : 0,
                      borderColor: 'primary.main',
                    }}
                    onClick={() => setSegmentColor(color)}
                  />
                ))}
              </Stack>
            </Box>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSegmentDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleSaveSegment} variant="contained">
            {editingSegment ? 'Update' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Auto-Cluster Dialog */}
      <Dialog
        open={autoClusterDialogOpen}
        onClose={() => !clustering && setAutoClusterDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ClusterIcon color="secondary" />
            Auto-Segment Dataset
          </Box>
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Automatically group rows into segments using ML clustering algorithms.
            Segments will be created based on the patterns in your numeric data.
          </Typography>

          <Stack spacing={3}>
            {/* Algorithm Selection */}
            <FormControl fullWidth>
              <InputLabel>Clustering Algorithm</InputLabel>
              <Select
                value={clusterAlgorithm}
                label="Clustering Algorithm"
                onChange={(e) => setClusterAlgorithm(e.target.value as any)}
                disabled={clustering}
              >
                <MenuItem value="kmeans">K-Means (fast, requires # of clusters)</MenuItem>
                <MenuItem value="dbscan">DBSCAN (auto-detects clusters, finds outliers)</MenuItem>
                <MenuItem value="hierarchical">Hierarchical (tree-based clustering)</MenuItem>
                <MenuItem value="gaussian_mixture">Gaussian Mixture (probabilistic)</MenuItem>
                <MenuItem value="mean_shift">Mean Shift (auto-detects # of clusters)</MenuItem>
              </Select>
            </FormControl>

            {/* Algorithm-specific parameters */}
            {(clusterAlgorithm === 'kmeans' || clusterAlgorithm === 'hierarchical' || clusterAlgorithm === 'gaussian_mixture') && (
              <Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                  <Typography variant="body2">
                    Number of Clusters: <strong>{numClusters}</strong>
                    {optimalK && <Chip label={`Suggested: ${optimalK}`} size="small" sx={{ ml: 1 }} color="info" />}
                  </Typography>
                  <Button
                    size="small"
                    variant="text"
                    onClick={handleFindOptimalClusters}
                    disabled={findingOptimal || clustering}
                    startIcon={findingOptimal ? <CircularProgress size={14} /> : null}
                  >
                    {findingOptimal ? 'Finding...' : 'Find Optimal'}
                  </Button>
                </Box>
                <Slider
                  value={numClusters}
                  onChange={(_, value) => setNumClusters(value as number)}
                  min={2}
                  max={15}
                  marks
                  valueLabelDisplay="auto"
                  disabled={clustering}
                />
              </Box>
            )}

            {clusterAlgorithm === 'dbscan' && (
              <>
                <Box>
                  <Typography variant="body2" gutterBottom>
                    Epsilon (neighborhood distance): <strong>{dbscanEps}</strong>
                  </Typography>
                  <Slider
                    value={dbscanEps}
                    onChange={(_, value) => setDbscanEps(value as number)}
                    min={0.1}
                    max={2.0}
                    step={0.1}
                    valueLabelDisplay="auto"
                    disabled={clustering}
                  />
                </Box>
                <Box>
                  <Typography variant="body2" gutterBottom>
                    Min Samples (points per cluster): <strong>{dbscanMinSamples}</strong>
                  </Typography>
                  <Slider
                    value={dbscanMinSamples}
                    onChange={(_, value) => setDbscanMinSamples(value as number)}
                    min={2}
                    max={20}
                    marks
                    valueLabelDisplay="auto"
                    disabled={clustering}
                  />
                </Box>
              </>
            )}

            {/* Segment Prefix */}
            <TextField
              label="Segment Name Prefix"
              value={segmentPrefix}
              onChange={(e) => setSegmentPrefix(e.target.value)}
              fullWidth
              helperText="Segments will be named: [Prefix] 1, [Prefix] 2, etc."
              disabled={clustering}
            />

            {/* Progress indicator */}
            {clustering && (
              <Box>
                <LinearProgress sx={{ mb: 1 }} />
                <Typography variant="body2" color="text.secondary" textAlign="center">
                  Running clustering algorithm...
                </Typography>
              </Box>
            )}

            {/* Results */}
            {clusterResult && (
              <Alert severity="success">
                <Typography variant="subtitle2" gutterBottom>
                  Clustering Complete
                </Typography>
                <Typography variant="body2">
                  Created {clusterResult.n_clusters} segments
                  {clusterResult.n_noise_points > 0 && ` (${clusterResult.n_noise_points} noise points)`}
                </Typography>
                {clusterResult.metrics?.silhouette_score && (
                  <Typography variant="body2">
                    Silhouette Score: {clusterResult.metrics.silhouette_score.toFixed(3)}
                    <Tooltip title="Score from -1 to 1. Higher is better. Above 0.5 is good clustering.">
                      <Chip label="?" size="small" sx={{ ml: 1, height: 16, fontSize: '0.7rem' }} />
                    </Tooltip>
                  </Typography>
                )}
              </Alert>
            )}
          </Stack>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={() => setAutoClusterDialogOpen(false)} disabled={clustering}>
            {clusterResult ? 'Close' : 'Cancel'}
          </Button>
          {!clusterResult && (
            <Button
              variant="contained"
              color="secondary"
              startIcon={clustering ? <CircularProgress size={16} color="inherit" /> : <AutoIcon />}
              onClick={handleRunAutoClustering}
              disabled={clustering}
            >
              {clustering ? 'Clustering...' : 'Run Auto-Segment'}
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
}
