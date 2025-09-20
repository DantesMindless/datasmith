import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Container,
  Grid,
  Card,
  CardContent,
  Paper,
  Chip,
  Button,
  IconButton,
  Tooltip,
  Alert,
  CircularProgress,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Fade,
  Menu,
  Badge,
  Avatar,
  ListItemIcon,
  ListItemText,
  Checkbox,
  FormControlLabel,
  Tabs,
  Tab,
} from '@mui/material';
import {
  Add,
  FilterList,
  Search,
  MoreVert,
  Edit,
  Delete,
  Analytics,
  CloudUpload,
  Assessment,
  DataObject,
  CheckCircle,
  Warning,
  Error,
  Info,
  Refresh,
  GetApp,
  Visibility,
  CleaningServices,
  Storage,
  Cable,
  TableView,
} from '@mui/icons-material';
import httpfetch from '../../utils/axios';
import DatasetAnalysisPage from './subpages/DatasetAnalysisPage';
import CreateDatasetPage from '../CreateDatasetPage';
import CreateConnection from '../CreateConnection';
import DataSourcesTables from '../DataSourcesTables';
import QueryTables from '../QueryTables';

interface Dataset {
  id: number;
  name: string;
  description?: string;
  dataset_type: string;
  dataset_purpose: string;
  data_quality: string;
  row_count?: number;
  column_count?: number;
  file_size_formatted?: string;
  quality_score?: number;
  is_processed: boolean;
  last_analyzed?: string;
  created_at: string;
}

interface FilterState {
  search: string;
  dataset_type: string;
  data_quality: string;
  dataset_purpose: string;
  is_processed: boolean | null;
}

const DATASET_TYPES = [
  { value: 'tabular', label: 'Tabular Data' },
  { value: 'image', label: 'Image Dataset' },
  { value: 'text', label: 'Text Dataset' },
  { value: 'time_series', label: 'Time Series' },
  { value: 'mixed', label: 'Mixed Dataset' },
];

const DATA_QUALITIES = [
  { value: 'excellent', label: 'Excellent (>95%)', color: 'success' },
  { value: 'good', label: 'Good (85-95%)', color: 'info' },
  { value: 'fair', label: 'Fair (70-85%)', color: 'warning' },
  { value: 'poor', label: 'Poor (<70%)', color: 'error' },
];

const DATASET_PURPOSES = [
  { value: 'classification', label: 'Classification' },
  { value: 'regression', label: 'Regression' },
  { value: 'clustering', label: 'Clustering' },
  { value: 'anomaly_detection', label: 'Anomaly Detection' },
  { value: 'recommendation', label: 'Recommendation' },
  { value: 'general', label: 'General Purpose' },
];

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`data-tabpanel-${index}`}
      aria-labelledby={`data-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function DatasetManagement() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [filteredDatasets, setFilteredDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedDataset, setSelectedDataset] = useState<number | null>(null);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [filterOpen, setFilterOpen] = useState(false);
  const [actionMenu, setActionMenu] = useState<null | HTMLElement>(null);
  const [selectedDatasetForMenu, setSelectedDatasetForMenu] = useState<Dataset | null>(null);

  const [filters, setFilters] = useState<FilterState>({
    search: '',
    dataset_type: '',
    data_quality: '',
    dataset_purpose: '',
    is_processed: null,
  });

  const fetchDatasets = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await httpfetch.get('datasets/');
      const datasetsData = response.data.results || response.data;
      setDatasets(datasetsData);
      setFilteredDatasets(datasetsData);
    } catch (err: any) {
      console.error('Error fetching datasets:', err);
      setError(err.response?.data?.error || err.message || 'Failed to fetch datasets');
    } finally {
      setLoading(false);
    }
  };

  const applyFilters = () => {
    let filtered = datasets.filter(dataset => {
      // Search filter
      if (filters.search && !dataset.name.toLowerCase().includes(filters.search.toLowerCase()) &&
          !dataset.description?.toLowerCase().includes(filters.search.toLowerCase())) {
        return false;
      }

      // Type filter
      if (filters.dataset_type && dataset.dataset_type !== filters.dataset_type) {
        return false;
      }

      // Quality filter
      if (filters.data_quality && dataset.data_quality !== filters.data_quality) {
        return false;
      }

      // Purpose filter
      if (filters.dataset_purpose && dataset.dataset_purpose !== filters.dataset_purpose) {
        return false;
      }

      // Processing status filter
      if (filters.is_processed !== null && dataset.is_processed !== filters.is_processed) {
        return false;
      }

      return true;
    });

    setFilteredDatasets(filtered);
  };

  const handleFilterChange = (key: keyof FilterState, value: any) => {
    const newFilters = { ...filters, [key]: value };
    setFilters(newFilters);
  };

  const clearFilters = () => {
    setFilters({
      search: '',
      dataset_type: '',
      data_quality: '',
      dataset_purpose: '',
      is_processed: null,
    });
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, dataset: Dataset) => {
    setActionMenu(event.currentTarget);
    setSelectedDatasetForMenu(dataset);
  };

  const handleMenuClose = () => {
    setActionMenu(null);
    setSelectedDatasetForMenu(null);
  };

  const handleAnalyzeDataset = (datasetId: number) => {
    setSelectedDataset(datasetId);
    setShowAnalysis(true);
    handleMenuClose();
  };

  const handleDeleteDataset = async (datasetId: number) => {
    if (!window.confirm('Are you sure you want to delete this dataset?')) {
      return;
    }

    try {
      await httpfetch.delete(`datasets/${datasetId}/`);
      await fetchDatasets(); // Refresh the list
      handleMenuClose();
    } catch (err: any) {
      console.error('Error deleting dataset:', err);
      setError('Failed to delete dataset');
    }
  };

  const getQualityIcon = (quality: string) => {
    switch (quality.toLowerCase()) {
      case 'excellent': return <CheckCircle color="success" />;
      case 'good': return <Info color="info" />;
      case 'fair': return <Warning color="warning" />;
      case 'poor': return <Error color="error" />;
      default: return <Info />;
    }
  };

  const getQualityColor = (quality: string) => {
    switch (quality.toLowerCase()) {
      case 'excellent': return 'success';
      case 'good': return 'info';
      case 'fair': return 'warning';
      case 'poor': return 'error';
      default: return 'default';
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  useEffect(() => {
    applyFilters();
  }, [filters, datasets]);

  if (showAnalysis && selectedDataset) {
    return (
      <DatasetAnalysisPage
        datasetId={selectedDataset}
        onBack={() => {
          setShowAnalysis(false);
          setSelectedDataset(null);
        }}
      />
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 4 }}>
        <Box>
          <Typography variant="h5" fontWeight={600}>
            Dataset Management
          </Typography>
        </Box>
        <Stack direction="row" spacing={2}>
          <Button
            variant="outlined"
            startIcon={<FilterList />}
            onClick={() => setFilterOpen(!filterOpen)}
            sx={{ position: 'relative' }}
          >
            Filters
            {Object.values(filters).some(f => f && f !== '') && (
              <Badge
                badgeContent=""
                color="primary"
                variant="dot"
                sx={{ position: 'absolute', top: 8, right: 8 }}
              />
            )}
          </Button>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={fetchDatasets}
            disabled={loading}
          >
            Refresh
          </Button>
        </Stack>
      </Box>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      {/* Filters Panel */}
      <Fade in={filterOpen}>
        <Card sx={{ mb: 3, display: filterOpen ? 'block' : 'none' }}>
          <CardContent>
            <Grid container spacing={3} alignItems="center">
              <Grid item xs={12} md={3}>
                <TextField
                  fullWidth
                  label="Search datasets"
                  variant="outlined"
                  size="small"
                  value={filters.search}
                  onChange={(e) => handleFilterChange('search', e.target.value)}
                  InputProps={{
                    startAdornment: <Search sx={{ mr: 1, color: 'text.secondary' }} />,
                  }}
                />
              </Grid>
              <Grid item xs={12} md={2}>
                <FormControl fullWidth size="small">
                  <InputLabel>Type</InputLabel>
                  <Select
                    value={filters.dataset_type}
                    label="Type"
                    onChange={(e) => handleFilterChange('dataset_type', e.target.value)}
                  >
                    <MenuItem value="">All Types</MenuItem>
                    {DATASET_TYPES.map(type => (
                      <MenuItem key={type.value} value={type.value}>
                        {type.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={2}>
                <FormControl fullWidth size="small">
                  <InputLabel>Quality</InputLabel>
                  <Select
                    value={filters.data_quality}
                    label="Quality"
                    onChange={(e) => handleFilterChange('data_quality', e.target.value)}
                  >
                    <MenuItem value="">All Qualities</MenuItem>
                    {DATA_QUALITIES.map(quality => (
                      <MenuItem key={quality.value} value={quality.value}>
                        {quality.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={2}>
                <FormControl fullWidth size="small">
                  <InputLabel>Purpose</InputLabel>
                  <Select
                    value={filters.dataset_purpose}
                    label="Purpose"
                    onChange={(e) => handleFilterChange('dataset_purpose', e.target.value)}
                  >
                    <MenuItem value="">All Purposes</MenuItem>
                    {DATASET_PURPOSES.map(purpose => (
                      <MenuItem key={purpose.value} value={purpose.value}>
                        {purpose.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={2}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={filters.is_processed === true}
                      onChange={(e) => handleFilterChange('is_processed', e.target.checked ? true : null)}
                    />
                  }
                  label="Processed Only"
                />
              </Grid>
              <Grid item xs={12} md={1}>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={clearFilters}
                  disabled={!Object.values(filters).some(f => f && f !== '')}
                >
                  Clear
                </Button>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Fade>

      {/* Dataset Statistics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <DataObject sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
              <Typography variant="h4" fontWeight={700}>
                {datasets.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Datasets
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <CheckCircle sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
              <Typography variant="h4" fontWeight={700}>
                {datasets.filter(d => d.is_processed).length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Processed
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Assessment sx={{ fontSize: 40, color: 'info.main', mb: 1 }} />
              <Typography variant="h4" fontWeight={700}>
                {datasets.filter(d => d.data_quality === 'excellent').length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Excellent Quality
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Warning sx={{ fontSize: 40, color: 'warning.main', mb: 1 }} />
              <Typography variant="h4" fontWeight={700}>
                {datasets.filter(d => ['poor', 'fair'].includes(d.data_quality)).length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Need Attention
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Dataset Table */}
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="h6" fontWeight={600}>
              Datasets ({filteredDatasets.length})
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {filteredDatasets.length !== datasets.length && `Filtered from ${datasets.length} total`}
            </Typography>
          </Box>

          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
              <CircularProgress />
            </Box>
          ) : (
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Dataset</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Purpose</TableCell>
                    <TableCell>Quality</TableCell>
                    <TableCell>Size</TableCell>
                    <TableCell>Rows</TableCell>
                    <TableCell>Columns</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Last Analyzed</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredDatasets.map((dataset) => (
                    <TableRow key={dataset.id} hover>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                          <Avatar
                            sx={{
                              bgcolor: 'primary.main',
                              width: 32,
                              height: 32,
                              fontSize: '0.875rem'
                            }}
                          >
                            {dataset.name.charAt(0).toUpperCase()}
                          </Avatar>
                          <Box>
                            <Typography variant="body2" fontWeight={600}>
                              {dataset.name}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {dataset.description || 'No description'}
                            </Typography>
                          </Box>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip label={dataset.dataset_type} size="small" variant="outlined" />
                      </TableCell>
                      <TableCell>
                        <Chip label={dataset.dataset_purpose} size="small" variant="outlined" />
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {getQualityIcon(dataset.data_quality)}
                          <Chip
                            label={dataset.data_quality}
                            size="small"
                            color={getQualityColor(dataset.data_quality) as any}
                          />
                        </Box>
                      </TableCell>
                      <TableCell>{dataset.file_size_formatted || '-'}</TableCell>
                      <TableCell>{dataset.row_count?.toLocaleString() || '-'}</TableCell>
                      <TableCell>{dataset.column_count || '-'}</TableCell>
                      <TableCell>
                        <Chip
                          label={dataset.is_processed ? 'Processed' : 'Pending'}
                          size="small"
                          color={dataset.is_processed ? 'success' : 'warning'}
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption">
                          {dataset.last_analyzed
                            ? new Date(dataset.last_analyzed).toLocaleDateString()
                            : 'Never'
                          }
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <IconButton
                          size="small"
                          onClick={(e) => handleMenuOpen(e, dataset)}
                        >
                          <MoreVert />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>

      {/* Action Menu */}
      <Menu
        anchorEl={actionMenu}
        open={Boolean(actionMenu)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => selectedDatasetForMenu && handleAnalyzeDataset(selectedDatasetForMenu.id)}>
          <ListItemIcon><Analytics fontSize="small" /></ListItemIcon>
          <ListItemText>Analyze Dataset</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleMenuClose}>
          <ListItemIcon><Visibility fontSize="small" /></ListItemIcon>
          <ListItemText>View Data</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleMenuClose}>
          <ListItemIcon><Edit fontSize="small" /></ListItemIcon>
          <ListItemText>Edit Metadata</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleMenuClose}>
          <ListItemIcon><CleaningServices fontSize="small" /></ListItemIcon>
          <ListItemText>Data Cleaning</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleMenuClose}>
          <ListItemIcon><GetApp fontSize="small" /></ListItemIcon>
          <ListItemText>Export Dataset</ListItemText>
        </MenuItem>
        <MenuItem
          onClick={() => selectedDatasetForMenu && handleDeleteDataset(selectedDatasetForMenu.id)}
          sx={{ color: 'error.main' }}
        >
          <ListItemIcon><Delete fontSize="small" color="error" /></ListItemIcon>
          <ListItemText>Delete Dataset</ListItemText>
        </MenuItem>
      </Menu>

    </Box>
  );
}

export default function AdvancedDatasetPage() {
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const tabConfig = [
    { label: 'Database Connections', icon: <Cable /> },
    { label: 'Create Connection', icon: <Add /> },
    { label: 'Query Interface', icon: <TableView /> },
    { label: 'Dataset Management', icon: <Storage /> },
    { label: 'Create Dataset', icon: <Add /> },
    { label: 'Data Analytics', icon: <Analytics /> },
  ];

  return (
    <Container maxWidth="xl" sx={{ py: 3, height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Page Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" fontWeight={700} gutterBottom>
          Data Management
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Manage your datasets, analyze data quality, and create intelligent data solutions
        </Typography>
      </Box>

      {/* Navigation Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs
          value={activeTab}
          onChange={handleTabChange}
          aria-label="data management tabs"
          sx={{
            '& .MuiTab-root': {
              textTransform: 'none',
              fontWeight: 500,
              fontSize: '1rem',
              minHeight: 56,
              '&.Mui-selected': {
                color: 'primary.main',
              },
            },
            '& .MuiTabs-indicator': {
              height: 3,
              borderRadius: '3px 3px 0 0',
            },
          }}
        >
          {tabConfig.map((tab, index) => (
            <Tab
              key={index}
              label={tab.label}
              icon={tab.icon}
              iconPosition="start"
              id={`data-tab-${index}`}
              aria-controls={`data-tabpanel-${index}`}
              sx={{ gap: 1 }}
            />
          ))}
        </Tabs>
      </Box>

      {/* Tab Content */}
      <Box sx={{ flex: 1, overflowY: 'auto' }}>
        <TabPanel value={activeTab} index={0}>
          <DataSourcesTables />
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          <CreateConnection />
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          <QueryTables />
        </TabPanel>

        <TabPanel value={activeTab} index={3}>
          <DatasetManagement />
        </TabPanel>

        <TabPanel value={activeTab} index={4}>
          <CreateDatasetPage />
        </TabPanel>

        <TabPanel value={activeTab} index={5}>
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 300 }}>
            <Typography variant="h6" color="text.secondary">
              Data Analytics Dashboard - Coming Soon
            </Typography>
          </Box>
        </TabPanel>
      </Box>
    </Container>
  );
}