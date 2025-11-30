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
  Divider,
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
  Fade,
  Tab,
  Tabs,
  Badge,
  TextField,
  TablePagination,
  InputAdornment,
  Checkbox,
  ListItemText,
} from '@mui/material';
import {
  ArrowBack,
  Analytics,
  Assessment,
  DataObject,
  Refresh,
  Download,
  Warning,
  CheckCircle,
  Error,
  Info,
  TableChart,
  BarChart as BarChartIcon,
  Timeline,
  BugReport,
  CleaningServices,
  Image as ImageIcon,
  TrendingUp,
  Search as SearchIcon,
  FilterList as FilterListIcon,
  ViewColumn as ViewColumnIcon,
} from '@mui/icons-material';
import httpfetch from '../../../utils/axios';
import ImageDatasetViewer from '../../ImageDatasetViewer';
import {
  ResponsiveContainer,
  BarChart as RechartsBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  PieChart as RechartsPieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  Histogram,
} from 'recharts';

interface DatasetAnalysisPageProps {
  datasetId?: string | number;
  onBack?: () => void;
}

interface DatasetData {
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
}

interface PreviewData {
  preview_data: any[];
  column_info: Record<string, any>;
  statistics: Record<string, any>;
  correlations?: any;
  distributions?: any;
}

interface QualityReport {
  quality_report: any;
  recommendations: any[];
  processing_errors?: string;
}

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#8dd1e1', '#d084d0'];

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <Box sx={{
        bgcolor: 'background.paper',
        p: 2,
        border: 1,
        borderColor: 'divider',
        borderRadius: 1,
        boxShadow: 2
      }}>
        <Typography variant="body2" fontWeight={600}>{label}</Typography>
        {payload.map((entry: any, index: number) => (
          <Typography key={index} variant="body2" sx={{ color: entry.color }}>
            {`${entry.dataKey}: ${entry.value}`}
          </Typography>
        ))}
      </Box>
    );
  }
  return null;
};

function DatasetAnalysisPage({ datasetId, onBack }: DatasetAnalysisPageProps) {
  const [dataset, setDataset] = useState<DatasetData | null>(null);
  const [previewData, setPreviewData] = useState<PreviewData | null>(null);
  const [qualityReport, setQualityReport] = useState<QualityReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState(0);
  const [refreshing, setRefreshing] = useState(false);
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [loadingRecommendations, setLoadingRecommendations] = useState(false);

  // Data preview state
  const [previewPage, setPreviewPage] = useState(0);
  const [previewRowsPerPage, setPreviewRowsPerPage] = useState(10);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);

  const fetchDatasetData = async (id: string | number) => {
    try {
      setLoading(true);
      setError('');

      // Fetch dataset details
      const [datasetResponse, previewResponse, qualityResponse] = await Promise.all([
        httpfetch.get(`datasets/${id}/`),
        httpfetch.get(`datasets/${id}/preview/`),
        httpfetch.get(`datasets/${id}/quality_report/`)
      ]);

      setDataset(datasetResponse.data);
      setPreviewData(previewResponse.data);
      setQualityReport(qualityResponse.data);

      // Fetch model recommendations
      fetchModelRecommendations(id);

    } catch (err: any) {
      console.error('Error fetching dataset data:', err);
      setError(err.response?.data?.error || err.message || 'Failed to fetch dataset data');
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    if (datasetId) {
      setRefreshing(true);
      await fetchDatasetData(datasetId);
      setRefreshing(false);
    }
  };

  const handleReanalyze = async () => {
    if (!datasetId) return;

    try {
      setRefreshing(true);
      await httpfetch.post(`datasets/${datasetId}/reanalyze/`);
      await fetchDatasetData(datasetId);
    } catch (err: any) {
      console.error('Error reanalyzing dataset:', err);
      setError('Failed to reanalyze dataset');
    } finally {
      setRefreshing(false);
    }
  };

  const fetchModelRecommendations = async (id: string | number) => {
    try {
      setLoadingRecommendations(true);
      const response = await httpfetch.get(`datasets/${id}/recommended_models/`);
      setRecommendations(response.data.recommendations || []);
    } catch (err: any) {
      console.error('Error fetching model recommendations:', err);
      setRecommendations([]);
    } finally {
      setLoadingRecommendations(false);
    }
  };

  useEffect(() => {
    if (datasetId) {
      fetchDatasetData(datasetId);
    }
  }, [datasetId]);

  // Initialize selected columns when preview data loads
  useEffect(() => {
    if (previewData?.preview_data?.[0] && selectedColumns.length === 0) {
      setSelectedColumns(Object.keys(previewData.preview_data[0]));
    }
  }, [previewData]);

  const getQualityColor = (quality: string) => {
    switch (quality.toLowerCase()) {
      case 'excellent': return 'success';
      case 'good': return 'info';
      case 'fair': return 'warning';
      case 'poor': return 'error';
      default: return 'default';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'high': return <Error color="error" />;
      case 'medium': return <Warning color="warning" />;
      case 'low': return <Info color="info" />;
      default: return <Info />;
    }
  };

  if (loading) {
    return (
      <Container maxWidth="xl" sx={{ py: 3, height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Stack alignItems="center" spacing={2}>
          <CircularProgress size={48} />
          <Typography variant="body1" color="text.secondary">
            Loading dataset analysis...
          </Typography>
        </Stack>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="xl" sx={{ py: 3 }}>
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
        {onBack && (
          <Button startIcon={<ArrowBack />} onClick={onBack}>
            Back to Datasets
          </Button>
        )}
      </Container>
    );
  }

  if (!dataset) {
    return (
      <Container maxWidth="xl" sx={{ py: 3 }}>
        <Typography variant="h6" color="text.secondary">
          No dataset selected
        </Typography>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 4 }}>
        {onBack && (
          <IconButton onClick={onBack} sx={{ mr: 2 }}>
            <ArrowBack />
          </IconButton>
        )}
        <Box sx={{ flex: 1 }}>
          <Typography variant="h4" fontWeight={700} gutterBottom>
            Dataset Analysis
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Comprehensive analysis and insights for your dataset
          </Typography>
        </Box>
        <Stack direction="row" spacing={2}>
          <Tooltip title="Refresh data">
            <IconButton onClick={handleRefresh} disabled={refreshing}>
              <Refresh />
            </IconButton>
          </Tooltip>
          <Button
            variant="outlined"
            startIcon={<Analytics />}
            onClick={handleReanalyze}
            disabled={refreshing}
          >
            Reanalyze
          </Button>
        </Stack>
      </Box>

      {/* Dataset Overview */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} md={8}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                <Box
                  sx={{
                    width: 48,
                    height: 48,
                    borderRadius: 2,
                    bgcolor: 'primary.main',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <DataObject sx={{ color: 'white', fontSize: 28 }} />
                </Box>
                <Box>
                  <Typography variant="h5" fontWeight={600}>
                    {dataset.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {dataset.description || 'No description provided'}
                  </Typography>
                </Box>
                <Chip
                  label={dataset.data_quality}
                  color={getQualityColor(dataset.data_quality)}
                  sx={{ ml: 'auto' }}
                />
              </Box>

              <Grid container spacing={3}>
                {dataset.dataset_type === 'image' ? (
                  <>
                    <Grid item xs={6} sm={4}>
                      <Typography variant="caption" color="text.secondary">
                        Total Images
                      </Typography>
                      <Typography variant="h6" fontWeight={600}>
                        {dataset.row_count?.toLocaleString() || 'N/A'}
                      </Typography>
                    </Grid>
                    <Grid item xs={6} sm={4}>
                      <Typography variant="caption" color="text.secondary">
                        File Size
                      </Typography>
                      <Typography variant="h6" fontWeight={600}>
                        {dataset.file_size_formatted || 'N/A'}
                      </Typography>
                    </Grid>
                    <Grid item xs={6} sm={4}>
                      <Typography variant="caption" color="text.secondary">
                        Type
                      </Typography>
                      <Typography variant="h6" fontWeight={600}>
                        {dataset.dataset_type}
                      </Typography>
                    </Grid>
                  </>
                ) : (
                  <>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="caption" color="text.secondary">
                        Rows
                      </Typography>
                      <Typography variant="h6" fontWeight={600}>
                        {dataset.row_count?.toLocaleString() || 'N/A'}
                      </Typography>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="caption" color="text.secondary">
                        Columns
                      </Typography>
                      <Typography variant="h6" fontWeight={600}>
                        {dataset.column_count || 'N/A'}
                      </Typography>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="caption" color="text.secondary">
                        File Size
                      </Typography>
                      <Typography variant="h6" fontWeight={600}>
                        {dataset.file_size_formatted || 'N/A'}
                      </Typography>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="caption" color="text.secondary">
                        Type
                      </Typography>
                      <Typography variant="h6" fontWeight={600}>
                        {dataset.dataset_type}
                      </Typography>
                    </Grid>
                  </>
                )}
              </Grid>
            </Grid>

            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'right' }}>
                <Stack direction="row" spacing={1} justifyContent="flex-end">
                  <Button variant="outlined" startIcon={<Download />} size="small">
                    Export
                  </Button>
                  <Button variant="contained" startIcon={<TableChart />} size="small">
                    View Data
                  </Button>
                </Stack>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Tabs Navigation */}
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={activeTab}
          onChange={(e, newValue) => setActiveTab(newValue)}
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label="Overview" icon={<Analytics />} />
          <Tab
            label={
              <Badge
                badgeContent={qualityReport?.recommendations?.length || 0}
                color="warning"
                sx={{ '& .MuiBadge-badge': { right: -12, top: 8 } }}
              >
                Quality Report
              </Badge>
            }
            icon={<Assessment />}
          />
          {dataset.dataset_type === 'image' && (
            <Tab label="Image Gallery" icon={<ImageIcon />} />
          )}
          {dataset.dataset_type !== 'image' && (
            <Tab label="Data Preview" icon={<TableChart />} />
          )}
          {dataset.dataset_type !== 'image' && (
            <Tab label="Statistics" icon={<BarChartIcon />} />
          )}
          {dataset.dataset_type !== 'image' && (
            <Tab label="Distributions" icon={<Timeline />} />
          )}
        </Tabs>
      </Paper>

      {/* Tab Content */}
      {activeTab === 0 && (
        <Grid container spacing={3}>
          {dataset.dataset_type === 'image' ? (
            /* Image Dataset Overview */
            <>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" fontWeight={600} gutterBottom>
                      Image Dataset Summary
                    </Typography>
                    <Stack spacing={2}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2" color="text.secondary">
                          Total Images
                        </Typography>
                        <Typography variant="body2" fontWeight={500}>
                          {dataset.row_count?.toLocaleString() || 'N/A'}
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2" color="text.secondary">
                          Dataset Size
                        </Typography>
                        <Typography variant="body2" fontWeight={500}>
                          {dataset.file_size_formatted || 'N/A'}
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2" color="text.secondary">
                          Processing Status
                        </Typography>
                        <Chip
                          label={dataset.is_processed ? 'Processed' : 'Processing'}
                          color={dataset.is_processed ? 'success' : 'warning'}
                          size="small"
                        />
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2" color="text.secondary">
                          Last Analyzed
                        </Typography>
                        <Typography variant="body2" fontWeight={500}>
                          {dataset.last_analyzed ? new Date(dataset.last_analyzed).toLocaleDateString() : 'Never'}
                        </Typography>
                      </Box>
                    </Stack>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent sx={{ textAlign: 'center' }}>
                    <ImageIcon sx={{ fontSize: 64, color: 'primary.main', mb: 2 }} />
                    <Typography variant="h4" fontWeight={700}>
                      {dataset.row_count?.toLocaleString() || '0'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Images in Dataset
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </>
          ) : (
            /* CSV/Tabular Dataset Overview */
            <>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" fontWeight={600} gutterBottom>
                      Dataset Statistics
                    </Typography>
                    {previewData?.statistics && (
                      <Stack spacing={2}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="body2" color="text.secondary">
                            Numeric Columns
                          </Typography>
                          <Typography variant="body2" fontWeight={500}>
                            {previewData.statistics.numeric_columns}
                          </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="body2" color="text.secondary">
                            Categorical Columns
                          </Typography>
                          <Typography variant="body2" fontWeight={500}>
                            {previewData.statistics.categorical_columns}
                          </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="body2" color="text.secondary">
                            Completeness
                          </Typography>
                          <Typography variant="body2" fontWeight={500}>
                            {previewData.statistics.completeness?.toFixed(1)}%
                          </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="body2" color="text.secondary">
                            Duplicate Rows
                          </Typography>
                          <Typography variant="body2" fontWeight={500}>
                            {previewData.statistics.duplicate_rows}
                          </Typography>
                        </Box>
                      </Stack>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* Column Types Distribution */}
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" fontWeight={600} gutterBottom>
                      Column Types
                    </Typography>
                    {previewData?.column_info && (
                      <Box sx={{ height: 300, pt: 2 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <RechartsPieChart>
                            <Pie
                              data={Object.entries(
                                Object.values(previewData.column_info).reduce((acc: any, col: any) => {
                                  acc[col.type] = (acc[col.type] || 0) + 1;
                                  return acc;
                                }, {})
                              ).map(([type, count]) => ({ type, count }))}
                              cx="50%"
                              cy="50%"
                              outerRadius={80}
                              fill="#8884d8"
                              dataKey="count"
                              label={({ type, percent }) => `${type}: ${(percent * 100).toFixed(0)}%`}
                            >
                              {Object.keys(previewData.column_info).map((_, index) => (
                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                              ))}
                            </Pie>
                            <RechartsTooltip />
                          </RechartsPieChart>
                        </ResponsiveContainer>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </>
          )}

          {/* Model Recommendations */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                  <Typography variant="h6" fontWeight={600} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <TrendingUp color="primary" />
                    Recommended Models for this Dataset
                  </Typography>
                  {loadingRecommendations && <CircularProgress size={20} />}
                </Box>

                {recommendations.length > 0 ? (
                  <Grid container spacing={2}>
                    {recommendations.map((rec, index) => (
                      <Grid item xs={12} sm={6} md={4} key={index}>
                        <Paper
                          elevation={0}
                          sx={{
                            p: 2,
                            border: '1px solid',
                            borderColor: 'divider',
                            borderRadius: 2,
                            height: '100%',
                            transition: 'all 0.2s',
                            '&:hover': {
                              borderColor: 'primary.main',
                              boxShadow: 2,
                            }
                          }}
                        >
                          <Stack spacing={1.5}>
                            {/* Rank Badge */}
                            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                              <Chip
                                label={`#${index + 1}`}
                                size="small"
                                color={index === 0 ? 'success' : index === 1 ? 'primary' : 'default'}
                                sx={{ fontWeight: 600 }}
                              />
                              <Chip
                                label={`${rec.compatibility_score}/100`}
                                size="small"
                                color={
                                  rec.compatibility_score >= 80 ? 'success' :
                                  rec.compatibility_score >= 60 ? 'primary' : 'warning'
                                }
                              />
                            </Box>

                            {/* Model Name */}
                            <Typography variant="subtitle1" fontWeight={600}>
                              {rec.model_type.split('_').map((word: string) =>
                                word.charAt(0).toUpperCase() + word.slice(1)
                              ).join(' ')}
                            </Typography>

                            {/* Model Properties */}
                            <Stack direction="row" spacing={1} flexWrap="wrap">
                              <Chip label={`Speed: ${rec.training_speed}`} size="small" variant="outlined" />
                              <Chip label={`Complexity: ${rec.complexity}`} size="small" variant="outlined" />
                            </Stack>

                            {/* Compatibility Status */}
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                              {rec.is_compatible ? (
                                <CheckCircle fontSize="small" color="success" />
                              ) : (
                                <Warning fontSize="small" color="warning" />
                              )}
                              <Typography variant="caption" fontWeight={500}>
                                {rec.is_compatible ? 'Compatible' : 'Partial Compatibility'}
                              </Typography>
                            </Box>

                            {/* Warnings */}
                            {rec.warnings && rec.warnings.length > 0 && (
                              <Alert severity="warning" sx={{ py: 0.5, fontSize: '0.75rem' }}>
                                {rec.warnings[0]}
                              </Alert>
                            )}

                            {/* Recommendation */}
                            {rec.recommendations && rec.recommendations.length > 0 && (
                              <Alert severity="info" icon={<Info fontSize="small" />} sx={{ py: 0.5, fontSize: '0.75rem' }}>
                                {rec.recommendations[0]}
                              </Alert>
                            )}
                          </Stack>
                        </Paper>
                      </Grid>
                    ))}
                  </Grid>
                ) : loadingRecommendations ? (
                  <Box sx={{ textAlign: 'center', py: 3 }}>
                    <CircularProgress />
                  </Box>
                ) : (
                  <Alert severity="info">
                    No model recommendations available for this dataset type.
                  </Alert>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {activeTab === 1 && qualityReport && (
        <Grid container spacing={3}>
          {/* Quality Score */}
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent sx={{ textAlign: 'center' }}>
                <CheckCircle sx={{ fontSize: 48, color: 'success.main', mb: 2 }} />
                <Typography variant="h4" fontWeight={700} color="success.main">
                  {qualityReport.quality_report?.completeness_score?.toFixed(1) || 'N/A'}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Data Completeness
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Recommendations */}
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  Quality Recommendations
                </Typography>
                {qualityReport.recommendations.length > 0 ? (
                  <Stack spacing={2}>
                    {qualityReport.recommendations.map((rec, index) => (
                      <Alert
                        key={index}
                        severity={rec.severity}
                        icon={getSeverityIcon(rec.severity)}
                      >
                        {rec.message}
                      </Alert>
                    ))}
                  </Stack>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No quality issues detected. Your dataset looks good!
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Quality Issues */}
          {qualityReport.quality_report?.potential_issues && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" fontWeight={600} gutterBottom>
                    Potential Issues
                  </Typography>
                  {qualityReport.quality_report.potential_issues.length > 0 ? (
                    <Stack spacing={1}>
                      {qualityReport.quality_report.potential_issues.map((issue: string, index: number) => (
                        <Box key={index} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <BugReport fontSize="small" color="warning" />
                          <Typography variant="body2">{issue}</Typography>
                        </Box>
                      ))}
                    </Stack>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      No issues detected
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      )}

      {/* Image Gallery Tab - Only for image datasets */}
      {activeTab === 2 && dataset.dataset_type === 'image' && datasetId && (
        <ImageDatasetViewer datasetId={typeof datasetId === 'string' ? datasetId : datasetId} />
      )}

      {/* Data Preview Tab - Only for non-image datasets */}
      {activeTab === 2 && dataset.dataset_type !== 'image' && (
        <>
          {!previewData || !previewData.preview_data || previewData.preview_data.length === 0 ? (
            <Card>
              <CardContent>
                <Box sx={{ textAlign: 'center', py: 8 }}>
                  <TableChart sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
                  <Typography variant="h6" color="text.secondary" gutterBottom>
                    No Preview Data Available
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    The dataset preview is empty or hasn't been processed yet.
                  </Typography>
                  <Button
                    variant="outlined"
                    startIcon={<Refresh />}
                    onClick={handleReanalyze}
                    sx={{ mt: 3 }}
                  >
                    Reanalyze Dataset
                  </Button>
                </Box>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent>
                {/* Preview Header with Controls */}
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
              <Typography variant="h6" fontWeight={600}>
                Data Preview
              </Typography>
              <Stack direction="row" spacing={2}>
                <TextField
                  size="small"
                  placeholder="Search..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <SearchIcon fontSize="small" />
                      </InputAdornment>
                    ),
                  }}
                  sx={{ width: 250 }}
                />
                <FormControl size="small" sx={{ minWidth: 200 }}>
                  <InputLabel>Columns</InputLabel>
                  <Select
                    multiple
                    value={selectedColumns}
                    onChange={(e) => setSelectedColumns(typeof e.target.value === 'string' ? e.target.value.split(',') : e.target.value)}
                    renderValue={(selected) => `${selected.length} selected`}
                    label="Columns"
                  >
                    {previewData.preview_data?.[0] && Object.keys(previewData.preview_data[0]).map((col) => (
                      <MenuItem key={col} value={col}>
                        <Checkbox checked={selectedColumns.indexOf(col) > -1} />
                        <ListItemText
                          primary={col}
                          secondary={previewData.column_info?.[col]?.type || 'unknown'}
                        />
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Stack>
            </Box>

            {/* Preview Stats */}
            <Box sx={{ display: 'flex', gap: 3, mb: 3, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Total Rows
                </Typography>
                <Typography variant="h6" fontWeight={600}>
                  {dataset.row_count?.toLocaleString() || 'N/A'}
                </Typography>
              </Box>
              <Divider orientation="vertical" flexItem />
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Showing
                </Typography>
                <Typography variant="h6" fontWeight={600}>
                  {previewData.preview_data?.length || 0} rows
                </Typography>
              </Box>
              <Divider orientation="vertical" flexItem />
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Selected Columns
                </Typography>
                <Typography variant="h6" fontWeight={600}>
                  {selectedColumns.length} / {Object.keys(previewData.preview_data?.[0] || {}).length}
                </Typography>
              </Box>
            </Box>

            {/* Data Table */}
            <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 600 }}>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 600, bgcolor: 'primary.50', minWidth: 60 }}>
                      #
                    </TableCell>
                    {previewData.preview_data?.[0] && selectedColumns
                      .filter(col => Object.keys(previewData.preview_data[0]).includes(col))
                      .map((col) => (
                        <TableCell key={col} sx={{ fontWeight: 600, bgcolor: 'primary.50', minWidth: 150 }}>
                          <Box>
                            <Typography variant="body2" fontWeight={600}>{col}</Typography>
                            <Chip
                              label={previewData.column_info?.[col]?.type || 'unknown'}
                              size="small"
                              sx={{ mt: 0.5, height: 20, fontSize: '0.65rem' }}
                            />
                          </Box>
                        </TableCell>
                      ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {previewData.preview_data
                    ?.filter((row) => {
                      if (!searchTerm) return true;
                      return Object.values(row).some(val =>
                        String(val).toLowerCase().includes(searchTerm.toLowerCase())
                      );
                    })
                    .slice(previewPage * previewRowsPerPage, previewPage * previewRowsPerPage + previewRowsPerPage)
                    .map((row, index) => (
                      <TableRow key={index} hover>
                        <TableCell sx={{ bgcolor: 'grey.50', fontWeight: 500 }}>
                          {previewPage * previewRowsPerPage + index + 1}
                        </TableCell>
                        {selectedColumns
                          .filter(col => Object.keys(row).includes(col))
                          .map((col) => {
                            const value = row[col];
                            const columnType = previewData.column_info?.[col]?.type;

                            return (
                              <TableCell key={col}>
                                {value !== null && value !== undefined ? (
                                  <Box>
                                    <Typography variant="body2">
                                      {String(value)}
                                    </Typography>
                                    {columnType === 'numeric' && (
                                      <Typography variant="caption" color="primary">
                                        {typeof value === 'number' ? value.toLocaleString() : value}
                                      </Typography>
                                    )}
                                  </Box>
                                ) : (
                                  <Typography variant="body2" color="text.disabled" fontStyle="italic">
                                    null
                                  </Typography>
                                )}
                              </TableCell>
                            );
                          })}
                      </TableRow>
                    ))}
                </TableBody>
              </Table>
            </TableContainer>

            {/* Pagination */}
            <TablePagination
              component="div"
              count={previewData.preview_data?.filter((row) => {
                if (!searchTerm) return true;
                return Object.values(row).some(val =>
                  String(val).toLowerCase().includes(searchTerm.toLowerCase())
                );
              }).length || 0}
              page={previewPage}
              onPageChange={(e, newPage) => setPreviewPage(newPage)}
              rowsPerPage={previewRowsPerPage}
              onRowsPerPageChange={(e) => {
                setPreviewRowsPerPage(parseInt(e.target.value, 10));
                setPreviewPage(0);
              }}
              rowsPerPageOptions={[5, 10, 25, 50]}
            />
          </CardContent>
        </Card>
          )}
        </>
      )}

      {/* Statistics Tab - Only for non-image datasets */}
      {activeTab === 3 && dataset.dataset_type !== 'image' && previewData?.column_info && (
        <Grid container spacing={3}>
          {/* Column Information */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  Column Statistics
                </Typography>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Column</TableCell>
                        <TableCell>Type</TableCell>
                        <TableCell>Unique Values</TableCell>
                        <TableCell>Null Count</TableCell>
                        <TableCell>Null %</TableCell>
                        <TableCell>Statistics</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(previewData.column_info).map(([col, info]: [string, any]) => (
                        <TableRow key={col}>
                          <TableCell>{col}</TableCell>
                          <TableCell>
                            <Chip label={info.type} size="small" />
                          </TableCell>
                          <TableCell>{info.unique_count}</TableCell>
                          <TableCell>{info.null_count}</TableCell>
                          <TableCell>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <LinearProgress
                                variant="determinate"
                                value={info.null_percentage}
                                sx={{ flex: 1, height: 8 }}
                                color={info.null_percentage > 30 ? 'error' : 'primary'}
                              />
                              <Typography variant="caption">
                                {info.null_percentage?.toFixed(1)}%
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell>
                            {info.type === 'numeric' && info.mean && (
                              <Typography variant="caption">
                                μ={info.mean.toFixed(2)}, σ={info.std?.toFixed(2)}
                              </Typography>
                            )}
                            {info.type === 'categorical' && info.top_values && (
                              <Typography variant="caption">
                                Top: {Object.keys(info.top_values)[0]}
                              </Typography>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Distributions Tab - Only for non-image datasets */}
      {activeTab === 4 && dataset.dataset_type !== 'image' && previewData?.distributions && (
        <Grid container spacing={3}>
          {Object.entries(previewData.distributions).slice(0, 6).map(([col, dist]: [string, any]) => (
            <Grid item xs={12} md={6} key={col}>
              <Card>
                <CardContent>
                  <Typography variant="h6" fontWeight={600} gutterBottom>
                    {col} Distribution
                  </Typography>
                  <Box sx={{ height: 300, pt: 2 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      {dist.type === 'histogram' ? (
                        <RechartsBarChart data={
                          dist.bins.slice(0, -1).map((bin: number, i: number) => ({
                            bin: bin.toFixed(2),
                            count: dist.counts[i]
                          }))
                        }>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="bin" />
                          <YAxis />
                          <RechartsTooltip content={<CustomTooltip />} />
                          <Bar dataKey="count" fill="#8884d8" />
                        </RechartsBarChart>
                      ) : (
                        <RechartsBarChart data={
                          dist.values.map((value: string, i: number) => ({
                            value: value.length > 15 ? value.substring(0, 15) + '...' : value,
                            count: dist.counts[i]
                          }))
                        }>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="value" />
                          <YAxis />
                          <RechartsTooltip content={<CustomTooltip />} />
                          <Bar dataKey="count" fill="#82ca9d" />
                        </RechartsBarChart>
                      )}
                    </ResponsiveContainer>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
    </Container>
  );
}

export default DatasetAnalysisPage;