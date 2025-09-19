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
} from '@mui/material';
import {
  ArrowBack,
  Analytics,
  TrendingUp,
  Assessment,
  DataObject,
  Refresh,
  Download,
  Info,
  Timeline,
  BarChart,
  PieChart,
  ShowChart,
} from '@mui/icons-material';
import httpfetch from '../../utils/axios';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  BarChart as RechartsBarChart,
  Bar,
  PieChart as RechartsPieChart,
  Pie,
  Cell,
  Area,
  AreaChart,
  RadialBarChart,
  RadialBar,
  ComposedChart,
  Scatter,
  ScatterChart,
  ReferenceLine,
} from 'recharts';

interface ModelAnalysisPageProps {
  modelId?: string | number;
  onBack?: () => void;
}

interface ModelData {
  id: number;
  name: string;
  model_type: string;
  status: string;
  accuracy?: number;
  dataset?: any;
  target_column?: string;
  config?: any;
  created_at?: string;
  updated_at?: string;
}

interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  confusion_matrix?: number[][];
  feature_importance?: Array<{ feature: string; importance: number }>;
  training_history?: Array<{ epoch: number; loss: number; accuracy: number; val_loss?: number; val_accuracy?: number }>;
  prediction_distribution?: Array<{ label: string; value: number }>;
}

interface ModelStatistics {
  total_predictions: number;
  avg_prediction_time: number;
  model_size: string;
  training_time: string;
  last_trained: string;
  dataset_info: {
    total_rows: number;
    total_features: number;
    target_distribution: Array<{ label: string; count: number }>;
  };
}

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#8dd1e1', '#d084d0'];

// Custom Tooltip Components
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
        <Typography variant="body2" fontWeight={600}>{`Epoch: ${label}`}</Typography>
        {payload.map((entry: any, index: number) => (
          <Typography key={index} variant="body2" sx={{ color: entry.color }}>
            {`${entry.dataKey}: ${entry.value?.toFixed(4)}`}
          </Typography>
        ))}
      </Box>
    );
  }
  return null;
};

const MetricsTooltip = ({ active, payload, label }: any) => {
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
        <Typography variant="body2">
          {`Value: ${(payload[0].value * 100).toFixed(1)}%`}
        </Typography>
      </Box>
    );
  }
  return null;
};

function ModelAnalysisPage({ modelId, onBack }: ModelAnalysisPageProps) {
  const [model, setModel] = useState<ModelData | null>(null);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [statistics, setStatistics] = useState<ModelStatistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedModel, setSelectedModel] = useState<string | number>(modelId || '');
  const [availableModels, setAvailableModels] = useState<ModelData[]>([]);
  const [refreshing, setRefreshing] = useState(false);

  const fetchAvailableModels = async () => {
    try {
      const response = await httpfetch.get('models/');
      setAvailableModels(response.data.results || response.data);
    } catch (err: any) {
      console.error('Error fetching models:', err);
    }
  };

  const fetchModelData = async (id: string | number) => {
    try {
      setLoading(true);
      setError('');

      console.log('Fetching model data for ID:', id);

      // Fetch model details
      const modelResponse = await httpfetch.get(`models/${id}/`);

      const modelData = modelResponse.data;
      console.log('Model data received:', modelData);
      setModel(modelData);

      // Try to fetch real analytics data, fall back to mock if needed
      try {
        const [metricsResponse, statisticsResponse] = await Promise.all([
          httpfetch.get(`models/${id}/metrics/`),
          httpfetch.get(`models/${id}/statistics/`)
        ]);

        console.log('Using real analytics data from backend');
        setMetrics(metricsResponse.data.metrics);
        setStatistics(statisticsResponse.data.statistics);
      } catch (analyticsError) {
        console.warn('Real analytics endpoints not available, using mock data:', analyticsError);
        setMetrics(generateMockMetrics(modelData));
        setStatistics(generateMockStatistics(modelData));
      }

    } catch (err: any) {
      console.error('Error fetching model data:', err);
      if (err.message === 'Request timeout') {
        setError('Request timed out. Please try again.');
      } else {
        setError(err.response?.data?.error || err.message || 'Failed to fetch model data');
      }
    } finally {
      setLoading(false);
    }
  };

  const generateMockMetrics = (modelData: ModelData): ModelMetrics => {
    const accuracy = modelData.accuracy || Math.random() * 0.3 + 0.7;
    return {
      accuracy,
      precision: accuracy + Math.random() * 0.1 - 0.05,
      recall: accuracy + Math.random() * 0.1 - 0.05,
      f1_score: accuracy + Math.random() * 0.08 - 0.04,
      confusion_matrix: [
        [85, 15],
        [12, 88]
      ],
      feature_importance: [
        { feature: 'Feature 1', importance: 0.35 },
        { feature: 'Feature 2', importance: 0.28 },
        { feature: 'Feature 3', importance: 0.22 },
        { feature: 'Feature 4', importance: 0.15 },
      ],
      training_history: Array.from({ length: 20 }, (_, i) => ({
        epoch: i + 1,
        loss: 1.2 - (i * 0.05) + Math.random() * 0.1,
        accuracy: 0.3 + (i * 0.035) + Math.random() * 0.05,
        val_loss: 1.3 - (i * 0.048) + Math.random() * 0.12,
        val_accuracy: 0.28 + (i * 0.032) + Math.random() * 0.06,
      })),
      prediction_distribution: [
        { label: 'Class A', value: 45 },
        { label: 'Class B', value: 35 },
        { label: 'Class C', value: 20 },
      ],
    };
  };

  const generateMockStatistics = (modelData: ModelData): ModelStatistics => {
    return {
      total_predictions: Math.floor(Math.random() * 10000) + 1000,
      avg_prediction_time: Math.random() * 100 + 10,
      model_size: `${(Math.random() * 50 + 5).toFixed(1)} MB`,
      training_time: `${Math.floor(Math.random() * 120 + 30)} minutes`,
      last_trained: modelData.updated_at || new Date().toISOString(),
      dataset_info: {
        total_rows: Math.floor(Math.random() * 50000) + 10000,
        total_features: Math.floor(Math.random() * 20) + 5,
        target_distribution: [
          { label: 'Positive', count: Math.floor(Math.random() * 3000) + 2000 },
          { label: 'Negative', count: Math.floor(Math.random() * 3000) + 2000 },
        ],
      },
    };
  };

  const handleRefresh = async () => {
    if (selectedModel) {
      setRefreshing(true);
      await fetchModelData(selectedModel);
      setRefreshing(false);
    }
  };

  const handleModelChange = (newModelId: string | number) => {
    setSelectedModel(newModelId);
    fetchModelData(newModelId);
  };

  useEffect(() => {
    fetchAvailableModels();
    if (modelId) {
      fetchModelData(modelId);
    } else {
      // If no model ID provided, just stop loading
      setLoading(false);
    }
  }, [modelId]);

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed': return 'success';
      case 'training': return 'warning';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  if (loading) {
    return (
      <Container maxWidth="xl" sx={{ py: 3, height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Stack alignItems="center" spacing={2}>
          <CircularProgress size={48} />
          <Typography variant="body1" color="text.secondary">
            Loading model analysis...
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
            Back to Models
          </Button>
        )}
      </Container>
    );
  }

  if (!model) {
    return (
      <Container maxWidth="xl" sx={{ py: 3 }}>
        <Box sx={{ textAlign: 'center', py: 8 }}>
          <Analytics sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            Select a model to analyze
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Choose a model from the dropdown to view detailed analysis
          </Typography>

          <FormControl sx={{ minWidth: 300, mb: 3 }}>
            <InputLabel>Select Model</InputLabel>
            <Select
              value={selectedModel}
              label="Select Model"
              onChange={(e) => handleModelChange(e.target.value)}
            >
              {availableModels.map((model) => (
                <MenuItem key={model.id} value={model.id}>
                  {model.name} ({model.model_type})
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {onBack && (
            <Box>
              <Button startIcon={<ArrowBack />} onClick={onBack}>
                Back to Models
              </Button>
            </Box>
          )}
        </Box>
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
            Model Analysis
          </Typography>
          <Typography variant="body1" color="text.secondary">
            In-depth analysis and performance metrics for your ML model
          </Typography>
        </Box>
        <Stack direction="row" spacing={2} alignItems="center">
          <FormControl sx={{ minWidth: 200 }}>
            <InputLabel size="small">Model</InputLabel>
            <Select
              size="small"
              value={selectedModel}
              label="Model"
              onChange={(e) => handleModelChange(e.target.value)}
            >
              {availableModels.map((model) => (
                <MenuItem key={model.id} value={model.id}>
                  {model.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Tooltip title="Refresh data">
            <IconButton onClick={handleRefresh} disabled={refreshing}>
              <Refresh />
            </IconButton>
          </Tooltip>
        </Stack>
      </Box>

      {/* Model Overview Card */}
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
                  <Analytics sx={{ color: 'white', fontSize: 28 }} />
                </Box>
                <Box>
                  <Typography variant="h5" fontWeight={600}>
                    {model.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {model.model_type} • Target: {model.target_column}
                  </Typography>
                </Box>
                <Chip
                  label={model.status}
                  color={getStatusColor(model.status)}
                  sx={{ ml: 'auto' }}
                />
              </Box>

              <Grid container spacing={3}>
                <Grid item xs={6} sm={3}>
                  <Typography variant="caption" color="text.secondary">
                    Accuracy
                  </Typography>
                  <Typography variant="h6" fontWeight={600} color="success.main">
                    {metrics?.accuracy ? `${(metrics.accuracy * 100).toFixed(1)}%` : 'N/A'}
                  </Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="caption" color="text.secondary">
                    Dataset
                  </Typography>
                  <Typography variant="body2" fontWeight={500}>
                    {model.dataset?.name || 'N/A'}
                  </Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="caption" color="text.secondary">
                    Created
                  </Typography>
                  <Typography variant="body2" fontWeight={500}>
                    {model.created_at ? new Date(model.created_at).toLocaleDateString() : 'N/A'}
                  </Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="caption" color="text.secondary">
                    Model Size
                  </Typography>
                  <Typography variant="body2" fontWeight={500}>
                    {statistics?.model_size || 'N/A'}
                  </Typography>
                </Grid>
              </Grid>
            </Grid>

            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'right' }}>
                <Stack direction="row" spacing={1} justifyContent="flex-end">
                  <Button variant="outlined" startIcon={<Download />} size="small">
                    Export Report
                  </Button>
                  <Button variant="contained" startIcon={<Assessment />} size="small">
                    Run Test
                  </Button>
                </Stack>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Performance Metrics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <TrendingUp sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
              <Typography variant="h4" fontWeight={700} color="success.main">
                {metrics?.accuracy ? `${(metrics.accuracy * 100).toFixed(1)}%` : 'N/A'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Accuracy
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Assessment sx={{ fontSize: 40, color: 'info.main', mb: 1 }} />
              <Typography variant="h4" fontWeight={700} color="info.main">
                {metrics?.precision ? `${(metrics.precision * 100).toFixed(1)}%` : 'N/A'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Precision
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <ShowChart sx={{ fontSize: 40, color: 'warning.main', mb: 1 }} />
              <Typography variant="h4" fontWeight={700} color="warning.main">
                {metrics?.recall ? `${(metrics.recall * 100).toFixed(1)}%` : 'N/A'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Recall
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <BarChart sx={{ fontSize: 40, color: 'secondary.main', mb: 1 }} />
              <Typography variant="h4" fontWeight={700} color="secondary.main">
                {metrics?.f1_score ? `${(metrics.f1_score * 100).toFixed(1)}%` : 'N/A'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                F1 Score
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts Grid */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {/* Training History */}
        {metrics?.training_history && (
          <Grid item xs={12} lg={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  Training & Validation Accuracy
                </Typography>
                <Box sx={{ height: 300, pt: 2 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={metrics.training_history}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis
                        dataKey="epoch"
                        tick={{ fontSize: 12 }}
                        tickLine={{ stroke: '#e0e0e0' }}
                      />
                      <YAxis
                        tick={{ fontSize: 12 }}
                        tickLine={{ stroke: '#e0e0e0' }}
                        domain={[0, 1]}
                      />
                      <RechartsTooltip content={<CustomTooltip />} />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="accuracy"
                        stroke="#8884d8"
                        strokeWidth={2}
                        name="Training Accuracy"
                        dot={{ r: 3 }}
                        activeDot={{ r: 5 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="val_accuracy"
                        stroke="#82ca9d"
                        strokeWidth={2}
                        name="Validation Accuracy"
                        dot={{ r: 3 }}
                        activeDot={{ r: 5 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Feature Importance */}
        {metrics?.feature_importance && (
          <Grid item xs={12} lg={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  Feature Importance
                </Typography>
                <Box sx={{ height: 300, pt: 2 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsBarChart
                      data={metrics.feature_importance}
                      layout="horizontal"
                      margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis
                        type="number"
                        tick={{ fontSize: 12 }}
                        tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                      />
                      <YAxis
                        type="category"
                        dataKey="feature"
                        tick={{ fontSize: 12 }}
                        width={75}
                      />
                      <RechartsTooltip
                        formatter={(value: any) => [`${(value * 100).toFixed(1)}%`, 'Importance']}
                      />
                      <Bar
                        dataKey="importance"
                        fill="#8884d8"
                        radius={[0, 4, 4, 0]}
                      />
                    </RechartsBarChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Prediction Distribution */}
        {metrics?.prediction_distribution && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  Prediction Distribution
                </Typography>
                <Box sx={{ height: 300, pt: 2 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsPieChart>
                      <Pie
                        data={metrics.prediction_distribution}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ label, percent }) => `${label}: ${(percent * 100).toFixed(1)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {metrics.prediction_distribution.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <RechartsTooltip formatter={(value: any) => [value, 'Count']} />
                      <Legend />
                    </RechartsPieChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Loss Curves */}
        {metrics?.training_history && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  Training & Validation Loss
                </Typography>
                <Box sx={{ height: 300, pt: 2 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={metrics.training_history}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis
                        dataKey="epoch"
                        tick={{ fontSize: 12 }}
                        tickLine={{ stroke: '#e0e0e0' }}
                      />
                      <YAxis
                        tick={{ fontSize: 12 }}
                        tickLine={{ stroke: '#e0e0e0' }}
                      />
                      <RechartsTooltip content={<CustomTooltip />} />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="loss"
                        stroke="#ff7300"
                        strokeWidth={2}
                        name="Training Loss"
                        dot={{ r: 3 }}
                        activeDot={{ r: 5 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="val_loss"
                        stroke="#8dd1e1"
                        strokeWidth={2}
                        name="Validation Loss"
                        dot={{ r: 3 }}
                        activeDot={{ r: 5 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Model Performance Radar Chart */}
        {metrics && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  Performance Metrics Overview
                </Typography>
                <Box sx={{ height: 300, pt: 2 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <RadialBarChart
                      cx="50%"
                      cy="50%"
                      innerRadius="20%"
                      outerRadius="80%"
                      data={[
                        { name: 'Accuracy', value: metrics.accuracy * 100, fill: '#8884d8' },
                        { name: 'Precision', value: metrics.precision * 100, fill: '#82ca9d' },
                        { name: 'Recall', value: metrics.recall * 100, fill: '#ffc658' },
                        { name: 'F1 Score', value: metrics.f1_score * 100, fill: '#ff7300' },
                      ]}
                    >
                      <RadialBar
                        minAngle={15}
                        label={{ position: 'insideStart', fill: '#fff' }}
                        background
                        clockWise
                        dataKey="value"
                      />
                      <Legend iconSize={10} />
                      <RechartsTooltip formatter={(value: any) => [`${value.toFixed(1)}%`, 'Score']} />
                    </RadialBarChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Combined Accuracy and Loss Over Time */}
        {metrics?.training_history && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  Training Progress Overview
                </Typography>
                <Box sx={{ height: 300, pt: 2 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={metrics.training_history}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis
                        dataKey="epoch"
                        tick={{ fontSize: 12 }}
                      />
                      <YAxis yAxisId="left" orientation="left" />
                      <YAxis yAxisId="right" orientation="right" />
                      <RechartsTooltip content={<CustomTooltip />} />
                      <Legend />
                      <Area
                        yAxisId="left"
                        type="monotone"
                        dataKey="accuracy"
                        fill="#8884d8"
                        fillOpacity={0.3}
                        stroke="#8884d8"
                        strokeWidth={2}
                        name="Accuracy"
                      />
                      <Line
                        yAxisId="right"
                        type="monotone"
                        dataKey="loss"
                        stroke="#ff7300"
                        strokeWidth={2}
                        name="Loss"
                        dot={{ r: 2 }}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>

      {/* Statistics and Details */}
      <Grid container spacing={3}>
        {/* Model Statistics */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" fontWeight={600} gutterBottom>
                Model Statistics
              </Typography>
              <Stack spacing={2}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">
                    Total Predictions
                  </Typography>
                  <Typography variant="body2" fontWeight={500}>
                    {statistics?.total_predictions?.toLocaleString() || 'N/A'}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">
                    Avg Prediction Time
                  </Typography>
                  <Typography variant="body2" fontWeight={500}>
                    {statistics?.avg_prediction_time ? `${statistics.avg_prediction_time.toFixed(2)}ms` : 'N/A'}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">
                    Training Time
                  </Typography>
                  <Typography variant="body2" fontWeight={500}>
                    {statistics?.training_time || 'N/A'}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">
                    Last Trained
                  </Typography>
                  <Typography variant="body2" fontWeight={500}>
                    {statistics?.last_trained ? new Date(statistics.last_trained).toLocaleDateString() : 'N/A'}
                  </Typography>
                </Box>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Dataset Information */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" fontWeight={600} gutterBottom>
                Dataset Information
              </Typography>
              <Stack spacing={2}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">
                    Total Rows
                  </Typography>
                  <Typography variant="body2" fontWeight={500}>
                    {statistics?.dataset_info?.total_rows?.toLocaleString() || 'N/A'}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">
                    Total Features
                  </Typography>
                  <Typography variant="body2" fontWeight={500}>
                    {statistics?.dataset_info?.total_features || 'N/A'}
                  </Typography>
                </Box>
                <Divider />
                <Typography variant="subtitle2" fontWeight={600}>
                  Target Distribution
                </Typography>
                {statistics?.dataset_info?.target_distribution?.map((item, index) => (
                  <Box key={index} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">
                      {item.label}
                    </Typography>
                    <Typography variant="body2" fontWeight={500}>
                      {item.count?.toLocaleString()}
                    </Typography>
                  </Box>
                ))}
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Enhanced Confusion Matrix with Heatmap Visualization */}
        {metrics?.confusion_matrix && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  Confusion Matrix
                </Typography>
                <TableContainer component={Paper} variant="outlined" sx={{ mb: 2 }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell></TableCell>
                        <TableCell align="center" sx={{ fontWeight: 600 }}>Predicted Negative</TableCell>
                        <TableCell align="center" sx={{ fontWeight: 600 }}>Predicted Positive</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell sx={{ fontWeight: 600 }}>Actual Negative</TableCell>
                        <TableCell align="center" sx={{ bgcolor: 'success.lighter' }}>
                          {metrics.confusion_matrix[0][0]}
                        </TableCell>
                        <TableCell align="center" sx={{ bgcolor: 'error.lighter' }}>
                          {metrics.confusion_matrix[0][1]}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell sx={{ fontWeight: 600 }}>Actual Positive</TableCell>
                        <TableCell align="center" sx={{ bgcolor: 'error.lighter' }}>
                          {metrics.confusion_matrix[1][0]}
                        </TableCell>
                        <TableCell align="center" sx={{ bgcolor: 'success.lighter' }}>
                          {metrics.confusion_matrix[1][1]}
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
                <Typography variant="body2" color="text.secondary">
                  True Positive Rate: {((metrics.confusion_matrix[1][1] / (metrics.confusion_matrix[1][1] + metrics.confusion_matrix[1][0])) * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  True Negative Rate: {((metrics.confusion_matrix[0][0] / (metrics.confusion_matrix[0][0] + metrics.confusion_matrix[0][1])) * 100).toFixed(1)}%
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Model Comparison & Benchmarks */}
        {statistics && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  Performance Benchmarks
                </Typography>
                <Box sx={{ height: 250, pt: 2 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsBarChart
                      data={[
                        { metric: 'Accuracy', current: (metrics?.accuracy || 0) * 100, baseline: 75, target: 90 },
                        { metric: 'Precision', current: (metrics?.precision || 0) * 100, baseline: 70, target: 85 },
                        { metric: 'Recall', current: (metrics?.recall || 0) * 100, baseline: 72, target: 88 },
                        { metric: 'F1 Score', current: (metrics?.f1_score || 0) * 100, baseline: 71, target: 87 },
                      ]}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="metric" tick={{ fontSize: 12 }} />
                      <YAxis tick={{ fontSize: 12 }} domain={[0, 100]} />
                      <RechartsTooltip formatter={(value: any) => [`${value.toFixed(1)}%`, '']} />
                      <Legend />
                      <Bar dataKey="current" fill="#8884d8" name="Current Model" />
                      <Bar dataKey="baseline" fill="#82ca9d" name="Baseline" />
                      <Bar dataKey="target" fill="#ffc658" name="Target" />
                    </RechartsBarChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Container>
  );
}

export default ModelAnalysisPage;