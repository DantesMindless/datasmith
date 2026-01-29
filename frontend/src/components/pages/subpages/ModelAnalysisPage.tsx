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
  FileDownload,
  Description,
  RestartAlt,
  PlayArrow,
  Warning,
  CheckCircle,
} from '@mui/icons-material';
import httpfetch from '../../../utils/axios';
import TrainingLogsViewerWebSocket from '../../TrainingLogsViewerWebSocket';
import { useTrainingWebSocket } from '../../../hooks/useTrainingWebSocket';
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
  dataset_name?: string;
  target_column?: string;
  config?: any;
  created_at?: string;
  updated_at?: string;
}

interface ModelMetrics {
  accuracy: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  confusion_matrix?: number[][];
  feature_importance?: Array<{ feature: string; importance: number }>;
  training_history?: Array<{ epoch: number; loss: number; accuracy: number; val_loss?: number; val_accuracy?: number }>;
  prediction_distribution?: Array<{ label: string; value: number }>;
  class_report?: Record<string, { precision: number; recall: number; 'f1-score': number; support: number }>;
  additional_metrics?: {
    training_samples?: number;
    test_samples?: number;
  };
  // Regression-specific fields
  is_regression?: boolean;
  r2_score?: number;
  rmse?: number;
  mae?: number;
  mse?: number;
  training_samples?: number;
  test_samples?: number;
  target_stats?: {
    mean: number;
    std: number;
    min: number;
    max: number;
  };
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
  const [retrainDialogOpen, setRetrainDialogOpen] = useState(false);
  const [availableDatasets, setAvailableDatasets] = useState<Array<{ id: string; name: string; column_info?: any[] }>>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [selectedTargetColumn, setSelectedTargetColumn] = useState<string>('');
  const [datasetColumns, setDatasetColumns] = useState<string[]>([]);
  const [resetting, setResetting] = useState(false);
  const [retraining, setRetraining] = useState(false);
  const [excludedColumns, setExcludedColumns] = useState<string[]>([]);
  const [columnRecommendations, setColumnRecommendations] = useState<any[]>([]);
  const [loadingRecommendations, setLoadingRecommendations] = useState(false);

  // WebSocket for live training metrics updates
  const { logs: wsLogs } = useTrainingWebSocket({
    modelId: String(selectedModel || modelId),
    enabled: !!(selectedModel || modelId) && model?.status === 'training',
    autoReconnect: true,
    onComplete: async () => {
      // Refresh metrics when training completes
      if (selectedModel) {
        await fetchModelData(selectedModel);
      }
    }
  });

  // Parse live metrics from WebSocket logs
  const parseLiveMetrics = (logs: any[]): Partial<ModelMetrics> | null => {
    if (!logs || logs.length === 0) return null;

    const trainingHistory: any[] = [];
    const epochMap = new Map();

    logs.forEach(log => {
      if (log.data && log.message.toLowerCase().includes('complete')) {
        // Extract epoch number from message like "Round 5 of 10 Complete:"
        const epochMatch = log.message.match(/round\s+(\d+)\s+of\s+(\d+)/i);
        if (epochMatch) {
          const epoch = parseInt(epochMatch[1]);
          const data: any = { epoch };

          // Parse metrics from log data
          if (log.data.train_loss) {
            data.loss = parseFloat(log.data.train_loss);
          }
          if (log.data.train_accuracy) {
            const acc = parseFloat(log.data.train_accuracy.toString().replace('%', ''));
            data.accuracy = acc / 100;
          }
          if (log.data.val_loss) {
            data.val_loss = parseFloat(log.data.val_loss);
          }
          if (log.data.val_accuracy) {
            const acc = parseFloat(log.data.val_accuracy.toString().replace('%', ''));
            data.val_accuracy = acc / 100;
          }

          epochMap.set(epoch, data);
        }
      }
    });

    // Convert map to sorted array
    if (epochMap.size > 0) {
      const sortedEpochs = Array.from(epochMap.keys()).sort((a, b) => a - b);
      sortedEpochs.forEach(epoch => {
        trainingHistory.push(epochMap.get(epoch));
      });

      return { training_history: trainingHistory };
    }

    return null;
  };

  // Disabled live metrics updates - only show graphs after training completes
  // useEffect(() => {
  //   if (wsLogs.length > 0 && model?.status === 'training') {
  //     const liveMetrics = parseLiveMetrics(wsLogs);
  //     if (liveMetrics && liveMetrics.training_history && liveMetrics.training_history.length > 0) {
  //       setMetrics(prev => ({
  //         ...prev!,
  //         training_history: liveMetrics.training_history
  //       }));
  //     }
  //   }
  // }, [wsLogs, model?.status]);

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

      // Fetch model details
      const modelResponse = await httpfetch.get(`models/${id}/`);

      const modelData = modelResponse.data;
      setModel(modelData);

      // Fetch real analytics data only - no mock fallback
      try {
        const [metricsResponse, statisticsResponse] = await Promise.all([
          httpfetch.get(`models/${id}/metrics/`),
          httpfetch.get(`models/${id}/statistics/`)
        ]);

        const realMetrics = metricsResponse.data.metrics;
        const realStats = statisticsResponse.data.statistics;

        // Set metrics if model is completed - sklearn models may not have training_history
        if (modelData.status === 'completed' || modelData.status === 'complete') {
          setMetrics(realMetrics);
        } else {
          setMetrics(null);
        }

        setStatistics(realStats);
      } catch (analyticsError) {
        setMetrics(null);
        setStatistics(null);
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

  const handleExportReport = () => {
    if (!model || !metrics) return;

    // Generate a comprehensive report
    const report = {
      title: `Model Analysis Report: ${model.name}`,
      generatedAt: new Date().toISOString(),
      model: {
        id: model.id,
        name: model.name,
        type: model.model_type,
        status: model.status,
        targetColumn: model.target_column,
        dataset: model.dataset_name || model.dataset?.name,
        createdAt: model.created_at,
      },
      performanceMetrics: {
        accuracy: metrics.accuracy ? `${(metrics.accuracy * 100).toFixed(2)}%` : 'N/A',
        precision: metrics.precision ? `${(metrics.precision * 100).toFixed(2)}%` : 'N/A',
        recall: metrics.recall ? `${(metrics.recall * 100).toFixed(2)}%` : 'N/A',
        f1Score: metrics.f1_score ? `${(metrics.f1_score * 100).toFixed(2)}%` : 'N/A',
      },
      confusionMatrix: metrics.confusion_matrix || null,
      featureImportance: metrics.feature_importance || [],
      predictionDistribution: metrics.prediction_distribution || [],
      classReport: metrics.class_report || null,
      additionalMetrics: metrics.additional_metrics || {},
      statistics: statistics || null,
    };

    // Convert to JSON and download
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `model_report_${model.name.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleDownloadModel = async () => {
    if (!model) return;

    try {
      const response = await httpfetch.get(`models/${model.id}/download/`, {
        responseType: 'blob',
      });

      // Create download link
      const blob = new Blob([response.data], { type: 'application/octet-stream' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      // Determine file extension based on model type
      let fileExtension = '.joblib'; // Default for sklearn models
      if (model.model_type === 'cnn' || model.model_type === 'neural_network') {
        fileExtension = '.pt'; // PyTorch models
      }
      
      link.download = `${model.name.replace(/\s+/g, '_')}${fileExtension}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err: any) {
      console.error('Error downloading model:', err);
      alert('Failed to download model file. The model may not have been saved yet.');
    }
  };

  const fetchAvailableDatasets = async () => {
    try {
      const response = await httpfetch.get('datasets/');
      const datasets = response.data.results || response.data || [];
      setAvailableDatasets(datasets.map((d: any) => ({ id: d.id, name: d.name, column_info: d.column_info })));
    } catch (err) {
      console.error('Error fetching datasets:', err);
    }
  };

  const parseColumnInfo = (columnInfo: any): string[] => {
    if (!columnInfo) return [];
    // Handle if it's a string (JSON)
    let parsed = columnInfo;
    if (typeof columnInfo === 'string') {
      try {
        parsed = JSON.parse(columnInfo);
      } catch {
        return [];
      }
    }
    // Handle if it's an array
    if (Array.isArray(parsed)) {
      return parsed.map((col: any) => col.name || col);
    }
    // Handle if it's an object with column names as keys
    if (typeof parsed === 'object') {
      return Object.keys(parsed);
    }
    return [];
  };

  const handleDatasetChange = (datasetId: string) => {
    setSelectedDataset(datasetId);
    // Update available columns based on selected dataset
    const dataset = availableDatasets.find(d => d.id === datasetId);
    const columns = parseColumnInfo(dataset?.column_info);
    setDatasetColumns(columns);
    // Reset target column and excluded columns when dataset changes
    setSelectedTargetColumn('');
    setExcludedColumns([]);
    setColumnRecommendations([]);
  };

  const fetchColumnRecommendations = async (datasetId: string, targetCol: string) => {
    if (!datasetId || !targetCol) return;

    try {
      setLoadingRecommendations(true);
      const response = await httpfetch.get(`datasets/${datasetId}/column_recommendations/?target_column=${targetCol}`);
      const recommendations = response.data.recommended_exclusions || [];
      setColumnRecommendations(recommendations);

      // Auto-exclude high confidence recommendations
      const autoExclude = recommendations
        .filter((rec: any) => rec.confidence === 'high')
        .map((rec: any) => rec.column);
      setExcludedColumns(autoExclude);
    } catch (err) {
      console.error('Error fetching column recommendations:', err);
      setColumnRecommendations([]);
    } finally {
      setLoadingRecommendations(false);
    }
  };

  const handleTargetColumnChange = (targetCol: string) => {
    setSelectedTargetColumn(targetCol);
    // Fetch column recommendations when target column changes
    if (selectedDataset && targetCol) {
      fetchColumnRecommendations(selectedDataset, targetCol);
    }
  };

  const handleResetModel = async () => {
    if (!model) return;

    if (!window.confirm(`Are you sure you want to reset "${model.name}"? This will clear all training data and the model file.`)) {
      return;
    }

    setResetting(true);
    try {
      await httpfetch.post(`models/${model.id}/reset/`);
      // Refresh model data
      await fetchModelData(model.id);
      alert('Model has been reset successfully. You can now retrain it.');
    } catch (err: any) {
      console.error('Error resetting model:', err);
      alert(err.response?.data?.error || 'Failed to reset model');
    } finally {
      setResetting(false);
    }
  };

  const handleOpenRetrainDialog = async () => {
    await fetchAvailableDatasets();
    // Pre-select the current dataset if available
    let datasetId = '';
    if (model?.dataset) {
      datasetId = typeof model.dataset === 'object' ? model.dataset.id : model.dataset;
      setSelectedDataset(datasetId);
      // Set columns for current dataset
      const datasets = await httpfetch.get('datasets/').then(r => r.data.results || r.data || []);
      const currentDataset = datasets.find((d: any) => d.id === datasetId);
      const columns = parseColumnInfo(currentDataset?.column_info);
      setDatasetColumns(columns);
    }
    // Pre-select the current target column
    if (model?.target_column) {
      setSelectedTargetColumn(model.target_column);
    }
    // Load current excluded columns from model's training_config
    const currentExcluded = model?.config?.excluded_columns || [];
    setExcludedColumns(currentExcluded);
    // Fetch column recommendations for the current dataset/target
    if (datasetId && model?.target_column) {
      fetchColumnRecommendations(datasetId, model.target_column);
    }
    setRetrainDialogOpen(true);
  };

  const handleRetrain = async () => {
    if (!model) return;

    setRetraining(true);
    try {
      // If dataset or target column changed, update them first
      const currentDatasetId = typeof model.dataset === 'object' ? model.dataset.id : model.dataset;
      const datasetChanged = selectedDataset && selectedDataset !== currentDatasetId;
      const targetChanged = selectedTargetColumn && selectedTargetColumn !== model.target_column;

      if (datasetChanged || targetChanged) {
        const updateData: any = {};
        if (datasetChanged) {
          updateData.dataset_id = selectedDataset;
        }
        if (targetChanged) {
          updateData.target_column = selectedTargetColumn;
        }
        await httpfetch.patch(`models/${model.id}/update_dataset/`, updateData);
      }

      // Update excluded columns in training_config
      const currentExcluded = model?.config?.excluded_columns || [];
      const excludedColumnsChanged = JSON.stringify(excludedColumns.sort()) !== JSON.stringify(currentExcluded.sort());
      if (excludedColumnsChanged) {
        await httpfetch.patch(`models/${model.id}/update_config/`, {
          training_config: { excluded_columns: excludedColumns }
        });
      }

      // Start training
      await httpfetch.post(`models/${model.id}/train/`);
      setRetrainDialogOpen(false);

      // Refresh model data to show training status
      await fetchModelData(model.id);
    } catch (err: any) {
      console.error('Error starting training:', err);
      alert(err.response?.data?.error || 'Failed to start training');
    } finally {
      setRetraining(false);
    }
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
                    {metrics?.is_regression ? 'R² Score' : 'Accuracy'}
                  </Typography>
                  <Typography variant="h6" fontWeight={600} color="success.main">
                    {metrics?.is_regression
                      ? (metrics?.r2_score !== undefined ? `${(metrics.r2_score * 100).toFixed(1)}%` : 'N/A')
                      : (metrics?.accuracy ? `${(metrics.accuracy * 100).toFixed(1)}%` : 'N/A')
                    }
                  </Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="caption" color="text.secondary">
                    Dataset
                  </Typography>
                  <Typography variant="body2" fontWeight={500}>
                    {model.dataset_name || model.dataset?.name || 'N/A'}
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
                <Stack direction="row" spacing={1} justifyContent="flex-end" flexWrap="wrap" useFlexGap>
                  <Button
                    variant="outlined"
                    color="warning"
                    startIcon={resetting ? <CircularProgress size={16} /> : <RestartAlt />}
                    size="small"
                    onClick={handleResetModel}
                    disabled={resetting || model.status === 'training'}
                  >
                    Reset
                  </Button>
                  <Button
                    variant="outlined"
                    color="primary"
                    startIcon={<PlayArrow />}
                    size="small"
                    onClick={handleOpenRetrainDialog}
                    disabled={model.status === 'training'}
                  >
                    Retrain
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<Description />}
                    size="small"
                    onClick={handleExportReport}
                    disabled={!metrics}
                  >
                    Export Report
                  </Button>
                  <Button
                    variant="contained"
                    startIcon={<FileDownload />}
                    size="small"
                    onClick={handleDownloadModel}
                    disabled={model.status !== 'complete' && model.status !== 'completed'}
                  >
                    Download Model
                  </Button>
                </Stack>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Performance Metrics - Different cards for Regression vs Classification */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {metrics?.is_regression ? (
          // Regression Metrics
          <>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <TrendingUp sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
                  <Typography variant="h4" fontWeight={700} color="success.main">
                    {metrics?.r2_score !== undefined ? `${(metrics.r2_score * 100).toFixed(1)}%` : 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    R² Score
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Assessment sx={{ fontSize: 40, color: 'info.main', mb: 1 }} />
                  <Typography variant="h4" fontWeight={700} color="info.main">
                    {metrics?.rmse !== undefined ? metrics.rmse.toLocaleString(undefined, { maximumFractionDigits: 2 }) : 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    RMSE
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <ShowChart sx={{ fontSize: 40, color: 'warning.main', mb: 1 }} />
                  <Typography variant="h4" fontWeight={700} color="warning.main">
                    {metrics?.mae !== undefined ? metrics.mae.toLocaleString(undefined, { maximumFractionDigits: 2 }) : 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    MAE
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <BarChart sx={{ fontSize: 40, color: 'secondary.main', mb: 1 }} />
                  <Typography variant="h4" fontWeight={700} color="secondary.main">
                    {metrics?.mse !== undefined ? metrics.mse.toLocaleString(undefined, { maximumFractionDigits: 0 }) : 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    MSE
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </>
        ) : (
          // Classification Metrics
          <>
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
          </>
        )}
      </Grid>

      {/* No Metrics Message */}
      {!metrics && (
        <Alert severity="info" sx={{ mb: 3 }}>
          <Typography variant="body2">
            📊 Training metrics will appear here after model training is complete.
            {model?.status === 'training' && ' Training is currently in progress - please wait for completion to view charts.'}
          </Typography>
        </Alert>
      )}

      {/* Charts Grid */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {/* Training History - Only show if accuracy data exists and has meaningful values (not all zeros) */}
        {metrics?.training_history && metrics.training_history.length > 0 &&
         metrics.training_history.some(h => h.accuracy !== undefined && h.accuracy !== null && h.accuracy > 0) && (
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
                      {metrics.training_history.some(h => h.val_accuracy !== undefined && h.val_accuracy !== null && h.val_accuracy > 0) && (
                        <Line
                          type="monotone"
                          dataKey="val_accuracy"
                          stroke="#82ca9d"
                          strokeWidth={2}
                          name="Validation Accuracy"
                          dot={{ r: 3 }}
                          activeDot={{ r: 5 }}
                        />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Feature Importance */}
        {metrics?.feature_importance && metrics.feature_importance.length > 0 && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  Feature Importance (Top {Math.min(metrics.feature_importance.length, 15)})
                </Typography>
                <Box sx={{ height: Math.max(400, Math.min(metrics.feature_importance.length, 15) * 35), pt: 2 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsBarChart
                      data={metrics.feature_importance.slice(0, 15)}
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 150, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis
                        type="number"
                        tick={{ fontSize: 12 }}
                        tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                        domain={[0, 'auto']}
                      />
                      <YAxis
                        type="category"
                        dataKey="feature"
                        tick={{ fontSize: 11 }}
                        width={140}
                        interval={0}
                      />
                      <RechartsTooltip
                        formatter={(value: any) => [`${(value * 100).toFixed(2)}%`, 'Importance']}
                        labelFormatter={(label) => `Feature: ${label}`}
                      />
                      <Bar
                        dataKey="importance"
                        fill="#8884d8"
                        radius={[0, 4, 4, 0]}
                      >
                        {metrics.feature_importance.slice(0, 15).map((entry: any, index: number) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Bar>
                    </RechartsBarChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Prediction Distribution */}
        {metrics?.prediction_distribution && metrics.prediction_distribution.length > 0 && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  {metrics.is_regression ? 'Predicted Value Distribution' : 'Prediction Distribution'}
                  <Chip
                    label={metrics.is_regression ? `${metrics.prediction_distribution.length} bins` : `${metrics.prediction_distribution.length} classes`}
                    size="small"
                    sx={{ ml: 1 }}
                  />
                </Typography>
                <Box sx={{ height: 400, pt: 2 }}>
                  {metrics.is_regression ? (
                    // For regression, show as bar chart (histogram)
                    <ResponsiveContainer width="100%" height="100%">
                      <RechartsBarChart
                        data={metrics.prediction_distribution}
                        margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis
                          dataKey="label"
                          tick={{ fontSize: 10 }}
                          angle={-45}
                          textAnchor="end"
                          height={80}
                        />
                        <YAxis tick={{ fontSize: 12 }} />
                        <RechartsTooltip formatter={(value: any) => [`${value} predictions`, 'Count']} />
                        <Bar dataKey="value" fill="#8884d8" radius={[4, 4, 0, 0]}>
                          {metrics.prediction_distribution.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Bar>
                      </RechartsBarChart>
                    </ResponsiveContainer>
                  ) : (
                    // For classification, show as pie chart
                    <ResponsiveContainer width="100%" height="100%">
                      <RechartsPieChart>
                        <Pie
                          data={metrics.prediction_distribution}
                          cx="50%"
                          cy="50%"
                          outerRadius={120}
                          fill="#8884d8"
                          dataKey="value"
                          nameKey="label"
                        >
                          {metrics.prediction_distribution.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <RechartsTooltip formatter={(value: any, name: any) => [value, name]} />
                      </RechartsPieChart>
                    </ResponsiveContainer>
                  )}
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Per-Class Metrics for sklearn models (shown when no training history) */}
        {metrics && !metrics.training_history?.length && metrics.class_report && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  Per-Class Performance
                </Typography>
                <Box sx={{ height: 300, pt: 2 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsBarChart
                      data={Object.entries(metrics.class_report)
                        .filter(([key]) => !['accuracy', 'macro avg', 'weighted avg'].includes(key))
                        .map(([label, values]) => ({
                          class: label,
                          precision: (values.precision * 100),
                          recall: (values.recall * 100),
                          f1: (values['f1-score'] * 100),
                        }))}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="class" tick={{ fontSize: 12 }} />
                      <YAxis tick={{ fontSize: 12 }} domain={[0, 100]} />
                      <RechartsTooltip formatter={(value: any) => [`${value.toFixed(1)}%`, '']} />
                      <Legend />
                      <Bar dataKey="precision" fill="#8884d8" name="Precision" />
                      <Bar dataKey="recall" fill="#82ca9d" name="Recall" />
                      <Bar dataKey="f1" fill="#ffc658" name="F1 Score" />
                    </RechartsBarChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Training/Test Split Info for sklearn models */}
        {metrics && !metrics.training_history?.length && metrics.additional_metrics && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  Data Split Overview
                </Typography>
                <Box sx={{ height: 300, pt: 2 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsPieChart>
                      <Pie
                        data={[
                          { name: 'Training Set', value: metrics.additional_metrics.training_samples || 0 },
                          { name: 'Test Set', value: metrics.additional_metrics.test_samples || 0 },
                        ]}
                        cx="50%"
                        cy="50%"
                        labelLine={true}
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        <Cell fill="#8884d8" />
                        <Cell fill="#82ca9d" />
                      </Pie>
                      <RechartsTooltip formatter={(value: any) => [`${value.toLocaleString()} samples`, '']} />
                      <Legend />
                    </RechartsPieChart>
                  </ResponsiveContainer>
                </Box>
                <Stack direction="row" spacing={4} justifyContent="center" sx={{ mt: 2 }}>
                  <Box textAlign="center">
                    <Typography variant="h5" fontWeight={600} color="primary">
                      {metrics.additional_metrics.training_samples?.toLocaleString() || 'N/A'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">Training Samples</Typography>
                  </Box>
                  <Box textAlign="center">
                    <Typography variant="h5" fontWeight={600} color="success.main">
                      {metrics.additional_metrics.test_samples?.toLocaleString() || 'N/A'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">Test Samples</Typography>
                  </Box>
                </Stack>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Loss Curves - Only show if loss data exists */}
        {metrics?.training_history && metrics.training_history.length > 0 &&
         metrics.training_history.some(h => h.loss !== undefined && h.loss !== null) && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  Training{metrics.training_history.some(h => h.val_loss !== undefined && h.val_loss !== null) ? ' & Validation' : ''} Loss
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
                      {metrics.training_history.some(h => h.val_loss !== undefined && h.val_loss !== null) && (
                        <Line
                          type="monotone"
                          dataKey="val_loss"
                          stroke="#8dd1e1"
                          strokeWidth={2}
                          name="Validation Loss"
                          dot={{ r: 3 }}
                          activeDot={{ r: 5 }}
                        />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Model Performance Radar Chart - Different for Regression vs Classification */}
        {metrics && !metrics.is_regression && (
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
                        { name: 'Accuracy', value: (metrics.accuracy || 0) * 100, fill: '#8884d8' },
                        { name: 'Precision', value: (metrics.precision || 0) * 100, fill: '#82ca9d' },
                        { name: 'Recall', value: (metrics.recall || 0) * 100, fill: '#ffc658' },
                        { name: 'F1 Score', value: (metrics.f1_score || 0) * 100, fill: '#ff7300' },
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

        {/* Regression Target Statistics */}
        {metrics?.is_regression && metrics.target_stats && metrics.target_stats.mean !== undefined && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  Target Value Statistics
                </Typography>
                <Box sx={{ height: 300, pt: 2, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                  <Stack spacing={3}>
                    <Box>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Mean Value
                      </Typography>
                      <Typography variant="h5" fontWeight={600} color="primary.main">
                        {metrics.target_stats.mean.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Standard Deviation
                      </Typography>
                      <Typography variant="h5" fontWeight={600} color="info.main">
                        ±{metrics.target_stats.std.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                      </Typography>
                    </Box>
                    <Stack direction="row" spacing={4}>
                      <Box>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Minimum
                        </Typography>
                        <Typography variant="h6" fontWeight={600} color="warning.main">
                          {metrics.target_stats.min.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                        </Typography>
                      </Box>
                      <Box>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Maximum
                        </Typography>
                        <Typography variant="h6" fontWeight={600} color="success.main">
                          {metrics.target_stats.max.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                        </Typography>
                      </Box>
                    </Stack>
                  </Stack>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Regression R² Score Gauge */}
        {metrics?.is_regression && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  Model Fit Quality (R² Score)
                </Typography>
                <Box sx={{ height: 300, pt: 2 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <RadialBarChart
                      cx="50%"
                      cy="50%"
                      innerRadius="40%"
                      outerRadius="80%"
                      startAngle={180}
                      endAngle={0}
                      data={[
                        { name: 'R² Score', value: Math.max(0, (metrics.r2_score || 0)) * 100, fill: metrics.r2_score && metrics.r2_score > 0.8 ? '#4caf50' : metrics.r2_score && metrics.r2_score > 0.5 ? '#ff9800' : '#f44336' },
                      ]}
                    >
                      <RadialBar
                        minAngle={15}
                        background
                        clockWise
                        dataKey="value"
                        cornerRadius={10}
                      />
                      <RechartsTooltip formatter={(value: any) => [`${value.toFixed(1)}%`, 'R² Score']} />
                    </RadialBarChart>
                  </ResponsiveContainer>
                  <Box sx={{ textAlign: 'center', mt: -8 }}>
                    <Typography variant="h3" fontWeight={700} color={metrics.r2_score && metrics.r2_score > 0.8 ? 'success.main' : metrics.r2_score && metrics.r2_score > 0.5 ? 'warning.main' : 'error.main'}>
                      {metrics.r2_score !== undefined ? `${(metrics.r2_score * 100).toFixed(1)}%` : 'N/A'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {metrics.r2_score && metrics.r2_score > 0.8 ? 'Excellent Fit' : metrics.r2_score && metrics.r2_score > 0.5 ? 'Moderate Fit' : 'Poor Fit'}
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Combined Accuracy and Loss Over Time - Only show if both accuracy and loss data exist with meaningful values */}
        {metrics?.training_history && metrics.training_history.length > 0 &&
         metrics.training_history.some(h => h.accuracy !== undefined && h.accuracy !== null && h.accuracy > 0) &&
         metrics.training_history.some(h => h.loss !== undefined && h.loss !== null) && (
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
                    {statistics?.avg_prediction_time !== undefined && statistics.avg_prediction_time > 0
                      ? `${statistics.avg_prediction_time.toFixed(2)}ms`
                      : 'Not tracked'}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">
                    Training Time
                  </Typography>
                  <Typography variant="body2" fontWeight={500}>
                    {statistics?.training_time && statistics.training_time !== 'N/A'
                      ? statistics.training_time
                      : 'Not recorded'}
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
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">
                    Training Samples
                  </Typography>
                  <Typography variant="body2" fontWeight={500}>
                    {(metrics?.additional_metrics?.training_samples || statistics?.dataset_info?.training_samples)?.toLocaleString() || 'N/A'}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">
                    Test Samples
                  </Typography>
                  <Typography variant="body2" fontWeight={500}>
                    {(metrics?.additional_metrics?.test_samples || statistics?.dataset_info?.test_samples)?.toLocaleString() || 'N/A'}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">
                    Data Quality Score
                  </Typography>
                  <Typography variant="body2" fontWeight={500}>
                    {statistics?.dataset_info?.data_quality_score ? `${statistics.dataset_info.data_quality_score.toFixed(1)}%` : 'N/A'}
                  </Typography>
                </Box>
                <Divider />
                <Typography variant="subtitle2" fontWeight={600}>
                  Prediction Distribution
                </Typography>
                {(metrics?.prediction_distribution?.length > 0 || statistics?.dataset_info?.target_distribution?.length > 0) ? (
                  (metrics?.prediction_distribution || statistics?.dataset_info?.target_distribution || []).map((item: any, index: number) => (
                    <Box key={index} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2" color="text.secondary">
                        {item.label}
                      </Typography>
                      <Typography variant="body2" fontWeight={500}>
                        {(item.count || item.value)?.toLocaleString()}
                      </Typography>
                    </Box>
                  ))
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No distribution data available
                  </Typography>
                )}
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Model Configuration */}
        {statistics?.model_info && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  Model Configuration
                </Typography>
                <Stack spacing={2}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">
                      Framework
                    </Typography>
                    <Chip
                      label={statistics.model_info.framework || 'Unknown'}
                      size="small"
                      color={statistics.model_info.framework === 'PyTorch' ? 'warning' : 'primary'}
                    />
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">
                      Model Type
                    </Typography>
                    <Typography variant="body2" fontWeight={500}>
                      {statistics.model_info.model_type?.split('_').map((w: string) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ') || 'N/A'}
                    </Typography>
                  </Box>
                  {statistics.model_info.hyperparameters && Object.keys(statistics.model_info.hyperparameters).length > 0 && (
                    <>
                      <Divider />
                      <Typography variant="subtitle2" fontWeight={600}>
                        Hyperparameters
                      </Typography>
                      {Object.entries(statistics.model_info.hyperparameters)
                        .filter(([key]) => !['features', 'excluded_columns', 'target'].includes(key))
                        .slice(0, 8)
                        .map(([key, value]) => {
                          // Format the value based on its type
                          let displayValue: string;
                          if (typeof value === 'boolean') {
                            displayValue = value ? 'Yes' : 'No';
                          } else if (Array.isArray(value)) {
                            // Handle layer configurations (fc_layers, conv_layers, etc.)
                            if (value.length > 0 && typeof value[0] === 'object') {
                              displayValue = value.map((layer: any, idx: number) => {
                                if (layer.units) {
                                  return `${layer.units}${layer.activation ? ` (${layer.activation})` : ''}`;
                                } else if (layer.out_channels) {
                                  return `${layer.out_channels}ch`;
                                }
                                return JSON.stringify(layer);
                              }).join(' → ');
                            } else {
                              displayValue = value.join(', ');
                            }
                          } else if (typeof value === 'object' && value !== null) {
                            displayValue = JSON.stringify(value);
                          } else {
                            displayValue = String(value);
                          }

                          return (
                            <Box key={key} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                              <Typography variant="body2" color="text.secondary">
                                {key.split('_').map((w: string) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
                              </Typography>
                              <Typography variant="body2" fontWeight={500} sx={{ maxWidth: '60%', textAlign: 'right' }}>
                                {displayValue}
                              </Typography>
                            </Box>
                          );
                        })}
                    </>
                  )}
                </Stack>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Enhanced Confusion Matrix with Heatmap Visualization */}
        {metrics?.confusion_matrix && metrics.confusion_matrix.length > 0 && (
          <Grid item xs={12} md={metrics.confusion_matrix.length > 3 ? 12 : 6}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  Confusion Matrix
                  {metrics.confusion_matrix.length > 2 && (
                    <Chip
                      label={`${metrics.confusion_matrix.length} classes`}
                      size="small"
                      sx={{ ml: 1 }}
                    />
                  )}
                </Typography>
                {(() => {
                  const matrix = metrics.confusion_matrix;
                  const numClasses = matrix.length;

                  // Get class labels from class_report or prediction_distribution, or use indices
                  const classLabels: string[] = [];
                  if (metrics.class_report) {
                    Object.keys(metrics.class_report).forEach(key => {
                      if (!['accuracy', 'macro avg', 'weighted avg'].includes(key)) {
                        classLabels.push(key);
                      }
                    });
                  }
                  // Fill in any missing labels with indices
                  while (classLabels.length < numClasses) {
                    classLabels.push(`Class ${classLabels.length}`);
                  }

                  // Calculate max value for color intensity
                  const maxVal = Math.max(...matrix.flat());

                  // Calculate overall accuracy from diagonal
                  const totalCorrect = matrix.reduce((sum, row, i) => sum + (row[i] || 0), 0);
                  const totalSamples = matrix.reduce((sum, row) => sum + row.reduce((s, v) => s + v, 0), 0);
                  const overallAccuracy = totalSamples > 0 ? (totalCorrect / totalSamples * 100).toFixed(1) : 'N/A';

                  return (
                    <>
                      <Box sx={{ overflowX: 'auto' }}>
                        <TableContainer component={Paper} variant="outlined" sx={{ mb: 2, minWidth: numClasses > 5 ? 600 : 'auto' }}>
                          <Table size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell sx={{ fontWeight: 600, bgcolor: 'grey.100' }}>Actual \ Predicted</TableCell>
                                {classLabels.slice(0, numClasses).map((label, i) => (
                                  <TableCell
                                    key={i}
                                    align="center"
                                    sx={{
                                      fontWeight: 600,
                                      bgcolor: 'grey.100',
                                      fontSize: numClasses > 5 ? '0.7rem' : '0.875rem',
                                      whiteSpace: 'nowrap'
                                    }}
                                  >
                                    {label.length > 10 ? `${label.slice(0, 8)}...` : label}
                                  </TableCell>
                                ))}
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {matrix.map((row, rowIdx) => (
                                <TableRow key={rowIdx}>
                                  <TableCell
                                    sx={{
                                      fontWeight: 600,
                                      bgcolor: 'grey.100',
                                      fontSize: numClasses > 5 ? '0.7rem' : '0.875rem',
                                      whiteSpace: 'nowrap'
                                    }}
                                  >
                                    {classLabels[rowIdx]?.length > 10 ? `${classLabels[rowIdx].slice(0, 8)}...` : classLabels[rowIdx]}
                                  </TableCell>
                                  {row.map((val, colIdx) => {
                                    const isDiagonal = rowIdx === colIdx;
                                    const intensity = maxVal > 0 ? val / maxVal : 0;
                                    return (
                                      <TableCell
                                        key={colIdx}
                                        align="center"
                                        sx={{
                                          bgcolor: isDiagonal
                                            ? `rgba(76, 175, 80, ${0.1 + intensity * 0.4})`  // Green for correct
                                            : val > 0
                                              ? `rgba(244, 67, 54, ${0.1 + intensity * 0.3})`  // Red for errors
                                              : 'transparent',
                                          fontWeight: isDiagonal ? 600 : 400,
                                          fontSize: numClasses > 5 ? '0.75rem' : '0.875rem',
                                        }}
                                      >
                                        {val}
                                      </TableCell>
                                    );
                                  })}
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      </Box>
                      <Stack direction="row" spacing={2} flexWrap="wrap">
                        <Typography variant="body2" color="text.secondary">
                          Overall Accuracy: <strong>{overallAccuracy}%</strong>
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Total Samples: <strong>{totalSamples.toLocaleString()}</strong>
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Correct: <strong>{totalCorrect.toLocaleString()}</strong>
                        </Typography>
                      </Stack>
                    </>
                  );
                })()}
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>

      {/* Training Logs Section */}
      <Box sx={{ mt: 4 }}>
        <Typography variant="h5" fontWeight={600} gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Timeline /> Training Logs
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          View detailed, real-time training logs and metrics for this model (WebSocket-powered)
        </Typography>
        <TrainingLogsViewerWebSocket
          modelId={String(model.id)}
          modelName={model.name}
          status={model.status}
        />
      </Box>

      {/* Retrain Dialog */}
      <Dialog open={retrainDialogOpen} onClose={() => setRetrainDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Retrain Model</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Retrain this model with the same dataset or select a different one. You can also change the target column and excluded columns.
          </Typography>
          <FormControl fullWidth sx={{ mt: 1 }}>
            <InputLabel>Dataset</InputLabel>
            <Select
              value={selectedDataset}
              label="Dataset"
              onChange={(e) => handleDatasetChange(e.target.value)}
            >
              {availableDatasets.map((dataset) => (
                <MenuItem key={dataset.id} value={dataset.id}>
                  {dataset.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl fullWidth sx={{ mt: 2 }}>
            <InputLabel>Target Column</InputLabel>
            <Select
              value={selectedTargetColumn}
              label="Target Column"
              onChange={(e) => handleTargetColumnChange(e.target.value)}
              disabled={datasetColumns.length === 0}
            >
              {datasetColumns.map((column) => (
                <MenuItem key={column} value={column}>
                  {column}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Column Exclusion Recommendations */}
          {selectedTargetColumn && (
            <Paper elevation={0} sx={{ p: 2, mt: 3, bgcolor: 'background.default', border: '1px solid', borderColor: 'divider' }}>
              <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Info fontSize="small" color="primary" />
                Feature Column Recommendations
              </Typography>

              {loadingRecommendations ? (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 1 }}>
                  <CircularProgress size={16} />
                  <Typography variant="body2" color="text.secondary">
                    Analyzing columns...
                  </Typography>
                </Box>
              ) : columnRecommendations.length > 0 ? (
                <>
                  <Alert severity="warning" sx={{ mb: 2 }}>
                    The following columns may not be suitable for training. Click to toggle exclusion:
                  </Alert>
                  <Stack spacing={1} sx={{ maxHeight: 250, overflowY: 'auto' }}>
                    {columnRecommendations.map((rec: any) => {
                      const isExcluded = excludedColumns.includes(rec.column);
                      return (
                        <Paper
                          key={rec.column}
                          elevation={0}
                          sx={{
                            p: 1.5,
                            border: '1px solid',
                            borderColor: isExcluded ? 'warning.main' : 'divider',
                            bgcolor: isExcluded ? 'warning.50' : 'background.paper',
                            cursor: 'pointer',
                            '&:hover': { borderColor: 'primary.main' }
                          }}
                          onClick={() => {
                            if (isExcluded) {
                              setExcludedColumns(excludedColumns.filter(col => col !== rec.column));
                            } else {
                              setExcludedColumns([...excludedColumns, rec.column]);
                            }
                          }}
                        >
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Box sx={{ flex: 1 }}>
                              <Typography variant="body2" fontWeight="medium">
                                {rec.column}
                                <Chip
                                  label={rec.confidence}
                                  size="small"
                                  color={rec.confidence === 'high' ? 'error' : 'warning'}
                                  sx={{ ml: 1, height: 20 }}
                                />
                              </Typography>
                              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                                {rec.reason}
                              </Typography>
                              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                                Type: {rec.dtype} • Unique: {rec.unique_values}
                              </Typography>
                            </Box>
                            <Chip
                              label={isExcluded ? "Excluded" : "Included"}
                              color={isExcluded ? "warning" : "default"}
                              size="small"
                            />
                          </Box>
                        </Paper>
                      );
                    })}
                  </Stack>
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
                    Click any column to toggle exclusion. Excluded columns will not be used for training.
                  </Typography>
                </>
              ) : (
                <Alert severity="success" icon={<CheckCircle />}>
                  All columns look good for training! No exclusions recommended.
                </Alert>
              )}

              {excludedColumns.length > 0 && (
                <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
                  <Typography variant="caption" color="text.secondary">
                    Currently excluded ({excludedColumns.length}): {excludedColumns.join(', ')}
                  </Typography>
                </Box>
              )}
            </Paper>
          )}

          {selectedDataset && model?.dataset && (
            (() => {
              const currentDatasetId = typeof model.dataset === 'object' ? model.dataset.id : model.dataset;
              const datasetChanged = selectedDataset !== currentDatasetId;
              const targetChanged = selectedTargetColumn && selectedTargetColumn !== model.target_column;
              const currentExcluded = model?.config?.excluded_columns || [];
              const excludedChanged = JSON.stringify(excludedColumns.sort()) !== JSON.stringify(currentExcluded.sort());
              if (datasetChanged || targetChanged || excludedChanged) {
                return (
                  <Alert severity="info" sx={{ mt: 2 }}>
                    {datasetChanged && `Dataset will be changed. `}
                    {targetChanged && `Target column will be changed to "${selectedTargetColumn}". `}
                    {excludedChanged && `Excluded columns will be updated (${excludedColumns.length} columns).`}
                  </Alert>
                );
              }
              return null;
            })()
          )}
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={() => setRetrainDialogOpen(false)} disabled={retraining}>
            Cancel
          </Button>
          <Button
            variant="contained"
            startIcon={retraining ? <CircularProgress size={16} /> : <PlayArrow />}
            onClick={handleRetrain}
            disabled={retraining || !selectedDataset || !selectedTargetColumn}
          >
            {retraining ? 'Starting...' : 'Start Training'}
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}

export default ModelAnalysisPage;