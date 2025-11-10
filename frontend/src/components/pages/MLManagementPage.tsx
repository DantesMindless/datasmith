import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Tabs, 
  Tab,
  Card,
  CardContent,
  CardActions,
  Button,
  Grid,
  Chip,
  LinearProgress,
  Alert,
  CircularProgress,
  Container,
  Fade,
  Stack,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Snackbar,
} from '@mui/material';
import {
  PlayArrow,
  Refresh,
  Analytics,
  DataObject,
  TrendingUp,
  Storage,
  Assessment,
  MoreVert,
  Delete,
  Edit,
  Download,
  FileCopy,
  CheckCircle,
  Error as ErrorIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import CreateModel from '../CreateModel';
import httpfetch from '../../utils/axios';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

interface MLManagementPageProps {
  onNavigateToAnalysis?: (modelId: number) => void;
}
// Authentication is now handled by httpfetch via JWT tokens

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`ml-tabpanel-${index}`}
      aria-labelledby={`ml-tab-${index}`}
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

function ModelsList({ onNavigateToAnalysis }: { onNavigateToAnalysis?: (modelId: number) => void }) {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [predictionDialog, setPredictionDialog] = useState({
    open: false,
    modelId: null,
    modelName: '',
    inputData: '',
    schema: null,
    tips: []
  });
  const [predictionResult, setPredictionResult] = useState({
    open: false,
    result: null,
    isLoading: false
  });
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success' as 'success' | 'error' | 'info' | 'warning'
  });
  const [optionsMenu, setOptionsMenu] = useState({
    anchorEl: null,
    modelId: null,
    model: null
  });

  const showNotification = (message: string, severity: 'success' | 'error' | 'info' | 'warning' = 'success') => {
    setSnackbar({ open: true, message, severity });
  };

  const fetchModels = async () => {
    try {
      setLoading(true);
      const response = await httpfetch.get('models/');
      setModels(response.data.results || response.data);
      setError('');
    } catch (err: any) {
      console.error('Error fetching models:', err);
      setError(err.response?.data?.error || 'Failed to fetch models');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed': return 'success';
      case 'training': return 'warning';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  const handleTrain = async (modelId: number) => {
    try {
      const response = await httpfetch.post(`models/${modelId}/train/`, {});
      showNotification(response.data.message || 'Training started', 'success');
      // Refresh models to show updated status
      await fetchModels();
    } catch (err: any) {
      console.error('Error training model:', err);
      showNotification(err.response?.data?.error || 'Failed to start training', 'error');
    }
  };

  const handleForceTrain = async (modelId: number) => {
    const confirmTraining = window.confirm(
      'This will start synchronous training (may take a while). Continue?'
    );

    if (!confirmTraining) return;

    try {
      const response = await httpfetch.post(`models/${modelId}/force_train/`, {});
      showNotification(`Training completed! Status: ${response.data.status}, Accuracy: ${response.data.accuracy || 'N/A'}`, 'success');
      // Refresh models to show updated status
      await fetchModels();
    } catch (err: any) {
      console.error('Error force training model:', err);
      showNotification(err.response?.data?.error || 'Failed to train model', 'error');
    }
  };

  // Helper function to parse CSV input (supports single or multiple rows)
  function parseCsvInput(csv: string): Record<string, string | number> | Array<Record<string, string | number>> {
    const lines = csv.trim().split('\n');
    if (lines.length < 2) throw new Error('CSV must have header and at least one data row');

    const headers = lines[0].split(',').map(h => h.trim());
    const dataRows = lines.slice(1); // Get all rows after header

    // Parse all data rows
    const parsedRows = dataRows.map(row => {
      const values = row.split(',').map(v => v.trim());
      if (headers.length !== values.length) throw new Error('Header and row length mismatch');

      const data: Record<string, string | number> = {};
      headers.forEach((h, i) => {
        data[h] = isNaN(Number(values[i])) ? values[i] : Number(values[i]);
      });
      return data;
    });

    // Return single object if only one row, otherwise return array
    return parsedRows.length === 1 ? parsedRows[0] : parsedRows;
  }

  const handlePredictClick = async (modelId: number, modelName: string) => {
    try {
      // Fetch prediction schema from the API
      const response = await httpfetch.get(`models/${modelId}/prediction_info/`);
      const schema = response.data.schema;

      let exampleData = '';
      if (schema?.input_type === 'image') {
        // For image models, provide image upload instruction
        exampleData = 'Image prediction - please upload an image file';
      } else if (schema?.input_features && schema.input_features.length > 0) {
        // Use actual feature names from the model
        const features = schema.input_features;
        const header = features.join(',');
        const sampleValues = features.map(() => '0.0').join(',');
        exampleData = `${header}\n${sampleValues}`;
      } else {
        // Fallback to generic example
        exampleData = `feature1,feature2,feature3\n1.0,2.0,3.0`;
      }

      setPredictionDialog({
        open: true,
        modelId,
        modelName,
        inputData: exampleData,
        schema: schema,
        tips: response.data.tips || []
      });
    } catch (err: any) {
      console.error('Error fetching prediction schema:', err);
      // Fallback to generic example if API call fails
      setPredictionDialog({
        open: true,
        modelId,
        modelName,
        inputData: `feature1,feature2,feature3\n1.0,2.0,3.0`,
        schema: null,
        tips: []
      });
    }
  };

  const handlePredictSubmit = async () => {
    try {
      setPredictionResult({ open: false, result: null, isLoading: true });

      let parsedData;
      let response;

      // Handle image prediction differently
      if (predictionDialog.schema?.input_type === 'image') {
        if (!predictionDialog.inputData || typeof predictionDialog.inputData !== 'object') {
          setPredictionResult({
            open: true,
            result: {
              success: false,
              error: 'Please select an image file.'
            },
            isLoading: false
          });
          return;
        }

        // Use FormData for file upload
        const formData = new FormData();
        formData.append('image', predictionDialog.inputData);

        response = await httpfetch.post(
          `models/${predictionDialog.modelId}/predict/`,
          formData,
          {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          }
        );

        parsedData = { filename: predictionDialog.inputData.name };
      } else {
        // Handle tabular data (CSV)
        try {
          parsedData = parseCsvInput(predictionDialog.inputData);
        } catch (parseError: any) {
          setPredictionResult({
            open: true,
            result: {
              success: false,
              error: 'Invalid CSV format. Please enter valid CSV data.'
            },
            isLoading: false
          });
          return;
        }

        const requestData = {
          data: parsedData
        };

        response = await httpfetch.post(`models/${predictionDialog.modelId}/predict/`, requestData);
      }

      // Close input dialog and show results
      setPredictionDialog({ open: false, modelId: null, modelName: '', inputData: '', schema: null, tips: [] });

      setPredictionResult({
        open: true,
        result: {
          success: true,
          prediction: response.data.prediction,
          predictions: response.data.predictions,
          count: response.data.count,
          inputData: parsedData,
          modelName: predictionDialog.modelName,
          modelId: predictionDialog.modelId
        },
        isLoading: false
      });

    } catch (err: any) {
      console.error('Error making prediction:', err);
      setPredictionResult({
        open: true,
        result: {
          success: false,
          error: err.response?.data?.error || 'Failed to make prediction'
        },
        isLoading: false
      });
    }
  };

  const handlePredictCancel = () => {
    setPredictionDialog({ open: false, modelId: null, modelName: '', inputData: '', schema: null, tips: [] });
  };

  const handleOptionsClick = (event, modelId, model) => {
    setOptionsMenu({
      anchorEl: event.currentTarget,
      modelId,
      model
    });
  };

  const handleOptionsClose = () => {
    setOptionsMenu({
      anchorEl: null,
      modelId: null,
      model: null
    });
  };

  const handleDeleteModel = async (modelId) => {
    const confirmDelete = window.confirm(
      'Are you sure you want to delete this model? This action cannot be undone.'
    );

    if (!confirmDelete) {
      handleOptionsClose();
      return;
    }

    try {
      const response = await httpfetch.delete(`models/${modelId}/`);
      // Check if deletion was successful (status 200, 204, or similar)
      if (response.status >= 200 && response.status < 300) {
        await fetchModels(); // Refresh the models list
        handleOptionsClose();
        showNotification('Model deleted successfully', 'success');
      } else {
        throw new Error(`Unexpected response status: ${response.status}`);
      }
    } catch (err: any) {
      console.error('Error deleting model:', err);
      // Only show error if it's actually a failure
      if (err.response?.status >= 400) {
        const errorMessage = typeof err.response?.data === 'object'
          ? err.response?.data?.error || err.response?.data?.detail || 'Failed to delete model'
          : err.response?.data || 'Failed to delete model';
        showNotification(errorMessage, 'error');
      } else {
        // For other errors, try refreshing to see if delete actually worked
        try {
          await fetchModels();
          showNotification('Model deletion completed - please verify in the list', 'warning');
        } catch {
          showNotification('Failed to delete model and unable to refresh list', 'error');
        }
      }
      handleOptionsClose();
    }
  };

  const handleDuplicateModel = async (model) => {
    try {
      const duplicateData = {
        name: `${model.name} (Copy)`,
        model_type: model.model_type,
        dataset: model.dataset,
        target_column: model.target_column,
        config: model.config || {}
      };

      await httpfetch.post('models/', duplicateData);
      await fetchModels(); // Refresh the models list
      handleOptionsClose();
      showNotification('Model duplicated successfully', 'success');
    } catch (err: any) {
      console.error('Error duplicating model:', err);
      showNotification(err.response?.data?.error || 'Failed to duplicate model', 'error');
      handleOptionsClose();
    }
  };

  const handleTest = async (modelId: number) => {
    try {
      const response = await httpfetch.post(`models/${modelId}/test/`, {});

      const results = response.data;
      const message = `${results.model_name}: ${(results.accuracy * 100).toFixed(2)}% accuracy`;

      showNotification(message, 'success');
    } catch (err: any) {
      console.error('Error testing model:', err);
      showNotification(err.response?.data?.error || 'Failed to test model', 'error');
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" fontWeight={600}>
          ML Models
        </Typography>
        <Tooltip title="Refresh models">
          <IconButton onClick={fetchModels} disabled={loading}>
            <Refresh />
          </IconButton>
        </Tooltip>
      </Box>
      
      {error && (
        <Fade in>
          <Alert severity="error" sx={{ mb: 3, borderRadius: 2 }}>
            {error}
          </Alert>
        </Fade>
      )}
      
      {loading ? (
        <Box display="flex" justifyContent="center" alignItems="center" py={8}>
          <Stack alignItems="center" spacing={2}>
            <CircularProgress size={48} />
            <Typography variant="body2" color="text.secondary">
              Loading your models...
            </Typography>
          </Stack>
        </Box>
      ) : models.length === 0 ? (
        <Card sx={{ textAlign: 'center', py: 6, borderStyle: 'dashed', borderWidth: 2, borderColor: 'divider' }}>
          <CardContent>
            <TrendingUp sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              No models yet
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Create your first ML model to get started with analytics
            </Typography>
            <Button variant="contained" startIcon={<Analytics />}>
              Create Model
            </Button>
          </CardContent>
        </Card>
      ) : (
        <Grid container spacing={3}>
          {models.map((model, index) => (
            <Grid item xs={12} sm={6} lg={4} key={model.id}>
              <Fade in timeout={300 + index * 100}>
                <Card 
                  sx={{ 
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    transition: 'all 0.3s ease-in-out',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
                    }
                  }}
                >
                  <CardContent sx={{ flex: 1, pb: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', mb: 2 }}>
                      <Box sx={{ flex: 1 }}>
                        <Typography variant="h6" fontWeight={600} gutterBottom noWrap>
                          {model.name}
                        </Typography>
                        <Chip
                          label={model.status}
                          color={getStatusColor(model.status)}
                          size="small"
                          sx={{ mb: 1.5 }}
                        />
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <IconButton
                          size="small"
                          onClick={(event) => handleOptionsClick(event, model.id, model)}
                          sx={{
                            width: 32,
                            height: 32,
                            '&:hover': {
                              bgcolor: 'action.hover'
                            }
                          }}
                        >
                          <MoreVert fontSize="small" />
                        </IconButton>
                        <Box
                          sx={{
                            width: 40,
                            height: 40,
                            borderRadius: 2,
                            bgcolor: 'primary.lighter',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            ml: 0.5,
                          }}
                        >
                          <Analytics color="primary" />
                        </Box>
                      </Box>
                    </Box>
                    
                    <Stack spacing={1.5}>
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          Model Type
                        </Typography>
                        <Typography variant="body2" fontWeight={500}>
                          {model.model_type || model.type}
                        </Typography>
                      </Box>
                      
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          Dataset
                        </Typography>
                        <Typography variant="body2" fontWeight={500}>
                          {model.dataset_name || 'N/A'}
                        </Typography>
                      </Box>
                      
                      {model.accuracy && (
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            Accuracy
                          </Typography>
                          <Typography variant="body2" fontWeight={500} color="success.main">
                            {(typeof model.accuracy === 'number' ? (model.accuracy * 100).toFixed(1) : model.accuracy)}%
                          </Typography>
                        </Box>
                      )}
                      
                      {model.status === 'Training' && (
                        <Box>
                          <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                            Training Progress
                          </Typography>
                          <LinearProgress sx={{ borderRadius: 1, height: 6 }} />
                        </Box>
                      )}
                    </Stack>
                  </CardContent>
                  
                  <CardActions sx={{ p: 2, pt: 0 }}>
                    {(model.status?.toLowerCase() === 'completed' ||
                      model.status?.toLowerCase() === 'complete' ||
                      model.status?.toLowerCase() === 'finished' ||
                      model.status?.toLowerCase() === 'success') ? (
                      <Stack direction="row" spacing={1} width="100%">
                        <Button
                          size="small"
                          variant="contained"
                          startIcon={<PlayArrow />}
                          onClick={() => handlePredictClick(model.id, model.name)}
                          sx={{ flex: 1 }}
                        >
                          Predict
                        </Button>
                        {onNavigateToAnalysis && (
                          <Button
                            size="small"
                            variant="outlined"
                            startIcon={<Analytics />}
                            onClick={() => onNavigateToAnalysis(model.id)}
                            sx={{ flex: 1 }}
                          >
                            Analyze
                          </Button>
                        )}
                        <Button
                          size="small"
                          variant="text"
                          onClick={() => handleTrain(model.id)}
                          sx={{ flex: 1 }}
                        >
                          Retrain
                        </Button>
                      </Stack>
                    ) : model.status?.toLowerCase() === 'pending' ? (
                      <Stack direction="row" spacing={1} width="100%">
                        <Button
                          size="small"
                          variant="contained"
                          startIcon={<PlayArrow />}
                          onClick={() => handleTrain(model.id)}
                          sx={{ flex: 1 }}
                        >
                          Start Training
                        </Button>
                        {onNavigateToAnalysis && (
                          <Button
                            size="small"
                            variant="outlined"
                            startIcon={<Analytics />}
                            onClick={() => onNavigateToAnalysis(model.id)}
                            sx={{ flex: 1 }}
                          >
                            View
                          </Button>
                        )}
                      </Stack>
                    ) : (
                      <Stack direction="row" spacing={1} width="100%">
                        {onNavigateToAnalysis && (
                          <Button
                            size="small"
                            variant="contained"
                            startIcon={<Analytics />}
                            onClick={() => onNavigateToAnalysis(model.id)}
                            sx={{ flex: 1 }}
                          >
                            Watch Training
                          </Button>
                        )}
                        <Chip
                          label="Training..."
                          color="warning"
                          size="small"
                          sx={{ flex: 1, height: 32 }}
                        />
                      </Stack>
                    )}
                  </CardActions>
                </Card>
              </Fade>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Prediction Dialog */}
      <Dialog open={predictionDialog.open} onClose={handlePredictCancel} maxWidth="md" fullWidth>
        <DialogTitle>
          Make Prediction - {predictionDialog.modelName}
        </DialogTitle>
        <DialogContent>
          {/* Show tips if available */}
          {predictionDialog.tips && predictionDialog.tips.length > 0 && (
            <Alert severity="info" sx={{ mb: 2 }}>
              <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                Tips for prediction:
              </Typography>
              <ul style={{ margin: 0, paddingLeft: '20px' }}>
                {predictionDialog.tips.map((tip, index) => (
                  <li key={index}>
                    <Typography variant="body2">{tip}</Typography>
                  </li>
                ))}
              </ul>
            </Alert>
          )}

          {/* Show required features for tabular data */}
          {predictionDialog.schema?.input_features && predictionDialog.schema?.input_type !== 'image' && (
            <Box sx={{ mb: 2, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
              <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                Required Features ({predictionDialog.schema.input_features.length}):
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {predictionDialog.schema.input_features.map((feature, index) => (
                  <Chip key={index} label={feature} size="small" color="primary" variant="outlined" />
                ))}
              </Box>
            </Box>
          )}

          {/* Image upload for image models */}
          {predictionDialog.schema?.input_type === 'image' ? (
            <Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Upload an image file for classification
              </Typography>
              <Button
                variant="outlined"
                component="label"
                fullWidth
                sx={{ mb: 2, p: 2, height: 100, borderStyle: 'dashed' }}
              >
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="body1" fontWeight={500}>
                    Click to select image
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Supported: JPG, PNG, BMP, GIF
                  </Typography>
                </Box>
                <input
                  type="file"
                  hidden
                  accept="image/*"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) {
                      setPredictionDialog(prev => ({
                        ...prev,
                        inputData: file
                      }));
                    }
                  }}
                />
              </Button>
              {predictionDialog.inputData && typeof predictionDialog.inputData === 'object' && (
                <Alert severity="success" sx={{ mb: 2 }}>
                  <Typography variant="body2">
                    Selected: {predictionDialog.inputData.name} ({(predictionDialog.inputData.size / 1024).toFixed(2)} KB)
                  </Typography>
                </Alert>
              )}
            </Box>
          ) : (
            // CSV input for tabular models
            <Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Enter the input data for prediction in <strong>CSV format</strong>:
              </Typography>
              <TextField
                fullWidth
                multiline
                rows={6}
                variant="outlined"
                value={predictionDialog.inputData}
                onChange={(e) => setPredictionDialog(prev => ({
                  ...prev,
                  inputData: e.target.value
                }))}
                placeholder={predictionDialog.inputData}
                sx={{ fontFamily: 'monospace' }}
              />
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                Example:<br />
                <code style={{ whiteSpace: 'pre-wrap' }}>{predictionDialog.inputData}</code>
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handlePredictCancel}>
            Cancel
          </Button>
          <Button
            onClick={handlePredictSubmit}
            variant="contained"
            startIcon={<PlayArrow />}
          >
            Predict
          </Button>
        </DialogActions>
      </Dialog>

      {/* Prediction Results Dialog */}
      <Dialog
        open={predictionResult.open}
        onClose={() => setPredictionResult({ open: false, result: null, isLoading: false })}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Assessment color="primary" />
            <Typography variant="h6" fontWeight={600}>
              Prediction Results
            </Typography>
          </Box>
        </DialogTitle>
        <DialogContent>
          {predictionResult.isLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', py: 4 }}>
              <Stack alignItems="center" spacing={2}>
                <CircularProgress size={48} />
                <Typography variant="body2" color="text.secondary">
                  Making prediction...
                </Typography>
              </Stack>
            </Box>
          ) : predictionResult.result?.success ? (
            <Box>
              {/* Success Alert */}
              <Alert severity="success" sx={{ mb: 3 }}>
                <Typography variant="subtitle2" fontWeight={600}>
                  Prediction completed successfully!
                </Typography>
              </Alert>

              {/* Model Info */}
              {predictionResult.result.modelName && (
                <Box sx={{ mb: 3, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    Model
                  </Typography>
                  <Typography variant="body1" fontWeight={600}>
                    {predictionResult.result.modelName}
                  </Typography>
                </Box>
              )}

              {/* Input Data - Only show for single predictions */}
              {!Array.isArray(predictionResult.result.inputData) && (
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                    Input Data:
                  </Typography>
                  <Card variant="outlined" sx={{ p: 2, bgcolor: 'grey.50' }}>
                    <Stack spacing={1}>
                      {Object.entries(predictionResult.result.inputData || {}).map(([key, value]) => (
                        <Box key={key} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="body2" color="text.secondary">
                            {key}:
                          </Typography>
                          <Typography variant="body2" fontWeight={500}>
                            {String(value)}
                          </Typography>
                        </Box>
                      ))}
                    </Stack>
                  </Card>
                </Box>
              )}

              {/* Prediction Result */}
              <Box>
                <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                  Prediction Results:
                </Typography>
                {predictionResult.result.prediction !== undefined ? (
                  // Single prediction
                  <Card
                    variant="outlined"
                    sx={{
                      p: 3,
                      bgcolor: 'primary.lighter',
                      border: '2px solid',
                      borderColor: 'primary.main'
                    }}
                  >
                    <Typography
                      variant="h4"
                      fontWeight={700}
                      color="primary.main"
                      sx={{ textAlign: 'center' }}
                    >
                      {String(predictionResult.result.prediction)}
                    </Typography>
                  </Card>
                ) : predictionResult.result.predictions && predictionResult.result.predictions.length > 0 ? (
                  // Multiple predictions - show as table with input data
                  <Box>
                    <Alert severity="info" sx={{ mb: 2 }}>
                      <Typography variant="body2">
                        Predicted {predictionResult.result.predictions.length} rows successfully
                      </Typography>
                    </Alert>
                    <Card variant="outlined" sx={{ overflow: 'hidden' }}>
                      <Box sx={{ overflowX: 'auto', maxHeight: 400 }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                          <thead>
                            <tr style={{ backgroundColor: '#f5f5f5', borderBottom: '2px solid #e0e0e0' }}>
                              <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: 600, fontSize: '0.875rem' }}>
                                #
                              </th>
                              {Array.isArray(predictionResult.result.inputData) && predictionResult.result.inputData[0] &&
                                Object.keys(predictionResult.result.inputData[0]).map(key => (
                                  <th key={key} style={{ padding: '12px 16px', textAlign: 'left', fontWeight: 600, fontSize: '0.875rem', color: '#666' }}>
                                    {key}
                                  </th>
                                ))
                              }
                              <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: 600, fontSize: '0.875rem', color: '#1976d2' }}>
                                Prediction
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {predictionResult.result.predictions.map((pred, index) => (
                              <tr
                                key={index}
                                style={{
                                  borderBottom: '1px solid #e0e0e0',
                                  backgroundColor: index % 2 === 0 ? '#fafafa' : 'white'
                                }}
                              >
                                <td style={{ padding: '12px 16px', fontSize: '0.875rem', fontWeight: 500 }}>
                                  {index + 1}
                                </td>
                                {Array.isArray(predictionResult.result.inputData) && predictionResult.result.inputData[index] &&
                                  Object.values(predictionResult.result.inputData[index]).map((value, i) => (
                                    <td key={i} style={{ padding: '12px 16px', fontSize: '0.875rem', color: '#666' }}>
                                      {String(value)}
                                    </td>
                                  ))
                                }
                                <td style={{ padding: '12px 16px', fontSize: '0.875rem', fontWeight: 600, color: '#1976d2' }}>
                                  {String(pred)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </Box>
                    </Card>
                  </Box>
                ) : (
                  <Alert severity="info">
                    Prediction completed but no results returned.
                  </Alert>
                )}
              </Box>
            </Box>
          ) : (
            <Alert severity="error">
              <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                Prediction Failed
              </Typography>
              <Typography variant="body2">
                {predictionResult.result?.error || 'An unknown error occurred'}
              </Typography>
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => setPredictionResult({ open: false, result: null, isLoading: false })}
            variant="contained"
          >
            Close
          </Button>
          {predictionResult.result?.success && (
            <Button
              onClick={() => {
                setPredictionResult({ open: false, result: null, isLoading: false });
                handlePredictClick(predictionResult.result.modelId, predictionResult.result.modelName);
              }}
              variant="outlined"
            >
              Make Another Prediction
            </Button>
          )}
        </DialogActions>
      </Dialog>

      {/* Options Menu */}
      <Menu
        anchorEl={optionsMenu.anchorEl}
        open={Boolean(optionsMenu.anchorEl)}
        onClose={handleOptionsClose}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
      >
        <MenuItem onClick={() => handleDuplicateModel(optionsMenu.model)}>
          <ListItemIcon>
            <FileCopy fontSize="small" />
          </ListItemIcon>
          <ListItemText>Duplicate Model</ListItemText>
        </MenuItem>
        <MenuItem onClick={() => handleDeleteModel(optionsMenu.modelId)}>
          <ListItemIcon>
            <Delete fontSize="small" color="error" />
          </ListItemIcon>
          <ListItemText sx={{ color: 'error.main' }}>Delete Model</ListItemText>
        </MenuItem>
      </Menu>

      {/* Notification Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={() => setSnackbar({ ...snackbar, open: false })}
          severity={snackbar.severity}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}


export default function MLManagementPage({ onNavigateToAnalysis }: MLManagementPageProps = {}) {
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const tabConfig = [
    { label: 'Models', icon: <Analytics /> },
    { label: 'Create Model', icon: <DataObject /> },
  ];

  return (
    <Container maxWidth="xl" sx={{ py: 3, height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Page Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" fontWeight={700} gutterBottom>
          Machine Learning
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Manage your ML models, datasets, and create new intelligent solutions
        </Typography>
      </Box>
      
      {/* Navigation Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs 
          value={activeTab} 
          onChange={handleTabChange} 
          aria-label="ml management tabs"
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
              id={`ml-tab-${index}`} 
              aria-controls={`ml-tabpanel-${index}`}
              sx={{ gap: 1 }}
            />
          ))}
        </Tabs>
      </Box>

      {/* Tab Content */}
      <Box sx={{ flex: 1, overflowY: 'auto' }}>
        <TabPanel value={activeTab} index={0}>
          <ModelsList onNavigateToAnalysis={onNavigateToAnalysis} />
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          <CreateModel />
        </TabPanel>
      </Box>
    </Container>
  );
}