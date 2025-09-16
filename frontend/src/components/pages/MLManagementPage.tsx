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
    inputData: ''
  });
  const [optionsMenu, setOptionsMenu] = useState({
    anchorEl: null,
    modelId: null,
    model: null
  });

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
      alert(response.data.message || 'Training started');
      // Refresh models to show updated status
      await fetchModels();
    } catch (err: any) {
      console.error('Error training model:', err);
      alert(err.response?.data?.error || 'Failed to start training');
    }
  };

  const handleForceTrain = async (modelId: number) => {
    const confirmTraining = window.confirm(
      'This will start synchronous training (may take a while). Continue?'
    );

    if (!confirmTraining) return;

    try {
      const response = await httpfetch.post(`models/${modelId}/force_train/`, {});
      alert(`Training completed!\nStatus: ${response.data.status}\nAccuracy: ${response.data.accuracy || 'N/A'}`);
      // Refresh models to show updated status
      await fetchModels();
    } catch (err: any) {
      console.error('Error force training model:', err);
      alert(err.response?.data?.error || 'Failed to train model');
    }
  };

  const handlePredictClick = (modelId: number, modelName: string) => {
    setPredictionDialog({
      open: true,
      modelId,
      modelName,
      inputData: JSON.stringify({
        feature1: 1.0,
        feature2: 2.0,
        feature3: 3.0
      }, null, 2)
    });
  };

  const handlePredictSubmit = async () => {
    try {
      let parsedData;
      try {
        parsedData = JSON.parse(predictionDialog.inputData);
      } catch (parseError) {
        alert('Invalid JSON format. Please enter valid JSON data.');
        return;
      }

      const requestData = {
        data: parsedData
      };

      const response = await httpfetch.post(`models/${predictionDialog.modelId}/predict/`, requestData);

      if (response.data.prediction !== undefined) {
        alert(`Prediction result: ${response.data.prediction}`);
      } else if (response.data.predictions) {
        alert(`Predictions: ${response.data.predictions.join(', ')}`);
      } else {
        alert('Prediction completed successfully');
      }

      setPredictionDialog({ open: false, modelId: null, modelName: '', inputData: '' });
    } catch (err: any) {
      console.error('Error making prediction:', err);
      alert(err.response?.data?.error || 'Failed to make prediction');
    }
  };

  const handlePredictCancel = () => {
    setPredictionDialog({ open: false, modelId: null, modelName: '', inputData: '' });
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
        alert('Model deleted successfully');
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
        alert(errorMessage);
      } else {
        // For other errors, try refreshing to see if delete actually worked
        try {
          await fetchModels();
          alert('Model deletion completed - please verify in the list');
        } catch {
          alert('Failed to delete model and unable to refresh list');
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
      alert('Model duplicated successfully');
    } catch (err: any) {
      console.error('Error duplicating model:', err);
      alert(err.response?.data?.error || 'Failed to duplicate model');
      handleOptionsClose();
    }
  };

  const handleTest = async (modelId: number) => {
    try {
      const response = await httpfetch.post(`models/${modelId}test/`, {});

      const results = response.data;
      const message = `Test Results for ${results.model_name}:
Accuracy: ${(results.accuracy * 100).toFixed(2)}%
Model Type: ${results.model_type}
Status: ${results.status}`;

      alert(message);
    } catch (err: any) {
      console.error('Error testing model:', err);
      alert(err.response?.data?.error || 'Failed to test model');
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
                          {model.dataset?.name || model.dataset || 'N/A'}
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
                      <Stack direction="column" spacing={1} width="100%">
                        <Stack direction="row" spacing={0.5}>
                          <Button
                            size="small"
                            variant="contained"
                            startIcon={<PlayArrow />}
                            onClick={() => handlePredictClick(model.id, model.name)}
                            sx={{ flex: 1 }}
                          >
                            Predict
                          </Button>
                          <Button
                            size="small"
                            variant="outlined"
                            startIcon={<Assessment />}
                            onClick={() => handleTest(model.id)}
                            sx={{ flex: 1 }}
                          >
                            Test
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
                        </Stack>
                        <Stack direction="row" spacing={1}>
                          <Button
                            size="small"
                            variant="text"
                            onClick={() => handleTrain(model.id)}
                            sx={{ flex: 1 }}
                          >
                            Retrain
                          </Button>
                          <Button
                            size="small"
                            variant="text"
                            onClick={() => handleForceTrain(model.id)}
                            sx={{ flex: 1, fontSize: '0.75rem' }}
                          >
                            Force
                          </Button>
                        </Stack>
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
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={() => handleForceTrain(model.id)}
                          title="Immediate synchronous training"
                        >
                          Force
                        </Button>
                      </Stack>
                    ) : (
                      <Stack direction="row" spacing={1} width="100%">
                        <Button size="small" disabled sx={{ flex: 1 }}>
                          Training in Progress...
                        </Button>
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={() => handleForceTrain(model.id)}
                          title="Reset and force training"
                        >
                          Reset & Force
                        </Button>
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
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Enter the input data for prediction in JSON format:
          </Typography>
          <TextField
            fullWidth
            multiline
            rows={10}
            variant="outlined"
            value={predictionDialog.inputData}
            onChange={(e) => setPredictionDialog(prev => ({
              ...prev,
              inputData: e.target.value
            }))}
            placeholder='{"feature1": 1.0, "feature2": 2.0}'
            sx={{ fontFamily: 'monospace' }}
          />
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            Example: {"{\"feature1\": 1.0, \"feature2\": 2.0, \"feature3\": 3.0}"}
          </Typography>
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
    </Box>
  );
}

function DatasetsList() {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [optionsMenu, setOptionsMenu] = useState({
    anchorEl: null,
    datasetId: null,
    dataset: null
  });

  const fetchDatasets = async () => {
    try {
      setLoading(true);
      const response = await httpfetch.get('datasets/');
      setDatasets(response.data.results || response.data);
      setError('');
    } catch (err: any) {
      console.error('Error fetching datasets:', err);
      setError(err.response?.data?.error || 'Failed to fetch datasets');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  const handleDatasetOptionsClick = (event, datasetId, dataset) => {
    setOptionsMenu({
      anchorEl: event.currentTarget,
      datasetId,
      dataset
    });
  };

  const handleDatasetOptionsClose = () => {
    setOptionsMenu({
      anchorEl: null,
      datasetId: null,
      dataset: null
    });
  };

  const handleDeleteDataset = async (datasetId) => {
    const confirmDelete = window.confirm(
      'Are you sure you want to delete this dataset? This action cannot be undone.'
    );

    if (!confirmDelete) {
      handleDatasetOptionsClose();
      return;
    }

    try {
      const response = await httpfetch.delete(`datasets/${datasetId}/`);
      // Check if deletion was successful (status 200, 204, or similar)
      if (response.status >= 200 && response.status < 300) {
        await fetchDatasets(); // Refresh the datasets list
        handleDatasetOptionsClose();
        alert('Dataset deleted successfully');
      } else {
        throw new Error(`Unexpected response status: ${response.status}`);
      }
    } catch (err: any) {
      console.error('Error deleting dataset:', err);
      // Only show error if it's actually a failure
      if (err.response?.status >= 400) {
        const errorMessage = typeof err.response?.data === 'object'
          ? err.response?.data?.error || err.response?.data?.detail || 'Failed to delete dataset'
          : err.response?.data || 'Failed to delete dataset';
        alert(errorMessage);
      } else {
        // For other errors, try refreshing to see if delete actually worked
        try {
          await fetchDatasets();
          alert('Dataset deletion completed - please verify in the list');
        } catch {
          alert('Failed to delete dataset and unable to refresh list');
        }
      }
      handleDatasetOptionsClose();
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" fontWeight={600}>
          Datasets
        </Typography>
        <Tooltip title="Refresh datasets">
          <IconButton onClick={fetchDatasets} disabled={loading}>
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
              Loading your datasets...
            </Typography>
          </Stack>
        </Box>
      ) : datasets.length === 0 ? (
        <Card sx={{ textAlign: 'center', py: 6, borderStyle: 'dashed', borderWidth: 2, borderColor: 'divider' }}>
          <CardContent>
            <Storage sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              No datasets yet
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Upload your first dataset to start training ML models
            </Typography>
            <Button variant="contained" startIcon={<DataObject />}>
              Upload Dataset
            </Button>
          </CardContent>
        </Card>
      ) : (
        <Grid container spacing={3}>
          {datasets.map((dataset, index) => (
            <Grid item xs={12} sm={6} lg={4} key={dataset.id}>
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
                          {dataset.name}
                        </Typography>
                        <Chip
                          label={dataset.file_type || dataset.type || 'Unknown'}
                          color="default"
                          size="small"
                          sx={{ mb: 1.5 }}
                        />
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <IconButton
                          size="small"
                          onClick={(event) => handleDatasetOptionsClick(event, dataset.id, dataset)}
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
                            bgcolor: 'secondary.lighter',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            ml: 0.5,
                          }}
                        >
                          <DataObject color="secondary" />
                        </Box>
                      </Box>
                    </Box>
                    
                    <Stack spacing={1.5}>
                      {dataset.file_size && (
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            File Size
                          </Typography>
                          <Typography variant="body2" fontWeight={500}>
                            {dataset.file_size}
                          </Typography>
                        </Box>
                      )}
                      
                      {dataset.rows_count && (
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            Rows
                          </Typography>
                          <Typography variant="body2" fontWeight={500}>
                            {dataset.rows_count.toLocaleString()}
                          </Typography>
                        </Box>
                      )}
                      
                      {dataset.columns_count && (
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            Columns
                          </Typography>
                          <Typography variant="body2" fontWeight={500}>
                            {dataset.columns_count}
                          </Typography>
                        </Box>
                      )}
                      
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          Created
                        </Typography>
                        <Typography variant="body2" fontWeight={500}>
                          {new Date(dataset.created_at || dataset.uploadDate).toLocaleDateString()}
                        </Typography>
                      </Box>
                    </Stack>
                  </CardContent>
                  
                  <CardActions sx={{ p: 2, pt: 0 }}>
                    <Stack direction="row" spacing={1} width="100%">
                      <Button 
                        size="small" 
                        variant="outlined"
                        sx={{ flex: 1 }}
                      >
                        View
                      </Button>
                      <Button 
                        size="small" 
                        variant="text"
                      >
                        Download
                      </Button>
                    </Stack>
                  </CardActions>
                </Card>
              </Fade>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Dataset Options Menu */}
      <Menu
        anchorEl={optionsMenu.anchorEl}
        open={Boolean(optionsMenu.anchorEl)}
        onClose={handleDatasetOptionsClose}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
      >
        <MenuItem onClick={() => handleDeleteDataset(optionsMenu.datasetId)}>
          <ListItemIcon>
            <Delete fontSize="small" color="error" />
          </ListItemIcon>
          <ListItemText sx={{ color: 'error.main' }}>Delete Dataset</ListItemText>
        </MenuItem>
      </Menu>
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
    { label: 'Datasets', icon: <Storage /> },
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
          <DatasetsList />
        </TabPanel>
        
        <TabPanel value={activeTab} index={2}>
          <CreateModel />
        </TabPanel>
      </Box>
    </Container>
  );
}