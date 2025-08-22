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
  CircularProgress
} from '@mui/material';
import CreateModel from '../CreateModel';
import httpfetch from '../../utils/axios';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}
const uname = "u@u.com";
const pass = "password";

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

function ModelsList() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchModels = async () => {
    try {
      setLoading(true);
      const response = await httpfetch.get('/models/', {
        auth: {
          username: uname,
          password: pass,
        },
      });
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
      const response = await httpfetch.post(`/models/${modelId}/train/`, {}, {
        auth: {
          username: uname,
          password: pass,
        },
      });
      alert(response.data.message || 'Training started');
      // Refresh models to show updated status
      await fetchModels();
    } catch (err: any) {
      console.error('Error training model:', err);
      alert(err.response?.data?.error || 'Failed to start training');
    }
  };

  const handlePredict = async (modelId: number) => {
    // TODO: Implement prediction UI/workflow
    alert('Prediction functionality coming soon!');
    console.log('Making prediction with model:', modelId);
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Your Models
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      
      {loading ? (
        <Box display="flex" justifyContent="center" p={4}>
          <CircularProgress />
        </Box>
      ) : models.length === 0 ? (
        <Typography variant="body1" color="textSecondary">
          No models created yet. Create your first model to get started!
        </Typography>
      ) : (
        <Grid container spacing={3}>
          {models.map((model) => (
            <Grid item xs={12} sm={6} md={4} key={model.id}>
              <Card>
                <CardContent>
                  <Typography variant="h6" component="div" gutterBottom>
                    {model.name}
                  </Typography>
                  
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    Type: {model.model_type || model.type}
                  </Typography>
                  
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    Dataset: {model.dataset?.name || model.dataset || 'N/A'}
                  </Typography>
                  
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
                    <Chip 
                      label={model.status} 
                      color={getStatusColor(model.status)}
                      size="small"
                    />
                    {model.accuracy && (
                      <Typography variant="body2" sx={{ ml: 2 }}>
                        Accuracy: {(typeof model.accuracy === 'number' ? (model.accuracy * 100).toFixed(1) : model.accuracy)}%
                      </Typography>
                    )}
                  </Box>
                  
                  {model.status === 'Training' && (
                    <LinearProgress sx={{ mt: 2 }} />
                  )}
                </CardContent>
                
                <CardActions>
                  {model.status === 'Completed' ? (
                    <>
                      <Button 
                        size="small" 
                        variant="contained" 
                        onClick={() => handlePredict(model.id)}
                      >
                        Predict
                      </Button>
                      <Button 
                        size="small" 
                        onClick={() => handleTrain(model.id)}
                      >
                        Retrain
                      </Button>
                    </>
                  ) : model.status === 'Pending' ? (
                    <Button 
                      size="small" 
                      variant="contained"
                      onClick={() => handleTrain(model.id)}
                    >
                      Train
                    </Button>
                  ) : (
                    <Button size="small" disabled>
                      Training...
                    </Button>
                  )}
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
    </Box>
  );
}

function DatasetsList() {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchDatasets = async () => {
    try {
      setLoading(true);
      const response = await httpfetch.get('/datasets/', {
        auth: {
          username: uname,
          password: pass,
        },
      });
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

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Your Datasets
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      
      {loading ? (
        <Box display="flex" justifyContent="center" p={4}>
          <CircularProgress />
        </Box>
      ) : datasets.length === 0 ? (
        <Typography variant="body1" color="textSecondary">
          No datasets uploaded yet. Upload datasets to train models!
        </Typography>
      ) : (
        <Grid container spacing={3}>
          {datasets.map((dataset) => (
            <Grid item xs={12} sm={6} md={4} key={dataset.id}>
              <Card>
                <CardContent>
                  <Typography variant="h6" component="div" gutterBottom>
                    {dataset.name}
                  </Typography>
                  
                  <Typography variant="body2" color="textSecondary">
                    Type: {dataset.file_type || dataset.type || 'Unknown'}
                  </Typography>
                  
                  {dataset.file_size && (
                    <Typography variant="body2" color="textSecondary">
                      Size: {dataset.file_size}
                    </Typography>
                  )}
                  
                  {dataset.rows_count && (
                    <Typography variant="body2" color="textSecondary">
                      Rows: {dataset.rows_count}
                    </Typography>
                  )}
                  
                  {dataset.columns_count && (
                    <Typography variant="body2" color="textSecondary">
                      Columns: {dataset.columns_count}
                    </Typography>
                  )}
                  
                  <Typography variant="body2" color="textSecondary">
                    Created: {new Date(dataset.created_at || dataset.uploadDate).toLocaleDateString()}
                  </Typography>
                </CardContent>
                
                <CardActions>
                  <Button size="small">View</Button>
                  <Button size="small">Download</Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
    </Box>
  );
}

export default function MLManagementPage() {
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  return (
    <Box sx={{ 
      width: '100%', 
      height: '100vh',
      pt: {
        xs: "calc(4px + var(--Header-height))",
        sm: "calc(4px + var(--Header-height))",
        md: 1,
      },
      pb: { xs: 1, sm: 1, md: 1 },
      display: "flex",
      flexDirection: "column",
      overflowY: 'scroll'
    }}>
      <Typography variant="h4" component="h1" gutterBottom sx={{ px: 3, pt: 2 }}>
        Machine Learning
      </Typography>
      
      <Box sx={{ borderBottom: 1, borderColor: 'divider', px: 3 }}>
        <Tabs value={activeTab} onChange={handleTabChange} aria-label="ml management tabs">
          <Tab label="Models" id="ml-tab-0" aria-controls="ml-tabpanel-0" />
          <Tab label="Datasets" id="ml-tab-1" aria-controls="ml-tabpanel-1" />
          <Tab label="Create Model" id="ml-tab-2" aria-controls="ml-tabpanel-2" />
        </Tabs>
      </Box>

      <TabPanel value={activeTab} index={0}>
        <ModelsList />
      </TabPanel>
      
      <TabPanel value={activeTab} index={1}>
        <DatasetsList />
      </TabPanel>
      
      <TabPanel value={activeTab} index={2}>
        <CreateModel />
      </TabPanel>
    </Box>
  );
}