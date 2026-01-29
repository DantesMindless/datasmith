import React, { useState, useEffect } from "react";
import {
  Container,
  Box,
  Typography,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  CircularProgress,
  Chip,
  Stack,
  Paper,
  Divider
} from "@mui/material";
import { Warning, CheckCircle, Info, TrendingUp } from "@mui/icons-material";
import { modelConfigSchemas, ConfigField } from "../utils/modelSchema";
import httpfetch from "../utils/axios";
import { useAppContext } from "../providers/useAppContext";

// Authentication is now handled by httpfetch via JWT tokens

export default function CreateModelPage() {
  const { cudaEnabled } = useAppContext();
  const [name, setName] = useState("");
  const [modelType, setModelType] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [targetColumn, setTargetColumn] = useState("");
  const [datasets, setDatasets] = useState([]);
  const [datasetColumns, setDatasetColumns] = useState<string[]>([]);
  const [configValues, setConfigValues] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [compatibilityInfo, setCompatibilityInfo] = useState<any>(null);
  const [loadingCompatibility, setLoadingCompatibility] = useState(false);
  const [excludedColumns, setExcludedColumns] = useState<string[]>([]);
  const [columnRecommendations, setColumnRecommendations] = useState<any[]>([]);
  const [loadingRecommendations, setLoadingRecommendations] = useState(false);

  const [dialogOpenKey, setDialogOpenKey] = useState<string | null>(null);
  const [layerDraft, setLayerDraft] = useState<Record<string, any>>({});

  const handleConfigChange = (key: string, value: any) => {
    setConfigValues((prev) => ({ ...prev, [key]: value }));
  };

  const openLayerDialog = (fieldKey: string) => {
    setDialogOpenKey(fieldKey);
    // Initialize with default values based on layer type
    // This ensures defaults are actually set, not just displayed
    if (fieldKey === "dense_layers" || fieldKey === "layer_config") {
      setLayerDraft({ units: 64, activation: "relu" });
    } else if (fieldKey === "conv_layers") {
      setLayerDraft({ out_channels: 32, kernel_size: 3, activation: "relu", dropout: 0 });
    } else if (fieldKey === "fc_layers") {
      setLayerDraft({ units: 128, activation: "relu", dropout: 0 });
    } else if (fieldKey === "pooling_layers") {
      setLayerDraft({ type: "max", pool_size: 2 });
    } else if (fieldKey === "dropout_layers") {
      setLayerDraft({ rate: 0.2 });
    } else {
      setLayerDraft({});
    }
  };

  const closeLayerDialog = () => {
    setDialogOpenKey(null);
    setLayerDraft({});
  };

  const saveLayer = () => {
    if (!dialogOpenKey) return;
    const currentArray = configValues[dialogOpenKey] || [];
    handleConfigChange(dialogOpenKey, [...currentArray, layerDraft]);
    closeLayerDialog();
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!datasetId) {
      setError("Please select a dataset");
      return;
    }

    // Get selected dataset to check if it's an image dataset
    const selectedDataset = datasets.find(d => d.id == datasetId);
    const isImageDataset = selectedDataset?.is_image_dataset || selectedDataset?.dataset_type === 'image';

    // Validate dataset type matches model type
    if (modelType === 'cnn' && !isImageDataset) {
      setError("CNN models require an image dataset. Please select an image dataset or choose a different model type.");
      return;
    }

    if (modelType !== 'cnn' && isImageDataset) {
      setError("This model type requires a tabular dataset (CSV). Please select a CSV dataset or use CNN for image data.");
      return;
    }

    // Only require target column for non-image datasets
    if (!isImageDataset && !targetColumn) {
      setError("Please select a target column");
      return;
    }

    const payload = {
      name,
      model_type: modelType,
      dataset: datasetId,
      target_column: targetColumn || 'class',  // Default to 'class' for image datasets
      training_config: {
        ...configValues,
        excluded_columns: excludedColumns,
        use_cuda: cudaEnabled,  // Use global CUDA setting
      },
    };
    
    try {
      setLoading(true);
      setError("");
      const response = await httpfetch.post('models/', payload);
      setSuccess("Model created successfully!");
      // Reset form
      setName("");
      setModelType("");
      setDatasetId("");
      setTargetColumn("");
      setDatasetColumns([]);
      setConfigValues({});
      setExcludedColumns([]);
      setColumnRecommendations([]);
    } catch (err: any) {
      console.error("Error creating model:", err);
      setError(err.response?.data?.error || "Failed to create model");
    } finally {
      setLoading(false);
    }
  };

  const fetchDatasets = async () => {
    try {
      const response = await httpfetch.get('datasets/');
      setDatasets(response.data.results || response.data);
    } catch (err) {
      console.error('Error fetching datasets:', err);
    }
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

  const fetchDatasetColumns = async (datasetId: string) => {
    try {
      const response = await httpfetch.get(`datasets/${datasetId}/columns/`);
      setDatasetColumns(response.data.columns || []);
    } catch (err) {
      console.error('Error fetching dataset columns:', err);
      // Fallback: try to get columns from dataset metadata
      try {
        const dataset = datasets.find(d => d.id == datasetId);
        if (dataset && dataset.columns) {
          setDatasetColumns(dataset.columns);
        } else {
          // Silently set empty columns - validation will handle compatibility
          setDatasetColumns([]);
        }
      } catch (fallbackErr) {
        // Silently fail - compatibility check will handle validation
        setDatasetColumns([]);
      }
    }
  };

  const checkCompatibility = async (modelTypeToCheck: string, datasetIdToCheck: string) => {
    if (!modelTypeToCheck || !datasetIdToCheck) {
      setCompatibilityInfo(null);
      return;
    }

    try {
      setLoadingCompatibility(true);
      const response = await httpfetch.post('models/check_compatibility/', {
        model_type: modelTypeToCheck,
        dataset_id: datasetIdToCheck
      });
      setCompatibilityInfo(response.data);
    } catch (err: any) {
      console.error('Error checking compatibility:', err);
      setCompatibilityInfo(null);
    } finally {
      setLoadingCompatibility(false);
    }
  };

  const handleDatasetChange = (newDatasetId: string) => {
    setDatasetId(newDatasetId);
    setTargetColumn(""); // Reset target column when dataset changes
    setDatasetColumns([]); // Clear existing columns
    setExcludedColumns([]); // Clear excluded columns
    setColumnRecommendations([]); // Clear recommendations

    if (newDatasetId) {
      fetchDatasetColumns(newDatasetId);
      // Check compatibility if model type is already selected
      if (modelType) {
        checkCompatibility(modelType, newDatasetId);
      }
    } else {
      setCompatibilityInfo(null);
    }
  };

  const handleModelTypeChange = (newModelType: string) => {
    setModelType(newModelType);

    // build initial config with defaults
    const defaults: Record<string, any> = {};
    const schema = modelConfigSchemas[newModelType] || [];
    schema.forEach((field) => {
      if (field.default !== undefined) {
        defaults[field.key] = field.default;
      }
    });

    setConfigValues(defaults);

    // Check compatibility if dataset is already selected
    if (datasetId) {
      checkCompatibility(newModelType, datasetId);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  useEffect(() => {
    if (datasetId && targetColumn && !isImageDataset) {
      fetchColumnRecommendations(datasetId, targetColumn);
    }
  }, [datasetId, targetColumn]);

  const fields: ConfigField[] = modelType ? modelConfigSchemas[modelType] || [] : [];

  // Check if selected dataset is an image dataset
  const selectedDataset = datasets.find(d => d.id == datasetId);
  const isImageDataset = selectedDataset?.is_image_dataset || selectedDataset?.dataset_type === 'image';

  return (
    <Container maxWidth="md">
      <Box
        sx={{
          mt: 4,
          p: 3,
          border: "1px solid #ddd",
          borderRadius: 2,
          boxShadow: 1,
        }}
      >
        <Typography variant="h5" gutterBottom>
          Create New Model
        </Typography>
        
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        {success && (
          <Alert severity="success" sx={{ mb: 2 }}>
            {success}
          </Alert>
        )}
        
        <form onSubmit={handleSubmit}>
          <TextField
            fullWidth
            label="Model Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            margin="normal"
            required
          />

          <FormControl fullWidth margin="normal">
            <InputLabel id="dataset-label">Dataset</InputLabel>
            <Select
              labelId="dataset-label"
              value={datasetId}
              onChange={(e) => handleDatasetChange(e.target.value)}
              required
            >
              {datasets.map((dataset: any) => (
                <MenuItem key={dataset.id} value={dataset.id}>
                  {dataset.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {!isImageDataset && datasetColumns.length > 0 && (
            <FormControl fullWidth margin="normal">
              <InputLabel id="target-column-label">Target Column</InputLabel>
              <Select
                labelId="target-column-label"
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                required
              >
                {datasetColumns.map((column: string) => (
                  <MenuItem key={column} value={column}>
                    {column}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          )}

          {/* Column Exclusion Recommendations */}
          {!isImageDataset && targetColumn && (
            <Paper elevation={0} sx={{ p: 2, mt: 2, bgcolor: 'background.default', border: '1px solid', borderColor: 'divider' }}>
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
                    The following columns may not be suitable for training. We recommend excluding them:
                  </Alert>
                  <Stack spacing={1}>
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
              ) : targetColumn ? (
                <Alert severity="success" icon={<CheckCircle />}>
                  All columns look good for training! No exclusions recommended.
                </Alert>
              ) : null}
            </Paper>
          )}

          {isImageDataset && (
            <Alert severity="info" sx={{ mt: 2 }}>
              <Typography variant="body2">
                <strong>Image Dataset Selected:</strong> Labels will be automatically extracted from folder names. No target column selection needed.
              </Typography>
            </Alert>
          )}

          <FormControl fullWidth margin="normal">
            <InputLabel id="model-type-label">Model Type</InputLabel>
            <Select
                labelId="model-type-label"
                value={modelType}
                onChange={(e) => handleModelTypeChange(e.target.value)}
                required
               >
              {/* ========== CLASSIFICATION ========== */}
              <MenuItem disabled sx={{ fontWeight: 'bold', bgcolor: 'action.hover' }}>
                <em>📊 Classification (for categories/labels)</em>
              </MenuItem>
              <MenuItem value="logistic_regression">Logistic Regression</MenuItem>
              <MenuItem value="naive_bayes">Naive Bayes</MenuItem>
              <MenuItem value="svm">Support Vector Classifier (SVC)</MenuItem>
              <MenuItem value="decision_tree">Decision Tree Classifier</MenuItem>
              <MenuItem value="random_forest">Random Forest Classifier</MenuItem>
              <MenuItem value="knn">K-Nearest Neighbors Classifier</MenuItem>
              <MenuItem value="gradient_boosting">Gradient Boosting Classifier</MenuItem>
              <MenuItem value="xgboost">XGBoost Classifier</MenuItem>
              <MenuItem value="lightgbm">LightGBM Classifier</MenuItem>
              <MenuItem value="adaboost">AdaBoost Classifier</MenuItem>

              {/* ========== REGRESSION ========== */}
              <MenuItem disabled sx={{ fontWeight: 'bold', bgcolor: 'action.hover', mt: 1 }}>
                <em>📈 Regression (for continuous values)</em>
              </MenuItem>
              <MenuItem value="linear_regression">Linear Regression</MenuItem>
              <MenuItem value="decision_tree_regressor">Decision Tree Regressor</MenuItem>
              <MenuItem value="random_forest_regressor">Random Forest Regressor</MenuItem>
              <MenuItem value="svr">Support Vector Regressor (SVR)</MenuItem>
              <MenuItem value="knn_regressor">K-Nearest Neighbors Regressor</MenuItem>
              <MenuItem value="gradient_boosting_regressor">Gradient Boosting Regressor</MenuItem>
              <MenuItem value="xgboost_regressor">XGBoost Regressor</MenuItem>
              <MenuItem value="lightgbm_regressor">LightGBM Regressor</MenuItem>
              <MenuItem value="adaboost_regressor">AdaBoost Regressor</MenuItem>

              {/* ========== DEEP LEARNING ========== */}
              <MenuItem disabled sx={{ fontWeight: 'bold', bgcolor: 'action.hover', mt: 1 }}>
                <em>🧠 Deep Learning</em>
              </MenuItem>
              <MenuItem value="neural_network">Neural Network (PyTorch)</MenuItem>
              <MenuItem value="cnn">Convolutional Neural Network (CNN)</MenuItem>
            </Select>
          </FormControl>

          {/* Compatibility Info Display */}
          {loadingCompatibility && (
            <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
              <CircularProgress size={20} />
              <Typography variant="body2" color="text.secondary">
                Checking compatibility...
              </Typography>
            </Box>
          )}

          {compatibilityInfo && compatibilityInfo.validation && (
            <Paper elevation={0} sx={{ mt: 2, p: 2, bgcolor: 'background.default', border: '1px solid', borderColor: 'divider' }}>
              <Stack spacing={1.5}>
                {/* Compatibility Status */}
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {compatibilityInfo.validation.is_compatible ? (
                    <CheckCircle color="success" fontSize="small" />
                  ) : (
                    <Warning color="error" fontSize="small" />
                  )}
                  <Typography variant="subtitle2" fontWeight={600}>
                    Compatibility: {compatibilityInfo.validation.is_compatible ? 'Compatible' : 'Not Compatible'}
                  </Typography>
                  <Chip
                    label={`Score: ${compatibilityInfo.validation.score}/100`}
                    size="small"
                    color={compatibilityInfo.validation.score >= 80 ? 'success' : compatibilityInfo.validation.score >= 50 ? 'warning' : 'error'}
                    sx={{ ml: 'auto' }}
                  />
                </Box>

                {/* Errors */}
                {compatibilityInfo.validation.errors && compatibilityInfo.validation.errors.length > 0 && (
                  <Box>
                    <Typography variant="caption" fontWeight={600} color="error.main" sx={{ display: 'block', mb: 0.5 }}>
                      Errors:
                    </Typography>
                    {compatibilityInfo.validation.errors.map((err: string, idx: number) => (
                      <Alert key={idx} severity="error" sx={{ py: 0.5, fontSize: '0.813rem' }}>
                        {err}
                      </Alert>
                    ))}
                  </Box>
                )}

                {/* Warnings */}
                {compatibilityInfo.validation.warnings && compatibilityInfo.validation.warnings.length > 0 && (
                  <Box>
                    <Typography variant="caption" fontWeight={600} color="warning.main" sx={{ display: 'block', mb: 0.5 }}>
                      Warnings:
                    </Typography>
                    {compatibilityInfo.validation.warnings.map((warn: string, idx: number) => (
                      <Alert key={idx} severity="warning" sx={{ py: 0.5, fontSize: '0.813rem' }}>
                        {warn}
                      </Alert>
                    ))}
                  </Box>
                )}

                {/* Recommendations */}
                {compatibilityInfo.validation.recommendations && compatibilityInfo.validation.recommendations.length > 0 && (
                  <Box>
                    <Typography variant="caption" fontWeight={600} color="info.main" sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
                      <TrendingUp fontSize="small" />
                      Recommendations:
                    </Typography>
                    {compatibilityInfo.validation.recommendations.map((rec: string, idx: number) => (
                      <Alert key={idx} severity="info" sx={{ py: 0.5, fontSize: '0.813rem' }}>
                        {rec}
                      </Alert>
                    ))}
                  </Box>
                )}
              </Stack>
            </Paper>
          )}

          {fields.length > 0 && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6" gutterBottom>
                Configuration
              </Typography>
              {fields.map((field) => {
 
                if (field.type === "select") {
                  const rawValue = configValues[field.key] ?? field.default;
                  // Handle null values for MUI Select - convert to empty string for display
                  const value = rawValue === null ? "__null__" : rawValue;
                  return (
                    <FormControl key={field.key} fullWidth margin="normal">
                      <InputLabel>{field.label}</InputLabel>
                      <Select
                        value={value}
                        label={field.label}
                        onChange={(e) => {
                          const newValue = e.target.value === "__null__" ? null : e.target.value;
                          handleConfigChange(field.key, newValue);
                        }}
                      >
                        {field.options?.map((opt) => (
                          <MenuItem key={String(opt.value)} value={opt.value === null ? "__null__" : opt.value}>
                            {opt.label}
                          </MenuItem>
                        ))}
                      </Select>
                      {field.helperText && (
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                          {field.helperText}
                        </Typography>
                      )}
                    </FormControl>
                  );
                }

                if (field.type === "layers") {
                  const layersArray = configValues[field.key] || field.default || [];
                  return (
                    <Box key={field.key} sx={{ mt: 3 }}>
                      <Typography variant="subtitle1" gutterBottom>
                        {field.label}
                      </Typography>

                      {layersArray.length === 0 && (
                        <Typography variant="body2" color="text.secondary">
                          No layers added yet
                        </Typography>
                      )}

                      {layersArray.map((layer: any, idx: number) => (
                        <Box
                          key={idx}
                          sx={{
                            border: "1px solid #ccc",
                            borderRadius: 1,
                            p: 2,
                            mb: 1,
                          }}
                        >
                          <Typography variant="body2" fontWeight="bold">
                            Layer {idx + 1}
                          </Typography>
                          {Object.entries(layer).map(([k, v]) => (
                            <Typography key={k} variant="body2">
                              {k}: {typeof v === 'object' && v !== null ? JSON.stringify(v) : String(v)}
                            </Typography>
                          ))}
                          <Button
                            color="error"
                            size="small"
                            onClick={() => {
                              const updated = layersArray.filter((_: any, i: number) => i !== idx);
                              handleConfigChange(field.key, updated);
                            }}
                          >
                            Remove
                          </Button>
                        </Box>
                      ))}

                      <Button
                        variant="outlined"
                        onClick={() => openLayerDialog(field.key)}
                        sx={{ mt: 1 }}
                      >
                        + Add Layer
                      </Button>
                    </Box>
                  );
                }

                if (field.type === "number" || field.type === "text") {
                  const value = configValues[field.key] ?? field.default ?? "";
                  return (
                    <TextField
                      key={field.key}
                      fullWidth
                      margin="normal"
                      label={field.label}
                      type={field.type}
                      value={value}
                      onChange={(e) => {
                        const newValue = field.type === "number" ? Number(e.target.value) : e.target.value;
                        handleConfigChange(field.key, newValue);
                      }}
                      helperText={field.helperText}
                    />
                  );
                }

                return null;
              })}
            </Box>
          )}

          <Box sx={{ mt: 4, textAlign: "right" }}>
            <Button 
              type="submit" 
              variant="contained" 
              size="large"
              disabled={loading}
              startIcon={loading ? <CircularProgress size={20} /> : null}
            >
              {loading ? 'Creating...' : 'Create Model'}
            </Button>
          </Box>
        </form>
      </Box>

      <Dialog open={dialogOpenKey !== null} onClose={closeLayerDialog}>
        <DialogTitle>Add Layer</DialogTitle>
        <DialogContent
          sx={{
            display: "flex",
            flexDirection: "column",
            gap: 2,
            pt: 2,
            minWidth: "300px",
          }}
        >
          {/* Dense/Hidden layers (for neural_network layer_config and dense_layers) */}
          {(dialogOpenKey === "dense_layers" || dialogOpenKey === "layer_config") && (
            <>
              <Box>
                <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                  Units
                </Typography>
                <TextField
                  fullWidth
                  size="small"
                  type="number"
                  placeholder="e.g. 128"
                  value={layerDraft.units || ""}
                  onChange={(e) =>
                    setLayerDraft({ ...layerDraft, units: Number(e.target.value) })
                  }
                  helperText="Number of neurons in this layer"
                />
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                  Activation
                </Typography>
                <TextField
                  select
                  fullWidth
                  size="small"
                  value={layerDraft.activation || "relu"}
                  onChange={(e) =>
                    setLayerDraft({ ...layerDraft, activation: e.target.value })
                  }
                >
                  <MenuItem value="relu">ReLU</MenuItem>
                  <MenuItem value="tanh">Tanh</MenuItem>
                  <MenuItem value="sigmoid">Sigmoid</MenuItem>
                  <MenuItem value="leaky_relu">Leaky ReLU</MenuItem>
                  <MenuItem value="softmax">Softmax</MenuItem>
                </TextField>
              </Box>
            </>
          )}

          {/* Convolution layers */}
          {dialogOpenKey === "conv_layers" && (
            <>
              <Box>
                <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                  Output Channels (Filters)
                </Typography>
                <TextField
                  fullWidth
                  size="small"
                  type="number"
                  placeholder="e.g. 32"
                  value={layerDraft.out_channels || ""}
                  onChange={(e) =>
                    setLayerDraft({ ...layerDraft, out_channels: Number(e.target.value) })
                  }
                  helperText="Number of output channels/filters"
                />
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                  Kernel Size
                </Typography>
                <TextField
                  fullWidth
                  size="small"
                  type="number"
                  placeholder="e.g. 3"
                  value={layerDraft.kernel_size || 3}
                  onChange={(e) =>
                    setLayerDraft({ ...layerDraft, kernel_size: Number(e.target.value) })
                  }
                  helperText="Size of the convolution kernel (e.g., 3 for 3x3)"
                />
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                  Activation
                </Typography>
                <TextField
                  select
                  fullWidth
                  size="small"
                  value={layerDraft.activation || "relu"}
                  onChange={(e) =>
                    setLayerDraft({ ...layerDraft, activation: e.target.value })
                  }
                >
                  <MenuItem value="relu">ReLU</MenuItem>
                  <MenuItem value="tanh">Tanh</MenuItem>
                  <MenuItem value="sigmoid">Sigmoid</MenuItem>
                  <MenuItem value="leaky_relu">Leaky ReLU</MenuItem>
                </TextField>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                  Dropout Rate
                </Typography>
                <TextField
                  fullWidth
                  size="small"
                  type="number"
                  placeholder="e.g. 0.2"
                  value={layerDraft.dropout || 0}
                  onChange={(e) =>
                    setLayerDraft({ ...layerDraft, dropout: Number(e.target.value) })
                  }
                  helperText="Dropout probability for this layer (0-0.5)"
                  inputProps={{ min: 0, max: 0.8, step: 0.1 }}
                />
              </Box>
            </>
          )}

          {/* FC layers for CNN */}
          {dialogOpenKey === "fc_layers" && (
            <>
              <Box>
                <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                  Units
                </Typography>
                <TextField
                  fullWidth
                  size="small"
                  type="number"
                  placeholder="e.g. 128"
                  value={layerDraft.units || ""}
                  onChange={(e) =>
                    setLayerDraft({ ...layerDraft, units: Number(e.target.value) })
                  }
                  helperText="Number of neurons in this FC layer"
                />
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                  Activation
                </Typography>
                <TextField
                  select
                  fullWidth
                  size="small"
                  value={layerDraft.activation || "relu"}
                  onChange={(e) =>
                    setLayerDraft({ ...layerDraft, activation: e.target.value })
                  }
                >
                  <MenuItem value="relu">ReLU</MenuItem>
                  <MenuItem value="tanh">Tanh</MenuItem>
                  <MenuItem value="sigmoid">Sigmoid</MenuItem>
                </TextField>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                  Dropout Rate
                </Typography>
                <TextField
                  fullWidth
                  size="small"
                  type="number"
                  placeholder="e.g. 0.5"
                  value={layerDraft.dropout || 0}
                  onChange={(e) =>
                    setLayerDraft({ ...layerDraft, dropout: Number(e.target.value) })
                  }
                  helperText="Dropout probability for this FC layer (0-0.8)"
                  inputProps={{ min: 0, max: 0.8, step: 0.1 }}
                />
              </Box>
            </>
          )}

          {/* Pooling layers */}
          {dialogOpenKey === "pooling_layers" && (
            <>
              <Box>
                <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                  Pool Type
                </Typography>
                <TextField
                  select
                  fullWidth
                  size="small"
                  value={layerDraft.type || "max"}
                  onChange={(e) =>
                    setLayerDraft({ ...layerDraft, type: e.target.value })
                  }
                >
                  <MenuItem value="max">Max Pooling</MenuItem>
                  <MenuItem value="avg">Average Pooling</MenuItem>
                </TextField>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                  Pool Size
                </Typography>
                <TextField
                  fullWidth
                  size="small"
                  type="number"
                  placeholder="e.g. 2"
                  value={layerDraft.pool_size || 2}
                  onChange={(e) =>
                    setLayerDraft({ ...layerDraft, pool_size: Number(e.target.value) })
                  }
                  helperText="Size of the pooling window"
                />
              </Box>
            </>
          )}

          {/* Dropout layers */}
          {dialogOpenKey === "dropout_layers" && (
            <Box>
              <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                Dropout Rate
              </Typography>
              <TextField
                fullWidth
                size="small"
                type="number"
                inputProps={{ step: 0.1, min: 0, max: 1 }}
                placeholder="e.g. 0.2"
                value={layerDraft.rate || 0.2}
                onChange={(e) =>
                  setLayerDraft({ ...layerDraft, rate: Number(e.target.value) })
                }
                helperText="Probability of dropping neurons (0-1)"
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={closeLayerDialog}>Cancel</Button>
          <Button variant="contained" onClick={saveLayer}>
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}
