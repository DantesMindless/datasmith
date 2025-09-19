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
  CircularProgress
} from "@mui/material";
import { modelConfigSchemas, ConfigField } from "../utils/modelSchema";
import httpfetch from "../utils/axios";

// Authentication is now handled by httpfetch via JWT tokens

export default function CreateModelPage() {
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

  const [dialogOpenKey, setDialogOpenKey] = useState<string | null>(null);
  const [layerDraft, setLayerDraft] = useState<Record<string, any>>({});

  const handleConfigChange = (key: string, value: any) => {
    setConfigValues((prev) => ({ ...prev, [key]: value }));
  };

  const openLayerDialog = (fieldKey: string) => {
    setDialogOpenKey(fieldKey);
    setLayerDraft({});
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

    if (!targetColumn) {
      setError("Please select a target column");
      return;
    }

    const payload = {
      name,
      model_type: modelType,
      dataset: datasetId,
      target_column: targetColumn,
      config: configValues,
    };
    
    try {
      setLoading(true);
      setError("");
      const response = await httpfetch.post('models/', payload);
      setSuccess("Model created successfully!");
      console.log("Model created:", response.data);
      // Reset form
      setName("");
      setModelType("");
      setDatasetId("");
      setTargetColumn("");
      setDatasetColumns([]);
      setConfigValues({});
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
          setError('Could not fetch dataset columns. Please select a different dataset.');
        }
      } catch (fallbackErr) {
        setError('Could not fetch dataset columns. Please select a different dataset.');
      }
    }
  };

  const handleDatasetChange = (newDatasetId: string) => {
    setDatasetId(newDatasetId);
    setTargetColumn(""); // Reset target column when dataset changes
    setDatasetColumns([]); // Clear existing columns

    if (newDatasetId) {
      fetchDatasetColumns(newDatasetId);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  const fields: ConfigField[] = modelType ? modelConfigSchemas[modelType] || [] : [];

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

          {datasetColumns.length > 0 && (
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

          <FormControl fullWidth margin="normal">
            <InputLabel id="model-type-label">Model Type</InputLabel>
            <Select
                labelId="model-type-label"
                value={modelType}
                onChange={(e) => {
                    const newType = e.target.value;
                    setModelType(newType);

                    // build initial config with defaults
                    const defaults: Record<string, any> = {};
                    const schema = modelConfigSchemas[newType] || [];
                    schema.forEach((field) => {
                    if (field.default !== undefined) {
                        defaults[field.key] = field.default;
                    }
                    });

                    setConfigValues(defaults);
                }}
                required
               >
              <MenuItem value="logistic_regression">Logistic Regression</MenuItem>
              <MenuItem value="random_forest">Random Forest</MenuItem>
              <MenuItem value="svm">Support Vector Machine</MenuItem>
              <MenuItem value="naive_bayes">Naive Bayes</MenuItem>
              <MenuItem value="knn">k-Nearest Neighbours</MenuItem>
              <MenuItem value="gradient_boosting">Gradient Boosting</MenuItem>
              <MenuItem value="neural_network">Neural Network (PyTorch)</MenuItem>
              <MenuItem value="cnn">Convolutional Neural Network (CNN)</MenuItem>
            </Select>
          </FormControl>

          {fields.length > 0 && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6" gutterBottom>
                Configuration
              </Typography>
              {fields.map((field) => {
 
                if (field.type === "select") {
                  const value = configValues[field.key] ?? field.default ?? "";
                  return (
                    <FormControl key={field.key} fullWidth margin="normal">
                      <InputLabel>{field.label}</InputLabel>
                      <Select
                        value={value}
                        onChange={(e) => handleConfigChange(field.key, e.target.value)}
                      >
                        {field.options?.map((opt) => (
                          <MenuItem key={opt.value} value={opt.value}>
                            {opt.label}
                          </MenuItem>
                        ))}
                      </Select>
                      {field.helperText && (
                        <Typography variant="caption" color="text.secondary">
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
                              {k}: {v}
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
                      onChange={(e) => handleConfigChange(field.key, e.target.value)}
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
            mt: 1,
            minWidth: "300px",
          }}
        >
          {dialogOpenKey === "dense_layers" && (
            <>
              <TextField
                label="Units"
                type="number"
                value={layerDraft.units || ""}
                onChange={(e) =>
                  setLayerDraft({ ...layerDraft, units: e.target.value })
                }
              />
              <TextField
                label="Activation"
                value={layerDraft.activation || ""}
                onChange={(e) =>
                  setLayerDraft({ ...layerDraft, activation: e.target.value })
                }
              />
            </>
          )}

          {dialogOpenKey === "conv_layers" && (
            <>
              <TextField
                label="Filters"
                type="number"
                value={layerDraft.filters || ""}
                onChange={(e) =>
                  setLayerDraft({ ...layerDraft, filters: e.target.value })
                }
              />
              <TextField
                label="Kernel Size"
                type="number"
                value={layerDraft.kernel_size || ""}
                onChange={(e) =>
                  setLayerDraft({ ...layerDraft, kernel_size: e.target.value })
                }
              />
              <TextField
                label="Stride"
                type="number"
                value={layerDraft.stride || ""}
                onChange={(e) =>
                  setLayerDraft({ ...layerDraft, stride: e.target.value })
                }
              />
              <TextField
                label="Activation"
                value={layerDraft.activation || ""}
                onChange={(e) =>
                  setLayerDraft({ ...layerDraft, activation: e.target.value })
                }
              />
            </>
          )}

          {dialogOpenKey === "pooling_layers" && (
            <>
              <TextField
                label="Type"
                value={layerDraft.type || ""}
                onChange={(e) =>
                  setLayerDraft({ ...layerDraft, type: e.target.value })
                }
              />
              <TextField
                label="Pool Size"
                type="number"
                value={layerDraft.pool_size || ""}
                onChange={(e) =>
                  setLayerDraft({ ...layerDraft, pool_size: e.target.value })
                }
              />
            </>
          )}

          {dialogOpenKey === "dropout_layers" && (
            <TextField
              label="Rate"
              type="number"
              value={layerDraft.rate || ""}
              onChange={(e) =>
                setLayerDraft({ ...layerDraft, rate: e.target.value })
              }
            />
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
