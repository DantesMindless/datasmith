import React, { useState } from 'react';
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
  Alert,
  CircularProgress,
  Paper,
  Grid,
  Chip,
  Stack,
} from '@mui/material';
import {
  CloudUpload,
  InsertDriveFile,
  Image,
  DataObject,
  Analytics,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import httpfetch from '../utils/axios';

const VisuallyHiddenInput = styled('input')({
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  height: 1,
  overflow: 'hidden',
  position: 'absolute',
  bottom: 0,
  left: 0,
  whiteSpace: 'nowrap',
  width: 1,
});

const UploadArea = styled(Paper)(({ theme }) => ({
  border: `2px dashed ${theme.palette.divider}`,
  borderRadius: theme.shape.borderRadius,
  padding: theme.spacing(4),
  textAlign: 'center',
  cursor: 'pointer',
  transition: 'all 0.3s ease',
  '&:hover': {
    borderColor: theme.palette.primary.main,
    backgroundColor: theme.palette.action.hover,
  },
  '&.dragover': {
    borderColor: theme.palette.primary.main,
    backgroundColor: theme.palette.primary.light,
    opacity: 0.8,
  },
}));

const DATASET_TYPES = [
  { value: 'tabular', label: 'Tabular Data', icon: <DataObject /> },
  { value: 'image', label: 'Image Dataset', icon: <Image /> },
  { value: 'text', label: 'Text Dataset', icon: <InsertDriveFile /> },
  { value: 'time_series', label: 'Time Series', icon: <Analytics /> },
  { value: 'mixed', label: 'Mixed Dataset', icon: <DataObject /> },
];

const DATASET_PURPOSES = [
  { value: 'classification', label: 'Classification' },
  { value: 'regression', label: 'Regression' },
  { value: 'clustering', label: 'Clustering' },
  { value: 'anomaly_detection', label: 'Anomaly Detection' },
  { value: 'recommendation', label: 'Recommendation' },
  { value: 'general', label: 'General Purpose' },
];

export default function CreateDatasetPage() {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    dataset_type: 'tabular',
    dataset_purpose: 'general',
    encoding: 'utf-8',
  });
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [filePreview, setFilePreview] = useState<any>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [dragOver, setDragOver] = useState(false);

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleFileSelect = async (file: File) => {
    setSelectedFile(file);
    setError('');

    // Auto-detect dataset type based on file
    const fileName = file.name.toLowerCase();
    if (fileName.endsWith('.csv') || fileName.endsWith('.xlsx') || fileName.endsWith('.json')) {
      setFormData(prev => ({ ...prev, dataset_type: 'tabular' }));
    } else if (fileName.endsWith('.zip') && (fileName.includes('image') || fileName.includes('img'))) {
      setFormData(prev => ({ ...prev, dataset_type: 'image' }));
    }

    // Generate preview for CSV files
    if (fileName.endsWith('.csv') && file.size < 5 * 1024 * 1024) { // Only for files < 5MB
      try {
        const text = await file.text();
        const lines = text.split('\n').slice(0, 6); // First 5 rows + header
        const preview = lines.map(line => line.split(',').slice(0, 5)); // First 5 columns
        setFilePreview(preview);
      } catch (err) {
        console.error('Error generating preview:', err);
      }
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setDragOver(false);
    const file = event.dataTransfer.files[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (event: React.DragEvent) => {
    event.preventDefault();
    setDragOver(false);
  };

  const validateForm = () => {
    if (!formData.name.trim()) {
      setError('Dataset name is required');
      return false;
    }
    if (!selectedFile) {
      setError('Please select a file to upload');
      return false;
    }
    return true;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) return;

    try {
      setUploading(true);
      setUploadProgress(0);
      setError('');

      const formDataToSend = new FormData();
      formDataToSend.append('name', formData.name);
      formDataToSend.append('description', formData.description);
      formDataToSend.append('dataset_type', formData.dataset_type);
      formDataToSend.append('dataset_purpose', formData.dataset_purpose);
      formDataToSend.append('encoding', formData.encoding);

      // Append file based on type
      if (formData.dataset_type === 'image') {
        formDataToSend.append('image_folder', selectedFile!);
      } else {
        formDataToSend.append('csv_file', selectedFile!);
      }

      const response = await httpfetch.post('datasets/', formDataToSend, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
        },
      });

      console.log('Dataset created successfully:', response.data);
      setSuccess('Dataset created successfully!');

      // Reset form
      setFormData({
        name: '',
        description: '',
        dataset_type: 'tabular',
        dataset_purpose: 'general',
        encoding: 'utf-8',
      });
      setSelectedFile(null);
      setFilePreview(null);

    } catch (err: any) {
      console.error('Error creating dataset:', err);
      setError(err.response?.data?.error || err.message || 'Failed to create dataset');
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const getFileIcon = (fileName: string) => {
    if (fileName.toLowerCase().includes('csv')) return <InsertDriveFile color="success" />;
    if (fileName.toLowerCase().includes('zip')) return <Image color="primary" />;
    return <InsertDriveFile color="action" />;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

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
          Create New Dataset
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
          <Grid container spacing={3}>
            {/* Basic Information */}
            <Grid item xs={12}>
              <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                Basic Information
              </Typography>
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Dataset Name"
                value={formData.name}
                onChange={(e) => handleInputChange('name', e.target.value)}
                required
                helperText="Give your dataset a descriptive name"
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Dataset Type</InputLabel>
                <Select
                  value={formData.dataset_type}
                  label="Dataset Type"
                  onChange={(e) => handleInputChange('dataset_type', e.target.value)}
                >
                  {DATASET_TYPES.map(type => (
                    <MenuItem key={type.value} value={type.value}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {type.icon}
                        {type.label}
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description"
                value={formData.description}
                onChange={(e) => handleInputChange('description', e.target.value)}
                multiline
                rows={2}
                helperText="Optional description of your dataset"
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Purpose</InputLabel>
                <Select
                  value={formData.dataset_purpose}
                  label="Purpose"
                  onChange={(e) => handleInputChange('dataset_purpose', e.target.value)}
                >
                  {DATASET_PURPOSES.map(purpose => (
                    <MenuItem key={purpose.value} value={purpose.value}>
                      {purpose.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Encoding</InputLabel>
                <Select
                  value={formData.encoding}
                  label="Encoding"
                  onChange={(e) => handleInputChange('encoding', e.target.value)}
                >
                  <MenuItem value="utf-8">UTF-8</MenuItem>
                  <MenuItem value="latin-1">Latin-1</MenuItem>
                  <MenuItem value="ascii">ASCII</MenuItem>
                  <MenuItem value="cp1252">Windows-1252</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            {/* File Upload */}
            <Grid item xs={12}>
              <Typography variant="subtitle1" fontWeight={600} gutterBottom sx={{ mt: 2 }}>
                File Upload
              </Typography>
            </Grid>

            <Grid item xs={12}>
              <UploadArea
                className={dragOver ? 'dragover' : ''}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => document.getElementById('file-upload')?.click()}
              >
                <VisuallyHiddenInput
                  id="file-upload"
                  type="file"
                  accept={formData.dataset_type === 'image' ? '.zip' : '.csv,.xlsx,.json'}
                  onChange={handleFileChange}
                />

                {selectedFile ? (
                  <Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 2, mb: 2 }}>
                      {getFileIcon(selectedFile.name)}
                      <Box sx={{ textAlign: 'left' }}>
                        <Typography variant="body2" fontWeight={600}>
                          {selectedFile.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {formatFileSize(selectedFile.size)}
                        </Typography>
                      </Box>
                    </Box>
                    <Stack direction="row" spacing={1} justifyContent="center">
                      <Chip label="File Selected" color="success" size="small" />
                      <Button
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedFile(null);
                          setFilePreview(null);
                        }}
                      >
                        Change File
                      </Button>
                    </Stack>
                  </Box>
                ) : (
                  <Box>
                    <CloudUpload sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                    <Typography variant="h6" gutterBottom>
                      Drop your file here or click to browse
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {formData.dataset_type === 'image'
                        ? 'Supported: ZIP files containing images'
                        : 'Supported: CSV, Excel, JSON files'
                      }
                    </Typography>
                  </Box>
                )}
              </UploadArea>
            </Grid>

            {/* File Preview */}
            {filePreview && (
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom>
                  File Preview
                </Typography>
                <Paper variant="outlined" sx={{ p: 2, maxHeight: 200, overflow: 'auto' }}>
                  <Box component="table" sx={{ width: '100%', fontSize: '0.75rem' }}>
                    {filePreview.map((row: string[], index: number) => (
                      <Box component="tr" key={index}>
                        {row.map((cell: string, cellIndex: number) => (
                          <Box
                            component={index === 0 ? "th" : "td"}
                            key={cellIndex}
                            sx={{
                              p: 0.5,
                              border: '1px solid',
                              borderColor: 'divider',
                              fontWeight: index === 0 ? 600 : 400,
                              bgcolor: index === 0 ? 'grey.50' : 'transparent'
                            }}
                          >
                            {cell.length > 20 ? cell.substring(0, 20) + '...' : cell}
                          </Box>
                        ))}
                      </Box>
                    ))}
                  </Box>
                </Paper>
              </Grid>
            )}

            {/* Upload Progress */}
            {uploading && (
              <Grid item xs={12}>
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" gutterBottom>
                    Uploading... {uploadProgress}%
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Box sx={{ width: '100%' }}>
                      <Box sx={{ height: 6, bgcolor: 'grey.200', borderRadius: 3, overflow: 'hidden' }}>
                        <Box
                          sx={{
                            height: '100%',
                            bgcolor: 'primary.main',
                            width: `${uploadProgress}%`,
                            transition: 'width 0.3s ease'
                          }}
                        />
                      </Box>
                    </Box>
                  </Box>
                </Box>
              </Grid>
            )}
          </Grid>

          <Box sx={{ mt: 4, textAlign: "right" }}>
            <Button
              type="submit"
              variant="contained"
              size="large"
              disabled={uploading}
              startIcon={uploading ? <CircularProgress size={20} /> : <CloudUpload />}
            >
              {uploading ? 'Creating...' : 'Create Dataset'}
            </Button>
          </Box>
        </form>
      </Box>
    </Container>
  );
}