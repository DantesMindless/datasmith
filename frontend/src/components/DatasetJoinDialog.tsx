import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  Stepper,
  Step,
  StepLabel,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  CircularProgress,
  Alert,
  Chip,
  Stack,
  Paper,
  Divider,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
} from '@mui/material';
import {
  MergeType,
  TableChart,
  Check,
  CallMerge,
  CallSplit,
  Merge,
} from '@mui/icons-material';
import { getDatasetColumns, joinDatasets, JoinType } from '../utils/requests';

interface Dataset {
  id: string;
  name: string;
  row_count?: number;
  column_count?: number;
  minio_csv_key?: string;
}

interface DatasetJoinDialogProps {
  open: boolean;
  onClose: () => void;
  datasets: Dataset[];
  onJoinComplete: () => void;
}

const steps = ['Select Datasets', 'Configure Join', 'Configure Output'];

const joinTypeInfo: Record<JoinType, { label: string; description: string; color: 'primary' | 'info' | 'warning' | 'secondary' }> = {
  inner: {
    label: 'Inner',
    description: 'Only rows with matching keys in both datasets',
    color: 'primary',
  },
  left: {
    label: 'Left Outer',
    description: 'All rows from left dataset, matching rows from right',
    color: 'info',
  },
  right: {
    label: 'Right Outer',
    description: 'All rows from right dataset, matching rows from left',
    color: 'warning',
  },
  outer: {
    label: 'Full Outer',
    description: 'All rows from both datasets, with nulls where no match',
    color: 'secondary',
  },
};

const DatasetJoinDialog: React.FC<DatasetJoinDialogProps> = ({
  open,
  onClose,
  datasets,
  onJoinComplete,
}) => {
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Step 1: Dataset selection
  const [leftDatasetId, setLeftDatasetId] = useState<string>('');
  const [rightDatasetId, setRightDatasetId] = useState<string>('');

  // Step 2: Column and join type selection
  const [leftColumns, setLeftColumns] = useState<string[]>([]);
  const [rightColumns, setRightColumns] = useState<string[]>([]);
  const [leftKeyColumn, setLeftKeyColumn] = useState<string>('');
  const [rightKeyColumn, setRightKeyColumn] = useState<string>('');
  const [joinType, setJoinType] = useState<JoinType>('inner');
  const [loadingColumns, setLoadingColumns] = useState(false);

  // Step 3: Output configuration
  const [resultName, setResultName] = useState<string>('');
  const [resultDescription, setResultDescription] = useState<string>('');

  // Filter to only show CSV datasets
  const csvDatasets = datasets.filter(d => d.minio_csv_key);

  // Reset state when dialog opens
  useEffect(() => {
    if (open) {
      setActiveStep(0);
      setLeftDatasetId('');
      setRightDatasetId('');
      setLeftColumns([]);
      setRightColumns([]);
      setLeftKeyColumn('');
      setRightKeyColumn('');
      setJoinType('inner');
      setResultName('');
      setResultDescription('');
      setError(null);
    }
  }, [open]);

  // Load columns when datasets are selected
  useEffect(() => {
    const loadColumns = async () => {
      if (leftDatasetId && rightDatasetId) {
        setLoadingColumns(true);
        setError(null);
        try {
          const [leftResult, rightResult] = await Promise.all([
            getDatasetColumns(leftDatasetId),
            getDatasetColumns(rightDatasetId),
          ]);
          setLeftColumns(leftResult.columns || []);
          setRightColumns(rightResult.columns || []);
        } catch (err) {
          setError('Failed to load column information');
        } finally {
          setLoadingColumns(false);
        }
      }
    };
    loadColumns();
  }, [leftDatasetId, rightDatasetId]);

  // Auto-generate result name
  useEffect(() => {
    if (leftDatasetId && rightDatasetId) {
      const leftDataset = datasets.find(d => d.id === leftDatasetId);
      const rightDataset = datasets.find(d => d.id === rightDatasetId);
      if (leftDataset && rightDataset) {
        setResultName(`${leftDataset.name} + ${rightDataset.name}`);
      }
    }
  }, [leftDatasetId, rightDatasetId, datasets]);

  const handleNext = () => {
    setActiveStep(prev => prev + 1);
  };

  const handleBack = () => {
    setActiveStep(prev => prev - 1);
  };

  const canProceedStep1 = leftDatasetId && rightDatasetId && leftDatasetId !== rightDatasetId;
  const canProceedStep2 = leftKeyColumn && rightKeyColumn;
  const canProceedStep3 = resultName.trim().length > 0;

  const handleJoin = async () => {
    setLoading(true);
    setError(null);
    try {
      await joinDatasets({
        left_dataset_id: leftDatasetId,
        right_dataset_id: rightDatasetId,
        left_key_column: leftKeyColumn,
        right_key_column: rightKeyColumn,
        join_type: joinType,
        result_name: resultName,
        result_description: resultDescription,
      });
      onJoinComplete();
      onClose();
    } catch (err: any) {
      const errorMessage = err.response?.data?.error || err.message || 'Failed to join datasets';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const getDatasetInfo = (datasetId: string) => {
    return datasets.find(d => d.id === datasetId);
  };

  const handleJoinTypeChange = (
    _event: React.MouseEvent<HTMLElement>,
    newJoinType: JoinType | null,
  ) => {
    if (newJoinType !== null) {
      setJoinType(newJoinType);
    }
  };

  const renderStepContent = () => {
    switch (activeStep) {
      case 0:
        return (
          <Box sx={{ py: 2 }}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Select two datasets to join. Both datasets must have CSV files.
            </Typography>

            <Stack spacing={3}>
              <FormControl fullWidth>
                <InputLabel>Left Dataset (Primary)</InputLabel>
                <Select
                  value={leftDatasetId}
                  onChange={(e) => setLeftDatasetId(e.target.value)}
                  label="Left Dataset (Primary)"
                >
                  {csvDatasets.map(dataset => (
                    <MenuItem
                      key={dataset.id}
                      value={dataset.id}
                      disabled={dataset.id === rightDatasetId}
                    >
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                        <span>{dataset.name}</span>
                        <Typography variant="caption" color="text.secondary">
                          {dataset.row_count?.toLocaleString()} rows
                        </Typography>
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                <MergeType sx={{ fontSize: 40, color: 'primary.main', transform: 'rotate(90deg)' }} />
              </Box>

              <FormControl fullWidth>
                <InputLabel>Right Dataset</InputLabel>
                <Select
                  value={rightDatasetId}
                  onChange={(e) => setRightDatasetId(e.target.value)}
                  label="Right Dataset"
                >
                  {csvDatasets.map(dataset => (
                    <MenuItem
                      key={dataset.id}
                      value={dataset.id}
                      disabled={dataset.id === leftDatasetId}
                    >
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                        <span>{dataset.name}</span>
                        <Typography variant="caption" color="text.secondary">
                          {dataset.row_count?.toLocaleString()} rows
                        </Typography>
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Stack>

            {leftDatasetId === rightDatasetId && leftDatasetId && (
              <Alert severity="error" sx={{ mt: 2 }}>
                Cannot join a dataset with itself
              </Alert>
            )}
          </Box>
        );

      case 1:
        return (
          <Box sx={{ py: 2 }}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Select the key columns and join type. The join type determines which rows are included in the result.
            </Typography>

            {loadingColumns ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                <CircularProgress />
              </Box>
            ) : (
              <Stack spacing={3}>
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    <TableChart sx={{ fontSize: 16, mr: 1, verticalAlign: 'middle' }} />
                    {getDatasetInfo(leftDatasetId)?.name}
                  </Typography>
                  <FormControl fullWidth size="small" sx={{ mt: 1 }}>
                    <InputLabel>Key Column</InputLabel>
                    <Select
                      value={leftKeyColumn}
                      onChange={(e) => setLeftKeyColumn(e.target.value)}
                      label="Key Column"
                    >
                      {leftColumns.map(col => (
                        <MenuItem key={col} value={col}>{col}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Paper>

                {/* Join Type Selection */}
                <Box>
                  <Typography variant="subtitle2" gutterBottom sx={{ mb: 1 }}>
                    Join Type
                  </Typography>
                  <ToggleButtonGroup
                    value={joinType}
                    exclusive
                    onChange={handleJoinTypeChange}
                    aria-label="join type"
                    fullWidth
                    size="small"
                  >
                    {(Object.keys(joinTypeInfo) as JoinType[]).map((type) => (
                      <ToggleButton
                        key={type}
                        value={type}
                        aria-label={type}
                        sx={{
                          py: 1,
                          flexDirection: 'column',
                          '&.Mui-selected': {
                            bgcolor: `${joinTypeInfo[type].color}.lighter`,
                            borderColor: `${joinTypeInfo[type].color}.main`,
                            '&:hover': {
                              bgcolor: `${joinTypeInfo[type].color}.light`,
                            },
                          },
                        }}
                      >
                        <Typography variant="caption" fontWeight={600}>
                          {joinTypeInfo[type].label}
                        </Typography>
                      </ToggleButton>
                    ))}
                  </ToggleButtonGroup>
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                    {joinTypeInfo[joinType].description}
                  </Typography>
                </Box>

                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    <TableChart sx={{ fontSize: 16, mr: 1, verticalAlign: 'middle' }} />
                    {getDatasetInfo(rightDatasetId)?.name}
                  </Typography>
                  <FormControl fullWidth size="small" sx={{ mt: 1 }}>
                    <InputLabel>Key Column</InputLabel>
                    <Select
                      value={rightKeyColumn}
                      onChange={(e) => setRightKeyColumn(e.target.value)}
                      label="Key Column"
                    >
                      {rightColumns.map(col => (
                        <MenuItem key={col} value={col}>{col}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Paper>
              </Stack>
            )}

            <Alert
              severity={joinType === 'inner' ? 'info' : 'warning'}
              sx={{ mt: 3 }}
            >
              {joinType === 'inner' && 'Only rows with matching key values in both datasets will be included.'}
              {joinType === 'left' && 'All rows from the left dataset will be included. Unmatched rows will have null values for right dataset columns.'}
              {joinType === 'right' && 'All rows from the right dataset will be included. Unmatched rows will have null values for left dataset columns.'}
              {joinType === 'outer' && 'All rows from both datasets will be included. Unmatched rows will have null values for missing columns.'}
            </Alert>
          </Box>
        );

      case 2:
        return (
          <Box sx={{ py: 2 }}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Configure the output dataset name and description.
            </Typography>

            <Stack spacing={3}>
              <TextField
                label="Result Dataset Name"
                value={resultName}
                onChange={(e) => setResultName(e.target.value)}
                fullWidth
                required
              />

              <TextField
                label="Description (Optional)"
                value={resultDescription}
                onChange={(e) => setResultDescription(e.target.value)}
                fullWidth
                multiline
                rows={3}
              />

              <Divider />

              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="subtitle2" gutterBottom>Join Summary</Typography>
                <Stack spacing={1}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">Left Dataset:</Typography>
                    <Typography variant="body2">{getDatasetInfo(leftDatasetId)?.name}</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">Right Dataset:</Typography>
                    <Typography variant="body2">{getDatasetInfo(rightDatasetId)?.name}</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">Join Condition:</Typography>
                    <Typography variant="body2">
                      {leftKeyColumn} = {rightKeyColumn}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">Join Type:</Typography>
                    <Chip
                      label={joinTypeInfo[joinType].label.toUpperCase()}
                      size="small"
                      color={joinTypeInfo[joinType].color}
                    />
                  </Box>
                </Stack>
              </Paper>
            </Stack>
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      PaperProps={{
        sx: { minHeight: '500px' }
      }}
    >
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <MergeType />
        Join Datasets
      </DialogTitle>

      <DialogContent dividers>
        <Stepper activeStep={activeStep} sx={{ mb: 3 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {renderStepContent()}
      </DialogContent>

      <DialogActions sx={{ px: 3, py: 2 }}>
        <Button onClick={onClose} disabled={loading}>
          Cancel
        </Button>
        <Box sx={{ flex: 1 }} />
        {activeStep > 0 && (
          <Button onClick={handleBack} disabled={loading}>
            Back
          </Button>
        )}
        {activeStep < steps.length - 1 ? (
          <Button
            variant="contained"
            onClick={handleNext}
            disabled={
              (activeStep === 0 && !canProceedStep1) ||
              (activeStep === 1 && !canProceedStep2)
            }
          >
            Next
          </Button>
        ) : (
          <Button
            variant="contained"
            onClick={handleJoin}
            disabled={!canProceedStep3 || loading}
            startIcon={loading ? <CircularProgress size={20} /> : <Check />}
          >
            {loading ? 'Joining...' : 'Join Datasets'}
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default DatasetJoinDialog;
