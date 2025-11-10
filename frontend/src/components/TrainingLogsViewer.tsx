/**
 * DEPRECATED: This component uses HTTP polling (every 2 seconds).
 * Use TrainingLogsViewerWebSocket for real-time WebSocket-based updates instead.
 *
 * This component is kept for backward compatibility only.
 */
import React, { useState, useEffect, useRef } from 'react';
import httpfetch from '../utils/axios';
import { Box, Paper, Typography, CircularProgress, LinearProgress, Chip, Divider, Accordion, AccordionSummary, AccordionDetails, IconButton, Tooltip, Tabs, Tab } from '@mui/material';
import { ExpandMore as ExpandMoreIcon, Timeline as TimelineIcon, CheckCircle, Error, Info, Warning, VerticalAlignBottom, VerticalAlignTop, ViewList, ViewModule, Download } from '@mui/icons-material';
import GroupedTrainingLogs from './GroupedTrainingLogs';

interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  data?: Record<string, any>;
}

interface TrainingMetrics {
  epochs: number[];
  losses: number[];
  accuracies: number[];
  val_accuracies: number[];
}

interface TrainingLogsData {
  model_id: string;
  model_name: string;
  status: string;
  logs: LogEntry[];
  metrics: TrainingMetrics;
  total_logs: number;
  training_log_text: string;
  accuracy: number | null;
}

interface TrainingLogsViewerProps {
  modelId: string;
  autoRefresh?: boolean;
  refreshInterval?: number; // in milliseconds
}

const TrainingLogsViewer: React.FC<TrainingLogsViewerProps> = ({
  modelId,
  autoRefresh = true,
  refreshInterval = 2000, // 2 seconds default
}) => {
  const [logsData, setLogsData] = useState<TrainingLogsData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState<boolean>(false); // Disabled by default to prevent auto-scroll on page entry
  const [viewMode, setViewMode] = useState<'grouped' | 'detailed'>('grouped'); // Default to grouped view

  const fetchTrainingLogs = async () => {
    try {
      const response = await httpfetch.get(`models/${modelId}/training_logs/`);
      setLogsData(response.data);
      setError(null);

      // Auto-scroll to bottom if enabled
      if (autoScroll && logsEndRef.current) {
        logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
      }
    } catch (err: any) {
      console.error('Error fetching training logs:', err);
      setError(err.response?.data?.error || 'Failed to fetch training logs');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTrainingLogs();

    // Set up auto-refresh if enabled and model is training
    let intervalId: NodeJS.Timeout | null = null;
    if (autoRefresh && logsData?.status === 'training') {
      intervalId = setInterval(fetchTrainingLogs, refreshInterval);
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [modelId, autoRefresh, refreshInterval, logsData?.status]);

  const getLevelIcon = (level: string) => {
    switch (level.toUpperCase()) {
      case 'SUCCESS':
        return <CheckCircle sx={{ color: '#4caf50', fontSize: 18 }} />;
      case 'ERROR':
        return <Error sx={{ color: '#f44336', fontSize: 18 }} />;
      case 'WARNING':
        return <Warning sx={{ color: '#ff9800', fontSize: 18 }} />;
      case 'PROGRESS':
        return <TimelineIcon sx={{ color: '#2196f3', fontSize: 18 }} />;
      case 'DEBUG':
        return <Info sx={{ color: '#673ab7', fontSize: 18 }} />;
      default:
        return <Info sx={{ color: '#9e9e9e', fontSize: 18 }} />;
    }
  };

  const getLevelColor = (level: string) => {
    switch (level.toUpperCase()) {
      case 'SUCCESS':
        return '#4caf50';
      case 'ERROR':
        return '#f44336';
      case 'WARNING':
        return '#ff9800';
      case 'PROGRESS':
        return '#2196f3';
      case 'INFO':
        return '#2196f3';
      case 'DEBUG':
        return '#673ab7';
      default:
        return '#9e9e9e';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'complete':
        return 'success';
      case 'training':
        return 'primary';
      case 'failed':
        return 'error';
      case 'pending':
        return 'warning';
      default:
        return 'default';
    }
  };

  const downloadLogs = () => {
    if (!logsData) return;

    // Format all logs as text
    const logLines = logsData.logs.map(entry => {
      const emoji = {
        'INFO': 'ℹ️',
        'SUCCESS': '✅',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'DEBUG': '🔍',
        'PROGRESS': '⏳'
      }[entry.level] || 'ℹ️';

      return `${entry.timestamp} ${emoji} [${entry.level}] ${entry.message}`;
    }).join('\n');

    // Create header
    const header = [
      '=' .repeat(80),
      `Training Logs: ${logsData.model_name}`,
      `Model ID: ${logsData.model_id}`,
      `Status: ${logsData.status.toUpperCase()}`,
      `Total Logs: ${logsData.total_logs}`,
      logsData.accuracy !== null ? `Accuracy: ${(logsData.accuracy * 100).toFixed(2)}%` : '',
      `Generated: ${new Date().toLocaleString()}`,
      '='.repeat(80),
      ''
    ].filter(line => line).join('\n');

    const fullText = header + '\n' + logLines;

    // Create blob and download
    const blob = new Blob([fullText], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `training_logs_${logsData.model_name.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  if (loading && !logsData) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={300}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Paper sx={{ p: 3, bgcolor: '#ffebee' }}>
        <Typography color="error">
          <Error sx={{ verticalAlign: 'middle', mr: 1 }} />
          {error}
        </Typography>
      </Paper>
    );
  }

  if (!logsData) {
    return (
      <Paper sx={{ p: 3 }}>
        <Typography color="textSecondary">No training logs available.</Typography>
      </Paper>
    );
  }

  const calculateProgress = () => {
    if (!logsData.logs || logsData.logs.length === 0) return 0;

    // Look for epoch information in logs
    const epochLogs = logsData.logs.filter(log =>
      log.message.includes('Epoch') && log.level === 'PROGRESS'
    );

    if (epochLogs.length === 0) return 0;

    // Try to extract current epoch and total epochs
    const lastEpochLog = epochLogs[epochLogs.length - 1];
    const match = lastEpochLog.message.match(/Epoch (\d+)\/(\d+)/);

    if (match) {
      const current = parseInt(match[1]);
      const total = parseInt(match[2]);
      return (current / total) * 100;
    }

    return 0;
  };

  const progress = calculateProgress();

  return (
    <Box>
      {/* Header with Status */}
      <Paper
        sx={{
          p: 2,
          mb: 2,
          background: 'linear-gradient(135deg, #f5f5f5 0%, #ede7f6 100%)',
          borderLeft: '4px solid #673ab7'
        }}
      >
        <Box display="flex" justifyContent="space-between" alignItems="center" flexWrap="wrap">
          <Box>
            <Typography variant="h6" gutterBottom sx={{ color: '#333' }}>
              Training Logs: {logsData.model_name}
            </Typography>
            <Typography variant="body2" sx={{ color: '#673ab7', fontWeight: 500 }}>
              Model ID: {logsData.model_id}
            </Typography>
          </Box>
          <Box>
            <Chip
              label={logsData.status.toUpperCase()}
              color={getStatusColor(logsData.status) as any}
              sx={{ mr: 1 }}
            />
            {logsData.accuracy !== null && (
              <Chip
                label={`Accuracy: ${(logsData.accuracy * 100).toFixed(2)}%`}
                color="success"
                variant="outlined"
              />
            )}
          </Box>
        </Box>

        {/* Progress Bar for Training */}
        {logsData.status === 'training' && progress > 0 && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" sx={{ color: '#673ab7', fontWeight: 600 }} gutterBottom>
              Training Progress: {progress.toFixed(1)}%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={progress}
              sx={{
                height: 8,
                borderRadius: 4,
                bgcolor: '#e1d5f0',
                '& .MuiLinearProgress-bar': {
                  background: 'linear-gradient(90deg, #673ab7 0%, #9575cd 100%)',
                  borderRadius: 4,
                }
              }}
            />
          </Box>
        )}
      </Paper>

      {/* Metrics Summary (if available) */}
      {logsData.metrics && logsData.metrics.epochs && logsData.metrics.epochs.length > 0 && (
        <Accordion defaultExpanded sx={{ border: '1px solid #e1d5f0' }}>
          <AccordionSummary
            expandIcon={<ExpandMoreIcon />}
            sx={{
              background: 'linear-gradient(135deg, #f3e5f5 0%, #ede7f6 100%)',
              '&:hover': { bgcolor: '#ede7f6' }
            }}
          >
            <Typography variant="subtitle1" fontWeight="bold" sx={{ color: '#673ab7' }}>
              📊 Training Metrics Summary
            </Typography>
          </AccordionSummary>
          <AccordionDetails sx={{ bgcolor: '#fafafa' }}>
            <Box display="flex" flexWrap="wrap" gap={3}>
              <Box
                sx={{
                  p: 2,
                  borderRadius: 2,
                  bgcolor: 'white',
                  border: '2px solid #e1d5f0',
                  minWidth: 150
                }}
              >
                <Typography variant="body2" sx={{ color: '#673ab7', fontWeight: 600 }}>
                  Epochs Completed
                </Typography>
                <Typography variant="h5" sx={{ color: '#673ab7', fontWeight: 700, mt: 1 }}>
                  {logsData.metrics.epochs.length}
                </Typography>
              </Box>
              {logsData.metrics.val_accuracies.length > 0 && (
                <Box
                  sx={{
                    p: 2,
                    borderRadius: 2,
                    bgcolor: 'white',
                    border: '2px solid #c8e6c9',
                    minWidth: 150
                  }}
                >
                  <Typography variant="body2" sx={{ color: '#4caf50', fontWeight: 600 }}>
                    Best Validation Accuracy
                  </Typography>
                  <Typography variant="h5" sx={{ color: '#4caf50', fontWeight: 700, mt: 1 }}>
                    {(Math.max(...logsData.metrics.val_accuracies)).toFixed(4)}
                  </Typography>
                </Box>
              )}
              {logsData.metrics.losses.length > 0 && (
                <Box
                  sx={{
                    p: 2,
                    borderRadius: 2,
                    bgcolor: 'white',
                    border: '2px solid #bbdefb',
                    minWidth: 150
                  }}
                >
                  <Typography variant="body2" sx={{ color: '#2196f3', fontWeight: 600 }}>
                    Latest Loss
                  </Typography>
                  <Typography variant="h5" sx={{ color: '#2196f3', fontWeight: 700, mt: 1 }}>
                    {logsData.metrics.losses[logsData.metrics.losses.length - 1].toFixed(4)}
                  </Typography>
                </Box>
              )}
            </Box>
          </AccordionDetails>
        </Accordion>
      )}

      {/* Detailed Logs */}
      <Paper
        sx={{
          mt: 2,
          border: '2px solid #673ab7',
          borderRadius: 2,
          overflow: 'hidden'
        }}
      >
        {/* Header with View Toggle */}
        <Box
          sx={{
            p: 1.5,
            background: 'linear-gradient(135deg, #f3e5f5 0%, #ede7f6 100%)',
            borderBottom: '2px solid #673ab7',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}
        >
          <Typography
            variant="subtitle1"
            sx={{
              color: '#673ab7',
              fontWeight: 700,
            }}
          >
            📋 Training Logs ({logsData.total_logs} entries)
          </Typography>

          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <Tabs
              value={viewMode}
              onChange={(e, newValue) => setViewMode(newValue)}
              sx={{
                minHeight: 36,
                '& .MuiTab-root': {
                  minHeight: 36,
                  minWidth: 100,
                  py: 0.5,
                  fontSize: '0.875rem'
                }
              }}
            >
              <Tab
                value="grouped"
                label="Grouped"
                icon={<ViewModule fontSize="small" />}
                iconPosition="start"
              />
              <Tab
                value="detailed"
                label="Detailed"
                icon={<ViewList fontSize="small" />}
                iconPosition="start"
              />
            </Tabs>

            <Box sx={{ display: 'flex', gap: 1 }}>
              <Tooltip title="Download all logs as text file">
                <IconButton
                  size="small"
                  onClick={downloadLogs}
                  sx={{
                    color: '#673ab7',
                    '&:hover': { bgcolor: 'rgba(103, 58, 183, 0.1)' }
                  }}
                >
                  <Download fontSize="small" />
                </IconButton>
              </Tooltip>

              {viewMode === 'detailed' && (
                <Tooltip title={autoScroll ? "Auto-scroll enabled" : "Auto-scroll disabled"}>
                  <IconButton
                    size="small"
                    onClick={() => setAutoScroll(!autoScroll)}
                    sx={{
                      color: autoScroll ? '#673ab7' : '#666',
                      '&:hover': { bgcolor: 'rgba(103, 58, 183, 0.1)' }
                    }}
                  >
                    {autoScroll ? <VerticalAlignBottom fontSize="small" /> : <VerticalAlignTop fontSize="small" />}
                  </IconButton>
                </Tooltip>
              )}
            </Box>
          </Box>
        </Box>

        {/* Grouped View */}
        {viewMode === 'grouped' && (
          <Box sx={{ maxHeight: 600, overflow: 'auto', p: 2, bgcolor: '#fafafa' }}>
            {logsData.logs.length === 0 ? (
              <Typography sx={{ color: 'text.secondary', textAlign: 'center', py: 4 }}>
                No training logs available yet. Logs will appear here during training.
              </Typography>
            ) : (
              <GroupedTrainingLogs logs={logsData.logs} status={logsData.status} />
            )}
          </Box>
        )}

        {/* Detailed View (Original) */}
        {viewMode === 'detailed' && (
          <Box sx={{ maxHeight: 600, overflow: 'auto', bgcolor: '#1e1e1e', color: '#fff', p: 2 }}>
            <Box
              sx={{
                mb: 2,
                p: 1.5,
                background: 'linear-gradient(135deg, #2a2a2a 0%, #3a2a4a 100%)',
                borderRadius: 1,
                border: '1px solid #673ab7'
              }}
            >
              <Typography
                variant="subtitle2"
                sx={{
                  color: '#b39ddb',
                  fontFamily: 'monospace',
                  fontWeight: 700,
                  letterSpacing: 0.5
                }}
              >
                📋 Raw Log Output
              </Typography>
            </Box>
            <Divider sx={{ bgcolor: '#673ab7', mb: 2, height: 2 }} />

          {logsData.logs.length === 0 ? (
            <Typography sx={{ color: '#aaa', fontFamily: 'monospace' }}>
              No detailed logs available yet. Logs will appear here during training.
            </Typography>
          ) : (
            <Box sx={{ fontFamily: 'monospace', fontSize: '0.875rem' }}>
              {logsData.logs.map((log, index) => (
                <Box key={index} sx={{ mb: 1.5 }}>
                  <Box display="flex" alignItems="flex-start" gap={1}>
                    {getLevelIcon(log.level)}
                    <Box flex={1}>
                      <Typography
                        component="span"
                        sx={{
                          color: '#888',
                          fontSize: '0.75rem',
                          fontFamily: 'monospace',
                        }}
                      >
                        {log.timestamp}
                      </Typography>
                      <Typography
                        component="span"
                        sx={{
                          color: getLevelColor(log.level),
                          fontWeight: 'bold',
                          ml: 1,
                          fontSize: '0.75rem',
                          fontFamily: 'monospace',
                        }}
                      >
                        [{log.level}]
                      </Typography>
                      <Typography
                        sx={{
                          color: '#fff',
                          fontFamily: 'monospace',
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                          mt: 0.5,
                        }}
                      >
                        {log.message}
                      </Typography>
                      {log.data && Object.keys(log.data).length > 0 && (
                        <Box
                          sx={{
                            mt: 0.5,
                            pl: 2,
                            borderLeft: '2px solid #444',
                            color: '#aaa',
                          }}
                        >
                          {Object.entries(log.data).map(([key, value]) => (
                            <Typography
                              key={key}
                              sx={{ fontSize: '0.75rem', fontFamily: 'monospace' }}
                            >
                              {key}: {typeof value === 'object' ? JSON.stringify(value) : value}
                            </Typography>
                          ))}
                        </Box>
                      )}
                    </Box>
                  </Box>
                </Box>
              ))}
              <div ref={logsEndRef} />
            </Box>
          )}

            {viewMode === 'detailed' && logsData.status === 'training' && (
              <Box
                sx={{
                  mt: 2,
                  p: 1.5,
                  background: 'linear-gradient(135deg, #2a2a2a 0%, #3a2a4a 100%)',
                  borderRadius: 1,
                  textAlign: 'center',
                  border: '1px solid #673ab7',
                }}
              >
                <CircularProgress
                  size={20}
                  sx={{
                    mr: 1,
                    verticalAlign: 'middle',
                    color: '#b39ddb'
                  }}
                />
                <Typography
                  component="span"
                  sx={{
                    color: '#b39ddb',
                    fontFamily: 'monospace',
                    fontSize: '0.875rem',
                    fontWeight: 600
                  }}
                >
                  Training in progress... (auto-refreshing every {refreshInterval / 1000}s)
                </Typography>
              </Box>
            )}
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default TrainingLogsViewer;
