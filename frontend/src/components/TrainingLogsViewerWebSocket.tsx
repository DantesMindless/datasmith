import React, { useState, useRef, useEffect } from 'react';
import { Box, Paper, Typography, CircularProgress, LinearProgress, Chip, Divider, Accordion, AccordionSummary, AccordionDetails, IconButton, Tooltip, Tabs, Tab, Alert } from '@mui/material';
import { ExpandMore as ExpandMoreIcon, Timeline as TimelineIcon, CheckCircle, Error, Info, Warning, VerticalAlignBottom, VerticalAlignTop, ViewList, ViewModule, Download, Refresh, WifiOff } from '@mui/icons-material';
import GroupedTrainingLogs from './GroupedTrainingLogs';
import { useTrainingWebSocket } from '../hooks/useTrainingWebSocket';

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

interface TrainingLogsViewerProps {
  modelId: string;
  modelName?: string;
  status?: string;
}

const TrainingLogsViewerWebSocket: React.FC<TrainingLogsViewerProps> = ({
  modelId,
  modelName = 'Model',
  status: initialStatus = 'training',
}) => {
  const [autoScroll, setAutoScroll] = useState<boolean>(false); // Disabled by default
  const [viewMode, setViewMode] = useState<'grouped' | 'detailed'>('grouped');
  const [status, setStatus] = useState<string>(initialStatus);
  const [accuracy, setAccuracy] = useState<number | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Use WebSocket hook for real-time logs
  const {
    logs,
    isConnected,
    error: wsError,
    reconnecting,
    reconnectAttempts,
    clearLogs,
    requestLogs,
    disconnect,
    connect
  } = useTrainingWebSocket({
    modelId,
    enabled: true,
    autoReconnect: true,
    onComplete: (data) => {
      setStatus(data.status || 'complete');
      if (data.accuracy !== undefined) {
        setAccuracy(data.accuracy);
      }
    },
    onError: (error) => {
      // Error is already handled by the hook
    }
  });

  // Auto-scroll to bottom if enabled
  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

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

  const calculateProgress = () => {
    if (!logs || logs.length === 0) return 0;

    // Look for epoch information in logs
    const epochLogs = logs.filter(log =>
      log.message.includes('Epoch') || log.message.includes('Round') && log.level === 'PROGRESS'
    );

    if (epochLogs.length === 0) return 0;

    // Try to extract current epoch and total epochs
    const lastEpochLog = epochLogs[epochLogs.length - 1];
    const match = lastEpochLog.message.match(/(?:Epoch|Round)\s+(\d+)\s+(?:of|\/)\s+(\d+)/i);

    if (match) {
      const current = parseInt(match[1]);
      const total = parseInt(match[2]);
      return (current / total) * 100;
    }

    return 0;
  };

  const downloadLogs = () => {
    // Format all logs as text
    const logLines = logs.map(entry => {
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
      '='.repeat(80),
      `Training Logs: ${modelName}`,
      `Model ID: ${modelId}`,
      `Status: ${status.toUpperCase()}`,
      `Total Logs: ${logs.length}`,
      accuracy !== null ? `Accuracy: ${(accuracy * 100).toFixed(2)}%` : '',
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
    link.download = `training_logs_${modelName.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const progress = calculateProgress();

  // Parse metrics from logs
  const parseMetrics = (): TrainingMetrics => {
    const metrics: TrainingMetrics = {
      epochs: [],
      losses: [],
      accuracies: [],
      val_accuracies: []
    };

    logs.forEach(log => {
      if (log.data) {
        if (log.data.train_loss) {
          try {
            metrics.losses.push(parseFloat(log.data.train_loss));
          } catch (e) { }
        }
        if (log.data.train_accuracy) {
          try {
            const acc = parseFloat(log.data.train_accuracy.toString().replace('%', '')) / 100;
            metrics.accuracies.push(acc);
          } catch (e) { }
        }
        if (log.data.val_accuracy) {
          try {
            const acc = parseFloat(log.data.val_accuracy.toString().replace('%', '')) / 100;
            metrics.val_accuracies.push(acc);
          } catch (e) { }
        }
      }
    });

    return metrics;
  };

  const metrics = parseMetrics();

  return (
    <Box>
      {/* Connection Status Alert */}
      {!isConnected && (
        <Alert
          severity={reconnecting ? 'warning' : 'error'}
          sx={{ mb: 2 }}
          action={
            <IconButton
              color="inherit"
              size="small"
              onClick={connect}
              disabled={reconnecting}
            >
              <Refresh />
            </IconButton>
          }
          icon={reconnecting ? <CircularProgress size={20} /> : <WifiOff />}
        >
          {reconnecting
            ? `Reconnecting... (Attempt ${reconnectAttempts})`
            : 'Disconnected from server. Click refresh to reconnect.'}
        </Alert>
      )}

      {/* WebSocket Error Alert */}
      {wsError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {wsError}
        </Alert>
      )}

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
              Training Logs: {modelName}
            </Typography>
            <Typography variant="body2" sx={{ color: '#673ab7', fontWeight: 500 }}>
              Model ID: {modelId}
            </Typography>
            <Typography variant="caption" sx={{ color: isConnected ? '#4caf50' : '#f44336', fontWeight: 600 }}>
              {isConnected ? '🟢 Live' : '🔴 Offline'}
            </Typography>
          </Box>
          <Box>
            <Chip
              label={status.toUpperCase()}
              color={getStatusColor(status) as any}
              sx={{ mr: 1 }}
            />
            {accuracy !== null && (
              <Chip
                label={`Accuracy: ${(accuracy * 100).toFixed(2)}%`}
                color="success"
                variant="outlined"
              />
            )}
          </Box>
        </Box>

        {/* Progress Bar for Training */}
        {status === 'training' && progress > 0 && (
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
      {metrics.epochs.length > 0 && (
        <Accordion defaultExpanded sx={{ border: '1px solid #e1d5f0', mb: 2 }}>
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
                  {metrics.epochs.length}
                </Typography>
              </Box>
              {metrics.val_accuracies.length > 0 && (
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
                    {(Math.max(...metrics.val_accuracies) * 100).toFixed(1)}%
                  </Typography>
                </Box>
              )}
              {metrics.losses.length > 0 && (
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
                    {metrics.losses[metrics.losses.length - 1].toFixed(4)}
                  </Typography>
                </Box>
              )}
            </Box>
          </AccordionDetails>
        </Accordion>
      )}

      {/* Logs Viewer */}
      <Paper
        sx={{
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
            📋 Training Logs ({logs.length} entries)
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

              <Tooltip title="Refresh logs">
                <IconButton
                  size="small"
                  onClick={requestLogs}
                  sx={{
                    color: '#673ab7',
                    '&:hover': { bgcolor: 'rgba(103, 58, 183, 0.1)' }
                  }}
                >
                  <Refresh fontSize="small" />
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
            {logs.length === 0 ? (
              <Typography sx={{ color: 'text.secondary', textAlign: 'center', py: 4 }}>
                {isConnected
                  ? 'No training logs available yet. Logs will appear here during training.'
                  : 'Waiting for connection...'}
              </Typography>
            ) : (
              <GroupedTrainingLogs logs={logs} status={status} />
            )}
          </Box>
        )}

        {/* Detailed View */}
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

            {logs.length === 0 ? (
              <Typography sx={{ color: '#aaa', fontFamily: 'monospace' }}>
                {isConnected
                  ? 'No detailed logs available yet. Logs will appear here during training.'
                  : 'Waiting for connection...'}
              </Typography>
            ) : (
              <Box sx={{ fontFamily: 'monospace', fontSize: '0.875rem' }}>
                {logs.map((log, index) => (
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

            {status === 'training' && isConnected && (
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
                  Training in progress... (live updates via WebSocket)
                </Typography>
              </Box>
            )}
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default TrainingLogsViewerWebSocket;
