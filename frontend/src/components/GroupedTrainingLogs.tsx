import React, { useState } from 'react';
import {
  Box,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Typography,
  Chip,
  LinearProgress,
  Paper,
  Stack,
  Divider,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  CheckCircle,
  Error,
  Warning,
  Info,
  Timeline,
  Speed,
  Storage,
  TrendingUp,
  Schedule,
} from '@mui/icons-material';

interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  data?: Record<string, any>;
}

interface GroupedTrainingLogsProps {
  logs: LogEntry[];
  status: string;
}

interface LogGroup {
  title: string;
  icon: React.ReactNode;
  logs: LogEntry[];
  timestamp: string;
  color: string;
  expanded: boolean;
}

const GroupedTrainingLogs: React.FC<GroupedTrainingLogsProps> = ({ logs, status }) => {
  // Group logs by phase
  const [expandedSections, setExpandedSections] = useState<Set<number>>(new Set([0, 1, 2])); // Expand first 3 by default

  const groupLogs = (): LogGroup[] => {
    const groups: LogGroup[] = [];
    let currentGroup: LogGroup | null = null;

    logs.forEach((log, index) => {
      const msg = log.message.toLowerCase();

      // Determine which group this log belongs to
      if (msg.includes('training started') || msg.includes('🚀')) {
        currentGroup = {
          title: 'Training Started',
          icon: <TrendingUp sx={{ color: '#2196f3' }} />,
          logs: [log],
          timestamp: log.timestamp,
          color: '#e3f2fd',
          expanded: true,
        };
        groups.push(currentGroup);
      } else if (msg.includes('preparing your data') || msg.includes('preparing data') || msg.includes('📊')) {
        currentGroup = {
          title: 'Data Preparation',
          icon: <Storage sx={{ color: '#673ab7' }} />,
          logs: [log],
          timestamp: log.timestamp,
          color: '#f3e5f5',
          expanded: true,
        };
        groups.push(currentGroup);
      } else if (msg.includes('dividing data') || msg.includes('✂️')) {
        if (currentGroup && currentGroup.title === 'Data Preparation') {
          currentGroup.logs.push(log);
        } else {
          currentGroup = {
            title: 'Data Split',
            icon: <Storage sx={{ color: '#673ab7' }} />,
            logs: [log],
            timestamp: log.timestamp,
            color: '#f3e5f5',
            expanded: true,
          };
          groups.push(currentGroup);
        }
      } else if (msg.includes('preparing data for learning') || msg.includes('🔧')) {
        if (currentGroup && currentGroup.title === 'Data Preparation') {
          currentGroup.logs.push(log);
        } else {
          currentGroup = {
            title: 'Data Preprocessing',
            icon: <Storage sx={{ color: '#673ab7' }} />,
            logs: [log],
            timestamp: log.timestamp,
            color: '#f3e5f5',
            expanded: true,
          };
          groups.push(currentGroup);
        }
      } else if (msg.includes('building the model') || msg.includes('🏗️')) {
        currentGroup = {
          title: 'Building the Model',
          icon: <Info sx={{ color: '#ff9800' }} />,
          logs: [log],
          timestamp: log.timestamp,
          color: '#fff3e0',
          expanded: true,
        };
        groups.push(currentGroup);
      } else if (msg.includes('starting to learn') || msg.includes('🎯')) {
        currentGroup = {
          title: 'Training Progress',
          icon: <Timeline sx={{ color: '#4caf50' }} />,
          logs: [log],
          timestamp: log.timestamp,
          color: '#e8f5e9',
          expanded: true,
        };
        groups.push(currentGroup);
      } else if (msg.includes('training round') || msg.includes('round') && msg.includes('complete')) {
        if (currentGroup && currentGroup.title === 'Training Progress') {
          currentGroup.logs.push(log);
        } else {
          currentGroup = {
            title: 'Training Progress',
            icon: <Timeline sx={{ color: '#4caf50' }} />,
            logs: [log],
            timestamp: log.timestamp,
            color: '#e8f5e9',
            expanded: true,
          };
          groups.push(currentGroup);
        }
      } else if (msg.includes('training complete') || msg.includes('✅')) {
        currentGroup = {
          title: 'Training Complete',
          icon: <CheckCircle sx={{ color: '#4caf50' }} />,
          logs: [log],
          timestamp: log.timestamp,
          color: '#e8f5e9',
          expanded: true,
        };
        groups.push(currentGroup);
      } else if (msg.includes('checking model performance') || msg.includes('🔍')) {
        currentGroup = {
          title: 'Model Validation',
          icon: <Speed sx={{ color: '#2196f3' }} />,
          logs: [log],
          timestamp: log.timestamp,
          color: '#e3f2fd',
          expanded: true,
        };
        groups.push(currentGroup);
      } else if (log.level === 'ERROR' || msg.includes('❌') || msg.includes('failed') || msg.includes('exception')) {
        currentGroup = {
          title: 'Error Occurred',
          icon: <Error sx={{ color: '#f44336' }} />,
          logs: [log],
          timestamp: log.timestamp,
          color: '#ffebee',
          expanded: true,
        };
        groups.push(currentGroup);
      } else if (currentGroup) {
        // Add to current group
        currentGroup.logs.push(log);
      } else {
        // Misc group
        currentGroup = {
          title: 'General Logs',
          icon: <Info sx={{ color: '#9e9e9e' }} />,
          logs: [log],
          timestamp: log.timestamp,
          color: '#f5f5f5',
          expanded: false,
        };
        groups.push(currentGroup);
      }
    });

    return groups;
  };

  const groups = groupLogs();

  const handleToggle = (index: number) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedSections(newExpanded);
  };

  const getTimeAgo = (timestamp: string): string => {
    try {
      const logTime = new Date(timestamp);
      const now = new Date();
      const diffMs = now.getTime() - logTime.getTime();
      const diffSec = Math.floor(diffMs / 1000);

      if (diffSec < 60) return `${diffSec}s ago`;
      if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
      return `${Math.floor(diffSec / 3600)}h ago`;
    } catch {
      return 'just now';
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
      case 'DEBUG':
        return '#673ab7';
      default:
        return '#9e9e9e';
    }
  };

  const extractMetrics = (logMessages: LogEntry[]) => {
    const metricsMap = new Map<string, { label: string; value: string; icon?: React.ReactNode }>();

    // Process logs in reverse order to get the most recent values
    [...logMessages].reverse().forEach(log => {
      const msg = log.message;

      // Extract accuracy (only if not already found)
      if (msg.includes('accuracy:') && !metricsMap.has('Training') && !metricsMap.has('Validation')) {
        const match = msg.match(/accuracy:\s*(\d+\.?\d*%)/i);
        if (match) {
          const type = msg.includes('training') ? 'Training' : msg.includes('validation') ? 'Validation' : 'Accuracy';
          if (!metricsMap.has(type)) {
            metricsMap.set(type, {
              label: type,
              value: match[1],
              icon: <TrendingUp fontSize="small" />
            });
          }
        }
      }

      // Extract time (only the most recent)
      if (msg.includes('time') && (msg.includes(':') || msg.includes('taken')) && !metricsMap.has('Time')) {
        const match = msg.match(/(\d+\.?\d*[smh])/);
        if (match) {
          metricsMap.set('Time', {
            label: 'Time',
            value: match[1],
            icon: <Schedule fontSize="small" />
          });
        }
      }

      // Extract round info (only the most recent)
      if (msg.includes('round') && msg.includes('of') && !metricsMap.has('Progress')) {
        const match = msg.match(/round\s+(\d+)\s+of\s+(\d+)/i);
        if (match) {
          const progress = (parseInt(match[1]) / parseInt(match[2])) * 100;
          metricsMap.set('Progress', {
            label: 'Progress',
            value: `${match[1]}/${match[2]} (${progress.toFixed(0)}%)`,
            icon: <Timeline fontSize="small" />
          });
        }
      }
    });

    return Array.from(metricsMap.values());
  };

  if (logs.length === 0) {
    return (
      <Box sx={{ p: 3, textAlign: 'center', color: 'text.secondary' }}>
        <Info sx={{ fontSize: 48, mb: 2, opacity: 0.5 }} />
        <Typography>No training logs available yet.</Typography>
        <Typography variant="body2">Logs will appear here when training starts.</Typography>
      </Box>
    );
  }

  return (
    <Box>
      {groups.map((group, index) => {
        const metrics = extractMetrics(group.logs);
        const isExpanded = expandedSections.has(index);

        return (
          <Accordion
            key={index}
            expanded={isExpanded}
            onChange={() => handleToggle(index)}
            sx={{
              mb: 1,
              border: '1px solid',
              borderColor: 'divider',
              borderRadius: 1,
              '&:before': { display: 'none' },
              boxShadow: 1,
            }}
          >
            <AccordionSummary
              expandIcon={<ExpandMoreIcon />}
              sx={{
                bgcolor: group.color,
                borderRadius: 1,
                '&:hover': { bgcolor: group.color, filter: 'brightness(0.98)' },
                minHeight: 64,
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', gap: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', flex: 1 }}>
                  {group.icon}
                  <Typography variant="subtitle1" fontWeight={600} sx={{ ml: 1.5 }}>
                    {group.title}
                  </Typography>
                  <Chip
                    label={getTimeAgo(group.timestamp)}
                    size="small"
                    sx={{ ml: 2, height: 20, fontSize: '0.7rem' }}
                  />
                </Box>

                {/* Show metrics inline */}
                {!isExpanded && metrics.length > 0 && (
                  <Box sx={{ display: 'flex', gap: 2, mr: 2 }}>
                    {metrics.slice(0, 3).map((metric, i) => (
                      <Box key={i} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        {metric.icon}
                        <Typography variant="body2" fontWeight={500}>
                          {metric.value}
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                )}
              </Box>
            </AccordionSummary>

            <AccordionDetails sx={{ bgcolor: '#fafafa', p: 2 }}>
              <Stack spacing={1.5}>
                {group.logs.map((log, logIndex) => {
                  const isMainMessage = logIndex === 0 || log.message.includes('✓') || log.message.includes('•');
                  const indent = log.message.startsWith('  ') ? 2 : log.message.startsWith('      ') ? 4 : 0;

                  return (
                    <Box
                      key={logIndex}
                      sx={{
                        pl: indent,
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: 1,
                      }}
                    >
                      {isMainMessage && (
                        <Box
                          sx={{
                            width: 4,
                            height: 4,
                            borderRadius: '50%',
                            bgcolor: getLevelColor(log.level),
                            mt: 1,
                            flexShrink: 0,
                          }}
                        />
                      )}
                      <Typography
                        variant="body2"
                        sx={{
                          fontFamily: log.message.includes('━') || log.message.includes('═') ? 'monospace' : 'inherit',
                          color: log.level === 'SUCCESS' ? 'success.main' :
                                 log.level === 'ERROR' ? 'error.main' :
                                 log.level === 'WARNING' ? 'warning.main' :
                                 'text.primary',
                          fontWeight: isMainMessage ? 500 : 400,
                        }}
                      >
                        {log.message}
                      </Typography>
                    </Box>
                  );
                })}

                {/* Show metrics as chips at bottom */}
                {metrics.length > 0 && (
                  <>
                    <Divider sx={{ my: 1 }} />
                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                      {metrics.map((metric, i) => (
                        <Chip
                          key={i}
                          icon={metric.icon}
                          label={`${metric.label}: ${metric.value}`}
                          size="small"
                          variant="outlined"
                          sx={{ fontWeight: 500 }}
                        />
                      ))}
                    </Box>
                  </>
                )}
              </Stack>
            </AccordionDetails>
          </Accordion>
        );
      })}
    </Box>
  );
};

export default GroupedTrainingLogs;
