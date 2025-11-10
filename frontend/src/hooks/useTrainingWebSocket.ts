/**
 * Custom React Hook for WebSocket-based training logs streaming
 *
 * Provides real-time training logs via WebSocket connection with automatic
 * reconnection, authentication, and event handling.
 */

import { useState, useEffect, useRef, useCallback } from 'react';

export interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  data?: Record<string, any>;
}

export interface TrainingMetrics {
  epochs: number[];
  losses: number[];
  accuracies: number[];
  val_accuracies: number[];
}

export interface WebSocketMessage {
  type: 'log_entry' | 'log_batch' | 'training_complete' | 'error' | 'pong';
  log?: LogEntry;
  logs?: LogEntry[];
  total?: number;
  model_id?: string;
  status?: string;
  accuracy?: number;
  message?: string;
  error?: string;
  timestamp?: string;
}

export interface UseTrainingWebSocketOptions {
  modelId: string;
  enabled?: boolean;
  autoReconnect?: boolean;
  reconnectDelay?: number;
  maxReconnectAttempts?: number;
  onConnect?: () => void;
  onDisconnect?: (event: CloseEvent) => void;
  onError?: (error: Event) => void;
  onComplete?: (data: { model_id?: string; status?: string; accuracy?: number; message?: string }) => void;
}

export interface UseTrainingWebSocketReturn {
  logs: LogEntry[];
  isConnected: boolean;
  error: string | null;
  reconnecting: boolean;
  reconnectAttempts: number;
  clearLogs: () => void;
  requestLogs: () => void;
  disconnect: () => void;
  connect: () => void;
}

/**
 * Custom hook for WebSocket connection to training logs
 *
 * @param options - Configuration options for the WebSocket connection
 * @returns Object containing logs, connection state, and control functions
 *
 * @example
 * ```tsx
 * const { logs, isConnected, error } = useTrainingWebSocket({
 *   modelId: '123',
 *   enabled: true,
 *   onComplete: (data) => {
 *     console.log('Training completed:', data);
 *   }
 * });
 * ```
 */
export const useTrainingWebSocket = (
  options: UseTrainingWebSocketOptions
): UseTrainingWebSocketReturn => {
  const {
    modelId,
    enabled = true,
    autoReconnect = true,
    reconnectDelay = 3000,
    maxReconnectAttempts = 5,
    onConnect,
    onDisconnect,
    onError,
    onComplete,
  } = options;

  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reconnecting, setReconnecting] = useState(false);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  /**
   * Get WebSocket URL with authentication token
   */
  const getWebSocketUrl = useCallback((): string => {
    // Get auth token from localStorage
    const token = localStorage.getItem('access_token');

    // Determine protocol (ws:// or wss://)
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

    // Get host (use environment variable or default to localhost)
    const host = import.meta.env.VITE_WS_HOST || window.location.hostname;
    const port = import.meta.env.VITE_WS_PORT || '8000';

    // Construct WebSocket URL
    let url = `${protocol}//${host}:${port}/ws/training/${modelId}/`;

    // Add token as query parameter if available
    if (token) {
      url += `?token=${token}`;
    }

    return url;
  }, [modelId]);

  /**
   * Send ping message to keep connection alive
   */
  const sendPing = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        action: 'ping',
        timestamp: new Date().toISOString()
      }));
    }
  }, []);

  /**
   * Request full log refresh from server
   */
  const requestLogs = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        action: 'request_logs'
      }));
    }
  }, []);

  /**
   * Clear all logs
   */
  const clearLogs = useCallback(() => {
    setLogs([]);
  }, []);

  /**
   * Handle incoming WebSocket messages
   */
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const data: WebSocketMessage = JSON.parse(event.data);

      switch (data.type) {
        case 'log_entry':
          // Single log entry - append to logs
          if (data.log) {
            setLogs((prevLogs) => [...prevLogs, data.log!]);
          }
          break;

        case 'log_batch':
          // Batch of log entries - replace all logs
          if (data.logs) {
            setLogs(data.logs);
          }
          break;

        case 'training_complete':
          // Training completed notification
          console.log('Training completed:', data);
          if (onComplete) {
            onComplete({
              model_id: data.model_id,
              status: data.status,
              accuracy: data.accuracy,
              message: data.message
            });
          }
          break;

        case 'error':
          // Error notification
          console.error('Training error:', data.error);
          setError(data.error || 'An error occurred during training');
          break;

        case 'pong':
          // Pong response to ping
          console.debug('Received pong');
          break;

        default:
          console.warn('Unknown message type:', data.type);
      }
    } catch (err) {
      console.error('Error parsing WebSocket message:', err);
    }
  }, [onComplete]);

  /**
   * Connect to WebSocket
   */
  const connect = useCallback(() => {
    if (!enabled || !modelId) {
      return;
    }

    // Clear any existing reconnect timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    try {
      const url = getWebSocketUrl();
      console.log('Connecting to WebSocket:', url);

      const ws = new WebSocket(url);

      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setError(null);
        setReconnecting(false);
        setReconnectAttempts(0);

        // Start ping interval to keep connection alive (every 30 seconds)
        pingIntervalRef.current = setInterval(sendPing, 30000);

        if (onConnect) {
          onConnect();
        }

        // Request existing logs
        requestLogs();
      };

      ws.onmessage = handleMessage;

      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError('WebSocket connection error');

        if (onError) {
          onError(event);
        }
      };

      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        setIsConnected(false);

        // Clear ping interval
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }

        if (onDisconnect) {
          onDisconnect(event);
        }

        // Attempt to reconnect if enabled and not manually closed
        if (autoReconnect && event.code !== 1000 && reconnectAttempts < maxReconnectAttempts) {
          setReconnecting(true);
          setReconnectAttempts((prev) => prev + 1);

          console.log(`Reconnecting in ${reconnectDelay}ms (attempt ${reconnectAttempts + 1}/${maxReconnectAttempts})`);

          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectDelay);
        } else if (reconnectAttempts >= maxReconnectAttempts) {
          setError('Maximum reconnection attempts reached');
          setReconnecting(false);
        }
      };

      wsRef.current = ws;
    } catch (err) {
      console.error('Error creating WebSocket:', err);
      setError('Failed to create WebSocket connection');
    }
  }, [
    enabled,
    modelId,
    autoReconnect,
    reconnectDelay,
    maxReconnectAttempts,
    reconnectAttempts,
    getWebSocketUrl,
    handleMessage,
    onConnect,
    onDisconnect,
    onError,
    sendPing,
    requestLogs
  ]);

  /**
   * Disconnect from WebSocket
   */
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnected');
      wsRef.current = null;
    }

    setIsConnected(false);
    setReconnecting(false);
  }, []);

  // Connect on mount and when dependencies change
  useEffect(() => {
    if (enabled && modelId) {
      connect();
    }

    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [enabled, modelId]); // Removed connect/disconnect to avoid infinite loop

  return {
    logs,
    isConnected,
    error,
    reconnecting,
    reconnectAttempts,
    clearLogs,
    requestLogs,
    disconnect,
    connect
  };
};
