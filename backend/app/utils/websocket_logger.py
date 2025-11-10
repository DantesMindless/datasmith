"""
WebSocket logger utility for sending real-time training logs.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from django.core.cache import cache

logger = logging.getLogger(__name__)


class WebSocketLogger:
    """
    Logger that sends training logs via WebSocket and caches them.

    Usage:
        ws_logger = WebSocketLogger(model_id='some-uuid')
        ws_logger.log('INFO', 'Training started', {'epochs': 10})
    """

    def __init__(self, model_id: str):
        """
        Initialize WebSocket logger.

        Args:
            model_id: Model ID to associate logs with
        """
        self.model_id = str(model_id)
        self.channel_layer = get_channel_layer()
        self.room_group_name = f'training_logs_{self.model_id}'
        self.cache_key = f'training_logs_{self.model_id}'

    def log(self, level: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a message and send it via WebSocket.

        Args:
            level: Log level (INFO, SUCCESS, WARNING, ERROR, DEBUG, PROGRESS)
            message: Log message
            data: Optional additional data dictionary
        """
        try:
            # Create log entry
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': level.upper(),
                'message': message,
                'data': data or {}
            }

            # Add to cache
            self._add_to_cache(log_entry)

            # Send via WebSocket
            if self.channel_layer:
                async_to_sync(self.channel_layer.group_send)(
                    self.room_group_name,
                    {
                        'type': 'training_log',
                        'log': log_entry
                    }
                )
            else:
                logger.warning("Channel layer not configured, logs will only be cached")

        except Exception as e:
            logger.error(f"Error sending WebSocket log: {e}", exc_info=True)

    def info(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log INFO level message."""
        self.log('INFO', message, data)

    def success(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log SUCCESS level message."""
        self.log('SUCCESS', message, data)

    def warning(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log WARNING level message."""
        self.log('WARNING', message, data)

    def error(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log ERROR level message."""
        self.log('ERROR', message, data)

    def debug(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log DEBUG level message."""
        self.log('DEBUG', message, data)

    def progress(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log PROGRESS level message."""
        self.log('PROGRESS', message, data)

    def complete(self, status: str = 'complete', accuracy: Optional[float] = None, message: str = 'Training completed successfully') -> None:
        """
        Send training completion notification.

        Args:
            status: Final status
            accuracy: Final accuracy
            message: Completion message
        """
        try:
            if self.channel_layer:
                async_to_sync(self.channel_layer.group_send)(
                    self.room_group_name,
                    {
                        'type': 'training_complete',
                        'model_id': self.model_id,
                        'status': status,
                        'accuracy': accuracy,
                        'message': message
                    }
                )
        except Exception as e:
            logger.error(f"Error sending completion notification: {e}", exc_info=True)

    def send_error(self, error_message: str) -> None:
        """
        Send training error notification.

        Args:
            error_message: Error message
        """
        try:
            # Log the error
            self.error(error_message)

            # Send error notification
            if self.channel_layer:
                async_to_sync(self.channel_layer.group_send)(
                    self.room_group_name,
                    {
                        'type': 'training_error',
                        'model_id': self.model_id,
                        'error': error_message,
                        'message': 'Training encountered an error'
                    }
                )
        except Exception as e:
            logger.error(f"Error sending error notification: {e}", exc_info=True)

    def clear_logs(self) -> None:
        """Clear cached logs for this model."""
        try:
            cache.delete(self.cache_key)
            logger.info(f"Cleared logs for model {self.model_id}")
        except Exception as e:
            logger.error(f"Error clearing logs: {e}", exc_info=True)

    def _add_to_cache(self, log_entry: Dict[str, Any]) -> None:
        """
        Add log entry to cache.

        Args:
            log_entry: Log entry to add
        """
        try:
            # Get existing logs
            logs = cache.get(self.cache_key, [])

            # Add new log
            logs.append(log_entry)

            # Limit to last 1000 logs to prevent memory issues
            if len(logs) > 1000:
                logs = logs[-1000:]

            # Update cache (1 hour timeout)
            cache.set(self.cache_key, logs, timeout=3600)

        except Exception as e:
            logger.error(f"Error adding log to cache: {e}", exc_info=True)
