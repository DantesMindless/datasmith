"""
WebSocket consumers for real-time communication.
"""

import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.core.cache import cache

logger = logging.getLogger(__name__)


class TrainingLogsConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for streaming real-time training logs.

    Connection URL: ws://localhost:8000/ws/training/{model_id}/

    Messages sent to client:
    - log_entry: Individual log entry
    - log_batch: Batch of log entries
    - training_complete: Training finished notification
    - error: Error notification
    """

    async def connect(self):
        """Handle WebSocket connection."""
        try:
            self.model_id = self.scope['url_route']['kwargs']['model_id']
            self.room_group_name = f'training_logs_{self.model_id}'

            # Check authentication
            user = self.scope.get('user')
            if not user or not user.is_authenticated:
                logger.warning(f"Unauthorized WebSocket connection attempt for model {self.model_id}")
                await self.close(code=4001)
                return

            # Verify model exists and user has permission
            has_permission = await self.check_model_permission(user)
            if not has_permission:
                logger.warning(f"User {user.id} denied access to model {self.model_id} logs")
                await self.close(code=4003)
                return

            # Accept the connection first
            await self.accept()
            logger.info(f"WebSocket connected: user={user.id}, model={self.model_id}, channel={self.channel_name}")

            # Join room group after accepting
            await self.channel_layer.group_add(
                self.room_group_name,
                self.channel_name
            )

            # Send existing logs to the newly connected client
            await self.send_existing_logs()

        except Exception as e:
            logger.error(f"Error in WebSocket connect: {e}", exc_info=True)
            await self.close(code=1011)

    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        # Leave room group
        if hasattr(self, 'room_group_name'):
            await self.channel_layer.group_discard(
                self.room_group_name,
                self.channel_name
            )
            logger.info(f"WebSocket disconnected: model={self.model_id}, code={close_code}")

    async def receive(self, text_data):
        """Handle messages received from WebSocket client."""
        try:
            data = json.loads(text_data)
            action = data.get('action')

            if action == 'ping':
                # Respond to ping to keep connection alive
                await self.send(text_data=json.dumps({
                    'type': 'pong',
                    'timestamp': data.get('timestamp')
                }))
            elif action == 'request_logs':
                # Client requesting full log refresh
                await self.send_existing_logs()
            else:
                logger.warning(f"Unknown action received: {action}")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received: {text_data}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}", exc_info=True)

    async def send_existing_logs(self):
        """Send all existing cached logs to the client."""
        try:
            logs = await self.get_cached_logs()
            if logs:
                await self.send(text_data=json.dumps({
                    'type': 'log_batch',
                    'logs': logs,
                    'total': len(logs)
                }))
            else:
                await self.send(text_data=json.dumps({
                    'type': 'log_batch',
                    'logs': [],
                    'total': 0,
                    'message': 'No logs available yet'
                }))
        except Exception as e:
            logger.error(f"Error sending existing logs: {e}", exc_info=True)

    # Receive message from room group
    async def training_log(self, event):
        """
        Handle training log events from the channel layer.

        Args:
            event: Event dict with log entry data
        """
        # Send log to WebSocket
        await self.send(text_data=json.dumps({
            'type': 'log_entry',
            'log': event['log']
        }))

    async def training_complete(self, event):
        """
        Handle training completion events.

        Args:
            event: Event dict with completion data
        """
        await self.send(text_data=json.dumps({
            'type': 'training_complete',
            'model_id': event.get('model_id'),
            'status': event.get('status'),
            'accuracy': event.get('accuracy'),
            'message': event.get('message', 'Training completed successfully')
        }))

    async def training_error(self, event):
        """
        Handle training error events.

        Args:
            event: Event dict with error data
        """
        await self.send(text_data=json.dumps({
            'type': 'error',
            'model_id': event.get('model_id'),
            'error': event.get('error'),
            'message': event.get('message', 'Training encountered an error')
        }))

    @database_sync_to_async
    def check_model_permission(self, user):
        """
        Check if user has permission to access model logs.

        Args:
            user: User instance

        Returns:
            bool: True if user has permission
        """
        from app.models.main import MLModel

        try:
            model = MLModel.objects.get(id=self.model_id, deleted=False)
            # For now, any authenticated user can view (adjust as needed)
            return True
        except MLModel.DoesNotExist:
            return False

    @database_sync_to_async
    def get_cached_logs(self):
        """
        Get cached logs from Redis.

        Returns:
            list: List of log entries
        """
        cache_key = f"training_logs_{self.model_id}"
        return cache.get(cache_key, [])
