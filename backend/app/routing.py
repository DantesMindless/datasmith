"""
WebSocket URL routing configuration.
"""

from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/training/(?P<model_id>[^/]+)/$', consumers.TrainingLogsConsumer.as_asgi()),
]
