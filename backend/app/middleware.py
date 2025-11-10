"""
WebSocket authentication middleware for Django Channels.
"""

from urllib.parse import parse_qs
from channels.db import database_sync_to_async
from channels.middleware import BaseMiddleware
from django.contrib.auth.models import AnonymousUser
from rest_framework_simplejwt.tokens import AccessToken
from django.contrib.auth import get_user_model

User = get_user_model()


@database_sync_to_async
def get_user_from_token(token_string):
    """
    Get user from JWT token.

    Args:
        token_string: JWT token string

    Returns:
        User instance or AnonymousUser
    """
    try:
        # Validate and decode the token
        access_token = AccessToken(token_string)
        user_id = access_token['user_id']

        # Get user from database
        user = User.objects.get(id=user_id)
        return user
    except Exception as e:
        print(f"Token validation error: {e}")
        return AnonymousUser()


class JWTAuthMiddleware(BaseMiddleware):
    """
    Custom middleware to authenticate WebSocket connections using JWT tokens.

    Supports token in:
    1. Query string: ?token=xxx
    2. Header: Authorization: Bearer xxx (not yet implemented for WebSockets)
    """

    async def __call__(self, scope, receive, send):
        # Get query string from scope
        query_string = scope.get('query_string', b'').decode()
        query_params = parse_qs(query_string)

        # Try to get token from query string
        token = query_params.get('token', [None])[0]

        if token:
            # Authenticate user with token
            scope['user'] = await get_user_from_token(token)
        else:
            # No token provided
            scope['user'] = AnonymousUser()

        return await super().__call__(scope, receive, send)


def JWTAuthMiddlewareStack(inner):
    """
    Convenience function to wrap the inner application with JWT auth middleware.

    Usage:
        application = ProtocolTypeRouter({
            "websocket": JWTAuthMiddlewareStack(
                URLRouter(websocket_urlpatterns)
            ),
        })
    """
    return JWTAuthMiddleware(inner)
