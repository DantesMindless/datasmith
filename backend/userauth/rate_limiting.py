"""
Rate limiting utilities for permission operations
"""

import time
from functools import wraps
from typing import Dict, Optional
from django.core.cache import cache
from django.http import JsonResponse
from rest_framework import status
from rest_framework.response import Response
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter using Django cache backend"""
    
    @staticmethod
    def _get_cache_key(user_id: str, operation: str) -> str:
        """Generate cache key for rate limiting"""
        return f"rate_limit:{user_id}:{operation}"
    
    @staticmethod
    def is_rate_limited(
        user_id: str, 
        operation: str, 
        max_requests: int = 10, 
        window_seconds: int = 60
    ) -> tuple[bool, Dict]:
        """
        Check if user is rate limited for specific operation
        
        Args:
            user_id: User ID
            operation: Operation name (e.g., 'grant_permission', 'bulk_operation')
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (is_limited, info_dict)
        """
        cache_key = RateLimiter._get_cache_key(user_id, operation)
        current_time = int(time.time())
        
        # Get current request data
        request_data = cache.get(cache_key, {
            'count': 0,
            'window_start': current_time,
            'requests': []
        })
        
        # Clean old requests outside the window
        window_start = current_time - window_seconds
        request_data['requests'] = [
            req_time for req_time in request_data['requests'] 
            if req_time > window_start
        ]
        request_data['count'] = len(request_data['requests'])
        
        # Check if rate limited
        if request_data['count'] >= max_requests:
            # Update cache with current data
            cache.set(cache_key, request_data, window_seconds)
            
            oldest_request = min(request_data['requests']) if request_data['requests'] else current_time
            reset_time = oldest_request + window_seconds
            
            return True, {
                'error': 'Rate limit exceeded',
                'limit': max_requests,
                'window_seconds': window_seconds,
                'current_count': request_data['count'],
                'reset_at': reset_time,
                'retry_after': reset_time - current_time
            }
        
        # Add current request
        request_data['requests'].append(current_time)
        request_data['count'] += 1
        
        # Update cache
        cache.set(cache_key, request_data, window_seconds)
        
        return False, {
            'limit': max_requests,
            'window_seconds': window_seconds,
            'current_count': request_data['count'],
            'remaining': max_requests - request_data['count']
        }
    
    @staticmethod
    def reset_rate_limit(user_id: str, operation: str):
        """Reset rate limit for specific user and operation"""
        cache_key = RateLimiter._get_cache_key(user_id, operation)
        cache.delete(cache_key)


def rate_limit(operation: str, max_requests: int = 10, window_seconds: int = 60):
    """
    Decorator for rate limiting API endpoints
    
    Args:
        operation: Operation name for tracking
        max_requests: Maximum requests allowed in window
        window_seconds: Time window in seconds
    """
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return view_func(request, *args, **kwargs)
            
            user_id = str(request.user.id)
            
            # Check rate limit
            is_limited, info = RateLimiter.is_rate_limited(
                user_id, operation, max_requests, window_seconds
            )
            
            if is_limited:
                logger.warning(
                    f"Rate limit exceeded for user {user_id} on operation {operation}",
                    extra={
                        'user_id': user_id,
                        'operation': operation,
                        'limit_info': info
                    }
                )
                
                response = Response(info, status=status.HTTP_429_TOO_MANY_REQUESTS)
                response['Retry-After'] = str(info['retry_after'])
                response['X-RateLimit-Limit'] = str(max_requests)
                response['X-RateLimit-Remaining'] = '0'
                response['X-RateLimit-Reset'] = str(info['reset_at'])
                return response
            
            # Add rate limit headers to successful responses
            response = view_func(request, *args, **kwargs)
            
            if hasattr(response, 'headers'):
                response['X-RateLimit-Limit'] = str(max_requests)
                response['X-RateLimit-Remaining'] = str(info['remaining'])
                response['X-RateLimit-Window'] = str(window_seconds)
            
            return response
        return wrapper
    return decorator


def permission_rate_limit(max_requests: int = 10, window_seconds: int = 60):
    """Specialized rate limiter for permission operations"""
    return rate_limit('permission_operations', max_requests, window_seconds)


def bulk_operation_rate_limit(max_requests: int = 3, window_seconds: int = 300):
    """Specialized rate limiter for bulk operations (more restrictive)"""
    return rate_limit('bulk_operations', max_requests, window_seconds)


def audit_rate_limit(max_requests: int = 50, window_seconds: int = 60):
    """Specialized rate limiter for audit log access"""
    return rate_limit('audit_access', max_requests, window_seconds)


class PermissionRateLimitMixin:
    """Mixin to add rate limiting to permission ViewSets"""
    
    def get_rate_limit_info(self, request):
        """Get current rate limit status for user"""
        if not request.user.is_authenticated:
            return {'error': 'Authentication required'}
        
        user_id = str(request.user.id)
        operations = ['permission_operations', 'bulk_operations', 'audit_access']
        
        status_info = {}
        for operation in operations:
            _, info = RateLimiter.is_rate_limited(
                user_id, operation, 
                max_requests=100,  # High limit for status check
                window_seconds=60
            )
            status_info[operation] = {
                'limit': info.get('limit', 'N/A'),
                'remaining': info.get('remaining', 'N/A'),
                'current_count': info.get('current_count', 0)
            }
        
        return status_info