"""
Structured logging utilities with correlation IDs for permission operations
"""

import uuid
import logging
import json
from typing import Dict, Any, Optional
from functools import wraps
from django.conf import settings
from django.http import HttpRequest
import threading

# Thread-local storage for correlation ID
_thread_locals = threading.local()


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records"""
    
    def filter(self, record):
        record.correlation_id = get_correlation_id()
        return True


def get_correlation_id() -> str:
    """Get current correlation ID from thread-local storage"""
    return getattr(_thread_locals, 'correlation_id', 'no-correlation-id')


def set_correlation_id(correlation_id: str):
    """Set correlation ID in thread-local storage"""
    _thread_locals.correlation_id = correlation_id


def generate_correlation_id() -> str:
    """Generate a new correlation ID"""
    return str(uuid.uuid4())[:8]


class PermissionLogger:
    """Centralized logging for permission operations with structured data"""
    
    def __init__(self):
        self.logger = logging.getLogger('userauth.permissions')
        
    def _get_base_context(self, user=None, request=None) -> Dict[str, Any]:
        """Get base context for all permission logs"""
        context = {
            'correlation_id': get_correlation_id(),
            'service': 'permission_system',
            'timestamp': None,  # Will be set by logging formatter
        }
        
        if user:
            context.update({
                'user_id': str(user.id),
                'username': user.username,
                'user_email': user.email,
                'user_role': user.role,
            })
            
        if request:
            context.update({
                'ip_address': self._get_client_ip(request),
                'user_agent': request.META.get('HTTP_USER_AGENT', ''),
                'method': request.method,
                'path': request.path,
            })
            
        return context
    
    def _get_client_ip(self, request: HttpRequest) -> str:
        """Extract client IP from request"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip
    
    def permission_granted(
        self, 
        user, 
        permission_type: str, 
        resource_id: str, 
        access_type: str,
        granted_by=None,
        request=None,
        **kwargs
    ):
        """Log permission granted event"""
        context = self._get_base_context(user, request)
        context.update({
            'event': 'permission_granted',
            'permission_type': permission_type,
            'resource_identifier': resource_id,
            'access_type': access_type,
            'granted_by_id': str(granted_by.id) if granted_by else None,
            'granted_by_email': granted_by.email if granted_by else None,
            **kwargs
        })
        
        self.logger.info(
            f'Permission granted: {permission_type}:{access_type} on {resource_id}',
            extra=context
        )
    
    def permission_revoked(
        self,
        user,
        permission_type: str,
        resource_id: str,
        access_type: str,
        revoked_by=None,
        request=None,
        **kwargs
    ):
        """Log permission revoked event"""
        context = self._get_base_context(user, request)
        context.update({
            'event': 'permission_revoked',
            'permission_type': permission_type,
            'resource_identifier': resource_id,
            'access_type': access_type,
            'revoked_by_id': str(revoked_by.id) if revoked_by else None,
            'revoked_by_email': revoked_by.email if revoked_by else None,
            **kwargs
        })
        
        self.logger.info(
            f'Permission revoked: {permission_type}:{access_type} on {resource_id}',
            extra=context
        )
    
    def permission_denied(
        self,
        user,
        permission_type: str,
        resource_id: str,
        access_type: str,
        reason: str,
        request=None,
        **kwargs
    ):
        """Log permission denied event"""
        context = self._get_base_context(user, request)
        context.update({
            'event': 'permission_denied',
            'permission_type': permission_type,
            'resource_identifier': resource_id,
            'access_type': access_type,
            'denial_reason': reason,
            **kwargs
        })
        
        self.logger.warning(
            f'Permission denied: {permission_type}:{access_type} on {resource_id} - {reason}',
            extra=context
        )
    
    def bulk_operation(
        self,
        operation_type: str,
        user,
        affected_count: int,
        operation_details: Dict[str, Any],
        request=None,
        **kwargs
    ):
        """Log bulk permission operations"""
        context = self._get_base_context(user, request)
        context.update({
            'event': 'bulk_permission_operation',
            'operation_type': operation_type,
            'affected_count': affected_count,
            'operation_details': operation_details,
            **kwargs
        })
        
        self.logger.info(
            f'Bulk operation: {operation_type} affected {affected_count} permissions',
            extra=context
        )
    
    def permission_check(
        self,
        user,
        permission_type: str,
        resource_id: str,
        access_type: str,
        result: bool,
        cache_hit: bool = False,
        request=None,
        **kwargs
    ):
        """Log permission check events (for debugging and monitoring)"""
        context = self._get_base_context(user, request)
        context.update({
            'event': 'permission_check',
            'permission_type': permission_type,
            'resource_identifier': resource_id,
            'access_type': access_type,
            'check_result': result,
            'cache_hit': cache_hit,
            **kwargs
        })
        
        # Use debug level for successful checks, info for denied checks
        level = logging.DEBUG if result else logging.INFO
        self.logger.log(
            level,
            f'Permission check: {permission_type}:{access_type} on {resource_id} = {result}',
            extra=context
        )
    
    def rate_limit_exceeded(
        self,
        user,
        operation: str,
        limit_info: Dict[str, Any],
        request=None,
        **kwargs
    ):
        """Log rate limit exceeded events"""
        context = self._get_base_context(user, request)
        context.update({
            'event': 'rate_limit_exceeded',
            'operation': operation,
            'limit_info': limit_info,
            **kwargs
        })
        
        self.logger.warning(
            f'Rate limit exceeded for operation: {operation}',
            extra=context
        )
    
    def security_violation(
        self,
        user,
        violation_type: str,
        details: Dict[str, Any],
        request=None,
        **kwargs
    ):
        """Log security violations"""
        context = self._get_base_context(user, request)
        context.update({
            'event': 'security_violation',
            'violation_type': violation_type,
            'violation_details': details,
            **kwargs
        })
        
        self.logger.error(
            f'Security violation: {violation_type}',
            extra=context
        )


# Global logger instance
permission_logger = PermissionLogger()


def with_correlation_id(view_func):
    """Decorator to add correlation ID to request processing"""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        # Generate or extract correlation ID
        correlation_id = (
            request.META.get('HTTP_X_CORRELATION_ID') or 
            request.META.get('HTTP_X_REQUEST_ID') or
            generate_correlation_id()
        )
        
        # Set in thread-local storage
        set_correlation_id(correlation_id)
        
        try:
            response = view_func(request, *args, **kwargs)
            
            # Add correlation ID to response headers
            if hasattr(response, '__setitem__'):
                response['X-Correlation-ID'] = correlation_id
            
            return response
        finally:
            # Clean up thread-local storage
            if hasattr(_thread_locals, 'correlation_id'):
                delattr(_thread_locals, 'correlation_id')
    
    return wrapper


def log_permission_operation(operation_type: str):
    """Decorator to automatically log permission operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_correlation_id = get_correlation_id()
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful operation
                permission_logger.logger.debug(
                    f'Permission operation completed: {operation_type}',
                    extra={
                        'correlation_id': start_correlation_id,
                        'operation': operation_type,
                        'function': func.__name__,
                        'status': 'success'
                    }
                )
                
                return result
                
            except Exception as e:
                # Log failed operation
                permission_logger.logger.error(
                    f'Permission operation failed: {operation_type}',
                    extra={
                        'correlation_id': start_correlation_id,
                        'operation': operation_type,
                        'function': func.__name__,
                        'status': 'error',
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    },
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            log_data['correlation_id'] = record.correlation_id
        
        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'exc_info',
                          'exc_text', 'stack_info', 'lineno', 'funcName',
                          'created', 'msecs', 'relativeCreated', 'thread',
                          'threadName', 'processName', 'process', 'getMessage',
                          'correlation_id'):
                log_data[key] = value
        
        return json.dumps(log_data, default=str)