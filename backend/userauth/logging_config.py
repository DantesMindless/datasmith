"""
Logging configuration for permission system with structured logging
"""

import os
from django.conf import settings

# Logging configuration for Django settings
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            '()': 'userauth.logging_utils.JSONFormatter',
        },
        'verbose': {
            'format': '{levelname} {asctime} [{correlation_id}] {name} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'filters': {
        'correlation_id': {
            '()': 'userauth.logging_utils.CorrelationIdFilter',
        },
        'require_debug_false': {
            '()': 'django.utils.log.RequireDebugFalse',
        },
        'require_debug_true': {
            '()': 'django.utils.log.RequireDebugTrue',
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
            'filters': ['correlation_id'],
        },
        'file_permissions': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(settings.BASE_DIR, 'logs', 'permissions.log'),
            'maxBytes': 1024*1024*10,  # 10MB
            'backupCount': 5,
            'formatter': 'json',
            'filters': ['correlation_id'],
        },
        'file_security': {
            'level': 'WARNING',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(settings.BASE_DIR, 'logs', 'security.log'),
            'maxBytes': 1024*1024*10,  # 10MB
            'backupCount': 10,
            'formatter': 'json',
            'filters': ['correlation_id'],
        },
        'mail_admins': {
            'level': 'ERROR',
            'class': 'django.utils.log.AdminEmailHandler',
            'filters': ['require_debug_false', 'correlation_id'],
            'formatter': 'verbose',
        }
    },
    'loggers': {
        'userauth.permissions': {
            'handlers': ['console', 'file_permissions', 'file_security'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'userauth.services': {
            'handlers': ['console', 'file_permissions'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'userauth.views': {
            'handlers': ['console', 'file_permissions'],
            'level': 'INFO',
            'propagate': False,
        },
        'django.security': {
            'handlers': ['console', 'file_security', 'mail_admins'],
            'level': 'WARNING',
            'propagate': False,
        },
        'django.request': {
            'handlers': ['console', 'mail_admins'],
            'level': 'ERROR',
            'propagate': False,
        },
        'root': {
            'handlers': ['console'],
            'level': 'INFO',
        },
    },
}


def setup_logging():
    """
    Setup logging configuration for the permission system
    Call this in Django settings.py or apps.py
    """
    import logging.config
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(settings.BASE_DIR, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Apply logging configuration
    logging.config.dictConfig(LOGGING_CONFIG)
    
    # Log that logging is configured
    logger = logging.getLogger('userauth.permissions')
    logger.info('Permission system logging configured successfully')