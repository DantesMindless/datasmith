"""
Celery tasks for bulk permission operations
"""

import logging
from typing import List, Dict, Any
from celery import shared_task
from django.contrib.auth import get_user_model
from django.db import transaction

from .services import PermissionService
from .models import AccessType, DataSourcePermission, ColumnPermission
from .logging_utils import permission_logger, set_correlation_id, generate_correlation_id
from .cache_manager import cache_manager

User = get_user_model()
logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def bulk_grant_column_permissions(
    self,
    user_ids: List[str],
    datasource_id: str,
    table_name: str,
    column_names: List[str],
    access_type: str = AccessType.READ,
    allow_in_segments: bool = False,
    granted_by_id: str = None,
    correlation_id: str = None
):
    """
    Celery task for bulk granting column permissions
    
    Args:
        user_ids: List of user IDs to grant permissions to
        datasource_id: DataSource ID
        table_name: Table name
        column_names: List of column names
        access_type: Access type (read, write, etc.)
        allow_in_segments: Whether to allow in ML segments
        granted_by_id: ID of user granting permissions
        correlation_id: Correlation ID for tracking
    """
    # Set correlation ID for logging
    if correlation_id:
        set_correlation_id(correlation_id)
    else:
        set_correlation_id(generate_correlation_id())
    
    try:
        logger.info(
            f"Starting bulk column permission grant task: {len(user_ids)} users, "
            f"{len(column_names)} columns",
            extra={
                'task_id': self.request.id,
                'user_count': len(user_ids),
                'column_count': len(column_names),
                'datasource_id': datasource_id,
                'table_name': table_name
            }
        )
        
        # Get valid users
        users = User.objects.filter(id__in=user_ids)
        granted_by = User.objects.get(id=granted_by_id) if granted_by_id else None
        
        # Use PermissionService bulk operation
        total_created, created_permissions = PermissionService.bulk_grant_permissions(
            users=list(users),
            datasource_id=datasource_id,
            table_name=table_name,
            column_names=column_names,
            access_type=access_type,
            allow_in_segments=allow_in_segments,
            granted_by=granted_by
        )
        
        # Log bulk operation
        if granted_by:
            permission_logger.bulk_operation(
                operation_type='bulk_grant_column_permissions',
                user=granted_by,
                affected_count=total_created,
                operation_details={
                    'target_user_count': len(user_ids),
                    'datasource_id': datasource_id,
                    'table_name': table_name,
                    'column_names': column_names,
                    'access_type': access_type,
                    'allow_in_segments': allow_in_segments,
                    'task_id': self.request.id
                }
            )
        
        # Warm cache for affected users
        for user in users:
            cache_manager.warm_user_cache(str(user.id), [datasource_id])
        
        result = {
            'status': 'success',
            'total_created': total_created,
            'created_permissions': created_permissions[:100],  # Limit response size
            'total_users_processed': len(users),
            'task_id': self.request.id
        }
        
        logger.info(
            f"Completed bulk column permission grant: {total_created} permissions created",
            extra=result
        )
        
        return result
        
    except Exception as exc:
        logger.error(
            f"Bulk column permission grant failed: {exc}",
            extra={
                'task_id': self.request.id,
                'error': str(exc),
                'user_ids': user_ids,
                'datasource_id': datasource_id
            },
            exc_info=True
        )
        
        # Log security violation if this looks suspicious
        if granted_by:
            permission_logger.security_violation(
                user=granted_by,
                violation_type='bulk_permission_failure',
                details={
                    'task_id': self.request.id,
                    'error': str(exc),
                    'attempted_user_count': len(user_ids),
                    'datasource_id': datasource_id
                }
            )
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying bulk permission task {self.request.id} (attempt {self.request.retries + 1})")
            raise self.retry(countdown=60 * (2 ** self.request.retries))  # Exponential backoff
        
        raise exc


@shared_task(bind=True)
def revoke_user_all_permissions(
    self,
    user_id: str,
    permission_types: List[str] = None,
    revoked_by_id: str = None,
    correlation_id: str = None
):
    """
    Celery task for revoking all permissions for a user
    
    Args:
        user_id: User ID to revoke permissions from
        permission_types: Types of permissions to revoke ['datasource', 'table', 'column']
        revoked_by_id: ID of user performing revocation
        correlation_id: Correlation ID for tracking
    """
    if correlation_id:
        set_correlation_id(correlation_id)
    else:
        set_correlation_id(generate_correlation_id())
    
    if permission_types is None:
        permission_types = ['datasource', 'table', 'column']
    
    try:
        user = User.objects.get(id=user_id)
        revoked_by = User.objects.get(id=revoked_by_id) if revoked_by_id else None
        
        logger.info(
            f"Starting permission revocation for user {user.email}",
            extra={
                'task_id': self.request.id,
                'target_user_id': user_id,
                'permission_types': permission_types
            }
        )
        
        revoked_counts = {
            'datasource': 0,
            'table': 0,
            'column': 0
        }
        
        with transaction.atomic():
            if 'datasource' in permission_types:
                # Revoke all datasource permissions
                datasource_perms = DataSourcePermission.objects.filter(user=user)
                for perm in datasource_perms:
                    count, _ = PermissionService.revoke_datasource_permission(
                        user=user,
                        datasource_id=perm.datasource_id,
                        access_type=perm.access_type,
                        revoked_by=revoked_by
                    )
                    revoked_counts['datasource'] += count
            
            if 'column' in permission_types:
                # Revoke all column permissions
                column_perms = ColumnPermission.objects.filter(user=user)
                for perm in column_perms:
                    count, _ = PermissionService.revoke_column_permission(
                        user=user,
                        datasource_id=perm.datasource_id,
                        table_name=perm.table_name,
                        column_name=perm.column_name,
                        access_type=perm.access_type,
                        revoked_by=revoked_by
                    )
                    revoked_counts['column'] += count
        
        # Clear all user caches
        cache_manager.invalidate_user_cache(str(user.id))
        
        # Log bulk revocation
        if revoked_by:
            permission_logger.bulk_operation(
                operation_type='revoke_all_user_permissions',
                user=revoked_by,
                affected_count=sum(revoked_counts.values()),
                operation_details={
                    'target_user_id': user_id,
                    'target_user_email': user.email,
                    'permission_types': permission_types,
                    'revoked_counts': revoked_counts,
                    'task_id': self.request.id
                }
            )
        
        result = {
            'status': 'success',
            'revoked_counts': revoked_counts,
            'total_revoked': sum(revoked_counts.values()),
            'user_email': user.email,
            'task_id': self.request.id
        }
        
        logger.info(
            f"Completed permission revocation for user {user.email}: {sum(revoked_counts.values())} permissions",
            extra=result
        )
        
        return result
        
    except Exception as exc:
        logger.error(
            f"Permission revocation failed for user {user_id}: {exc}",
            extra={
                'task_id': self.request.id,
                'user_id': user_id,
                'error': str(exc)
            },
            exc_info=True
        )
        raise exc


@shared_task(bind=True)
def copy_user_permissions(
    self,
    source_user_id: str,
    target_user_ids: List[str],
    permission_types: List[str] = None,
    copied_by_id: str = None,
    correlation_id: str = None
):
    """
    Celery task for copying permissions from one user to multiple users
    
    Args:
        source_user_id: ID of user to copy permissions from
        target_user_ids: List of user IDs to copy permissions to
        permission_types: Types of permissions to copy
        copied_by_id: ID of user performing the copy
        correlation_id: Correlation ID for tracking
    """
    if correlation_id:
        set_correlation_id(correlation_id)
    else:
        set_correlation_id(generate_correlation_id())
    
    if permission_types is None:
        permission_types = ['datasource', 'table', 'column']
    
    try:
        source_user = User.objects.get(id=source_user_id)
        target_users = User.objects.filter(id__in=target_user_ids)
        copied_by = User.objects.get(id=copied_by_id) if copied_by_id else None
        
        logger.info(
            f"Starting permission copy from {source_user.email} to {len(target_users)} users",
            extra={
                'task_id': self.request.id,
                'source_user_id': source_user_id,
                'target_user_count': len(target_users),
                'permission_types': permission_types
            }
        )
        
        results = []
        total_copied = 0
        
        for target_user in target_users:
            copied_counts = PermissionService.copy_user_permissions(
                source_user=source_user,
                target_user=target_user,
                permission_types=permission_types,
                copied_by=copied_by
            )
            
            user_total = sum(copied_counts.values())
            total_copied += user_total
            
            results.append({
                'target_user_id': str(target_user.id),
                'target_user_email': target_user.email,
                'copied_counts': copied_counts,
                'total_copied': user_total
            })
            
            # Warm cache for target user
            cache_manager.warm_user_cache(str(target_user.id))
        
        # Log bulk copy operation
        if copied_by:
            permission_logger.bulk_operation(
                operation_type='copy_user_permissions',
                user=copied_by,
                affected_count=total_copied,
                operation_details={
                    'source_user_id': source_user_id,
                    'source_user_email': source_user.email,
                    'target_user_count': len(target_users),
                    'permission_types': permission_types,
                    'task_id': self.request.id
                }
            )
        
        result = {
            'status': 'success',
            'source_user_email': source_user.email,
            'target_results': results,
            'total_permissions_copied': total_copied,
            'task_id': self.request.id
        }
        
        logger.info(
            f"Completed permission copy: {total_copied} permissions copied to {len(target_users)} users",
            extra=result
        )
        
        return result
        
    except Exception as exc:
        logger.error(
            f"Permission copy failed: {exc}",
            extra={
                'task_id': self.request.id,
                'source_user_id': source_user_id,
                'target_user_ids': target_user_ids,
                'error': str(exc)
            },
            exc_info=True
        )
        raise exc


@shared_task
def warm_user_caches(user_ids: List[str], datasources: List[str] = None):
    """
    Celery task to warm caches for multiple users
    
    Args:
        user_ids: List of user IDs to warm caches for
        datasources: Optional list of datasource IDs to focus on
    """
    logger.info(f"Starting cache warming for {len(user_ids)} users")
    
    warmed_count = 0
    for user_id in user_ids:
        try:
            cache_manager.warm_user_cache(user_id, datasources)
            warmed_count += 1
        except Exception as e:
            logger.warning(f"Failed to warm cache for user {user_id}: {e}")
    
    logger.info(f"Cache warming completed: {warmed_count}/{len(user_ids)} users")
    
    return {
        'status': 'success',
        'users_requested': len(user_ids),
        'users_warmed': warmed_count,
        'datasources': datasources
    }


@shared_task
def cleanup_expired_permissions():
    """
    Celery task to clean up expired permissions
    """
    from datetime import timezone as tz
    from datetime import datetime
    
    logger.info("Starting expired permissions cleanup")
    
    now = datetime.now(tz.utc)
    
    # Clean expired datasource permissions
    expired_ds = DataSourcePermission.objects.filter(expires_at__lt=now)
    ds_count = expired_ds.count()
    expired_ds.delete()
    
    # Clean expired column permissions
    expired_col = ColumnPermission.objects.filter(expires_at__lt=now)
    col_count = expired_col.count()
    expired_col.delete()
    
    # Clear permission cache to ensure consistency
    cleared_keys = cache_manager.clear_all_permission_cache()
    
    result = {
        'status': 'success',
        'expired_datasource_permissions': ds_count,
        'expired_column_permissions': col_count,
        'total_expired': ds_count + col_count,
        'cache_keys_cleared': cleared_keys
    }
    
    logger.info(
        f"Expired permissions cleanup completed: {ds_count + col_count} permissions removed",
        extra=result
    )
    
    return result