"""
Permission Services

Business logic for permission management, separated from models and views
following Django best practices.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from django.contrib.auth import get_user_model
from django.db import transaction
from django.core.cache import cache
from django.conf import settings

from .logging_utils import permission_logger, log_permission_operation
from .cache_manager import cache_manager

from .models import (
    UserRole, AccessType, DataSourcePermission, 
    TablePermission, ColumnPermission, PermissionAuditLog
)

User = get_user_model()
logger = logging.getLogger(__name__)

# Configuration constants
ROLE_HIERARCHY = {
    UserRole.SUPER_ADMIN: 5,
    UserRole.DATABASE_ADMIN: 4,
    UserRole.DATA_MANAGER: 3,
    UserRole.ANALYST: 2,
    UserRole.VIEWER: 1
}

PERMISSION_CACHE_TTL = getattr(settings, 'PERMISSION_CACHE_TTL', 300)  # 5 minutes


class PermissionService:
    """Service class for permission management operations"""
    
    @staticmethod
    def get_user_role_level(user: User) -> int:
        """Get numeric role level for user"""
        return ROLE_HIERARCHY.get(user.role, 0)
    
    @staticmethod
    def has_role_or_higher(user: User, required_role: str) -> bool:
        """Check if user has required role or higher in hierarchy"""
        user_level = PermissionService.get_user_role_level(user)
        required_level = ROLE_HIERARCHY.get(required_role, 0)
        return user_level >= required_level
    
    @staticmethod
    def get_accessible_datasources(user: User):
        """Get datasources user can access with optimized queries"""
        if PermissionService.has_role_or_higher(user, UserRole.DATABASE_ADMIN):
            from datasource.models import DataSource
            return DataSource.objects.filter(deleted=False)
        
        # Use select_related for performance
        permitted_ids = DataSourcePermission.objects.filter(
            user=user,
            access_type__in=[AccessType.READ, AccessType.WRITE, AccessType.ADMIN]
        ).select_related('user').values_list('datasource_id', flat=True)
        
        from datasource.models import DataSource
        return DataSource.objects.filter(
            id__in=permitted_ids,
            deleted=False
        )
    
    @staticmethod
    def can_access_datasource(user: User, datasource_id: str) -> bool:
        """Check datasource access with advanced caching"""
        # Check cache first
        cached_result = cache_manager.get_user_datasource_access(
            str(user.id), str(datasource_id)
        )
        
        if cached_result is not None:
            permission_logger.permission_check(
                user=user,
                permission_type='datasource',
                resource_id=str(datasource_id),
                access_type='read',
                result=cached_result,
                cache_hit=True
            )
            return cached_result
        
        # Check role first (fastest path)
        if PermissionService.has_role_or_higher(user, UserRole.DATABASE_ADMIN):
            cache_manager.set_user_datasource_access(
                str(user.id), str(datasource_id), True
            )
            permission_logger.permission_check(
                user=user,
                permission_type='datasource',
                resource_id=str(datasource_id),
                access_type='read',
                result=True,
                cache_hit=False,
                check_method='role_hierarchy'
            )
            return True
        
        # Check explicit permissions
        has_permission = DataSourcePermission.objects.filter(
            user=user,
            datasource_id=datasource_id,
            access_type__in=[AccessType.READ, AccessType.WRITE, AccessType.ADMIN]
        ).exists()
        
        # Cache the result
        cache_manager.set_user_datasource_access(
            str(user.id), str(datasource_id), has_permission
        )
        
        permission_logger.permission_check(
            user=user,
            permission_type='datasource',
            resource_id=str(datasource_id),
            access_type='read',
            result=has_permission,
            cache_hit=False,
            check_method='explicit_permission'
        )
        
        return has_permission
    
    @staticmethod
    def get_accessible_columns(user: User, datasource_id: str, table_name: str) -> List[str]:
        """Get accessible columns with advanced caching"""
        # Check cache first
        cached_result = cache_manager.get_user_columns(
            str(user.id), str(datasource_id), table_name
        )
        
        if cached_result is not None:
            return cached_result
        
        if PermissionService.has_role_or_higher(user, UserRole.DATABASE_ADMIN):
            # Empty list means all columns accessible for admin users
            cache_manager.set_user_columns(
                str(user.id), str(datasource_id), table_name, []
            )
            return []
        
        columns = list(
            ColumnPermission.objects.filter(
                user=user,
                datasource_id=datasource_id,
                table_name=table_name,
                access_type__in=[AccessType.READ, AccessType.WRITE]
            ).values_list('column_name', flat=True)
        )
        
        # Cache the result
        cache_manager.set_user_columns(
            str(user.id), str(datasource_id), table_name, columns
        )
        
        return columns
    
    @staticmethod
    @transaction.atomic
    @log_permission_operation('grant_datasource_permission')
    def grant_datasource_permission(
        user: User,
        datasource_id: str,
        access_type: str,
        granted_by: User,
        expires_at: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """Grant datasource permission with proper transaction management"""
        try:
            permission, created = DataSourcePermission.objects.get_or_create(
                user=user,
                datasource_id=datasource_id,
                access_type=access_type,
                defaults={
                    'granted_by': granted_by,
                    'expires_at': expires_at
                }
            )
            
            # Log the operation with structured logging
            permission_logger.permission_granted(
                user=user,
                permission_type='datasource',
                resource_id=str(datasource_id),
                access_type=access_type,
                granted_by=granted_by,
                expires_at=expires_at.isoformat() if expires_at else None,
                permission_created=created
            )
            
            # Log to audit table
            PermissionAuditLog.objects.create(
                user=user,
                action='granted',
                permission_type='datasource',
                resource_identifier=str(datasource_id),
                details={
                    'access_type': access_type,
                    'expires_at': expires_at.isoformat() if expires_at else None
                },
                performed_by=granted_by
            )
            
            # Invalidate cache using cache manager
            cache_manager.invalidate_user_cache(
                str(user.id), str(datasource_id)
            )
            
            return created, "Permission granted successfully" if created else "Permission already exists"
            
        except Exception as e:
            logger.error(f"Failed to grant datasource permission: {e}")
            permission_logger.security_violation(
                user=user,
                violation_type='permission_grant_failure',
                details={
                    'datasource_id': str(datasource_id),
                    'access_type': access_type,
                    'error': str(e)
                }
            )
            return False, str(e)
    
    @staticmethod
    @transaction.atomic
    def grant_column_permission(
        user: User,
        datasource_id: str,
        table_name: str,
        column_name: str,
        access_type: str = AccessType.READ,
        allow_in_segments: bool = False,
        granted_by: Optional[User] = None,
        expires_at: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """Grant column permission with transaction management"""
        try:
            permission, created = ColumnPermission.objects.get_or_create(
                user=user,
                datasource_id=datasource_id,
                table_name=table_name,
                column_name=column_name,
                access_type=access_type,
                defaults={
                    'granted_by': granted_by,
                    'allow_in_segments': allow_in_segments,
                    'expires_at': expires_at
                }
            )
            
            # Log the operation
            if granted_by:
                PermissionAuditLog.objects.create(
                    user=user,
                    action='granted',
                    permission_type='column',
                    resource_identifier=f"{datasource_id}.{table_name}.{column_name}",
                    details={
                        'access_type': access_type,
                        'allow_in_segments': allow_in_segments,
                        'expires_at': expires_at.isoformat() if expires_at else None
                    },
                    performed_by=granted_by
                )
            
            # Invalidate cache using cache manager
            cache_manager.invalidate_user_cache(
                str(user.id), str(datasource_id), table_name
            )
            
            return created, "Permission granted successfully" if created else "Permission already exists"
            
        except Exception as e:
            logger.error(f"Failed to grant column permission: {e}")
            return False, str(e)
    
    @staticmethod
    def check_permission_expiry(user: User) -> Dict[str, int]:
        """Check and clean expired permissions"""
        now = timezone.now()
        
        expired_datasource = DataSourcePermission.objects.filter(
            user=user,
            expires_at__lt=now
        ).count()
        
        expired_column = ColumnPermission.objects.filter(
            user=user,
            expires_at__lt=now
        ).count()
        
        # Delete expired permissions
        DataSourcePermission.objects.filter(user=user, expires_at__lt=now).delete()
        ColumnPermission.objects.filter(user=user, expires_at__lt=now).delete()
        
        # Clear caches for this user
        # In production, you'd want more sophisticated cache invalidation
        cache.delete_many([
            key for key in cache._cache.keys() 
            if key.startswith(f"user_datasource_access_{user.id}_") or 
               key.startswith(f"user_columns_{user.id}_")
        ])
        
        return {
            'expired_datasource': expired_datasource,
            'expired_column': expired_column
        }
    
    @staticmethod
    @transaction.atomic
    def revoke_datasource_permission(
        user: User,
        datasource_id: str,
        access_type: str = None,
        revoked_by: User = None
    ) -> Tuple[int, str]:
        """Revoke datasource permission with proper transaction management"""
        try:
            filter_kwargs = {
                'user': user,
                'datasource_id': datasource_id
            }
            
            if access_type:
                filter_kwargs['access_type'] = access_type
            
            deleted_count, _ = DataSourcePermission.objects.filter(
                **filter_kwargs
            ).delete()
            
            if deleted_count > 0 and revoked_by:
                # Log the operation
                PermissionAuditLog.objects.create(
                    user=user,
                    action='revoked',
                    permission_type='datasource',
                    resource_identifier=str(datasource_id),
                    details={
                        'access_type': access_type or 'all',
                        'deleted_count': deleted_count
                    },
                    performed_by=revoked_by
                )
                
                # Clear cache
                cache_key = f"user_datasource_access_{user.id}_{datasource_id}"
                cache.delete(cache_key)
            
            return deleted_count, f"Revoked {deleted_count} permission(s) successfully"
            
        except Exception as e:
            logger.error(f"Failed to revoke datasource permission: {e}")
            return 0, str(e)
    
    @staticmethod
    @transaction.atomic
    def revoke_column_permission(
        user: User,
        datasource_id: str,
        table_name: str,
        column_name: str,
        access_type: str = None,
        revoked_by: User = None
    ) -> Tuple[int, str]:
        """Revoke column permission with transaction management"""
        try:
            filter_kwargs = {
                'user': user,
                'datasource_id': datasource_id,
                'table_name': table_name,
                'column_name': column_name
            }
            
            if access_type:
                filter_kwargs['access_type'] = access_type
            
            deleted_count, _ = ColumnPermission.objects.filter(
                **filter_kwargs
            ).delete()
            
            if deleted_count > 0 and revoked_by:
                # Log the operation
                PermissionAuditLog.objects.create(
                    user=user,
                    action='revoked',
                    permission_type='column',
                    resource_identifier=f"{datasource_id}.{table_name}.{column_name}",
                    details={
                        'access_type': access_type or 'all',
                        'deleted_count': deleted_count
                    },
                    performed_by=revoked_by
                )
                
                # Clear cache
                cache_key = f"user_columns_{user.id}_{datasource_id}_{table_name}"
                cache.delete(cache_key)
            
            return deleted_count, f"Revoked {deleted_count} permission(s) successfully"
            
        except Exception as e:
            logger.error(f"Failed to revoke column permission: {e}")
            return 0, str(e)
    
    @staticmethod
    @transaction.atomic
    def bulk_grant_permissions(
        users: List[User],
        datasource_id: str,
        table_name: str,
        column_names: List[str],
        access_type: str = AccessType.READ,
        allow_in_segments: bool = False,
        granted_by: User = None
    ) -> Tuple[int, List[str]]:
        """Bulk grant column permissions with optimization"""
        created_permissions = []
        total_created = 0
        
        try:
            with transaction.atomic():
                for user in users:
                    for column_name in column_names:
                        success, message = PermissionService.grant_column_permission(
                            user=user,
                            datasource_id=datasource_id,
                            table_name=table_name,
                            column_name=column_name,
                            access_type=access_type,
                            allow_in_segments=allow_in_segments,
                            granted_by=granted_by
                        )
                        
                        if success:
                            created_permissions.append(f"{user.email}:{column_name}")
                            total_created += 1
                        
                        # Clear user's column cache
                        cache_key = f"user_columns_{user.id}_{datasource_id}_{table_name}"
                        cache.delete(cache_key)
            
            return total_created, created_permissions
            
        except Exception as e:
            logger.error(f"Failed to bulk grant permissions: {e}")
            return 0, []
    
    @staticmethod
    def copy_user_permissions(
        source_user: User,
        target_user: User,
        permission_types: List[str] = None,
        copied_by: User = None
    ) -> Dict[str, int]:
        """Copy permissions from one user to another"""
        if permission_types is None:
            permission_types = ['datasource', 'table', 'column']
        
        copied_counts = {
            'datasource': 0,
            'table': 0,
            'column': 0
        }
        
        try:
            with transaction.atomic():
                if 'datasource' in permission_types:
                    for perm in DataSourcePermission.objects.filter(user=source_user):
                        _, created = DataSourcePermission.objects.get_or_create(
                            user=target_user,
                            datasource_id=perm.datasource_id,
                            access_type=perm.access_type,
                            defaults={
                                'granted_by': copied_by or perm.granted_by,
                                'expires_at': perm.expires_at
                            }
                        )
                        if created:
                            copied_counts['datasource'] += 1
                
                if 'table' in permission_types:
                    for perm in TablePermission.objects.filter(user=source_user):
                        _, created = TablePermission.objects.get_or_create(
                            user=target_user,
                            datasource_id=perm.datasource_id,
                            table_name=perm.table_name,
                            access_type=perm.access_type,
                            defaults={
                                'granted_by': copied_by or perm.granted_by,
                                'expires_at': perm.expires_at
                            }
                        )
                        if created:
                            copied_counts['table'] += 1
                
                if 'column' in permission_types:
                    for perm in ColumnPermission.objects.filter(user=source_user):
                        _, created = ColumnPermission.objects.get_or_create(
                            user=target_user,
                            datasource_id=perm.datasource_id,
                            table_name=perm.table_name,
                            column_name=perm.column_name,
                            access_type=perm.access_type,
                            defaults={
                                'granted_by': copied_by or perm.granted_by,
                                'allow_in_segments': perm.allow_in_segments,
                                'expires_at': perm.expires_at
                            }
                        )
                        if created:
                            copied_counts['column'] += 1
                
                # Log the bulk copy operation
                if copied_by:
                    total_copied = sum(copied_counts.values())
                    if total_copied > 0:
                        PermissionAuditLog.objects.create(
                            user=target_user,
                            action='granted',
                            permission_type='bulk_copy',
                            resource_identifier=f"copy_from_{source_user.id}",
                            details={
                                'source_user': source_user.email,
                                'copied_counts': copied_counts,
                                'total_copied': total_copied
                            },
                            performed_by=copied_by
                        )
                
                # Clear target user's caches
                cache_keys = [
                    key for key in cache._cache.keys() 
                    if key.startswith(f"user_datasource_access_{target_user.id}_") or 
                       key.startswith(f"user_columns_{target_user.id}_")
                ]
                cache.delete_many(cache_keys)
            
            return copied_counts
            
        except Exception as e:
            logger.error(f"Failed to copy user permissions: {e}")
            return {'datasource': 0, 'table': 0, 'column': 0}
    
    @staticmethod
    def invalidate_user_cache(user: User, datasource_id: str = None, table_name: str = None):
        """Smart cache invalidation for specific user permissions"""
        try:
            if datasource_id and table_name:
                # Invalidate specific table column cache
                cache_key = f"user_columns_{user.id}_{datasource_id}_{table_name}"
                cache.delete(cache_key)
            elif datasource_id:
                # Invalidate specific datasource cache and all related column caches
                cache_key = f"user_datasource_access_{user.id}_{datasource_id}"
                cache.delete(cache_key)
                
                # Find and delete all column caches for this datasource
                pattern_keys = [
                    key for key in cache._cache.keys() 
                    if key.startswith(f"user_columns_{user.id}_{datasource_id}_")
                ]
                cache.delete_many(pattern_keys)
            else:
                # Invalidate all permission caches for user
                all_keys = [
                    key for key in cache._cache.keys() 
                    if key.startswith(f"user_datasource_access_{user.id}_") or 
                       key.startswith(f"user_columns_{user.id}_")
                ]
                cache.delete_many(all_keys)
                
        except Exception as e:
            logger.error(f"Failed to invalidate cache for user {user.id}: {e}")
    
    @staticmethod
    def get_permission_statistics() -> Dict[str, Any]:
        """Get comprehensive permission system statistics"""
        try:
            from datetime import timedelta
            from django.utils import timezone
            
            now = timezone.now()
            week_ago = now - timedelta(days=7)
            
            stats = {
                'total_users': User.objects.count(),
                'active_users': User.objects.filter(is_active_user=True).count(),
                'users_by_role': {
                    role[0]: User.objects.filter(role=role[0]).count()
                    for role in UserRole.choices
                },
                'permission_counts': {
                    'datasource_permissions': DataSourcePermission.objects.count(),
                    'table_permissions': TablePermission.objects.count(),
                    'column_permissions': ColumnPermission.objects.count(),
                    'total_permissions': (
                        DataSourcePermission.objects.count() +
                        TablePermission.objects.count() +
                        ColumnPermission.objects.count()
                    )
                },
                'recent_activity': {
                    'permissions_granted_last_week': PermissionAuditLog.objects.filter(
                        timestamp__gte=week_ago,
                        action='granted'
                    ).count(),
                    'permissions_revoked_last_week': PermissionAuditLog.objects.filter(
                        timestamp__gte=week_ago,
                        action='revoked'
                    ).count(),
                    'total_audit_logs': PermissionAuditLog.objects.count()
                },
                'expiring_permissions': {
                    'datasource_expiring_soon': DataSourcePermission.objects.filter(
                        expires_at__lt=now + timedelta(days=7),
                        expires_at__gt=now
                    ).count(),
                    'column_expiring_soon': ColumnPermission.objects.filter(
                        expires_at__lt=now + timedelta(days=7),
                        expires_at__gt=now
                    ).count()
                },
                'cache_stats': {
                    'cache_ttl': PERMISSION_CACHE_TTL,
                    'estimated_cache_keys': len([
                        key for key in cache._cache.keys() 
                        if 'user_datasource_access_' in key or 'user_columns_' in key
                    ]) if hasattr(cache, '_cache') else 0
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get permission statistics: {e}")
            return {}
    
    @staticmethod
    def get_user_permission_summary(user: User) -> Dict[str, Any]:
        """Get comprehensive permission summary with optimized queries"""
        # Check cache first
        cached_result = cache_manager.get_user_permission_summary(str(user.id))
        
        if cached_result is not None:
            return cached_result
        
        # Use select_related and prefetch_related for performance
        datasource_perms = DataSourcePermission.objects.filter(user=user).select_related('granted_by')
        table_perms = TablePermission.objects.filter(user=user).select_related('granted_by')
        column_perms = ColumnPermission.objects.filter(user=user).select_related('granted_by')
        
        result = {
            'user_id': user.id,
            'role': user.role,
            'role_level': PermissionService.get_user_role_level(user),
            'permissions': {
                'datasource_permissions': list(datasource_perms.values(
                    'datasource_id', 'access_type', 'granted_at', 'expires_at'
                )),
                'table_permissions': list(table_perms.values(
                    'datasource_id', 'table_name', 'access_type', 'granted_at'
                )),
                'column_permissions': list(column_perms.values(
                    'datasource_id', 'table_name', 'column_name', 
                    'access_type', 'allow_in_segments', 'granted_at'
                ))
            },
            'stats': {
                'total_datasource_permissions': datasource_perms.count(),
                'total_table_permissions': table_perms.count(),
                'total_column_permissions': column_perms.count(),
                'segment_accessible_columns': column_perms.filter(allow_in_segments=True).count()
            }
        }
        
        # Cache the result
        cache_manager.set_user_permission_summary(str(user.id), result)
        return result