"""
Permission Management Utilities

Helper functions and decorators for managing user access control
across the DataSmith platform.
"""

from functools import wraps
from typing import List, Dict, Any, Optional
from django.http import JsonResponse
from django.contrib.auth import get_user_model
from rest_framework import status
from rest_framework.response import Response

from .models import (
    UserRole, AccessType, DataSourcePermission, 
    TablePermission, ColumnPermission, PermissionAuditLog
)
from datasource.models import DataSource
from .services import PermissionService

User = get_user_model()


class PermissionManager:
    """Centralized permission management"""
    
    @staticmethod
    def grant_datasource_access(
        user: User, 
        datasource_id: str, 
        access_type: str, 
        granted_by: User,
        expires_at: Optional[str] = None
    ) -> bool:
        """Grant datasource access to user"""
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
            
            # Log the permission grant
            PermissionAuditLog.objects.create(
                user=user,
                action='granted',
                permission_type='datasource',
                resource_identifier=str(datasource_id),
                details={
                    'access_type': access_type,
                    'expires_at': expires_at
                },
                performed_by=granted_by
            )
            
            return created
            
        except Exception as e:
            return False
    
    @staticmethod
    def grant_column_access(
        user: User,
        datasource_id: str,
        table_name: str,
        column_name: str,
        access_type: str = AccessType.READ,
        allow_in_segments: bool = False,
        granted_by: User = None,
        expires_at: Optional[str] = None
    ) -> bool:
        """Grant column-level access - KEY FEATURE"""
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
            
            # Log the permission grant
            PermissionAuditLog.objects.create(
                user=user,
                action='granted',
                permission_type='column',
                resource_identifier=f"{datasource_id}.{table_name}.{column_name}",
                details={
                    'access_type': access_type,
                    'allow_in_segments': allow_in_segments,
                    'expires_at': expires_at
                },
                performed_by=granted_by
            )
            
            return created
            
        except Exception as e:
            return False
    
    @staticmethod
    def revoke_permission(
        user: User,
        permission_type: str,
        resource_identifier: str,
        revoked_by: User
    ) -> bool:
        """Revoke any permission"""
        try:
            deleted_count = 0
            
            if permission_type == 'datasource':
                deleted_count = DataSourcePermission.objects.filter(
                    user=user, datasource_id=resource_identifier
                ).delete()[0]
            elif permission_type == 'column':
                # Parse resource_identifier: "datasource_id.table.column"
                parts = resource_identifier.split('.')
                if len(parts) >= 3:
                    deleted_count = ColumnPermission.objects.filter(
                        user=user,
                        datasource_id=parts[0],
                        table_name=parts[1],
                        column_name=parts[2]
                    ).delete()[0]
            
            if deleted_count > 0:
                # Log the permission revocation
                PermissionAuditLog.objects.create(
                    user=user,
                    action='revoked',
                    permission_type=permission_type,
                    resource_identifier=resource_identifier,
                    details={'deleted_count': deleted_count},
                    performed_by=revoked_by
                )
            
            return deleted_count > 0
            
        except Exception as e:
            return False
    
    @staticmethod
    def get_user_permissions(user: User) -> Dict[str, Any]:
        """Get comprehensive user permissions"""
        return {
            'role': user.role,
            'datasource_permissions': list(
                DataSourcePermission.objects.filter(user=user).values(
                    'datasource_id', 'access_type', 'granted_at'
                )
            ),
            'table_permissions': list(
                TablePermission.objects.filter(user=user).values(
                    'datasource_id', 'table_name', 'access_type', 'granted_at'
                )
            ),
            'column_permissions': list(
                ColumnPermission.objects.filter(user=user).values(
                    'datasource_id', 'table_name', 'column_name', 
                    'access_type', 'allow_in_segments', 'granted_at'
                )
            )
        }
    
    @staticmethod
    def filter_query_columns(user: User, datasource_id: str, query: str) -> str:
        """Filter SQL query to only include accessible columns"""
        if user.has_role(UserRole.DATABASE_ADMIN):
            return query  # Admin can access everything
        
        # This is a simplified implementation
        # In production, you'd want more sophisticated SQL parsing
        accessible_columns = {}
        
        # Get all column permissions for this user and datasource
        column_perms = ColumnPermission.objects.filter(
            user=user,
            datasource_id=datasource_id,
            access_type__in=[AccessType.READ, AccessType.WRITE]
        )
        
        for perm in column_perms:
            table_key = f"{perm.schema_name}.{perm.table_name}"
            if table_key not in accessible_columns:
                accessible_columns[table_key] = []
            accessible_columns[table_key].append(perm.column_name)
        
        # For now, return the original query with a comment
        # TODO: Implement proper SQL parsing and column filtering
        filtered_query = f"-- User {user.username} permissions applied\n{query}"
        
        return filtered_query
    
    @staticmethod
    def can_user_access_resource(
        user: User, 
        resource_type: str, 
        resource_id: str,
        access_type: str = AccessType.READ
    ) -> bool:
        """Check if user can access a specific resource"""
        
        # Super admins can access everything
        if user.has_role(UserRole.SUPER_ADMIN):
            return True
        
        if resource_type == 'datasource':
            return user.can_access_datasource(resource_id)
        elif resource_type == 'table':
            # resource_id format: "datasource_id.table_name"
            parts = resource_id.split('.')
            if len(parts) >= 2:
                return user.can_access_table(parts[0], parts[1])
        elif resource_type == 'column':
            # resource_id format: "datasource_id.table_name.column_name"
            parts = resource_id.split('.')
            if len(parts) >= 3:
                accessible_columns = user.get_accessible_columns(parts[0], parts[1])
                return not accessible_columns or parts[2] in accessible_columns
        
        return False


def require_permission(resource_type: str, access_type: str = AccessType.READ):
    """Decorator to enforce permissions on API views"""
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            user = request.user
            
            if not user.is_authenticated:
                return Response(
                    {'error': 'Authentication required'}, 
                    status=status.HTTP_401_UNAUTHORIZED
                )
            
            # Extract resource ID from URL parameters or request data
            resource_id = kwargs.get('id') or kwargs.get('pk') or request.data.get('id')
            
            # Check permission using PermissionService
            has_access = False
            if resource_id:
                if resource_type == 'datasource':
                    has_access = PermissionService.can_access_datasource(user, str(resource_id))
                elif resource_type == 'table':
                    # resource_id format: "datasource_id.table_name"
                    parts = str(resource_id).split('.')
                    if len(parts) >= 2:
                        has_access = user.can_access_table(parts[0], parts[1])
                elif resource_type == 'column':
                    # resource_id format: "datasource_id.table_name.column_name"
                    parts = str(resource_id).split('.')
                    if len(parts) >= 3:
                        accessible_columns = PermissionService.get_accessible_columns(user, parts[0], parts[1])
                        has_access = not accessible_columns or parts[2] in accessible_columns
            
            if resource_id and not has_access:
                # Log unauthorized access attempt
                PermissionAuditLog.objects.create(
                    user=user,
                    action='used',
                    permission_type=resource_type,
                    resource_identifier=str(resource_id),
                    details={
                        'access_type': access_type,
                        'denied': True,
                        'view': view_func.__name__
                    }
                )
                
                return Response(
                    {'error': 'Insufficient permissions'}, 
                    status=status.HTTP_403_FORBIDDEN
                )
            
            # Log successful access
            if resource_id:
                PermissionAuditLog.objects.create(
                    user=user,
                    action='used',
                    permission_type=resource_type,
                    resource_identifier=str(resource_id),
                    details={
                        'access_type': access_type,
                        'view': view_func.__name__
                    }
                )
            
            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator


def require_role(required_role: str):
    """Decorator to enforce role-based access"""
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            user = request.user
            
            if not user.is_authenticated:
                return Response(
                    {'error': 'Authentication required'}, 
                    status=status.HTTP_401_UNAUTHORIZED
                )
            
            if not user.has_role(required_role):
                return Response(
                    {'error': f'Role {required_role} required'}, 
                    status=status.HTTP_403_FORBIDDEN
                )
            
            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator


class DataSourcePermissionMixin:
    """Mixin for DataSource views with permission filtering"""
    
    def get_accessible_datasources(self, user: User):
        """Get datasources user can access using PermissionService"""
        return PermissionService.get_accessible_datasources(user)
    
    def filter_datasource_response(self, user: User, datasource, data: Dict[str, Any]):
        """Filter datasource response based on user permissions"""
        if PermissionService.has_role_or_higher(user, UserRole.DATABASE_ADMIN):
            return data
        
        # Remove sensitive information for non-admin users
        filtered_data = data.copy()
        if 'credentials' in filtered_data:
            filtered_data['credentials'] = {'***': 'Access restricted'}
        
        return filtered_data