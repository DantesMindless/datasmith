"""
Permission Management API Views

REST API endpoints for managing user roles, permissions, and access control
across the DataSmith platform.
"""

from django.contrib.auth import get_user_model
from django.db.models import Count, Q
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView

from .models import (
    UserRole, AccessType, UserGroup, DataSourcePermission,
    TablePermission, ColumnPermission, PermissionAuditLog
)
from django.utils import timezone
from datetime import timedelta
from .serializers import (
    UserSerializer, UserGroupSerializer, DataSourcePermissionSerializer,
    TablePermissionSerializer, ColumnPermissionSerializer, PermissionAuditLogSerializer,
    BulkPermissionSerializer, PermissionSummarySerializer
)
from .permissions import (
    require_role, require_permission, 
    DataSourcePermissionMixin
)
from .services import PermissionService
from .rate_limiting import (
    permission_rate_limit, bulk_operation_rate_limit, 
    audit_rate_limit, PermissionRateLimitMixin
)
from .logging_utils import with_correlation_id, permission_logger
from datasource.models import DataSource

User = get_user_model()


class UserManagementViewSet(viewsets.ModelViewSet):
    """ViewSet for managing users with role-based access"""
    
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filter users based on requester's role"""
        user = self.request.user
        
        if user.has_role(UserRole.SUPER_ADMIN):
            return User.objects.all()
        elif user.has_role(UserRole.DATABASE_ADMIN):
            # Database admins can see users in their department
            return User.objects.filter(
                Q(department=user.department) | Q(role__in=[UserRole.ANALYST, UserRole.VIEWER])
            )
        else:
            # Regular users can only see themselves
            return User.objects.filter(id=user.id)
    
    @require_role(UserRole.DATABASE_ADMIN)
    @with_correlation_id
    @action(detail=True, methods=['post'])
    def assign_role(self, request, pk=None):
        """Assign role to a user"""
        user = self.get_object()
        new_role = request.data.get('role')
        
        if new_role not in [choice[0] for choice in UserRole.choices]:
            return Response(
                {'error': 'Invalid role'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Prevent non-super-admins from creating super-admins
        if new_role == UserRole.SUPER_ADMIN and not request.user.has_role(UserRole.SUPER_ADMIN):
            return Response(
                {'error': 'Insufficient permissions'}, 
                status=status.HTTP_403_FORBIDDEN
            )
        
        old_role = user.role
        user.role = new_role
        user.save()
        
        # Log role change with structured logging
        permission_logger.permission_granted(
            user=user,
            permission_type='role',
            resource_id='role_change',
            access_type=new_role,
            granted_by=request.user,
            request=request,
            old_role=old_role,
            new_role=new_role
        )
        
        # Log to audit table
        PermissionAuditLog.objects.create(
            user=user,
            action='modified',
            permission_type='role',
            resource_identifier=f"role_change",
            details={'old_role': old_role, 'new_role': new_role},
            performed_by=request.user
        )
        
        return Response({'message': f'Role updated to {new_role}'})
    
    @action(detail=True, methods=['get'])
    def permissions_summary(self, request, pk=None):
        """Get comprehensive permissions summary for a user"""
        user = self.get_object()
        
        # Check if requester can view this user's permissions
        if not (request.user.has_role(UserRole.DATABASE_ADMIN) or request.user == user):
            return Response(
                {'error': 'Insufficient permissions'}, 
                status=status.HTTP_403_FORBIDDEN
            )
        
        permissions = PermissionService.get_user_permission_summary(user)['permissions']
        
        summary = {
            'user': UserSerializer(user).data,
            'role': user.role,
            'datasource_count': len(permissions['datasource_permissions']),
            'table_count': len(permissions['table_permissions']),
            'column_count': len(permissions['column_permissions']),
            'segment_accessible_columns': len([
                p for p in permissions['column_permissions'] 
                if p.get('allow_in_segments', False)
            ]),
            'last_permission_granted': None,
            'permissions': permissions
        }
        
        # Get last permission granted date
        latest_perm = PermissionAuditLog.objects.filter(
            user=user, action='granted'
        ).order_by('-timestamp').first()
        
        if latest_perm:
            summary['last_permission_granted'] = latest_perm.timestamp
        
        return Response(summary)


class UserGroupViewSet(viewsets.ModelViewSet):
    """ViewSet for managing user groups"""
    
    queryset = UserGroup.objects.all()
    serializer_class = UserGroupSerializer
    permission_classes = [IsAuthenticated]
    
    @require_role(UserRole.DATA_MANAGER)
    def create(self, request, *args, **kwargs):
        """Create a new user group"""
        request.data['created_by'] = request.user.id
        return super().create(request, *args, **kwargs)
    
    @require_role(UserRole.DATA_MANAGER)
    @action(detail=True, methods=['post'])
    def add_users(self, request, pk=None):
        """Add users to a group"""
        group = self.get_object()
        user_ids = request.data.get('user_ids', [])
        
        users = User.objects.filter(id__in=user_ids)
        group.users.add(*users)
        
        return Response({
            'message': f'Added {len(users)} users to group {group.name}'
        })
    
    @require_role(UserRole.DATA_MANAGER)
    @action(detail=True, methods=['post'])
    def remove_users(self, request, pk=None):
        """Remove users from a group"""
        group = self.get_object()
        user_ids = request.data.get('user_ids', [])
        
        users = User.objects.filter(id__in=user_ids)
        group.users.remove(*users)
        
        return Response({
            'message': f'Removed {len(users)} users from group {group.name}'
        })


class DataSourcePermissionViewSet(viewsets.ModelViewSet):
    """ViewSet for managing datasource permissions"""
    
    queryset = DataSourcePermission.objects.all()
    serializer_class = DataSourcePermissionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filter permissions based on user role"""
        user = self.request.user
        
        if user.has_role(UserRole.DATABASE_ADMIN):
            return DataSourcePermission.objects.all()
        else:
            return DataSourcePermission.objects.filter(user=user)
    
    @require_role(UserRole.DATABASE_ADMIN)
    @permission_rate_limit(max_requests=20, window_seconds=60)
    def create(self, request, *args, **kwargs):
        """Grant datasource permission"""
        request.data['granted_by'] = request.user.id
        
        # Validate datasource exists
        datasource_id = request.data.get('datasource_id')
        if not DataSource.objects.filter(id=datasource_id, deleted=False).exists():
            return Response(
                {'error': 'DataSource not found'}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        return super().create(request, *args, **kwargs)


class ColumnPermissionViewSet(viewsets.ModelViewSet):
    """ViewSet for managing column-level permissions - KEY FEATURE"""
    
    queryset = ColumnPermission.objects.all()
    serializer_class = ColumnPermissionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filter column permissions based on user role"""
        user = self.request.user
        
        if user.has_role(UserRole.DATABASE_ADMIN):
            return ColumnPermission.objects.all()
        else:
            return ColumnPermission.objects.filter(user=user)
    
    @require_role(UserRole.DATABASE_ADMIN)
    @permission_rate_limit(max_requests=20, window_seconds=60)
    def create(self, request, *args, **kwargs):
        """Grant column permission"""
        request.data['granted_by'] = request.user.id
        return super().create(request, *args, **kwargs)
    
    @require_role(UserRole.DATABASE_ADMIN)
    @bulk_operation_rate_limit(max_requests=5, window_seconds=300)
    @action(detail=False, methods=['post'])
    def bulk_grant(self, request):
        """Grant permissions to multiple columns for multiple users"""
        serializer = BulkPermissionSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        user_ids = data['user_ids']
        datasource_id = data['datasource_id']
        column_names = data.get('column_names', [])
        table_name = data.get('table_name')
        access_type = data['access_type']
        allow_in_segments = data['allow_in_segments']
        
        # Get valid users
        users_list = User.objects.filter(id__in=user_ids)
        
        # Use PermissionService bulk operation
        total_created, created_permissions = PermissionService.bulk_grant_permissions(
            users=list(users_list),
            datasource_id=datasource_id,
            table_name=table_name,
            column_names=column_names,
            access_type=access_type,
            allow_in_segments=allow_in_segments,
            granted_by=request.user
        )
        
        return Response({
            'message': f'Granted {total_created} column permissions',
            'granted_permissions': created_permissions
        })
    
    @action(detail=False, methods=['get'])
    def by_datasource(self, request):
        """Get column permissions by datasource"""
        datasource_id = request.query_params.get('datasource_id')
        if not datasource_id:
            return Response(
                {'error': 'datasource_id parameter required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        permissions = self.get_queryset().filter(datasource_id=datasource_id)
        serializer = self.get_serializer(permissions, many=True)
        
        # Group by table and column for easier frontend consumption
        grouped = {}
        for perm in serializer.data:
            table_name = perm['table_name']
            column_name = perm['column_name']
            
            if table_name not in grouped:
                grouped[table_name] = {}
            if column_name not in grouped[table_name]:
                grouped[table_name][column_name] = []
            
            grouped[table_name][column_name].append(perm)
        
        return Response({
            'datasource_id': datasource_id,
            'permissions_by_table': grouped,
            'total_permissions': len(serializer.data)
        })


class PermissionAuditViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for viewing permission audit logs"""
    
    queryset = PermissionAuditLog.objects.all()
    serializer_class = PermissionAuditLogSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filter audit logs based on user role"""
        user = self.request.user
        
        if user.has_role(UserRole.DATABASE_ADMIN):
            return PermissionAuditLog.objects.all().order_by('-timestamp')
        else:
            return PermissionAuditLog.objects.filter(
                user=user
            ).order_by('-timestamp')
    
    @audit_rate_limit(max_requests=30, window_seconds=60)
    @action(detail=False, methods=['get'])
    def recent_activity(self, request):
        """Get recent permission activity"""
        limit = int(request.query_params.get('limit', 50))
        logs = self.get_queryset()[:limit]
        serializer = self.get_serializer(logs, many=True)
        
        return Response({
            'recent_activity': serializer.data,
            'total_logs': self.get_queryset().count()
        })
    
    @audit_rate_limit(max_requests=20, window_seconds=60)
    @action(detail=False, methods=['get'])
    def user_activity(self, request):
        """Get permission activity for a specific user"""
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response(
                {'error': 'user_id parameter required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Check if requester can view this user's activity
        if not request.user.has_role(UserRole.DATABASE_ADMIN):
            if str(request.user.id) != user_id:
                return Response(
                    {'error': 'Insufficient permissions'}, 
                    status=status.HTTP_403_FORBIDDEN
                )
        
        logs = PermissionAuditLog.objects.filter(
            user_id=user_id
        ).order_by('-timestamp')[:100]
        
        serializer = self.get_serializer(logs, many=True)
        
        return Response({
            'user_id': user_id,
            'activity': serializer.data,
            'total_actions': logs.count()
        })


class PermissionManagementAPI(APIView):
    """Utility API for permission management operations"""
    
    permission_classes = [IsAuthenticated]
    
    @require_role(UserRole.DATABASE_ADMIN)
    @bulk_operation_rate_limit(max_requests=3, window_seconds=300)
    def post(self, request):
        """Perform bulk permission operations"""
        operation = request.data.get('operation')
        
        if operation == 'revoke_user_permissions':
            user_id = request.data.get('user_id')
            permission_type = request.data.get('permission_type', 'all')
            
            try:
                user = User.objects.get(id=user_id)
                
                revoked_count = 0
                
                # Use PermissionService for proper transaction management
                if permission_type in ['all', 'datasource']:
                    # Get all datasource permissions for this user
                    datasource_perms = DataSourcePermission.objects.filter(user=user)
                    for perm in datasource_perms:
                        count, _ = PermissionService.revoke_datasource_permission(
                            user=user,
                            datasource_id=perm.datasource_id,
                            access_type=perm.access_type,
                            revoked_by=request.user
                        )
                        revoked_count += count
                
                if permission_type in ['all', 'table']:
                    revoked_count += TablePermission.objects.filter(user=user).delete()[0]
                
                if permission_type in ['all', 'column']:
                    # Get all column permissions for this user
                    column_perms = ColumnPermission.objects.filter(user=user)
                    for perm in column_perms:
                        count, _ = PermissionService.revoke_column_permission(
                            user=user,
                            datasource_id=perm.datasource_id,
                            table_name=perm.table_name,
                            column_name=perm.column_name,
                            access_type=perm.access_type,
                            revoked_by=request.user
                        )
                        revoked_count += count
                
                # Log bulk revocation
                PermissionAuditLog.objects.create(
                    user=user,
                    action='revoked',
                    permission_type='bulk',
                    resource_identifier='all_permissions',
                    details={'revoked_count': revoked_count, 'permission_type': permission_type},
                    performed_by=request.user
                )
                
                return Response({
                    'message': f'Revoked {revoked_count} permissions for user {user.email}'
                })
                
            except User.DoesNotExist:
                return Response(
                    {'error': 'User not found'}, 
                    status=status.HTTP_404_NOT_FOUND
                )
        
        elif operation == 'copy_permissions':
            source_user_id = request.data.get('source_user_id')
            target_user_id = request.data.get('target_user_id')
            
            try:
                source_user = User.objects.get(id=source_user_id)
                target_user = User.objects.get(id=target_user_id)
                
                # Use PermissionService for proper permission copying
                copied_counts = PermissionService.copy_user_permissions(
                    source_user=source_user,
                    target_user=target_user,
                    permission_types=['datasource', 'table', 'column'],
                    copied_by=request.user
                )
                
                copied_count = sum(copied_counts.values())
                
                return Response({
                    'message': f'Copied {copied_count} permissions from {source_user.email} to {target_user.email}',
                    'details': copied_counts
                })
                
            except User.DoesNotExist:
                return Response(
                    {'error': 'User not found'}, 
                    status=status.HTTP_404_NOT_FOUND
                )
        
        else:
            return Response(
                {'error': 'Invalid operation'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
    
    def get(self, request):
        """Get permission management statistics"""
        stats = {
            'total_users': User.objects.count(),
            'users_by_role': {
                role[0]: User.objects.filter(role=role[0]).count()
                for role in UserRole.choices
            },
            'total_permissions': {
                'datasource': DataSourcePermission.objects.count(),
                'table': TablePermission.objects.count(),
                'column': ColumnPermission.objects.count(),
            },
            'recent_activity_count': PermissionAuditLog.objects.filter(
                timestamp__gte=timezone.now() - timedelta(days=7)
            ).count() if 'timezone' in globals() else 0
        }
        
        return Response(stats)