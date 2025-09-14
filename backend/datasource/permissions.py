from rest_framework.permissions import BasePermission, SAFE_METHODS
from rest_framework.request import Request
from rest_framework.views import APIView
from typing import Any, Optional
from .models import DataSource
from userauth.permissions import PermissionManager
from userauth.models import AccessType


class IsOwnerOrReadOnly(BasePermission):
    """
    Custom permission to allow only owners of a datasource to edit it.
    Allows read-only access for other authenticated users with proper permissions.
    """
    
    def has_permission(self, request: Request, view: APIView) -> bool:
        """
        Check if user has general permission to access datasource endpoints.
        
        Args:
            request: The HTTP request
            view: The view being accessed
            
        Returns:
            Boolean indicating if user has permission
        """
        # Allow read-only access for any authenticated user
        if request.method in SAFE_METHODS:
            return request.user and request.user.is_authenticated
        # Allow write access only if authenticated
        return request.user and request.user.is_authenticated

    def has_object_permission(self, request: Request, view: APIView, obj: DataSource) -> bool:
        """
        Check if user has permission to access a specific datasource object.
        
        Args:
            request: The HTTP request
            view: The view being accessed
            obj: The datasource object being accessed
            
        Returns:
            Boolean indicating if user has object-level permission
        """
        # Allow read-only access for any authenticated user with read permissions
        if request.method in SAFE_METHODS:
            return PermissionManager.can_user_access_resource(
                request.user, 'datasource', str(obj.id), AccessType.READ
            )
        # Allow write access only to the owner of the datasource
        return obj.user_id == request.user.id or obj.created_by == request.user.id


class CanAccessDatasource(BasePermission):
    """
    Permission class for datasource access based on the custom permission system.
    """
    
    def has_permission(self, request: Request, view: APIView) -> bool:
        """
        Check if user has general permission to access datasource endpoints.
        
        Args:
            request: The HTTP request
            view: The view being accessed
            
        Returns:
            Boolean indicating if user has permission
        """
        return request.user and request.user.is_authenticated

    def has_object_permission(self, request: Request, view: APIView, obj: DataSource) -> bool:
        """
        Check if user has permission to access a specific datasource object.
        
        Args:
            request: The HTTP request  
            view: The view being accessed
            obj: The datasource object being accessed
            
        Returns:
            Boolean indicating if user has object-level permission
        """
        # Determine required access type based on request method
        if request.method in SAFE_METHODS:
            access_type = AccessType.READ
        elif request.method in ['PUT', 'PATCH']:
            access_type = AccessType.WRITE  
        elif request.method == 'DELETE':
            access_type = AccessType.DELETE
        else:
            access_type = AccessType.WRITE
            
        return PermissionManager.can_user_access_resource(
            request.user, 'datasource', str(obj.id), access_type
        )