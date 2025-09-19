from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from .permission_views import (
    UserManagementViewSet, UserGroupViewSet, DataSourcePermissionViewSet,
    ColumnPermissionViewSet, PermissionAuditViewSet, PermissionManagementAPI
)

# Create router for permission viewsets
router = DefaultRouter()
router.register(r'users', UserManagementViewSet)
router.register(r'groups', UserGroupViewSet)
router.register(r'datasource-permissions', DataSourcePermissionViewSet)
router.register(r'column-permissions', ColumnPermissionViewSet)
router.register(r'audit-logs', PermissionAuditViewSet)

urlpatterns = [
    # Authentication endpoints
    path("login/", views.login),
    path("logout/", views.logout),
    path("refresh/", views.refresh_token),
    path("signup/", views.signup),
    path("test_token/", views.test_token),
    
    # Permission management endpoints
    path("permissions/", include(router.urls)),
    path("permissions/manage/", PermissionManagementAPI.as_view(), name="permission-management"),
]
