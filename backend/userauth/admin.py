from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import (
    CustomUser, UserGroup, DataSourcePermission, TablePermission,
    ColumnPermission, PermissionAuditLog
)


@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    """Enhanced user admin with role management"""
    
    fieldsets = UserAdmin.fieldsets + (
        ('DataSmith Profile', {
            'fields': ('role', 'department', 'is_active_user', 'last_login_ip')
        }),
    )
    
    list_display = ('email', 'username', 'role', 'department', 'is_active_user', 'date_joined')
    list_filter = ('role', 'department', 'is_active_user', 'date_joined')
    search_fields = ('email', 'username', 'department')
    ordering = ('-date_joined',)


@admin.register(UserGroup)
class UserGroupAdmin(admin.ModelAdmin):
    """Admin for user groups"""
    
    list_display = ('name', 'description', 'user_count', 'created_by', 'created_at')
    list_filter = ('created_at', 'created_by')
    search_fields = ('name', 'description')
    filter_horizontal = ('users',)
    
    def user_count(self, obj):
        return obj.users.count()
    user_count.short_description = 'User Count'


@admin.register(DataSourcePermission)
class DataSourcePermissionAdmin(admin.ModelAdmin):
    """Admin for datasource permissions"""
    
    list_display = ('user', 'datasource_id', 'access_type', 'granted_by', 'granted_at', 'expires_at')
    list_filter = ('access_type', 'granted_at', 'expires_at', 'granted_by')
    search_fields = ('user__email', 'user__username', 'datasource_id')
    date_hierarchy = 'granted_at'
    
    def has_change_permission(self, request, obj=None):
        # Only allow changes by admins
        return request.user.is_superuser or request.user.role in ['super_admin', 'database_admin']


@admin.register(TablePermission)
class TablePermissionAdmin(admin.ModelAdmin):
    """Admin for table permissions"""
    
    list_display = ('user', 'datasource_id', 'table_name', 'access_type', 'granted_by', 'granted_at')
    list_filter = ('access_type', 'granted_at', 'schema_name')
    search_fields = ('user__email', 'table_name', 'schema_name')
    date_hierarchy = 'granted_at'


@admin.register(ColumnPermission)  
class ColumnPermissionAdmin(admin.ModelAdmin):
    """Admin for column permissions - KEY FEATURE"""
    
    list_display = (
        'user', 'datasource_id', 'table_name', 'column_name', 
        'access_type', 'allow_in_segments', 'granted_by', 'granted_at'
    )
    list_filter = (
        'access_type', 'allow_in_segments', 'granted_at', 
        'schema_name', 'table_name'
    )
    search_fields = ('user__email', 'table_name', 'column_name')
    date_hierarchy = 'granted_at'
    
    fieldsets = (
        ('Permission Details', {
            'fields': ('user', 'datasource_id', 'schema_name', 'table_name', 'column_name')
        }),
        ('Access Configuration', {
            'fields': ('access_type', 'allow_in_segments', 'expires_at')
        }),
        ('Audit Information', {
            'fields': ('granted_by',),
            'classes': ('collapse',)
        })
    )
    
    def get_readonly_fields(self, request, obj=None):
        if obj:  # Editing existing permission
            return ('granted_by', 'granted_at')
        return ('granted_by',)
    
    def save_model(self, request, obj, form, change):
        if not change:  # Creating new permission
            obj.granted_by = request.user
        super().save_model(request, obj, form, change)


@admin.register(PermissionAuditLog)
class PermissionAuditLogAdmin(admin.ModelAdmin):
    """Admin for permission audit logs"""
    
    list_display = (
        'user', 'action', 'permission_type', 'resource_identifier',
        'performed_by', 'timestamp'
    )
    list_filter = ('action', 'permission_type', 'timestamp', 'performed_by')
    search_fields = ('user__email', 'resource_identifier', 'details')
    date_hierarchy = 'timestamp'
    readonly_fields = ('timestamp',)
    
    def has_add_permission(self, request):
        return False  # Audit logs are created automatically
    
    def has_change_permission(self, request, obj=None):
        return False  # Audit logs should not be modified
    
    def has_delete_permission(self, request, obj=None):
        # Only allow deletion by superusers (for cleanup)
        return request.user.is_superuser


# Customize admin site
admin.site.site_header = "DataSmith Administration"
admin.site.site_title = "DataSmith Admin"
admin.site.index_title = "Welcome to DataSmith Administration"
