import uuid
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.conf import settings
from core.models import BaseModel


class UserRole(models.TextChoices):
    """User role definitions with hierarchy"""
    SUPER_ADMIN = "super_admin", "Super Admin"
    DATABASE_ADMIN = "database_admin", "Database Admin"
    DATA_MANAGER = "data_manager", "Data Manager"
    ANALYST = "analyst", "Analyst"
    VIEWER = "viewer", "Viewer"


class PermissionLevel(models.TextChoices):
    """Permission granularity levels"""
    DATASOURCE = "datasource", "DataSource"
    TABLE = "table", "Table"
    COLUMN = "column", "Column"
    SEGMENT = "segment", "Segment"
    ML_MODEL = "ml_model", "ML Model"


class AccessType(models.TextChoices):
    """Types of access permissions"""
    READ = "read", "Read"
    WRITE = "write", "Write"
    DELETE = "delete", "Delete"
    EXECUTE = "execute", "Execute"
    ADMIN = "admin", "Admin"


class CustomUser(AbstractUser):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField(unique=True)
    
    # Enhanced user fields for access control
    role = models.CharField(
        max_length=20,
        choices=UserRole.choices,
        default=UserRole.VIEWER
    )
    department = models.CharField(max_length=100, blank=True)
    is_active_user = models.BooleanField(default=True)
    last_login_ip = models.GenericIPAddressField(null=True, blank=True)

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username"]

    def __str__(self):
        return self.email
    
    def has_role(self, role: str) -> bool:
        """Check if user has specific role or higher"""
        role_hierarchy = {
            UserRole.SUPER_ADMIN: 5,
            UserRole.DATABASE_ADMIN: 4,
            UserRole.DATA_MANAGER: 3,
            UserRole.ANALYST: 2,
            UserRole.VIEWER: 1
        }
        user_level = role_hierarchy.get(self.role, 0)
        required_level = role_hierarchy.get(role, 0)
        return user_level >= required_level
    
    def can_access_datasource(self, datasource_id: str) -> bool:
        """Check if user can access a datasource"""
        if self.has_role(UserRole.DATABASE_ADMIN):
            return True
        return DataSourcePermission.objects.filter(
            user=self,
            datasource_id=datasource_id,
            access_type__in=[AccessType.READ, AccessType.WRITE, AccessType.ADMIN]
        ).exists()
    
    def can_access_table(self, datasource_id: str, table_name: str) -> bool:
        """Check if user can access a specific table"""
        if self.has_role(UserRole.DATABASE_ADMIN):
            return True
        return TablePermission.objects.filter(
            user=self,
            datasource_id=datasource_id,
            table_name=table_name,
            access_type__in=[AccessType.READ, AccessType.WRITE, AccessType.ADMIN]
        ).exists()
    
    def get_accessible_columns(self, datasource_id: str, table_name: str) -> list:
        """Get list of columns user can access"""
        if self.has_role(UserRole.DATABASE_ADMIN):
            return []  # Empty list means all columns accessible
        
        column_perms = ColumnPermission.objects.filter(
            user=self,
            datasource_id=datasource_id,
            table_name=table_name,
            access_type__in=[AccessType.READ, AccessType.WRITE]
        )
        return [perm.column_name for perm in column_perms]


class UserGroup(BaseModel):
    """User groups for managing permissions"""
    
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField()
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE,
        related_name='created_groups'
    )
    users = models.ManyToManyField(
        settings.AUTH_USER_MODEL,
        related_name='datasmith_groups',
        blank=True
    )
    
    def __str__(self):
        return self.name


class DataSourcePermission(BaseModel):
    """DataSource-level permissions"""
    
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='datasource_permissions'
    )
    group = models.ForeignKey(
        'UserGroup',
        on_delete=models.CASCADE,
        null=True, blank=True,
        related_name='datasource_permissions'
    )
    datasource_id = models.UUIDField()  # References DataSource
    access_type = models.CharField(
        max_length=10,
        choices=AccessType.choices
    )
    granted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='granted_datasource_permissions'
    )
    granted_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        unique_together = ('user', 'datasource_id', 'access_type')
    
    def __str__(self):
        return f"{self.user.username} - {self.access_type} on DataSource {self.datasource_id}"


class TablePermission(BaseModel):
    """Table-level permissions"""
    
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='table_permissions'
    )
    group = models.ForeignKey(
        'UserGroup',
        on_delete=models.CASCADE,
        null=True, blank=True,
        related_name='table_permissions'
    )
    datasource_id = models.UUIDField()
    schema_name = models.CharField(max_length=100, default='public')
    table_name = models.CharField(max_length=100)
    access_type = models.CharField(
        max_length=10,
        choices=AccessType.choices
    )
    granted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='granted_table_permissions'
    )
    granted_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        unique_together = ('user', 'datasource_id', 'table_name', 'access_type')
    
    def __str__(self):
        return f"{self.user.username} - {self.access_type} on {self.schema_name}.{self.table_name}"


class ColumnPermission(BaseModel):
    """Column-level permissions - KEY FEATURE for your requirement"""
    
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='column_permissions'
    )
    group = models.ForeignKey(
        'UserGroup',
        on_delete=models.CASCADE,
        null=True, blank=True,
        related_name='column_permissions'
    )
    datasource_id = models.UUIDField()
    schema_name = models.CharField(max_length=100, default='public')
    table_name = models.CharField(max_length=100)
    column_name = models.CharField(max_length=100)
    access_type = models.CharField(
        max_length=10,
        choices=AccessType.choices,
        default=AccessType.READ
    )
    granted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='granted_column_permissions'
    )
    granted_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    
    # CRITICAL: Allow column in ML segments even without direct read access
    allow_in_segments = models.BooleanField(default=False)
    
    class Meta:
        unique_together = ('user', 'datasource_id', 'table_name', 'column_name', 'access_type')
    
    def __str__(self):
        return f"{self.user.username} - {self.access_type} on {self.table_name}.{self.column_name}"


class PermissionAuditLog(models.Model):
    """Audit log for all permission changes"""
    
    ACTION_CHOICES = [
        ('granted', 'Granted'),
        ('revoked', 'Revoked'),
        ('modified', 'Modified'),
        ('used', 'Used')
    ]
    
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='permission_audit_logs'
    )
    action = models.CharField(max_length=10, choices=ACTION_CHOICES)
    permission_type = models.CharField(max_length=20)
    resource_identifier = models.CharField(max_length=255)
    details = models.JSONField()
    performed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='performed_permission_actions',
        null=True, blank=True
    )
    timestamp = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.action} - {self.permission_type} for {self.user.username}"
