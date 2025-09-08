from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import (
    UserRole, AccessType, UserGroup, DataSourcePermission,
    TablePermission, ColumnPermission, PermissionAuditLog, CustomUser
)

User = get_user_model()


class UserSerializer(serializers.ModelSerializer):
    """Enhanced user serializer with roles"""
    class Meta:
        model = CustomUser
        fields = [
            "id", "username", "password", "email", "first_name", "last_name",
            "role", "department", "is_active_user", "date_joined", "last_login"
        ]
        extra_kwargs = {"password": {"write_only": True}}
        read_only_fields = ('id', 'date_joined', 'last_login')

    def create(self, validated_data):
        password = validated_data.pop("password")
        user = CustomUser(**validated_data)
        user.set_password(password)
        user.save()
        return user


class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = CustomUser
        fields = ("email", "username", "password", "role", "department")

    def create(self, validated_data):
        password = validated_data.pop("password")
        user = CustomUser(**validated_data)
        user.set_password(password)
        user.save()
        return user


class UserGroupSerializer(serializers.ModelSerializer):
    """Serializer for user groups"""
    users = UserSerializer(many=True, read_only=True)
    user_count = serializers.SerializerMethodField()
    
    class Meta:
        model = UserGroup
        fields = ('id', 'name', 'description', 'users', 'user_count', 'created_at')
        read_only_fields = ('id', 'created_at')
    
    def get_user_count(self, obj):
        return obj.users.count()


class DataSourcePermissionSerializer(serializers.ModelSerializer):
    """Serializer for datasource permissions"""
    user_email = serializers.CharField(source='user.email', read_only=True)
    granted_by_email = serializers.CharField(source='granted_by.email', read_only=True)
    
    class Meta:
        model = DataSourcePermission
        fields = (
            'id', 'user', 'user_email', 'datasource_id', 'access_type',
            'granted_by', 'granted_by_email', 'granted_at', 'expires_at'
        )
        read_only_fields = ('id', 'granted_at', 'user_email', 'granted_by_email')


class TablePermissionSerializer(serializers.ModelSerializer):
    """Serializer for table permissions"""
    user_email = serializers.CharField(source='user.email', read_only=True)
    granted_by_email = serializers.CharField(source='granted_by.email', read_only=True)
    
    class Meta:
        model = TablePermission
        fields = (
            'id', 'user', 'user_email', 'datasource_id', 'schema_name', 
            'table_name', 'access_type', 'granted_by', 'granted_by_email',
            'granted_at', 'expires_at'
        )
        read_only_fields = ('id', 'granted_at', 'user_email', 'granted_by_email')


class ColumnPermissionSerializer(serializers.ModelSerializer):
    """Serializer for column permissions - KEY FEATURE"""
    user_email = serializers.CharField(source='user.email', read_only=True)
    granted_by_email = serializers.CharField(source='granted_by.email', read_only=True)
    
    class Meta:
        model = ColumnPermission
        fields = (
            'id', 'user', 'user_email', 'datasource_id', 'schema_name',
            'table_name', 'column_name', 'access_type', 'allow_in_segments',
            'granted_by', 'granted_by_email', 'granted_at', 'expires_at'
        )
        read_only_fields = ('id', 'granted_at', 'user_email', 'granted_by_email')


class PermissionAuditLogSerializer(serializers.ModelSerializer):
    """Serializer for permission audit logs"""
    user_email = serializers.CharField(source='user.email', read_only=True)
    performed_by_email = serializers.CharField(source='performed_by.email', read_only=True)
    
    class Meta:
        model = PermissionAuditLog
        fields = (
            'id', 'user', 'user_email', 'action', 'permission_type',
            'resource_identifier', 'details', 'performed_by', 'performed_by_email',
            'timestamp', 'ip_address'
        )
        read_only_fields = ('id', 'timestamp', 'user_email', 'performed_by_email')


class BulkPermissionSerializer(serializers.Serializer):
    """Serializer for bulk permission operations"""
    user_ids = serializers.ListField(
        child=serializers.UUIDField(),
        help_text="List of user IDs to grant permissions to"
    )
    datasource_id = serializers.UUIDField(required=False)
    table_name = serializers.CharField(max_length=100, required=False)
    column_names = serializers.ListField(
        child=serializers.CharField(max_length=100),
        required=False,
        help_text="List of column names for bulk column permissions"
    )
    access_type = serializers.ChoiceField(
        choices=AccessType.choices,
        default=AccessType.READ
    )
    allow_in_segments = serializers.BooleanField(default=False)
    expires_at = serializers.DateTimeField(required=False)


class PermissionSummarySerializer(serializers.Serializer):
    """Serializer for user permission summary"""
    user = UserSerializer()
    role = serializers.CharField()
    datasource_count = serializers.IntegerField()
    table_count = serializers.IntegerField()
    column_count = serializers.IntegerField()
    segment_accessible_columns = serializers.IntegerField()
    last_permission_granted = serializers.DateTimeField()
    permissions = serializers.DictField()
