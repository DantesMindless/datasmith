"""
Database indexes for permission system optimization
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('userauth', '0002_customuser_department_customuser_is_active_user_and_more'),
    ]

    operations = [
        # DataSourcePermission indexes
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_datasource_perm_user_datasource ON userauth_datasourcepermission (user_id, datasource_id);",
            reverse_sql="DROP INDEX IF EXISTS idx_datasource_perm_user_datasource;"
        ),
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_datasource_perm_user_access ON userauth_datasourcepermission (user_id, access_type);",
            reverse_sql="DROP INDEX IF EXISTS idx_datasource_perm_user_access;"
        ),
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_datasource_perm_expires ON userauth_datasourcepermission (expires_at) WHERE expires_at IS NOT NULL;",
            reverse_sql="DROP INDEX IF EXISTS idx_datasource_perm_expires;"
        ),
        
        # TablePermission indexes
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_table_perm_user_datasource_table ON userauth_tablepermission (user_id, datasource_id, table_name);",
            reverse_sql="DROP INDEX IF EXISTS idx_table_perm_user_datasource_table;"
        ),
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_table_perm_user_access ON userauth_tablepermission (user_id, access_type);",
            reverse_sql="DROP INDEX IF EXISTS idx_table_perm_user_access;"
        ),
        
        # ColumnPermission indexes - CRITICAL for performance
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_column_perm_user_datasource_table ON userauth_columnpermission (user_id, datasource_id, table_name);",
            reverse_sql="DROP INDEX IF EXISTS idx_column_perm_user_datasource_table;"
        ),
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_column_perm_user_column ON userauth_columnpermission (user_id, datasource_id, table_name, column_name);",
            reverse_sql="DROP INDEX IF EXISTS idx_column_perm_user_column;"
        ),
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_column_perm_segments ON userauth_columnpermission (user_id, allow_in_segments) WHERE allow_in_segments = true;",
            reverse_sql="DROP INDEX IF EXISTS idx_column_perm_segments;"
        ),
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_column_perm_access_type ON userauth_columnpermission (user_id, access_type);",
            reverse_sql="DROP INDEX IF EXISTS idx_column_perm_access_type;"
        ),
        
        # PermissionAuditLog indexes
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_audit_log_user_timestamp ON userauth_permissionauditlog (user_id, timestamp DESC);",
            reverse_sql="DROP INDEX IF EXISTS idx_audit_log_user_timestamp;"
        ),
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_audit_log_action_timestamp ON userauth_permissionauditlog (action, timestamp DESC);",
            reverse_sql="DROP INDEX IF EXISTS idx_audit_log_action_timestamp;"
        ),
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_audit_log_performed_by ON userauth_permissionauditlog (performed_by_id, timestamp DESC);",
            reverse_sql="DROP INDEX IF EXISTS idx_audit_log_performed_by;"
        ),
        
        # CustomUser indexes
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_user_role_active ON userauth_customuser (role, is_active_user) WHERE is_active_user = true;",
            reverse_sql="DROP INDEX IF EXISTS idx_user_role_active;"
        ),
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_user_department ON userauth_customuser (department) WHERE department != '';",
            reverse_sql="DROP INDEX IF EXISTS idx_user_department;"
        ),
        
        # UserGroup indexes
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_usergroup_created_by ON userauth_usergroup (created_by_id, created_at DESC);",
            reverse_sql="DROP INDEX IF EXISTS idx_usergroup_created_by;"
        ),
    ]