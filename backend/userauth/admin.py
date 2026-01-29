from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser


@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    """Custom user admin"""
    list_display = ('email', 'username', 'is_staff', 'date_joined')
    list_filter = ('is_staff', 'is_superuser', 'date_joined')
    search_fields = ('email', 'username')
    ordering = ('-date_joined',)


# Customize admin site
admin.site.site_header = "DataSmith Administration"
admin.site.site_title = "DataSmith Admin"
admin.site.index_title = "Welcome to DataSmith Administration"
