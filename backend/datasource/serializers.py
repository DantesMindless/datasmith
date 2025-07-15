from typing import Dict
from rest_framework import serializers
from .models import DataSource
from .constants.choices import DatasourceTypeChoices


class DatasourceViewSerializer(serializers.ModelSerializer):
    user = serializers.SerializerMethodField()
    created_by = serializers.SerializerMethodField()

    class Meta:
        model = DataSource
        fields = [
            "id",
            "name",
            "type",
            "user",
            "created_by",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["created_at", "updated_at"]

    def get_user(self, obj: DataSource) -> str:
        return obj.user.username or "System"

    def get_created_by(self, obj: DataSource) -> str:
        return obj.created_by.username or "System"


class DataSourceSerializer(serializers.ModelSerializer):
    """
    Serializer for the DataSource model.

    Includes validation for the `credentials` field to ensure compatibility with
    the selected datasource type.
    """

    class Meta:
        model = DataSource
        fields = "__all__"
        read_only_fields = ["created_at", "updated_at"]

    def validate_type(self, value):
        # Normalize to uppercase to match TextChoices
        value = value.upper()
        if value not in DatasourceTypeChoices.choices_list():
            raise serializers.ValidationError(
                f"Invalid datasource type: '{value}'. Valid types are: {DatasourceTypeChoices.choices_list()}"
            )
        return value

    def validate_credentials(self, value):
        if not isinstance(value, dict):
            raise serializers.ValidationError("Credentials must be a dictionary")
        if not value:
            raise serializers.ValidationError("Credentials cannot be empty")
        if not (
            adapter := DatasourceTypeChoices.get_adapter(self.initial_data.get("type"))
        ):
            raise serializers.ValidationError("Connection type")
        _, message = adapter.verify_params(value)
        if message:
            raise serializers.ValidationError(f"Missing credentials: {message}")
        return value
