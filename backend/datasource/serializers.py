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


class DatasourceEditSerializer(serializers.ModelSerializer):
    """Serializer for editing datasource - includes credentials."""
    user = serializers.SerializerMethodField()
    created_by = serializers.SerializerMethodField()

    class Meta:
        model = DataSource
        fields = [
            "id",
            "name",
            "description",
            "type",
            "credentials",
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

    def validate_credentials(self, value: Dict[str, str]) -> Dict[str, str]:
        """
        Validates the `credentials` field.

        Ensures the credentials are a dictionary, not empty, and contain the
        required parameters for the selected datasource type.

        Args:
            value (Dict[str, str]): The credentials dictionary.

        Returns:
            Dict[str, str]: The validated credentials dictionary.

        Raises:
            serializers.ValidationError: If the credentials are invalid or incomplete.
        """
        if not isinstance(value, dict):
            raise serializers.ValidationError("Credentials must be a dictionary.")
        if not value:
            raise serializers.ValidationError("Credentials cannot be empty.")

        datasource_type = self.initial_data.get("type")
        adapter = DatasourceTypeChoices.get_adapter(datasource_type)

        if not adapter:
            raise serializers.ValidationError(
                f"Invalid connection type: {datasource_type}. "
                f"Supported types are: {', '.join(DatasourceTypeChoices.choices_list())}."
            )

        is_valid, missing_params = adapter.verify_params(value)
        if not is_valid:
            raise serializers.ValidationError(
                detail=missing_params, code="missing_required_credentials"
            )

        return value
