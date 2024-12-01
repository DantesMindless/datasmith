from rest_framework import serializers
from .models import DataSource
from .constants.choices import DatasourceTypeChoices


class DataSourceSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataSource
        fields = "__all__"
        read_only_fields = ["created_at", "updated_at"]

    def validate_type(self, value):
        if value not in DatasourceTypeChoices.choices_list():
            raise serializers.ValidationError("Invalid datasource type")
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
        print(adapter.verify_params(value))
        _, message = adapter.verify_params(value)
        if message:
            raise serializers.ValidationError(f"Missing credentials: {message}")
        return value
