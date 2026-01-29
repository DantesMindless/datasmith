from rest_framework import serializers
from .models import CustomUser


class UserSerializer(serializers.ModelSerializer):
    """User serializer for authentication"""
    class Meta:
        model = CustomUser
        fields = [
            "id", "username", "password", "email", "first_name", "last_name",
            "date_joined", "last_login"
        ]
        extra_kwargs = {"password": {"write_only": True}}
        read_only_fields = ('id', 'date_joined', 'last_login')

    def create(self, validated_data):
        password = validated_data.pop("password")
        user = CustomUser(**validated_data)
        user.set_password(password)
        user.save()
        return user
