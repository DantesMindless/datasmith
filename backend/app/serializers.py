from rest_framework import serializers

from .models.main import MyModel  # Replace with your actual model


class MyModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = MyModel
        fields = "__all__"
