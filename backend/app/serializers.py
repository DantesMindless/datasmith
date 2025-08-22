from rest_framework import serializers
from .models.main import Dataset, MLModel, TrainingRun


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ['id', 'name', 'csv_file', 'image_folder', 'is_image_dataset', 'created_at', 'created_by']
        read_only_fields = ['id', 'created_at', 'created_by']


class MLModelSerializer(serializers.ModelSerializer):
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)
    
    class Meta:
        model = MLModel
        fields = [
            'id', 'name', 'dataset', 'dataset_name', 'target_column', 
            'training_config', 'status', 'model_type', 'test_size', 
            'random_state', 'max_iter', 'created_at', 'created_by'
        ]
        read_only_fields = ['id', 'created_at', 'created_by', 'status']


class TrainingRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingRun
        fields = ['id', 'model', 'history']
        read_only_fields = ['id']
