from rest_framework import serializers
from .models.main import Dataset, MLModel, TrainingRun


class DatasetSerializer(serializers.ModelSerializer):
    file_size_formatted = serializers.SerializerMethodField()
    quality_score = serializers.SerializerMethodField()

    class Meta:
        model = Dataset
        fields = [
            'id', 'name', 'description', 'csv_file', 'image_folder',
            'dataset_type', 'dataset_purpose', 'data_quality',
            'row_count', 'column_count', 'file_size', 'file_size_formatted',
            'encoding', 'is_processed', 'last_analyzed', 'quality_score',
            'is_image_dataset', 'created_at', 'created_by'
        ]
        read_only_fields = [
            'id', 'created_at', 'created_by', 'row_count', 'column_count',
            'file_size', 'is_processed', 'last_analyzed'
        ]

    def get_file_size_formatted(self, obj):
        """Format file size in human readable format"""
        if not obj.file_size:
            return None

        size = obj.file_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    def get_quality_score(self, obj):
        """Get quality score percentage"""
        if not obj.statistics:
            return None
        return obj.statistics.get('completeness', 0)


class MLModelSerializer(serializers.ModelSerializer):
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)
    
    class Meta:
        model = MLModel
        fields = [
            'id', 'name', 'dataset', 'dataset_name', 'target_column',
            'training_config', 'status', 'model_type', 'test_size',
            'random_state', 'max_iter', 'accuracy', 'training_log',
            'created_at', 'created_by'
        ]
        read_only_fields = ['id', 'created_at', 'created_by', 'status', 'accuracy', 'training_log']


class TrainingRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingRun
        fields = ['id', 'model', 'history']
        read_only_fields = ['id']
