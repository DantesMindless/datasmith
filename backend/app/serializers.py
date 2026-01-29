from rest_framework import serializers
from .models.main import Dataset, MLModel, TrainingRun, DataSegment, RowSegmentLabel


class DatasetSerializer(serializers.ModelSerializer):
    file_size_formatted = serializers.SerializerMethodField()
    quality_score = serializers.SerializerMethodField()
    source_dataset_names = serializers.SerializerMethodField()

    # Accept file uploads in the request
    csv_file = serializers.FileField(write_only=True, required=False)
    image_folder = serializers.FileField(write_only=True, required=False)

    class Meta:
        model = Dataset
        fields = [
            'id', 'name', 'description', 'csv_file', 'image_folder',
            'minio_csv_key', 'minio_zip_key', 'minio_images_prefix',
            'dataset_type', 'dataset_purpose', 'data_quality',
            'row_count', 'column_count', 'file_size', 'file_size_formatted',
            'encoding', 'is_processed', 'processing_errors', 'quality_score',
            'is_image_dataset', 'created_at', 'created_by',
            'source_datasets', 'source_dataset_names', 'join_metadata',
            'model_recommendations', 'column_info', 'statistics'
        ]
        read_only_fields = [
            'id', 'created_at', 'created_by', 'row_count', 'column_count',
            'file_size', 'is_processed', 'processing_errors',
            'minio_csv_key', 'minio_zip_key', 'minio_images_prefix',
            'source_datasets', 'source_dataset_names', 'join_metadata',
            'model_recommendations', 'column_info', 'statistics'
        ]

    def create(self, validated_data):
        """Handle file uploads to MinIO before creating dataset"""
        csv_file = validated_data.pop('csv_file', None)
        image_folder = validated_data.pop('image_folder', None)

        # Create the dataset instance (without saving to DB yet)
        dataset = Dataset(**validated_data)

        # Upload files to MinIO if provided
        if csv_file:
            from core.storage_utils import upload_file_to_minio
            minio_key = f"datasets/{dataset.id}/csv/{csv_file.name}"
            upload_file_to_minio(csv_file, minio_key, 'text/csv')
            dataset.minio_csv_key = minio_key

        if image_folder:
            from core.storage_utils import upload_file_to_minio
            minio_key = f"datasets/{dataset.id}/zip/{image_folder.name}"
            upload_file_to_minio(image_folder, minio_key, 'application/zip')
            dataset.minio_zip_key = minio_key

        # Save the dataset (this will trigger extraction for images)
        dataset.save()
        return dataset

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

    def get_source_dataset_names(self, obj):
        """Get names of source datasets for lineage display"""
        return [ds.name for ds in obj.source_datasets.all()]


class DatasetJoinSerializer(serializers.Serializer):
    """Serializer for validating dataset join requests"""
    JOIN_TYPE_CHOICES = [
        ('inner', 'Inner Join'),
        ('left', 'Left Outer Join'),
        ('right', 'Right Outer Join'),
        ('outer', 'Full Outer Join'),
    ]

    left_dataset_id = serializers.UUIDField()
    right_dataset_id = serializers.UUIDField()
    left_key_column = serializers.CharField(max_length=255)
    right_key_column = serializers.CharField(max_length=255)
    join_type = serializers.ChoiceField(choices=JOIN_TYPE_CHOICES, default='inner')
    result_name = serializers.CharField(max_length=255)
    result_description = serializers.CharField(required=False, allow_blank=True, default='')

    def validate(self, data):
        """Validate that both datasets exist and have CSV files"""
        left_id = data.get('left_dataset_id')
        right_id = data.get('right_dataset_id')

        if left_id == right_id:
            raise serializers.ValidationError("Cannot join a dataset with itself")

        try:
            left_dataset = Dataset.objects.get(id=left_id, deleted=False)
        except Dataset.DoesNotExist:
            raise serializers.ValidationError({"left_dataset_id": "Dataset not found"})

        try:
            right_dataset = Dataset.objects.get(id=right_id, deleted=False)
        except Dataset.DoesNotExist:
            raise serializers.ValidationError({"right_dataset_id": "Dataset not found"})

        if not left_dataset.minio_csv_key:
            raise serializers.ValidationError({"left_dataset_id": "Dataset has no CSV file"})

        if not right_dataset.minio_csv_key:
            raise serializers.ValidationError({"right_dataset_id": "Dataset has no CSV file"})

        data['left_dataset'] = left_dataset
        data['right_dataset'] = right_dataset
        return data


class MLModelSerializer(serializers.ModelSerializer):
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)

    class Meta:
        model = MLModel
        fields = [
            'id', 'name', 'dataset', 'dataset_name', 'target_column',
            'training_config', 'status', 'model_type', 'test_size',
            'random_state', 'max_iter', 'accuracy', 'training_log',
            'prediction_schema', 'created_at', 'created_by'
        ]
        read_only_fields = ['id', 'created_at', 'created_by', 'status', 'accuracy', 'training_log', 'prediction_schema']


class TrainingRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingRun
        fields = ['id', 'model', 'history']
        read_only_fields = ['id']


class DataSegmentSerializer(serializers.ModelSerializer):
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)
    label_count = serializers.SerializerMethodField()

    class Meta:
        model = DataSegment
        fields = [
            'id', 'dataset', 'dataset_name', 'name', 'description', 'color',
            'row_count', 'label_count', 'rules', 'created_at', 'created_by', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'created_by', 'updated_at', 'row_count']

    def get_label_count(self, obj):
        """Get actual count of labeled rows"""
        return obj.row_labels.count()


class RowSegmentLabelSerializer(serializers.ModelSerializer):
    segment_name = serializers.CharField(source='segment.name', read_only=True)
    segment_color = serializers.CharField(source='segment.color', read_only=True)
    assigned_by_username = serializers.CharField(source='assigned_by.username', read_only=True, allow_null=True)

    class Meta:
        model = RowSegmentLabel
        fields = [
            'id', 'segment', 'segment_name', 'segment_color', 'row_index',
            'row_data', 'assigned_by', 'assigned_by_username', 'assignment_method',
            'confidence', 'notes', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class BulkRowSegmentLabelSerializer(serializers.Serializer):
    """Serializer for bulk assigning rows to a segment"""
    segment_id = serializers.UUIDField()
    row_indices = serializers.ListField(
        child=serializers.IntegerField(min_value=0),
        allow_empty=False
    )
    assignment_method = serializers.ChoiceField(
        choices=['manual', 'rule', 'ml'],
        default='manual'
    )
    confidence = serializers.FloatField(min_value=0.0, max_value=1.0, default=1.0)
    notes = serializers.CharField(required=False, allow_blank=True)
