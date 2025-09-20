import os
import zipfile
import shutil
from pathlib import Path
from django.db import models
from django.conf import settings
from core.models import (
    BaseModel,
)
from app.models.choices import ModelStatus, ModelType, DatasetType, DatasetPurpose, DataQuality, ColumnType


class Dataset(BaseModel):
    # Basic Information
    name = models.CharField(max_length=255, default="Unnamed Dataset")
    description = models.TextField(blank=True, null=True)

    # File Storage
    csv_file = models.FileField(upload_to="csv_datasets/", blank=True, null=True)
    image_folder = models.FileField(upload_to="image_zips/", blank=True, null=True)
    extracted_path = models.CharField(max_length=512, blank=True, null=True)

    # Dataset Classification
    dataset_type = models.CharField(
        max_length=20,
        choices=DatasetType.choices,
        default=DatasetType.TABULAR
    )
    dataset_purpose = models.CharField(
        max_length=20,
        choices=DatasetPurpose.choices,
        default=DatasetPurpose.GENERAL
    )
    data_quality = models.CharField(
        max_length=20,
        choices=DataQuality.choices,
        default=DataQuality.UNKNOWN
    )

    # Metadata and Statistics
    row_count = models.IntegerField(null=True, blank=True)
    column_count = models.IntegerField(null=True, blank=True)
    file_size = models.BigIntegerField(null=True, blank=True)  # in bytes
    encoding = models.CharField(max_length=50, default="utf-8")

    # Analysis Results
    column_info = models.JSONField(default=dict, blank=True)  # Column types, null counts, etc.
    statistics = models.JSONField(default=dict, blank=True)  # Summary statistics
    quality_report = models.JSONField(default=dict, blank=True)  # Data quality analysis
    preview_data = models.JSONField(default=list, blank=True)  # First few rows for preview

    # Processing Status
    is_processed = models.BooleanField(default=False)
    processing_errors = models.TextField(blank=True, null=True)
    last_analyzed = models.DateTimeField(null=True, blank=True)

    # Legacy field for backward compatibility
    is_image_dataset = models.BooleanField(default=False)

    def save(self, *args, **kwargs):
        # Set dataset type based on file type
        if self.image_folder and not self.dataset_type:
            self.dataset_type = DatasetType.IMAGE
            self.is_image_dataset = True
        elif self.csv_file and not self.dataset_type:
            self.dataset_type = DatasetType.TABULAR

        super().save(*args, **kwargs)

        # Handle image dataset extraction
        if self.image_folder and self.image_folder.name.endswith(".zip"):
            extract_to = os.path.join(
                settings.MEDIA_ROOT, "image_datasets", str(self.id)
            )
            os.makedirs(extract_to, exist_ok=True)

            temp_dir = os.path.join(extract_to, "tmp_extraction")
            with zipfile.ZipFile(self.image_folder.path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            top = next(Path(temp_dir).iterdir(), None)
            if top and top.is_dir():
                for item in top.iterdir():
                    shutil.move(str(item), extract_to)
                shutil.rmtree(temp_dir)
            else:
                for item in Path(temp_dir).iterdir():
                    shutil.move(str(item), extract_to)
                shutil.rmtree(temp_dir)

            self.extracted_path = extract_to
            super().save(update_fields=["extracted_path"])

        # Analyze CSV file
        if self.csv_file and not self.is_processed:
            self.analyze_dataset()

    def analyze_dataset(self):
        """Comprehensive dataset analysis"""
        if not self.csv_file:
            return

        try:
            import pandas as pd
            from django.utils import timezone

            # Read the dataset
            df = pd.read_csv(self.csv_file.path, encoding=self.encoding)

            # Basic statistics
            self.row_count = len(df)
            self.column_count = len(df.columns)
            self.file_size = os.path.getsize(self.csv_file.path)

            # Column analysis
            column_info = {}
            for col in df.columns:
                col_data = df[col]
                col_info = {
                    'type': self._detect_column_type(col_data),
                    'null_count': int(col_data.isnull().sum()),
                    'null_percentage': float(col_data.isnull().sum() / len(df) * 100),
                    'unique_count': int(col_data.nunique()),
                    'dtype': str(col_data.dtype)
                }

                # Add type-specific statistics
                if col_info['type'] == ColumnType.NUMERIC:
                    col_info.update({
                        'mean': float(col_data.mean()) if pd.api.types.is_numeric_dtype(col_data) else None,
                        'std': float(col_data.std()) if pd.api.types.is_numeric_dtype(col_data) else None,
                        'min': float(col_data.min()) if pd.api.types.is_numeric_dtype(col_data) else None,
                        'max': float(col_data.max()) if pd.api.types.is_numeric_dtype(col_data) else None,
                        'median': float(col_data.median()) if pd.api.types.is_numeric_dtype(col_data) else None
                    })
                elif col_info['type'] == ColumnType.CATEGORICAL:
                    value_counts = col_data.value_counts().head(10).to_dict()
                    col_info['top_values'] = {str(k): int(v) for k, v in value_counts.items()}

                column_info[col] = col_info

            self.column_info = column_info

            # Overall statistics
            numeric_cols = df.select_dtypes(include=['number']).columns
            self.statistics = {
                'numeric_columns': len(numeric_cols),
                'categorical_columns': len(df.columns) - len(numeric_cols),
                'total_null_values': int(df.isnull().sum().sum()),
                'completeness': float((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
                'memory_usage': int(df.memory_usage(deep=True).sum()),
                'duplicate_rows': int(df.duplicated().sum())
            }

            # Data quality assessment
            completeness = self.statistics['completeness']
            if completeness >= 95:
                self.data_quality = DataQuality.EXCELLENT
            elif completeness >= 85:
                self.data_quality = DataQuality.GOOD
            elif completeness >= 70:
                self.data_quality = DataQuality.FAIR
            else:
                self.data_quality = DataQuality.POOR

            # Quality report
            self.quality_report = {
                'completeness_score': completeness,
                'duplicate_rows': int(df.duplicated().sum()),
                'columns_with_nulls': [col for col, info in column_info.items() if info['null_count'] > 0],
                'highly_null_columns': [col for col, info in column_info.items() if info['null_percentage'] > 50],
                'potential_issues': self._identify_data_issues(df, column_info)
            }

            # Preview data (first 10 rows)
            preview_df = df.head(10)
            self.preview_data = preview_df.to_dict('records')

            # Detect dataset purpose
            self.dataset_purpose = self._detect_dataset_purpose(df, column_info)

            self.is_processed = True
            self.last_analyzed = timezone.now()
            self.processing_errors = None

        except Exception as e:
            self.processing_errors = str(e)
            self.is_processed = False

        # Save without triggering analysis again
        super().save(update_fields=[
            'row_count', 'column_count', 'file_size', 'column_info',
            'statistics', 'quality_report', 'preview_data', 'dataset_purpose',
            'data_quality', 'is_processed', 'last_analyzed', 'processing_errors'
        ])

    def _detect_column_type(self, series):
        """Detect the type of a column"""
        import pandas as pd
        import re

        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return ColumnType.DATETIME

        # Check for boolean
        if pd.api.types.is_bool_dtype(series):
            return ColumnType.BOOLEAN

        # Check for numeric
        if pd.api.types.is_numeric_dtype(series):
            return ColumnType.NUMERIC

        # Check string patterns
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            series_str = series.astype(str).str.lower()
            sample = series_str.dropna().head(100)

            # Email pattern
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if sample.str.match(email_pattern).sum() / len(sample) > 0.8:
                return ColumnType.EMAIL

            # URL pattern
            url_pattern = r'^https?://'
            if sample.str.match(url_pattern).sum() / len(sample) > 0.8:
                return ColumnType.URL

            # Image path pattern
            img_pattern = r'\.(jpg|jpeg|png|gif|bmp|tiff)$'
            if sample.str.match(img_pattern).sum() / len(sample) > 0.8:
                return ColumnType.IMAGE_PATH

            # Categorical (limited unique values)
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.1 and series.nunique() < 50:
                return ColumnType.CATEGORICAL

            # Text (high unique ratio)
            if unique_ratio > 0.8:
                return ColumnType.TEXT

            return ColumnType.CATEGORICAL

        return ColumnType.UNKNOWN

    def _detect_dataset_purpose(self, df, column_info):
        """Detect the likely purpose of the dataset"""
        # Look for target-like column names
        target_keywords = ['target', 'label', 'class', 'category', 'outcome', 'result', 'prediction']

        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in target_keywords):
                col_type = column_info[col]['type']
                unique_count = column_info[col]['unique_count']

                if col_type == ColumnType.CATEGORICAL and unique_count <= 10:
                    return DatasetPurpose.CLASSIFICATION
                elif col_type == ColumnType.NUMERIC:
                    return DatasetPurpose.REGRESSION

        # Check for time series indicators
        datetime_cols = [col for col, info in column_info.items() if info['type'] == ColumnType.DATETIME]
        if datetime_cols:
            return DatasetPurpose.TIME_SERIES

        return DatasetPurpose.GENERAL

    def _identify_data_issues(self, df, column_info):
        """Identify potential data quality issues"""
        issues = []

        # High null percentage
        for col, info in column_info.items():
            if info['null_percentage'] > 30:
                issues.append(f"Column '{col}' has {info['null_percentage']:.1f}% missing values")

        # Duplicate rows
        if df.duplicated().sum() > 0:
            issues.append(f"Dataset contains {df.duplicated().sum()} duplicate rows")

        # Single value columns
        for col, info in column_info.items():
            if info['unique_count'] == 1:
                issues.append(f"Column '{col}' has only one unique value")

        # Potential encoding issues
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].astype(str).str.contains('�').any():
                issues.append(f"Column '{col}' may have encoding issues")

        return issues

    def __str__(self):
        return self.name


class MLModel(BaseModel):
    name = models.CharField(max_length=255, default="Unnamed Model")
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    target_column = models.CharField(max_length=255, default="target")
    training_config = models.JSONField(default=dict, blank=True)
    status = models.CharField(
        max_length=50, choices=ModelStatus.choices, default=ModelStatus.PENDING
    )
    model_file = models.FileField(
        upload_to="trained_models/",
        null=True,
        blank=True,
        help_text="Path to the trained model file",
    )
    training_log = models.TextField(
        blank=True, null=True, default="Training not started."
    )
    model_type = models.CharField(
        max_length=50, choices=ModelType.choices, default=ModelType.LOGISTIC_REGRESSION
    )
    test_size = models.FloatField(default=0.2)
    random_state = models.IntegerField(default=42)
    max_iter = models.IntegerField(default=1000)
    accuracy = models.FloatField(null=True, blank=True, help_text="Model accuracy score")

    def __str__(self):
        return self.name


class TrainingRun(models.Model):
    model = models.OneToOneField(
        "MLModel", on_delete=models.CASCADE, related_name="training_run"
    )
    history = models.JSONField(default=list, blank=True)

    def add_entry(self, status: str, accuracy: float = None, error: str = None):
        from django.utils.timezone import now

        entry = {
            "timestamp": now().isoformat(),
            "status": status,
            "accuracy": accuracy,
            "error": error,
        }
        self.history.append(entry)
        self.save()
