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

    # MinIO Storage (S3-compatible object storage)
    minio_csv_key = models.CharField(max_length=512, blank=True, null=True, help_text="MinIO key for CSV file")
    minio_zip_key = models.CharField(max_length=512, blank=True, null=True, help_text="MinIO key for ZIP archive")
    minio_images_prefix = models.CharField(max_length=512, blank=True, null=True, help_text="MinIO prefix for extracted images")

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
    segment_labels = models.JSONField(default=dict, blank=True)  # Segmentation/clustering results
    model_recommendations = models.JSONField(default=dict, blank=True)  # Recommended ML models based on dataset analysis

    # Lineage Tracking
    source_datasets = models.ManyToManyField(
        'self', symmetrical=False, related_name='derived_datasets', blank=True
    )
    join_metadata = models.JSONField(default=dict, blank=True)

    # Processing Status
    is_processed = models.BooleanField(default=False)
    processing_errors = models.TextField(blank=True, null=True)

    # Legacy field for backward compatibility
    is_image_dataset = models.BooleanField(default=False)

    def save(self, *args, **kwargs):
        import logging
        logger = logging.getLogger(__name__)

        is_new_upload = self._state.adding

        # Set dataset type based on MinIO keys
        if is_new_upload:
            if self.minio_zip_key and not self.dataset_type:
                self.dataset_type = DatasetType.IMAGE
                self.is_image_dataset = True
                self.processing_errors = "Queued for extraction..."
                self.is_processed = False
                logger.info(f"[DATASET SAVE] Set dataset_type to IMAGE")
            elif self.minio_csv_key and not self.dataset_type:
                self.dataset_type = DatasetType.TABULAR
                logger.info(f"[DATASET SAVE] Set dataset_type to TABULAR")

        super().save(*args, **kwargs)

        # Handle image dataset extraction asynchronously with Celery
        if is_new_upload and self.dataset_type == DatasetType.IMAGE and self.minio_zip_key:
            try:
                from app.functions.celery_tasks import extract_image_dataset_task
                logger.info(f"Queuing Celery task for dataset {self.id} extraction")
                extract_image_dataset_task.delay(str(self.id))
                logger.info(f"Successfully queued extraction task for dataset {self.id}")
            except Exception as e:
                logger.warning(f"Celery not available, using fallback threading: {e}")
                import threading
                thread = threading.Thread(target=self._extract_sync)
                thread.daemon = True
                thread.start()
                logger.info(f"Started fallback thread extraction for dataset {self.id}")

        # Analyze CSV file
        if is_new_upload and self.dataset_type == DatasetType.TABULAR and not self.is_processed:
            self.analyze_dataset()

    def _extract_sync(self):
        """Fallback synchronous extraction method - MinIO only"""
        import tempfile
        import logging
        logger = logging.getLogger(__name__)

        if not self.minio_zip_key:
            logger.error("[EXTRACT] No ZIP file available")
            return

        try:
            # Download ZIP from MinIO
            logger.info(f"[EXTRACT] Downloading ZIP from MinIO: {self.minio_zip_key}")
            zip_bytes = self.get_zip_bytes()
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            temp_zip.write(zip_bytes)
            temp_zip.close()
            zip_path = temp_zip.name
            logger.info(f"[EXTRACT] ZIP downloaded to temp file: {zip_path}")

            # Create temp extraction directory
            extract_to = tempfile.mkdtemp(prefix=f'dataset_{self.id}_')
            logger.info(f"[EXTRACT] Extracting to temp directory: {extract_to}")

            # Extract ZIP
            temp_dir = os.path.join(extract_to, "tmp_extraction")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
            logger.info(f"[EXTRACT] ZIP extracted to: {temp_dir}")

            # Handle nested directories (if ZIP contains a single folder, unwrap it)
            top = next(Path(temp_dir).iterdir(), None)
            if top and top.is_dir() and len(list(Path(temp_dir).iterdir())) == 1:
                for item in top.iterdir():
                    shutil.move(str(item), extract_to)
                shutil.rmtree(temp_dir)
                logger.info(f"[EXTRACT] Unwrapped nested directory")
            else:
                for item in Path(temp_dir).iterdir():
                    shutil.move(str(item), extract_to)
                shutil.rmtree(temp_dir)

            # Count images and calculate total size
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
            image_count = 0
            total_size = 0
            for root, dirs, files in os.walk(extract_to):
                level = root.replace(extract_to, '').count(os.sep)
                if level < 3:
                    for f in files:
                        file_path = Path(root) / f
                        if file_path.suffix.lower() in image_extensions:
                            image_count += 1
                            total_size += file_path.stat().st_size
                else:
                    del dirs[:]

            logger.info(f"[EXTRACT] Found {image_count} images, total size: {total_size} bytes")

            # Upload extracted images to MinIO
            from core.storage_utils import upload_directory_to_minio
            minio_prefix = f"datasets/{self.id}/images/"
            logger.info(f"[EXTRACT] Uploading extracted images to MinIO: {minio_prefix}")
            upload_directory_to_minio(extract_to, minio_prefix)
            logger.info(f"[EXTRACT] Images uploaded to MinIO successfully")

            # Update dataset
            self.minio_images_prefix = minio_prefix
            self.row_count = image_count
            self.file_size = total_size
            self.is_processed = True
            self.processing_errors = None

            # Clean up temp files
            shutil.rmtree(extract_to)
            os.unlink(zip_path)
            logger.info(f"[EXTRACT] Cleaned up temp files")

            super().save(update_fields=["minio_images_prefix", "row_count", "file_size", "is_processed", "processing_errors"])
            logger.info(f"[EXTRACT] Extraction completed successfully")

        except Exception as e:
            logger.error(f"[EXTRACT] Extraction failed: {str(e)}")
            self.processing_errors = f"Extraction failed: {str(e)}"
            super().save(update_fields=["processing_errors"])

    def get_csv_file_object(self):
        """
        Get a file-like object for reading CSV data from MinIO.
        Returns an io.BytesIO object that can be used with pd.read_csv()
        """
        if not self.minio_csv_key:
            return None

        from core.storage_utils import download_from_minio
        import io

        csv_bytes = download_from_minio(self.minio_csv_key)
        return io.BytesIO(csv_bytes)

    @property
    def csv_file(self):
        """
        Backward compatibility property that mimics FileField behavior.
        Returns a mock object with .path attribute for code expecting FileField.
        """
        if not self.minio_csv_key:
            return None

        class CSVFileMock:
            def __init__(self, dataset):
                self.dataset = dataset
                self._temp_path = None

            @property
            def path(self):
                """Download CSV to temp file and return path"""
                if self._temp_path is None:
                    from core.storage_utils import download_from_minio
                    import tempfile

                    csv_bytes = download_from_minio(self.dataset.minio_csv_key)

                    # Create temp file
                    fd, self._temp_path = tempfile.mkstemp(suffix='.csv')
                    with open(fd, 'wb') as f:
                        f.write(csv_bytes)

                return self._temp_path

            @property
            def name(self):
                """Return the MinIO key as name"""
                return self.dataset.minio_csv_key

            def __bool__(self):
                return bool(self.dataset.minio_csv_key)

        return CSVFileMock(self)

    def analyze_dataset(self):
        """Comprehensive dataset analysis for tabular datasets"""
        import logging
        logger = logging.getLogger(__name__)

        if not self.minio_csv_key:
            # No CSV file to analyze - this is expected for image datasets
            logger.info(f"[ANALYZE] Dataset {self.id} has no CSV file, skipping analysis")
            return False

        logger.info(f"[ANALYZE] Starting analysis for dataset {self.id}")
        csv_path = None
        try:
            import pandas as pd
            from django.utils import timezone

            # Get CSV file path (downloads from MinIO to temp file)
            csv_path = self.get_csv_path()

            # Read the dataset
            df = pd.read_csv(csv_path, encoding=self.encoding)

            # Basic statistics
            self.row_count = len(df)
            self.column_count = len(df.columns)

            # Get file size from MinIO
            from core.storage_utils import get_minio_client
            client = get_minio_client()
            response = client.head_object(
                Bucket=os.getenv("MINIO_BUCKET_NAME"),
                Key=self.minio_csv_key
            )
            self.file_size = response['ContentLength']

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

            # Preview data (first 100 rows for better caching)
            preview_df = df.head(100)
            self.preview_data = preview_df.to_dict('records')
            # Convert any numpy types to Python types for JSON serialization
            # Also handle NaN values which can't be serialized to JSON
            import math
            def convert_value(v):
                if hasattr(v, 'item'):
                    v = v.item()
                if isinstance(v, float) and math.isnan(v):
                    return None
                return v

            self.preview_data = [
                {k: convert_value(v) for k, v in row.items()}
                for row in self.preview_data
            ]

            # Detect dataset purpose
            self.dataset_purpose = self._detect_dataset_purpose(df, column_info)

            # Generate model recommendations
            self.model_recommendations = self._generate_model_recommendations(df, column_info)

            self.is_processed = True
            self.processing_errors = None
            logger.info(f"[ANALYZE] Successfully analyzed dataset {self.id}: {self.row_count} rows, {self.column_count} columns")

        except Exception as e:
            logger.error(f"[ANALYZE] Failed to analyze dataset {self.id}: {str(e)}", exc_info=True)
            self.processing_errors = str(e)
            self.is_processed = False

        finally:
            # Cleanup temp file
            if csv_path:
                self.cleanup_temp_files(csv_path)

        # Save without triggering analysis again
        super().save(update_fields=[
            'row_count', 'column_count', 'file_size', 'column_info',
            'statistics', 'quality_report', 'preview_data', 'dataset_purpose',
            'data_quality', 'is_processed', 'processing_errors',
            'model_recommendations'
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

    def _generate_model_recommendations(self, df, column_info):
        """
        Generate ML model recommendations based on dataset characteristics.
        Returns a dict with recommended models and compatibility scores.
        """
        import numpy as np

        recommendations = {
            'recommended_models': [],
            'not_recommended': [],
            'dataset_characteristics': {},
            'reasoning': []
        }

        # Check if this is an image dataset first
        if self.dataset_type == DatasetType.IMAGE or self.is_image_dataset:
            return self._generate_image_model_recommendations()

        # Analyze dataset characteristics for tabular data
        n_rows = len(df)
        n_features = len(df.columns) - 1  # Exclude target
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        n_numeric = len(numeric_cols)
        n_categorical = len(categorical_cols)

        # Check for potential target columns
        target_candidates = []
        for col, info in column_info.items():
            if info['type'] == ColumnType.CATEGORICAL and info['unique_count'] <= 20:
                target_candidates.append({
                    'column': col,
                    'unique_values': info['unique_count'],
                    'task_type': 'classification'
                })
            elif info['type'] == ColumnType.NUMERIC:
                # Could be regression target
                target_candidates.append({
                    'column': col,
                    'task_type': 'regression'
                })

        # Determine likely task type
        task_type = 'classification'  # Default
        if self.dataset_purpose == DatasetPurpose.REGRESSION:
            task_type = 'regression'

        recommendations['dataset_characteristics'] = {
            'rows': n_rows,
            'features': n_features,
            'numeric_features': n_numeric,
            'categorical_features': n_categorical,
            'likely_task': task_type,
            'target_candidates': target_candidates[:5]  # Top 5 candidates
        }

        # Model scoring based on dataset characteristics
        model_scores = {}

        # Random Forest
        rf_score = 85  # Base score - good all-around
        if n_rows >= 1000:
            rf_score += 5
        if n_features >= 5:
            rf_score += 5
        if n_categorical > 0:
            rf_score += 3  # Handles categorical well with encoding
        model_scores['random_forest'] = {
            'score': min(rf_score, 100),
            'name': 'Random Forest',
            'reasons': [
                'Excellent for tabular data',
                'Handles non-linear relationships',
                'Provides feature importance',
                'Robust to outliers'
            ]
        }

        # Gradient Boosting
        gb_score = 88  # Base score - often best for tabular
        if n_rows >= 1000:
            gb_score += 5
        if n_features >= 5:
            gb_score += 5
        if task_type == 'classification':
            gb_score += 2
        model_scores['gradient_boosting'] = {
            'score': min(gb_score, 100),
            'name': 'Gradient Boosting',
            'reasons': [
                'Best performer on structured/tabular data',
                'Learns complex patterns effectively',
                'Handles threshold-based rules well',
                'State-of-the-art for competitions'
            ]
        }

        # Logistic/Linear Regression
        lr_score = 70
        if n_features <= 20:
            lr_score += 10
        if n_rows >= 500:
            lr_score += 5
        model_name = 'Logistic Regression' if task_type == 'classification' else 'Linear Regression'
        model_scores['logistic_regression'] = {
            'score': lr_score,
            'name': model_name,
            'reasons': [
                'Fast training and prediction',
                'Highly interpretable',
                'Good baseline model',
                'Works well with linear relationships'
            ]
        }

        # Neural Network
        nn_score = 60  # Base - needs more data
        if n_rows >= 5000:
            nn_score += 15
        if n_rows >= 10000:
            nn_score += 10
        if n_features >= 10:
            nn_score += 5
        if n_rows < 1000:
            nn_score -= 20  # Penalize for small datasets
        model_scores['neural_network'] = {
            'score': max(nn_score, 30),
            'name': 'Neural Network',
            'reasons': [
                'Can learn complex non-linear patterns',
                'Flexible architecture',
                'Good for large datasets',
                'Requires more data for best performance'
            ]
        }

        # SVM
        svm_score = 65
        if n_rows <= 10000:
            svm_score += 10  # SVM works better on smaller datasets
        if n_rows > 50000:
            svm_score -= 20  # Too slow for large datasets
        if n_features <= 50:
            svm_score += 5
        model_scores['svm'] = {
            'score': max(svm_score, 30),
            'name': 'Support Vector Machine',
            'reasons': [
                'Effective in high-dimensional spaces',
                'Works well with clear margins',
                'Memory efficient',
                'Best for small to medium datasets'
            ]
        }

        # Decision Tree
        dt_score = 70
        if n_rows >= 500:
            dt_score += 5
        model_scores['decision_tree'] = {
            'score': dt_score,
            'name': 'Decision Tree',
            'reasons': [
                'Highly interpretable',
                'No feature scaling required',
                'Handles non-linear relationships',
                'Fast training'
            ]
        }

        # Sort by score and categorize
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['score'], reverse=True)

        for model_key, model_info in sorted_models:
            model_entry = {
                'model_type': model_key,
                'name': model_info['name'],
                'compatibility_score': model_info['score'],
                'reasons': model_info['reasons']
            }

            if model_info['score'] >= 70:
                recommendations['recommended_models'].append(model_entry)
            else:
                recommendations['not_recommended'].append(model_entry)

        # Add overall reasoning
        recommendations['reasoning'] = [
            f"Dataset has {n_rows:,} rows and {n_features} features",
            f"Contains {n_numeric} numeric and {n_categorical} categorical features",
            f"Detected task type: {task_type}",
        ]

        if n_rows < 1000:
            recommendations['reasoning'].append("Small dataset - tree-based models recommended over neural networks")
        elif n_rows >= 10000:
            recommendations['reasoning'].append("Large dataset - neural networks become more viable")

        if n_categorical > n_numeric:
            recommendations['reasoning'].append("Many categorical features - consider tree-based models")

        return recommendations

    def _generate_image_model_recommendations(self):
        """
        Generate ML model recommendations for image datasets.
        Returns a dict with recommended image-compatible models.
        """
        recommendations = {
            'recommended_models': [],
            'not_recommended': [],
            'dataset_characteristics': {},
            'reasoning': []
        }

        # Estimate dataset size (for image datasets we consider number of images)
        estimated_images = getattr(self, 'row_count', 0) or 1000  # Default fallback

        recommendations['dataset_characteristics'] = {
            'estimated_images': estimated_images,
            'dataset_type': 'image',
            'task_type': 'classification'
        }

        # Image model scoring
        model_scores = {}

        # CNN (Basic)
        cnn_score = 85
        if estimated_images >= 1000:
            cnn_score += 5
        if estimated_images >= 5000:
            cnn_score += 5
        model_scores['cnn'] = {
            'score': min(cnn_score, 100),
            'name': 'Convolutional Neural Network (CNN)',
            'reasons': [
                'Designed specifically for image data',
                'Learns spatial features automatically',
                'Good for custom image classification',
                'Full control over architecture'
            ]
        }

        # Sort by score and categorize
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['score'], reverse=True)

        for model_key, model_info in sorted_models:
            model_entry = {
                'model_type': model_key,
                'name': model_info['name'],
                'compatibility_score': model_info['score'],
                'reasons': model_info['reasons']
            }

            if model_info['score'] >= 70:
                recommendations['recommended_models'].append(model_entry)
            else:
                recommendations['not_recommended'].append(model_entry)

        # Add reasoning for image datasets
        recommendations['reasoning'] = [
            f"Image dataset with approximately {estimated_images:,} images",
            "Image classification task detected",
            "Convolutional models required for spatial feature extraction",
        ]

        if estimated_images < 500:
            recommendations['reasoning'].append("Small image dataset - transfer learning strongly recommended")
        elif estimated_images >= 5000:
            recommendations['reasoning'].append("Large image dataset - both custom CNN and transfer learning viable")

        recommendations['reasoning'].append("All recommended models work exclusively with image data")

        return recommendations

    def get_csv_bytes(self):
        """Get CSV file as bytes from MinIO"""
        if not self.minio_csv_key:
            raise ValueError("No CSV file available")
        from core.storage_utils import download_from_minio
        return download_from_minio(self.minio_csv_key)

    def get_csv_path(self):
        """Get CSV file path for pandas - downloads from MinIO to temp file"""
        import tempfile

        if not self.minio_csv_key:
            raise ValueError("No CSV file available")

        from core.storage_utils import download_from_minio
        csv_bytes = download_from_minio(self.minio_csv_key)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='wb')
        temp_file.write(csv_bytes)
        temp_file.close()
        return temp_file.name

    def get_image_directory(self):
        """Get image directory path for PyTorch - downloads from MinIO to temp directory"""
        import tempfile

        if not self.minio_images_prefix:
            raise ValueError("No image directory available")

        from core.storage_utils import download_minio_directory
        temp_dir = tempfile.mkdtemp(prefix=f'dataset_{self.id}_')
        download_minio_directory(self.minio_images_prefix, temp_dir)
        return temp_dir

    def get_zip_bytes(self):
        """Get ZIP file as bytes from MinIO"""
        if not self.minio_zip_key:
            raise ValueError("No ZIP file available")
        from core.storage_utils import download_from_minio
        return download_from_minio(self.minio_zip_key)

    def cleanup_temp_files(self, path):
        """Clean up temporary files created by helper methods"""
        import shutil
        import os

        if path and os.path.exists(path):
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)

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
    prediction_schema = models.JSONField(
        default=dict,
        blank=True,
        help_text="Schema describing required input fields for prediction"
    )
    analytics_data = models.JSONField(
        default=dict,
        blank=True,
        help_text="Detailed analytics data including metrics, confusion matrix, feature importance, etc."
    )

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


class DataSegment(BaseModel):
    """
    Represents a segment/group for manual data labeling and categorization.
    Used for organizing dataset rows into meaningful groups.
    """
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name='segments'
    )
    name = models.CharField(max_length=255, help_text="Segment name (e.g., 'High Value Customers')")
    description = models.TextField(blank=True, null=True, help_text="Description of this segment")
    color = models.CharField(max_length=7, default="#3b82f6", help_text="Hex color for UI visualization")

    # Segment metadata
    row_count = models.IntegerField(default=0, help_text="Number of rows in this segment")
    rules = models.JSONField(
        default=dict,
        blank=True,
        help_text="Optional automated rules for segment assignment (filters, conditions)"
    )

    class Meta:
        ordering = ['created_at']
        unique_together = [['dataset', 'name']]

    def __str__(self):
        return f"{self.dataset.name} - {self.name} ({self.row_count} rows)"


class RowSegmentLabel(BaseModel):
    """
    Maps individual dataset rows to segments.
    Supports manual labeling and automated assignment.
    """
    segment = models.ForeignKey(
        DataSegment,
        on_delete=models.CASCADE,
        related_name='row_labels'
    )
    row_index = models.IntegerField(help_text="Index of the row in the dataset")
    row_data = models.JSONField(default=dict, help_text="Snapshot of row data for quick reference")

    # Assignment metadata
    assigned_by = models.ForeignKey(
        'userauth.CustomUser',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text="User who assigned this label"
    )
    assignment_method = models.CharField(
        max_length=20,
        choices=[
            ('manual', 'Manual'),
            ('rule', 'Rule-based'),
            ('ml', 'ML-based'),
        ],
        default='manual'
    )
    confidence = models.FloatField(
        default=1.0,
        help_text="Confidence score (1.0 for manual, varies for automated)"
    )
    notes = models.TextField(blank=True, null=True, help_text="Optional notes about this assignment")

    class Meta:
        ordering = ['row_index']
        unique_together = [['segment', 'row_index']]
        indexes = [
            models.Index(fields=['row_index']),
            models.Index(fields=['segment', 'row_index']),
        ]

    def __str__(self):
        return f"Row {self.row_index} → {self.segment.name}"
