import logging
import json
from datetime import datetime
from typing import Any, Dict, List
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import viewsets, status
from rest_framework.decorators import action, authentication_classes, permission_classes
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import SessionAuthentication
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.shortcuts import get_object_or_404
from django.contrib.auth.models import AnonymousUser
from django.http import FileResponse, Http404
from rest_framework.decorators import api_view, permission_classes, authentication_classes

from .models.main import Dataset, MLModel, TrainingRun
from .models.choices import ModelStatus
from .serializers import DatasetSerializer, MLModelSerializer, TrainingRunSerializer, DatasetJoinSerializer
from .functions.celery_tasks import train_cnn_task, train_nn_task, train_sklearn_task
from .models.choices import ModelType
from .validators.model_compatibility import ModelCompatibilityValidator

logger = logging.getLogger(__name__)


class DatasetViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing datasets with proper authentication and pagination.
    
    Provides CRUD operations for datasets with user-based filtering.
    Only authenticated users can access their datasets.
    """
    queryset = Dataset.objects.all().order_by('id')
    serializer_class = DatasetSerializer
    parser_classes = (MultiPartParser, FormParser, JSONParser)
    authentication_classes = [JWTAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer) -> None:
        """
        Create a new dataset, ensuring it's associated with the authenticated user.
        
        Args:
            serializer: The dataset serializer instance
            
        Raises:
            ValueError: If user is not authenticated
        """
        if isinstance(self.request.user, AnonymousUser):
            raise ValueError("Anonymous users cannot create datasets")
        serializer.save(created_by=self.request.user)

    def get_queryset(self):
        """
        Filter datasets based on user authentication status.

        Returns:
            QuerySet of datasets accessible to the current user
        """
        logger.info(f"Dataset access - User: {self.request.user}, Authenticated: {self.request.user.is_authenticated}")
        if self.request.user.is_authenticated:
            return Dataset.objects.filter(deleted=False, created_by=self.request.user).order_by('id')
        else:
            return Dataset.objects.none()

    def destroy(self, request, *args, **kwargs):
        """
        Delete a dataset and clean up its associated files from MinIO storage.
        
        Args:
            request: The HTTP request
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Response indicating deletion status
        """
        from core.storage_utils import delete_from_minio, get_minio_client
        import os
        
        instance = self.get_object()
        
        # Clean up MinIO files before deleting the database record
        files_deleted = []
        deletion_errors = []
        
        try:
            # Delete individual files first
            files_to_delete = []
            
            # Add specific file keys
            if instance.minio_csv_key:
                files_to_delete.append(instance.minio_csv_key)
            if instance.minio_zip_key:
                files_to_delete.append(instance.minio_zip_key)
            
            # Delete individual files
            for file_key in files_to_delete:
                try:
                    delete_from_minio(file_key)
                    files_deleted.append(file_key)
                    logger.info(f"Deleted dataset file from storage: {file_key}")
                except Exception as e:
                    deletion_errors.append(f"Failed to delete {file_key}: {str(e)}")
                    logger.warning(f"Failed to delete dataset file {file_key}: {e}")
            
            # Delete entire dataset folder if it exists (for images and other files)
            dataset_id = str(instance.id)
            dataset_folder_prefix = f"datasets/{dataset_id}/"
            
            try:
                # List and delete all objects with this prefix
                client = get_minio_client()
                bucket = os.getenv("MINIO_BUCKET_NAME")
                
                paginator = client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=bucket, Prefix=dataset_folder_prefix)
                
                objects_to_delete = []
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            objects_to_delete.append({'Key': obj['Key']})
                
                # Delete objects in batches
                if objects_to_delete:
                    # Split into batches of 1000 (AWS/MinIO limit)
                    for i in range(0, len(objects_to_delete), 1000):
                        batch = objects_to_delete[i:i+1000]
                        client.delete_objects(
                            Bucket=bucket,
                            Delete={'Objects': batch}
                        )
                        files_deleted.extend([obj['Key'] for obj in batch])
                    
                    logger.info(f"Deleted {len(objects_to_delete)} files from dataset folder: {dataset_folder_prefix}")
                else:
                    logger.info(f"No files found in dataset folder: {dataset_folder_prefix}")
                    
            except Exception as e:
                deletion_errors.append(f"Failed to delete dataset folder: {str(e)}")
                logger.warning(f"Failed to delete dataset folder {dataset_folder_prefix}: {e}")
            
            # Delete the database record
            dataset_name = instance.name
            dataset_id = instance.id
            self.perform_destroy(instance)
            
            logger.info(f"Successfully deleted dataset {dataset_id} ({dataset_name}) and cleaned up {len(files_deleted)} files")
            
            return Response(
                {
                    "message": f"Dataset '{dataset_name}' deleted successfully",
                    "files_deleted": len(files_deleted),
                    "deletion_errors": deletion_errors if deletion_errors else None
                }, 
                status=status.HTTP_204_NO_CONTENT
            )
            
        except Exception as e:
            logger.error(f"Error during dataset deletion: {e}")
            return Response(
                {"error": f"Failed to delete dataset: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['get'])
    def columns(self, request, pk=None) -> Response:
        """
        Get column names for a specific dataset.

        Args:
            request: The HTTP request
            pk: Primary key of the dataset

        Returns:
            Response with column names or error
        """
        dataset = self.get_object()

        try:
            import pandas as pd

            # Read CSV file to get column names from MinIO
            if dataset.minio_csv_key:
                csv_obj = dataset.get_csv_file_object()
                df = pd.read_csv(csv_obj)
                columns = df.columns.tolist()

                return Response({
                    "columns": columns,
                    "count": len(columns)
                })
            else:
                return Response(
                    {"error": "No CSV file associated with this dataset"},
                    status=status.HTTP_400_BAD_REQUEST
                )

        except Exception as e:
            logger.error(f"Error reading dataset columns for dataset {dataset.id}: {e}")
            return Response(
                {"error": f"Failed to read dataset columns: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['get'])
    def column_recommendations(self, request, pk=None) -> Response:
        """
        Get recommendations for which columns to exclude from training.

        Analyzes the dataset and identifies columns that should likely be excluded:
        - Datetime/timestamp columns
        - ID-like columns (patient_id, test_id, etc.)
        - High cardinality categorical columns

        Returns:
            Response with recommended exclusions and reasoning
        """
        dataset = self.get_object()
        target_column = request.query_params.get('target_column')

        try:
            import pandas as pd
            import re

            if not dataset.minio_csv_key:
                return Response(
                    {"error": "No CSV file associated with this dataset"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            csv_obj = dataset.get_csv_file_object()
            df = pd.read_csv(csv_obj)

            recommendations = []

            for col in df.columns:
                if target_column and col == target_column:
                    continue

                reason = None
                confidence = "medium"

                # Check for datetime columns
                if df[col].dtype == 'datetime64[ns]':
                    reason = "Datetime column - not suitable for direct ML training"
                    confidence = "high"
                elif df[col].dtype == 'object':
                    # Try to parse as datetime
                    try:
                        pd.to_datetime(df[col].iloc[:100])
                        reason = "Timestamp string - should be feature-engineered first"
                        confidence = "high"
                    except Exception:
                        pass

                # Check for ID-like columns (high cardinality, unique values)
                if not reason:
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio > 0.95:
                        # Pattern matching for common ID column names
                        if re.search(r'(^|_)(id|key|uuid|guid)($|_)', col, re.IGNORECASE):
                            reason = "ID column - unique identifier with no predictive value"
                            confidence = "high"
                        else:
                            reason = f"High cardinality ({df[col].nunique()} unique values) - likely not useful"
                            confidence = "medium"

                # Check for constant columns
                if not reason and df[col].nunique() == 1:
                    reason = "Constant column - no variance, provides no information"
                    confidence = "high"

                if reason:
                    recommendations.append({
                        "column": col,
                        "reason": reason,
                        "confidence": confidence,
                        "dtype": str(df[col].dtype),
                        "unique_values": int(df[col].nunique()),
                        "sample_values": df[col].head(3).tolist()
                    })

            return Response({
                "total_columns": len(df.columns),
                "recommended_exclusions": recommendations,
                "excluded_count": len(recommendations)
            })

        except Exception as e:
            logger.error(f"Error analyzing columns for dataset {dataset.id}: {e}")
            return Response(
                {"error": f"Failed to analyze columns: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['get'])
    def preview(self, request, pk=None) -> Response:
        """
        Get preview data and analysis for a dataset.

        Query Parameters:
            offset (int): Starting row index for pagination
            limit (int): Number of rows to return

        Returns:
            Response with preview data, column info, and basic statistics
        """
        dataset = self.get_object()

        if not dataset.is_processed and dataset.dataset_type != 'image':
            dataset.analyze_dataset()

        # Different response for image datasets
        if dataset.dataset_type == 'image':
            return Response({
                'preview_data': [],
                'column_info': {},
                'statistics': {
                    'total_images': dataset.row_count or 0,
                    'dataset_size': dataset.file_size or 0
                },
                'data_quality': dataset.data_quality or 'Good',
                'dataset_type': dataset.dataset_type,
                'dataset_purpose': dataset.dataset_purpose,
                'row_count': dataset.row_count,
                'column_count': 0,
                'file_size': dataset.file_size
            })

        # Handle pagination for CSV datasets
        offset = request.query_params.get('offset')
        limit = request.query_params.get('limit')

        logger.info(f"Preview request - RAW offset: '{offset}', RAW limit: '{limit}'")
        logger.info(f"All query params: {dict(request.query_params)}")

        # If pagination parameters are provided, load from CSV dynamically
        if offset is not None or limit is not None:
            offset = int(offset) if offset is not None else 0
            limit = int(limit) if limit is not None else 100

            logger.info(f"Loading paginated data - PARSED offset: {offset}, PARSED limit: {limit}")

            try:
                import pandas as pd
                if dataset.minio_csv_key:
                    csv_obj = dataset.get_csv_file_object()
                    df = pd.read_csv(csv_obj)
                    # Get the requested slice
                    df_slice = df.iloc[offset:offset + limit]
                    preview_data = df_slice.to_dict('records')
                    # Convert any numpy types to Python types
                    preview_data = [
                        {k: (v.item() if hasattr(v, 'item') else v) for k, v in row.items()}
                        for row in preview_data
                    ]
                    logger.info(f"Returning {len(preview_data)} rows from CSV")
                else:
                    # Fall back to stored preview data
                    preview_data = dataset.preview_data[offset:offset + limit]
                    logger.info(f"Returning {len(preview_data)} rows from stored preview_data (no CSV)")
            except Exception as e:
                logger.error(f"Error loading paginated data: {e}")
                # Fall back to stored preview data
                preview_data = dataset.preview_data[offset:offset + limit] if dataset.preview_data else []
                logger.info(f"Error occurred, returning {len(preview_data)} rows from fallback")
        else:
            # No pagination params, return stored preview data
            preview_data = dataset.preview_data
            logger.info(f"No pagination params, returning {len(preview_data)} stored rows")

        return Response({
            'preview_data': preview_data,
            'column_info': dataset.column_info,
            'statistics': dataset.statistics,
            'data_quality': dataset.data_quality,
            'dataset_type': dataset.dataset_type,
            'dataset_purpose': dataset.dataset_purpose,
            'row_count': dataset.row_count,
            'column_count': dataset.column_count,
            'file_size': dataset.file_size
        })

    @action(detail=True, methods=['get'])
    def quality_report(self, request, pk=None) -> Response:
        """
        Get comprehensive data quality report.

        Returns:
            Response with detailed quality analysis
        """
        dataset = self.get_object()

        if not dataset.is_processed and dataset.dataset_type != 'image':
            dataset.analyze_dataset()

        # Different response for image datasets
        if dataset.dataset_type == 'image':
            return Response({
                'quality_report': {
                    'total_images': dataset.row_count or 0,
                    'total_size': dataset.file_size or 0,
                    'minio_images_prefix': dataset.minio_images_prefix,
                    'completeness_score': 100.0 if dataset.is_processed else 0.0
                },
                'data_quality': dataset.data_quality or 'Good',
                'processing_errors': dataset.processing_errors,
                'recommendations': []
            })

        return Response({
            'quality_report': dataset.quality_report,
            'data_quality': dataset.data_quality,
            'processing_errors': dataset.processing_errors,
            'recommendations': self._get_quality_recommendations(dataset)
        })

    @action(detail=True, methods=['get'])
    def statistics(self, request, pk=None) -> Response:
        """
        Get detailed statistical analysis of the dataset.

        Returns:
            Response with comprehensive statistics
        """
        dataset = self.get_object()

        if not dataset.is_processed and dataset.dataset_type != 'image':
            dataset.analyze_dataset()

        # Different response for image datasets
        if dataset.dataset_type == 'image':
            return Response({
                'basic_stats': {
                    'total_images': dataset.row_count or 0,
                    'total_size': dataset.file_size or 0,
                    'avg_image_size': (dataset.file_size / dataset.row_count) if dataset.row_count else 0,
                },
                'column_stats': {},
                'correlations': {},
                'distributions': {}
            })

        return Response({
            'basic_stats': dataset.statistics,
            'column_stats': dataset.column_info,
            'correlations': self._calculate_correlations(dataset),
            'distributions': self._get_distributions(dataset)
        })

    @action(detail=True, methods=['get'])
    def export(self, request, pk=None) -> Response:
        """
        Export dataset as CSV file download.

        Returns:
            FileResponse with CSV content
        """
        dataset = self.get_object()

        try:
            import pandas as pd
            from django.http import HttpResponse

            if not dataset.minio_csv_key:
                return Response(
                    {"error": "No CSV file available for this dataset"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Get CSV data from MinIO
            csv_obj = dataset.get_csv_file_object()
            df = pd.read_csv(csv_obj)

            # Create HTTP response with CSV
            response = HttpResponse(content_type='text/csv')
            filename = f"{dataset.name.replace(' ', '_')}.csv"
            response['Content-Disposition'] = f'attachment; filename="{filename}"'

            # Write CSV to response
            df.to_csv(response, index=False)

            return response

        except Exception as e:
            logger.error(f"Error exporting dataset {dataset.id}: {e}")
            return Response(
                {"error": f"Failed to export dataset: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['post'])
    def compare(self, request) -> Response:
        """
        Compare multiple datasets side by side.

        Request body:
            dataset_ids: List of dataset IDs to compare

        Returns:
            Comparison data for all selected datasets
        """
        dataset_ids = request.data.get('dataset_ids', [])

        if not dataset_ids or len(dataset_ids) < 2:
            return Response(
                {"error": "At least 2 datasets are required for comparison"},
                status=status.HTTP_400_BAD_REQUEST
            )

        if len(dataset_ids) > 5:
            return Response(
                {"error": "Cannot compare more than 5 datasets at once"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            datasets = Dataset.objects.filter(id__in=dataset_ids, deleted=False)

            if datasets.count() != len(dataset_ids):
                return Response(
                    {"error": "Some datasets not found"},
                    status=status.HTTP_404_NOT_FOUND
                )

            comparison_data = []
            for dataset in datasets:
                data = {
                    'id': str(dataset.id),
                    'name': dataset.name,
                    'dataset_type': dataset.dataset_type,
                    'dataset_purpose': dataset.dataset_purpose,
                    'data_quality': dataset.data_quality,
                    'quality_score': dataset.quality_score,
                    'row_count': dataset.row_count,
                    'column_count': dataset.column_count,
                    'file_size': dataset.file_size,
                    'file_size_formatted': dataset.file_size_formatted,
                    'created_at': dataset.created_at.isoformat(),
                    'is_processed': dataset.is_processed,
                }

                # Add column info if CSV dataset
                if dataset.minio_csv_key:
                    try:
                        import pandas as pd
                        csv_obj = dataset.get_csv_file_object()
                        df = pd.read_csv(csv_obj)

                        data['columns'] = list(df.columns)
                        data['numeric_columns'] = list(df.select_dtypes(include=['number']).columns)
                        data['categorical_columns'] = list(df.select_dtypes(include=['object']).columns)
                        data['missing_values'] = df.isnull().sum().to_dict()
                        data['duplicate_rows'] = int(df.duplicated().sum())
                    except Exception as e:
                        logger.error(f"Error reading dataset {dataset.id} for comparison: {e}")
                        data['columns'] = []

                comparison_data.append(data)

            return Response({
                'datasets': comparison_data,
                'comparison_count': len(comparison_data)
            })

        except Exception as e:
            logger.error(f"Error comparing datasets: {e}")
            return Response(
                {"error": f"Failed to compare datasets: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['get'])
    def sample(self, request, pk=None) -> Response:
        """
        Get a random sample of the dataset.

        Query params:
            - size: number of rows to sample (default: 100)
            - random_state: seed for reproducible sampling
        """
        dataset = self.get_object()

        if not dataset.minio_csv_key:
            return Response(
                {"error": "No CSV file associated with this dataset"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            import pandas as pd

            size = int(request.query_params.get('size', 100))
            random_state = request.query_params.get('random_state', 42)

            csv_obj = dataset.get_csv_file_object()
            df = pd.read_csv(csv_obj)

            if len(df) <= size:
                sample_df = df
            else:
                sample_df = df.sample(n=size, random_state=int(random_state))

            return Response({
                'sample_data': sample_df.to_dict('records'),
                'sample_size': len(sample_df),
                'total_size': len(df),
                'columns': list(df.columns)
            })

        except Exception as e:
            logger.error(f"Error sampling dataset {pk}: {e}", exc_info=True)
            return Response(
                {"error": f"Failed to sample dataset: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST
            )

    @action(detail=True, methods=['post'])
    def reanalyze(self, request, pk=None) -> Response:
        """
        Force reanalysis of the dataset.

        Returns:
            Response with updated analysis results
        """
        dataset = self.get_object()

        # Check if dataset has a CSV file
        if not dataset.minio_csv_key:
            return Response({
                'error': 'Dataset has no CSV file to analyze'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Run analysis
        dataset.analyze_dataset()

        # Refresh from database to get updated values
        dataset.refresh_from_db()

        return Response({
            'message': 'Dataset reanalyzed successfully',
            'statistics': dataset.statistics,
            'data_quality': dataset.data_quality,
            'is_processed': dataset.is_processed,
            'processing_errors': dataset.processing_errors
        })

    @action(detail=False, methods=['post'])
    def join(self, request) -> Response:
        """
        Join two CSV datasets using the specified join type on key columns.

        Request body:
        {
            "left_dataset_id": "uuid",
            "right_dataset_id": "uuid",
            "left_key_column": "patient_id",
            "right_key_column": "patient_id",
            "join_type": "inner",  // "inner", "left", "right", or "outer"
            "result_name": "Patients + Blood Tests",
            "result_description": "Combined patient and blood test data"
        }

        Returns:
            Response with the newly created joined dataset
        """
        import pandas as pd
        import io
        from datetime import datetime
        from core.storage_utils import upload_to_minio

        # Validate request data
        serializer = DatasetJoinSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated_data = serializer.validated_data
        left_dataset = validated_data['left_dataset']
        right_dataset = validated_data['right_dataset']
        left_key = validated_data['left_key_column']
        right_key = validated_data['right_key_column']
        join_type = validated_data.get('join_type', 'inner')
        result_name = validated_data['result_name']
        result_description = validated_data.get('result_description', '')

        # Human-readable join type names
        join_type_names = {
            'inner': 'Inner',
            'left': 'Left Outer',
            'right': 'Right Outer',
            'outer': 'Full Outer'
        }
        join_type_display = join_type_names.get(join_type, 'Inner')

        try:
            # Load both CSVs from MinIO
            left_csv = left_dataset.get_csv_file_object()
            right_csv = right_dataset.get_csv_file_object()

            left_df = pd.read_csv(left_csv)
            right_df = pd.read_csv(right_csv)

            # Validate key columns exist
            if left_key not in left_df.columns:
                return Response(
                    {"error": f"Column '{left_key}' not found in left dataset"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            if right_key not in right_df.columns:
                return Response(
                    {"error": f"Column '{right_key}' not found in right dataset"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Handle column name conflicts with suffixes
            left_cols = set(left_df.columns) - {left_key}
            right_cols = set(right_df.columns) - {right_key}
            overlapping = left_cols & right_cols

            # Perform join with specified type
            if left_key == right_key:
                result_df = pd.merge(
                    left_df, right_df,
                    on=left_key,
                    how=join_type,
                    suffixes=('_left', '_right')
                )
            else:
                result_df = pd.merge(
                    left_df, right_df,
                    left_on=left_key,
                    right_on=right_key,
                    how=join_type,
                    suffixes=('_left', '_right')
                )

            # For inner join, warn if no results
            if len(result_df) == 0 and join_type == 'inner':
                return Response(
                    {"error": "Join produced no results. Check that key columns have matching values."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Create CSV in memory
            csv_buffer = io.StringIO()
            result_df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            csv_bytes = csv_content.encode('utf-8')
            csv_buffer.close()

            # Upload to MinIO
            filename = f"joined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            minio_key = f"datasets/{request.user.id}/{filename}"
            upload_to_minio(csv_bytes, minio_key, content_type='text/csv')

            # Build join metadata
            join_metadata = {
                'join_type': join_type,
                'join_type_display': join_type_display,
                'left_dataset_id': str(left_dataset.id),
                'left_dataset_name': left_dataset.name,
                'left_key_column': left_key,
                'left_row_count': len(left_df),
                'right_dataset_id': str(right_dataset.id),
                'right_dataset_name': right_dataset.name,
                'right_key_column': right_key,
                'right_row_count': len(right_df),
                'result_row_count': len(result_df),
                'overlapping_columns': list(overlapping),
                'joined_at': datetime.now().isoformat()
            }

            # Create Dataset with lineage tracking
            new_dataset = Dataset(
                name=result_name,
                description=result_description or f"{join_type_display} join of '{left_dataset.name}' and '{right_dataset.name}'",
                created_by=request.user,
                minio_csv_key=minio_key,
                row_count=len(result_df),
                column_count=len(result_df.columns),
                file_size=len(csv_bytes),
                join_metadata=join_metadata
            )
            new_dataset.save()

            # Add source datasets for lineage
            new_dataset.source_datasets.add(left_dataset, right_dataset)

            # Return success response
            return Response({
                "success": True,
                "message": "Datasets joined successfully",
                "dataset": DatasetSerializer(new_dataset).data,
                "join_stats": {
                    "left_rows": len(left_df),
                    "right_rows": len(right_df),
                    "result_rows": len(result_df),
                    "result_columns": len(result_df.columns),
                    "join_type": join_type_display
                }
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(f"Error joining datasets: {e}", exc_info=True)
            return Response(
                {"error": f"Failed to join datasets: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['get'])
    def extraction_status(self, request, pk=None) -> Response:
        """
        Get the extraction status for an image dataset.

        Returns:
            Response with extraction status, progress, and any errors
        """
        dataset = self.get_object()

        if dataset.dataset_type != 'image':
            return Response(
                {"error": "This endpoint is only for image datasets"},
                status=status.HTTP_400_BAD_REQUEST
            )

        return Response({
            'dataset_id': dataset.id,
            'dataset_name': dataset.name,
            'is_processed': dataset.is_processed,
            'minio_images_prefix': dataset.minio_images_prefix,
            'row_count': dataset.row_count,  # Image count
            'processing_errors': dataset.processing_errors,
            'status': 'completed' if dataset.is_processed else (
                'error' if dataset.processing_errors and 'failed' in dataset.processing_errors.lower()
                else 'processing'
            )
        })

    @action(detail=True, methods=['post'])
    def retry_extraction(self, request, pk=None) -> Response:
        """
        Retry extraction for a failed image dataset.

        Returns:
            Response indicating extraction retry status
        """
        from .functions.celery_tasks import extract_image_dataset_task

        dataset = self.get_object()

        if dataset.dataset_type != 'image':
            return Response(
                {"error": "This endpoint is only for image datasets"},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not dataset.minio_zip_key:
            return Response(
                {"error": "No image ZIP file found"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Reset status
            dataset.processing_errors = "Retrying extraction..."
            dataset.is_processed = False
            dataset.save(update_fields=['processing_errors', 'is_processed'])

            # Queue extraction task
            extract_image_dataset_task.delay(dataset.id)

            return Response({
                'message': 'Extraction retry queued successfully',
                'dataset_id': dataset.id,
                'status': 'processing'
            })

        except Exception as e:
            logger.error(f"Error retrying extraction for dataset {pk}: {e}", exc_info=True)
            return Response(
                {"error": f"Failed to retry extraction: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['get'])
    def images(self, request, pk=None) -> Response:
        """
        Get list of images in an image dataset.

        Query params:
            - page: page number (default: 1)
            - page_size: images per page (default: 50)

        Returns:
            Response with image file list and metadata
        """
        dataset = self.get_object()

        if dataset.dataset_type != 'image':
            return Response(
                {"error": "This endpoint is only for image datasets"},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not dataset.minio_images_prefix:
            return Response(
                {"error": "Image dataset has not been extracted yet"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            import os
            from core.storage_utils import get_minio_client

            # Get pagination parameters
            page = int(request.query_params.get('page', 1))
            page_size = int(request.query_params.get('page_size', 50))

            # Supported image extensions
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

            # Get all image files from MinIO
            client = get_minio_client()
            bucket = os.getenv("MINIO_BUCKET_NAME")
            prefix = dataset.minio_images_prefix

            image_files = []
            paginator = client.get_paginator('list_objects_v2')

            for page_data in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page_data.get('Contents', []):
                    key = obj['Key']
                    # Get relative path from the prefix
                    relative_path = key[len(prefix):]
                    if not relative_path:
                        continue

                    # Check if it's an image file
                    filename = relative_path.split('/')[-1]
                    ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
                    if ext not in image_extensions:
                        continue

                    # Get the parent directory (classification label/class)
                    path_parts = relative_path.strip('/').split('/')
                    parent_dir = path_parts[-2] if len(path_parts) > 1 else None

                    image_files.append({
                        'filename': filename,
                        'path': relative_path,
                        'size': obj['Size'],
                        'extension': ext,
                        'class': parent_dir
                    })

            # Sort by filename
            image_files.sort(key=lambda x: x['filename'])

            # Paginate
            total_images = len(image_files)
            total_pages = (total_images + page_size - 1) // page_size
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_images = image_files[start_idx:end_idx]

            return Response({
                'images': paginated_images,
                'total_images': total_images,
                'page': page,
                'page_size': page_size,
                'total_pages': total_pages,
                'dataset_id': dataset.id,
                'dataset_name': dataset.name
            })

        except Exception as e:
            logger.error(f"Error listing images for dataset {pk}: {e}", exc_info=True)
            return Response(
                {"error": f"Failed to list images: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def _get_quality_recommendations(self, dataset):
        """Generate data quality recommendations"""
        recommendations = []

        if not dataset.quality_report:
            return recommendations

        quality_report = dataset.quality_report

        # High null columns
        if quality_report.get('highly_null_columns'):
            recommendations.append({
                'type': 'missing_data',
                'severity': 'high',
                'message': f"Consider removing or imputing columns with >50% missing values: {', '.join(quality_report['highly_null_columns'])}",
                'columns': quality_report['highly_null_columns']
            })

        # Duplicate rows
        if quality_report.get('duplicate_rows', 0) > 0:
            recommendations.append({
                'type': 'duplicates',
                'severity': 'medium',
                'message': f"Remove {quality_report['duplicate_rows']} duplicate rows to improve data quality",
                'count': quality_report['duplicate_rows']
            })

        # Data quality score
        completeness = quality_report.get('completeness_score', 0)
        if completeness < 80:
            recommendations.append({
                'type': 'completeness',
                'severity': 'high',
                'message': f"Data completeness is {completeness:.1f}%. Consider data cleaning or imputation strategies",
                'score': completeness
            })

        # Column-specific recommendations
        for col, info in dataset.column_info.items():
            if info.get('unique_count') == 1:
                recommendations.append({
                    'type': 'constant_column',
                    'severity': 'low',
                    'message': f"Column '{col}' has only one unique value and may not be useful for analysis",
                    'column': col
                })

        return recommendations

    def _calculate_correlations(self, dataset):
        """Calculate correlation matrix for numeric columns"""
        if not dataset.minio_csv_key:
            return {}

        try:
            import pandas as pd

            csv_obj = dataset.get_csv_file_object()
            df = pd.read_csv(csv_obj)
            numeric_df = df.select_dtypes(include=['number'])

            if len(numeric_df.columns) < 2:
                return {}

            corr_matrix = numeric_df.corr()

            # Convert to format suitable for visualization
            correlations = []
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:  # Avoid duplicates
                        correlations.append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': float(corr_matrix.loc[col1, col2])
                        })

            return {
                'matrix': corr_matrix.to_dict(),
                'pairs': correlations,
                'strong_correlations': [
                    pair for pair in correlations
                    if abs(pair['correlation']) > 0.7
                ]
            }

        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return {}

    def _get_distributions(self, dataset):
        """Get distribution data for visualization"""
        if not dataset.minio_csv_key:
            return {}

        try:
            import pandas as pd
            import numpy as np

            csv_obj = dataset.get_csv_file_object()
            df = pd.read_csv(csv_obj)
            distributions = {}

            # Numeric columns - histograms
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols[:10]:  # Limit to first 10 for performance
                series = df[col].dropna()
                if len(series) > 0:
                    hist, bins = np.histogram(series, bins=20)
                    distributions[col] = {
                        'type': 'histogram',
                        'bins': bins.tolist(),
                        'counts': hist.tolist(),
                        'mean': float(series.mean()),
                        'std': float(series.std())
                    }

            # Categorical columns - value counts
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols[:10]:  # Limit to first 10
                series = df[col].dropna()
                if len(series) > 0:
                    value_counts = series.value_counts().head(20)  # Top 20 values
                    distributions[col] = {
                        'type': 'categorical',
                        'values': value_counts.index.tolist(),
                        'counts': value_counts.values.tolist(),
                        'total_unique': int(series.nunique())
                    }

            return distributions

        except Exception as e:
            logger.error(f"Error calculating distributions: {e}")
            return {}

    @action(detail=True, methods=['get'])
    def compatible_models(self, request, pk=None) -> Response:
        """
        Get list of compatible model types for this dataset.

        Returns:
            Response with compatible models and their details
        """
        dataset = self.get_object()

        try:
            compatible = ModelCompatibilityValidator.get_compatible_models_for_dataset_type(
                dataset.dataset_type
            )

            model_details = []
            for model_type in compatible:
                info = ModelCompatibilityValidator.get_model_info(model_type)
                model_details.append({
                    'model_type': model_type,
                    'complexity': info.get('complexity'),
                    'training_speed': info.get('training_speed'),
                    'interpretability': info.get('interpretability'),
                    'min_samples': info.get('min_samples'),
                    'compatible_purposes': info.get('compatible_purposes', [])
                })

            return Response({
                'dataset_id': dataset.id,
                'dataset_name': dataset.name,
                'dataset_type': dataset.dataset_type,
                'compatible_models': model_details,
                'total_compatible': len(model_details)
            })

        except Exception as e:
            logger.error(f"Error getting compatible models for dataset {pk}: {e}")
            return Response(
                {"error": f"Failed to get compatible models: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['get'])
    def recommended_models(self, request, pk=None) -> Response:
        """
        Get AI-recommended model types for this dataset based on characteristics.

        Returns:
            Response with ranked model recommendations
        """
        dataset = self.get_object()

        try:
            # Use stored recommendations if available (generated during dataset analysis)
            if dataset.model_recommendations and dataset.model_recommendations.get('recommended_models'):
                stored_recs = dataset.model_recommendations
                formatted_recommendations = []

                for rec in stored_recs.get('recommended_models', []):
                    formatted_recommendations.append({
                        'model_type': rec.get('model_type'),
                        'name': rec.get('name'),
                        'compatibility_score': rec.get('compatibility_score'),
                        'is_compatible': rec.get('compatibility_score', 0) >= 70,
                        'warnings': [],
                        'recommendations': rec.get('reasons', []),
                        'complexity': 'medium',
                        'training_speed': 'fast' if rec.get('model_type') in ['logistic_regression', 'decision_tree'] else 'medium',
                        'interpretability': 'high' if rec.get('model_type') in ['logistic_regression', 'decision_tree'] else 'medium'
                    })

                return Response({
                    'dataset_id': dataset.id,
                    'dataset_name': dataset.name,
                    'dataset_type': dataset.dataset_type,
                    'dataset_purpose': dataset.dataset_purpose,
                    'row_count': dataset.row_count,
                    'column_count': dataset.column_count,
                    'recommendations': formatted_recommendations,
                    'dataset_characteristics': stored_recs.get('dataset_characteristics', {}),
                    'reasoning': stored_recs.get('reasoning', [])
                })

            # Fallback to ModelCompatibilityValidator if no stored recommendations
            recommendations = ModelCompatibilityValidator.get_recommended_models(dataset, top_n=5)

            formatted_recommendations = []
            for model_type, score, validation in recommendations:
                model_info = ModelCompatibilityValidator.get_model_info(model_type)
                formatted_recommendations.append({
                    'model_type': model_type,
                    'compatibility_score': score,
                    'is_compatible': validation['is_compatible'],
                    'warnings': validation['warnings'],
                    'recommendations': validation['recommendations'],
                    'complexity': model_info.get('complexity'),
                    'training_speed': model_info.get('training_speed'),
                    'interpretability': model_info.get('interpretability')
                })

            return Response({
                'dataset_id': dataset.id,
                'dataset_name': dataset.name,
                'dataset_type': dataset.dataset_type,
                'dataset_purpose': dataset.dataset_purpose,
                'row_count': dataset.row_count,
                'column_count': dataset.column_count,
                'recommendations': formatted_recommendations
            })

        except Exception as e:
            logger.error(f"Error getting model recommendations for dataset {pk}: {e}")
            return Response(
                {"error": f"Failed to get model recommendations: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class MLModelViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing ML models with proper authentication and pagination.
    
    Provides CRUD operations for ML models with user-based filtering.
    Only authenticated users can access their models.
    """
    queryset = MLModel.objects.all().order_by('id')
    serializer_class = MLModelSerializer
    authentication_classes = [JWTAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer) -> None:
        """
        Create a new ML model, ensuring it's associated with the authenticated user.
        
        Args:
            serializer: The ML model serializer instance
            
        Raises:
            ValueError: If user is not authenticated
        """
        if isinstance(self.request.user, AnonymousUser):
            raise ValueError("Anonymous users cannot create ML models")
        serializer.save(created_by=self.request.user)

    def get_queryset(self):
        """
        Filter ML models based on user authentication status.

        Returns:
            QuerySet of ML models accessible to the current user
        """
        logger.info(f"MLModel access - User: {self.request.user}, Authenticated: {self.request.user.is_authenticated}")
        if self.request.user.is_authenticated:
            return MLModel.objects.filter(deleted=False, created_by=self.request.user).order_by('id')
        else:
            return MLModel.objects.none()

    def destroy(self, request, *args, **kwargs):
        """
        Delete a model and clean up its associated files from MinIO storage.
        
        Args:
            request: The HTTP request
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Response indicating deletion status
        """
        from core.storage_utils import delete_from_minio
        
        instance = self.get_object()
        
        # Clean up MinIO files before deleting the database record
        files_deleted = []
        deletion_errors = []
        
        try:
            # Delete main model file if exists
            if instance.model_file and instance.model_file.name:
                try:
                    delete_from_minio(instance.model_file.name)
                    files_deleted.append(instance.model_file.name)
                    logger.info(f"Deleted model file from storage: {instance.model_file.name}")
                except Exception as e:
                    deletion_errors.append(f"Failed to delete model file: {str(e)}")
                    logger.warning(f"Failed to delete model file {instance.model_file.name}: {e}")
            
            # Delete additional model files (encoder, scaler, etc.) if they exist
            model_id = str(instance.id)
            additional_files = [
                f"trained_models/{model_id}_encoder.joblib",
                f"trained_models/{model_id}_scaler.joblib", 
                f"trained_models/{model_id}_label_encoder.joblib"
            ]
            
            for file_path in additional_files:
                try:
                    delete_from_minio(file_path)
                    files_deleted.append(file_path)
                    logger.info(f"Deleted additional model file: {file_path}")
                except Exception as e:
                    # Don't log errors for files that might not exist
                    pass
            
            # Clear cache entries
            from django.core.cache import cache
            cache_key = f"training_logs_{instance.id}"
            cache.delete(cache_key)
            
            # Delete the database record
            model_name = instance.name
            model_id = instance.id
            self.perform_destroy(instance)
            
            logger.info(f"Successfully deleted model {model_id} ({model_name}) and cleaned up {len(files_deleted)} files")
            
            return Response(
                {
                    "message": f"Model '{model_name}' deleted successfully",
                    "files_deleted": len(files_deleted),
                    "deletion_errors": deletion_errors if deletion_errors else None
                }, 
                status=status.HTTP_204_NO_CONTENT
            )
            
        except Exception as e:
            logger.error(f"Error during model deletion: {e}")
            return Response(
                {"error": f"Failed to delete model: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['get'])
    def validate_compatibility(self, request, pk=None) -> Response:
        """
        Validate if the model type is compatible with its dataset.

        Returns:
            Response with validation results, warnings, and recommendations
        """
        model = self.get_object()

        try:
            validation = ModelCompatibilityValidator.validate_compatibility(
                model.model_type,
                model.dataset
            )

            return Response({
                'model_id': model.id,
                'model_name': model.name,
                'model_type': model.model_type,
                'dataset_id': model.dataset.id,
                'dataset_name': model.dataset.name,
                'dataset_type': model.dataset.dataset_type,
                'validation': validation
            })

        except Exception as e:
            logger.error(f"Error validating compatibility for model {pk}: {e}")
            return Response(
                {"error": f"Failed to validate compatibility: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['post'])
    def check_compatibility(self, request) -> Response:
        """
        Check compatibility before creating a model.

        Request body:
            {
                "model_type": "random_forest",
                "dataset_id": "uuid-here"
            }

        Returns:
            Response with validation results
        """
        model_type = request.data.get('model_type')
        dataset_id = request.data.get('dataset_id')

        if not model_type or not dataset_id:
            return Response(
                {"error": "Both model_type and dataset_id are required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            dataset = Dataset.objects.get(id=dataset_id)
            validation = ModelCompatibilityValidator.validate_compatibility(
                model_type,
                dataset
            )

            return Response({
                'model_type': model_type,
                'dataset_id': dataset.id,
                'dataset_name': dataset.name,
                'dataset_type': dataset.dataset_type,
                'validation': validation
            })

        except Dataset.DoesNotExist:
            return Response(
                {"error": "Dataset not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Error checking compatibility: {e}")
            return Response(
                {"error": f"Failed to check compatibility: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['post'])
    def train(self, request, pk=None) -> Response:
        """
        Start training for a specific ML model.
        
        Args:
            request: The HTTP request
            pk: Primary key of the model to train
            
        Returns:
            Response indicating training start status or error
        """
        model = self.get_object()
        
        if model.status == ModelStatus.TRAINING:
            return Response(
                {"error": "Model is already training"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Validate dataset type matches model type
        is_cnn = model.model_type == ModelType.CNN
        is_image_dataset = model.dataset.is_image_dataset or bool(model.dataset.minio_images_prefix)
        has_csv = bool(model.dataset.minio_csv_key)

        if is_cnn and not is_image_dataset:
            return Response(
                {"error": "CNN models require an image dataset. The selected dataset appears to be a tabular dataset."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not is_cnn and not has_csv:
            return Response(
                {"error": "This model type requires a tabular dataset with a CSV file. The selected dataset appears to be an image dataset."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            model.status = ModelStatus.TRAINING
            model.save()

            # Create or get training run
            run, _ = TrainingRun.objects.get_or_create(model=model)
            run.add_entry(status=ModelStatus.TRAINING)

            # Try to queue task with Celery, fallback to synchronous execution
            try:
                if is_image_dataset:
                    train_cnn_task.delay(model.id)
                elif model.model_type == ModelType.NEURAL_NETWORK:
                    train_nn_task.delay(model.id)
                else:
                    train_sklearn_task.delay(model.id)

                logger.info(f"Training task queued for model {model.id}")

            except Exception as celery_error:
                logger.warning(f"Celery not available, falling back to synchronous training: {celery_error}")

                # Fallback to synchronous training
                try:
                    if model.dataset.is_image_dataset:
                        train_cnn_task(model.id)
                    elif model.model_type == ModelType.NEURAL_NETWORK:
                        train_nn_task(model.id)
                    else:
                        train_sklearn_task(model.id)

                    logger.info(f"Synchronous training completed for model {model.id}")

                except Exception as training_error:
                    logger.error(f"Training failed: {training_error}")
                    model.status = ModelStatus.FAILED
                    model.training_log = f"Training failed: {training_error}"
                    model.save()
                    run.add_entry(status=ModelStatus.FAILED, error=str(training_error))
                    raise

            return Response({
                "message": f"Training started for model '{model.name}'",
                "status": model.status
            })

        except Exception as e:
            model.status = ModelStatus.FAILED
            model.training_log = f"Error starting training: {e}"
            model.save()
            run.add_entry(status=ModelStatus.FAILED, error=str(e))
            
            return Response(
                {"error": f"Failed to start training: {e}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['post'])
    def force_train(self, request, pk=None) -> Response:
        """
        Force synchronous training (bypass Celery) for development/testing.

        Args:
            request: The HTTP request
            pk: Primary key of the model to train

        Returns:
            Response indicating training result
        """
        from .functions.celery_tasks import train_sklearn_task, train_nn_task, train_cnn_task

        model = self.get_object()

        if model.status == ModelStatus.TRAINING:
            return Response(
                {"error": "Model is already training"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            model.status = ModelStatus.TRAINING
            model.save()

            # Create or get training run
            run, _ = TrainingRun.objects.get_or_create(model=model)
            run.add_entry(status=ModelStatus.TRAINING)

            # Execute training synchronously
            logger.info(f"Force training model {model.id} synchronously")

            if model.dataset.is_image_dataset:
                result = train_cnn_task(model.id)
            elif model.model_type == ModelType.NEURAL_NETWORK:
                result = train_nn_task(model.id)
            else:
                result = train_sklearn_task(model.id)

            # Refresh model from database
            model.refresh_from_db()

            return Response({
                "message": f"Training completed for model '{model.name}'",
                "status": model.status,
                "accuracy": model.accuracy,
                "training_log": model.training_log
            })

        except Exception as e:
            logger.error(f"Force training failed for model {model.id}: {e}")
            model.status = ModelStatus.FAILED
            model.training_log = f"Force training failed: {e}"
            model.save()
            run.add_entry(status=ModelStatus.FAILED, error=str(e))

            return Response(
                {"error": f"Training failed: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['get'])
    def training_status(self, request, pk=None) -> Response:
        """
        Get training status for a specific ML model.

        Args:
            request: The HTTP request
            pk: Primary key of the model

        Returns:
            Response with training status and log information
        """
        model = self.get_object()
        training_run = getattr(model, 'training_run', None)

        return Response({
            "status": model.status,
            "training_log": model.training_log,
            "history": training_run.history if training_run else []
        })

    @action(detail=True, methods=['get'])
    def training_logs(self, request, pk=None) -> Response:
        """
        Get real-time detailed training logs for a specific ML model.

        This endpoint retrieves structured, detailed logs from the training process
        including data loading, preprocessing, model architecture, epoch-by-epoch
        progress, and evaluation metrics.

        Args:
            request: The HTTP request
            pk: Primary key of the model

        Returns:
            Response with detailed training logs, metrics, and progress information
        """
        from django.core.cache import cache

        model = self.get_object()
        cache_key = f"training_logs_{model.id}"

        # Get logs from cache (updated in real-time during training)
        cached_logs = cache.get(cache_key, [])

        # Parse metrics from logs if available
        metrics = {
            'epochs': [],
            'losses': [],
            'accuracies': [],
            'val_accuracies': []
        }

        for log_entry in cached_logs:
            # Extract epoch number from messages like "✓ Round 5 of 10 Complete:"
            message = log_entry.get('message', '')
            if 'round' in message.lower() and 'complete' in message.lower():
                import re
                match = re.search(r'round\s+(\d+)\s+of\s+(\d+)', message, re.IGNORECASE)
                if match:
                    epoch_num = int(match.group(1))
                    if epoch_num not in metrics['epochs']:
                        metrics['epochs'].append(epoch_num)

            if log_entry.get('data'):
                data = log_entry['data']
                if 'train_loss' in data:
                    try:
                        metrics['losses'].append(float(data['train_loss']))
                    except (ValueError, TypeError):
                        pass
                if 'train_accuracy' in data:
                    try:
                        acc_str = data['train_accuracy'].replace('%', '')
                        metrics['accuracies'].append(float(acc_str))
                    except (ValueError, TypeError, AttributeError):
                        pass
                if 'val_accuracy' in data:
                    try:
                        acc_str = data['val_accuracy'].replace('%', '')
                        metrics['val_accuracies'].append(float(acc_str))
                    except (ValueError, TypeError, AttributeError):
                        pass

        return Response({
            "model_id": str(model.id),
            "model_name": model.name,
            "status": model.status,
            "logs": cached_logs,
            "metrics": metrics,
            "total_logs": len(cached_logs),
            "training_log_text": model.training_log or "No logs available yet",
            "accuracy": model.accuracy,
            "created_at": model.created_at.isoformat() if model.created_at else None
        })

    @action(detail=True, methods=['get'])
    def prediction_info(self, request, pk=None) -> Response:
        """
        Get prediction schema and tips for making predictions with this model.

        Args:
            request: The HTTP request
            pk: Primary key of the model

        Returns:
            Response with prediction schema, tips, and example usage
        """
        model = self.get_object()

        if model.status != ModelStatus.COMPLETE:
            return Response(
                {
                    "error": "Model is not ready for predictions",
                    "status": model.status,
                    "message": "Please wait for the model to complete training"
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        # Build comprehensive prediction info
        prediction_info = {
            "model_id": model.id,
            "model_name": model.name,
            "model_type": model.model_type,
            "schema": model.prediction_schema,
            "tips": []
        }

        # Add model-specific tips
        if model.dataset.is_image_dataset:
            prediction_info["tips"] = [
                "Upload an image file in one of the supported formats",
                f"Image will be automatically resized to {model.prediction_schema.get('input_size', 64)}x{model.prediction_schema.get('input_size', 64)}",
                f"Model will classify the image into one of these categories: {', '.join(model.prediction_schema.get('output_classes', []))}",
                "Supported formats: JPG, PNG, BMP, GIF"
            ]
            prediction_info["example_request"] = {
                "data": "path/to/image.jpg or upload image file"
            }
        else:
            # Tabular data model
            input_features = model.prediction_schema.get('input_features', [])
            prediction_info["tips"] = [
                f"Provide values for all {len(input_features)} required features",
                "You can send a single record (dict) or multiple records (list of dicts)",
                "Feature names must match exactly as shown in the schema",
                "Missing features will cause an error"
            ]

            if 'output_classes' in model.prediction_schema:
                prediction_info["tips"].append(
                    f"Model will predict one of: {', '.join(model.prediction_schema.get('output_classes', []))}"
                )

            # Create example with proper structure
            example_single = {feature: 0.0 for feature in input_features}
            prediction_info["example_request"] = {
                "single_prediction": {
                    "data": example_single
                },
                "batch_prediction": {
                    "data": [example_single, example_single]
                }
            }

        return Response(prediction_info)

    @action(detail=True, methods=['post'])
    def predict(self, request, pk=None) -> Response:
        """
        Make predictions using a trained ML model.

        Args:
            request: The HTTP request containing prediction data
            pk: Primary key of the model to use for prediction

        Returns:
            Response with prediction results or error
        """
        from .functions.prediction import predict_nn, predict_sklearn_model, predict_cnn
        import pandas as pd
        import json

        model = self.get_object()

        if model.status != ModelStatus.COMPLETE:
            return Response(
                {"error": "Model must be trained and completed before making predictions"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Check for uploaded image file first
            if 'image' in request.FILES:
                # Handle image file upload
                # Check if this is an image model (CNN) or dataset is marked as image dataset
                is_image_model = (
                    model.model_type == ModelType.CNN or
                    model.dataset.is_image_dataset or
                    bool(model.dataset.minio_images_prefix)
                )

                if not is_image_model:
                    return Response(
                        {"error": "This model does not support image predictions"},
                        status=status.HTTP_400_BAD_REQUEST
                    )

                import tempfile
                import os
                from PIL import Image

                # Save uploaded file temporarily
                uploaded_file = request.FILES['image']

                # Validate it's an image
                try:
                    img = Image.open(uploaded_file)
                    img.verify()
                    uploaded_file.seek(0)  # Reset file pointer after verify
                except Exception:
                    return Response(
                        {"error": "Invalid image file"},
                        status=status.HTTP_400_BAD_REQUEST
                    )

                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    for chunk in uploaded_file.chunks():
                        tmp_file.write(chunk)
                    tmp_path = tmp_file.name

                try:
                    # Make prediction
                    prediction = predict_cnn(model, tmp_path)

                    return Response({
                        "prediction": prediction,
                        "filename": uploaded_file.name
                    })
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            # Get prediction data from request
            data = request.data.get('data')
            if not data:
                return Response(
                    {"error": "No prediction data provided"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Handle different input types
            if isinstance(data, str):
                # For image path or single text input
                if model.dataset.is_image_dataset:
                    prediction = predict_cnn(model, data)
                    return Response({"prediction": prediction})
                else:
                    return Response(
                        {"error": "Invalid data format for non-image model"},
                        status=status.HTTP_400_BAD_REQUEST
                    )

            elif isinstance(data, (list, dict)):
                # Convert to DataFrame
                if isinstance(data, dict):
                    df = pd.DataFrame([data])
                else:
                    df = pd.DataFrame(data)

                # Make predictions based on model type
                if model.model_type == ModelType.NEURAL_NETWORK:
                    result_df = predict_nn(model, df)
                else:
                    result_df = predict_sklearn_model(model, df)

                # Convert result to JSON
                predictions = result_df['prediction'].tolist() if 'prediction' in result_df.columns else []

                return Response({
                    "predictions": predictions,
                    "count": len(predictions)
                })

            else:
                return Response(
                    {"error": "Invalid data format. Expected string, dict, or list"},
                    status=status.HTTP_400_BAD_REQUEST
                )

        except Exception as e:
            logger.error(f"Prediction error for model {model.id}: {e}")
            return Response(
                {"error": f"Prediction failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['post'])
    def test(self, request, pk=None) -> Response:
        """
        Test a trained ML model with test dataset.

        Args:
            request: The HTTP request
            pk: Primary key of the model to test

        Returns:
            Response with test results or error
        """
        model = self.get_object()

        if model.status != ModelStatus.COMPLETE:
            return Response(
                {"error": "Model must be trained and completed before testing"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # For now, return the existing accuracy if available
            # This can be expanded to run actual testing on new data
            test_results = {
                "model_name": model.name,
                "model_type": model.model_type,
                "accuracy": model.accuracy if model.accuracy else 0.0,
                "status": model.status,
                "test_timestamp": model.updated_at.isoformat()
            }

            # If training log contains test metrics, parse and include them
            if model.training_log:
                try:
                    # Try to parse JSON metrics from training log
                    import re
                    accuracy_match = re.search(r'accuracy[:\s]+([0-9.]+)', model.training_log)
                    if accuracy_match:
                        test_results["accuracy"] = float(accuracy_match.group(1))
                except Exception:
                    pass

            return Response(test_results)

        except Exception as e:
            logger.error(f"Test error for model {model.id}: {e}")
            return Response(
                {"error": f"Model testing failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def _generate_model_metrics(self, model) -> Dict[str, Any]:
        """Extract real metrics from training logs and saved analytics data."""
        from django.core.cache import cache
        from app.models.choices import REGRESSION_MODELS
        import re

        # Determine if this is a regression model - check both analytics flag and model_type
        is_regression_model = model.model_type in [m.value for m in REGRESSION_MODELS]

        # Try to use saved analytics data first
        if model.analytics_data and (model.analytics_data.get('accuracy') is not None or model.analytics_data.get('r2_score') is not None):
            analytics = model.analytics_data
            # Use is_regression from analytics, or fall back to checking model_type
            is_regression = analytics.get('is_regression', is_regression_model)

            base_metrics = {
                "accuracy": analytics.get('accuracy', model.accuracy or 0.0),
                "feature_importance": analytics.get('feature_importance', []),
                "training_history": analytics.get('training_history', []),
                "prediction_distribution": analytics.get('prediction_distribution', []),
                "additional_metrics": {
                    "training_samples": analytics.get('training_samples', 0),
                    "test_samples": analytics.get('test_samples', 0),
                },
                "is_regression": is_regression,
            }

            if is_regression:
                # Add regression-specific metrics
                # For backwards compatibility: if r2_score is not stored, use accuracy as r2_score
                r2_score = analytics.get('r2_score')
                if r2_score is None:
                    r2_score = analytics.get('accuracy', 0.0)
                base_metrics.update({
                    "r2_score": r2_score,
                    "rmse": analytics.get('rmse', 0.0),
                    "mae": analytics.get('mae', 0.0),
                    "mse": analytics.get('mse', 0.0),
                    "target_stats": analytics.get('target_stats', {}),
                })
                # Update base_metrics to reflect it's a regression model
                base_metrics["is_regression"] = True
            else:
                # Add classification-specific metrics
                base_metrics.update({
                    "precision": analytics.get('precision', 0.0),
                    "recall": analytics.get('recall', 0.0),
                    "f1_score": analytics.get('f1_score', 0.0),
                    "confusion_matrix": analytics.get('confusion_matrix', []),
                    "class_report": analytics.get('class_report', {}),
                })

            return base_metrics

        # Fallback: Try to get real training data from cache
        cache_key = f"training_logs_{model.id}"
        cached_logs = cache.get(cache_key, [])

        # Extract real training history from logs
        training_history = []
        epoch_data = {}  # Store data by epoch number

        for log_entry in cached_logs:
            message = log_entry.get('message', '')
            data = log_entry.get('data', {})

            # Look for "Round X of Y Complete:" messages
            match = re.search(r'round\s+(\d+)\s+of\s+(\d+)\s+complete', message, re.IGNORECASE)
            if match and data:
                epoch_num = int(match.group(1))

                # Extract metrics from this epoch
                epoch_metrics = {"epoch": epoch_num}

                if 'train_loss' in data:
                    try:
                        epoch_metrics['loss'] = float(data['train_loss'])
                    except (ValueError, TypeError):
                        pass

                if 'train_accuracy' in data:
                    try:
                        acc_str = data['train_accuracy'].replace('%', '')
                        epoch_metrics['accuracy'] = float(acc_str) / 100.0
                    except (ValueError, TypeError, AttributeError):
                        pass

                if 'val_loss' in data:
                    try:
                        epoch_metrics['val_loss'] = float(data['val_loss'])
                    except (ValueError, TypeError):
                        pass

                if 'val_accuracy' in data:
                    try:
                        acc_str = data['val_accuracy'].replace('%', '')
                        epoch_metrics['val_accuracy'] = float(acc_str) / 100.0
                    except (ValueError, TypeError, AttributeError):
                        pass

                epoch_data[epoch_num] = epoch_metrics

        # Convert epoch_data dict to sorted list
        if epoch_data:
            training_history = [epoch_data[epoch] for epoch in sorted(epoch_data.keys())]

        # If no real training history data, return None - no mock data
        if not training_history:
            return {
                "accuracy": model.accuracy if model.accuracy else 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "confusion_matrix": [],
                "feature_importance": [],
                "training_history": [],
                "prediction_distribution": [],
                "additional_metrics": {}
            }

        # Use real base accuracy from last epoch
        if training_history and 'val_accuracy' in training_history[-1]:
            base_accuracy = training_history[-1]['val_accuracy']
        elif model.accuracy:
            base_accuracy = model.accuracy
        else:
            base_accuracy = 0.0

        return {
            "accuracy": round(base_accuracy, 4),
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "confusion_matrix": [],
            "feature_importance": [],
            "training_history": training_history,
            "prediction_distribution": [],
            "additional_metrics": {}
        }

    def _generate_model_statistics(self, model) -> Dict[str, Any]:
        """Return real model statistics from stored data."""
        from core.storage_utils import get_minio_client
        import os

        # Get model file size from MinIO if available
        model_size = "N/A"
        model_file_path = model.model_file.name if model.model_file else None
        if model_file_path:
            try:
                client = get_minio_client()
                response = client.head_object(
                    Bucket=os.getenv("MINIO_BUCKET_NAME", "mediafiles"),
                    Key=model_file_path
                )
                size_bytes = response.get('ContentLength', 0)
                if size_bytes > 0:
                    if size_bytes < 1024:
                        model_size = f"{size_bytes} B"
                    elif size_bytes < 1024 * 1024:
                        model_size = f"{size_bytes / 1024:.1f} KB"
                    else:
                        model_size = f"{size_bytes / (1024 * 1024):.2f} MB"
            except Exception as e:
                logger.debug(f"Could not get model file size: {e}")

        # Get dataset info
        dataset = model.dataset
        dataset_rows = 0
        dataset_features = 0
        data_quality_score = 0.0
        target_distribution = []

        if dataset:
            dataset_rows = dataset.row_count or 0
            dataset_features = dataset.column_count or 0
            if dataset.statistics:
                data_quality_score = dataset.statistics.get('completeness', 0.0)

        # Get training info from analytics_data
        analytics = model.analytics_data or {}
        training_samples = analytics.get('training_samples', 0)
        test_samples = analytics.get('test_samples', 0)

        # Get prediction distribution for target
        if analytics.get('prediction_distribution'):
            target_distribution = analytics.get('prediction_distribution', [])

        # Get training time from analytics_data first, fallback to TrainingRun
        training_time = "N/A"

        # Check analytics_data for training time (saved during training)
        if analytics.get('training_time'):
            training_time = analytics.get('training_time')
        else:
            # Fallback: Calculate from TrainingRun timestamps
            try:
                latest_run = model.training_runs.order_by('-created_at').first()
                if latest_run and latest_run.created_at and latest_run.updated_at:
                    duration = (latest_run.updated_at - latest_run.created_at).total_seconds()
                    if duration < 60:
                        training_time = f"{duration:.1f}s"
                    elif duration < 3600:
                        training_time = f"{duration / 60:.1f}m"
                    else:
                        training_time = f"{duration / 3600:.1f}h"
            except Exception:
                pass

        # Get framework based on model type
        nn_types = ['neural_network', 'cnn', 'transfer_learning']
        framework = "PyTorch" if model.model_type in nn_types else "scikit-learn"

        return {
            "total_predictions": 0,  # Would need prediction logging to track this
            "avg_prediction_time": 0.0,
            "model_size": model_size,
            "training_time": training_time,
            "last_trained": model.updated_at.isoformat() if model.updated_at else None,
            "framework": framework,
            "dataset_info": {
                "total_rows": dataset_rows,
                "total_features": dataset_features,
                "training_samples": training_samples,
                "test_samples": test_samples,
                "target_distribution": target_distribution,
                "data_quality_score": data_quality_score,
            },
            "model_info": {
                "model_type": model.model_type,
                "accuracy": model.accuracy,
                "version": "1.0.0",
                "framework": framework,
                "hyperparameters": model.training_config or {}
            }
        }

    @action(detail=True, methods=['get'])
    def metrics(self, request, pk=None) -> Response:
        """
        Get comprehensive performance metrics for a specific ML model.

        Args:
            request: The HTTP request
            pk: Primary key of the model

        Returns:
            Response with detailed performance metrics
        """
        model = self.get_object()

        try:
            metrics_data = self._generate_model_metrics(model)

            logger.info(f"Generated metrics for model {model.id} ({model.name})")

            return Response({
                "model_id": model.id,
                "model_name": model.name,
                "model_type": model.model_type,
                "status": model.status,
                "metrics": metrics_data,
                "generated_at": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Error generating metrics for model {model.id}: {e}")
            return Response(
                {"error": f"Failed to generate model metrics: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['get'])
    def statistics(self, request, pk=None) -> Response:
        """
        Get comprehensive usage statistics for a specific ML model.

        Args:
            request: The HTTP request
            pk: Primary key of the model

        Returns:
            Response with detailed usage statistics
        """
        model = self.get_object()

        try:
            stats_data = self._generate_model_statistics(model)

            logger.info(f"Generated statistics for model {model.id} ({model.name})")

            return Response({
                "model_id": model.id,
                "model_name": model.name,
                "model_type": model.model_type,
                "status": model.status,
                "statistics": stats_data,
                "generated_at": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Error generating statistics for model {model.id}: {e}")
            return Response(
                {"error": f"Failed to generate model statistics: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['get'])
    def analytics(self, request, pk=None) -> Response:
        """
        Get complete analytics package (metrics + statistics) for a model.

        Args:
            request: The HTTP request
            pk: Primary key of the model

        Returns:
            Response with comprehensive analytics data
        """
        model = self.get_object()

        try:
            metrics_data = self._generate_model_metrics(model)
            stats_data = self._generate_model_statistics(model)

            logger.info(f"Generated complete analytics for model {model.id} ({model.name})")

            return Response({
                "model_id": model.id,
                "model_name": model.name,
                "model_type": model.model_type,
                "status": model.status,
                "created_at": model.created_at.isoformat() if model.created_at else None,
                "updated_at": model.updated_at.isoformat() if model.updated_at else None,
                "metrics": metrics_data,
                "statistics": stats_data,
                "generated_at": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Error generating analytics for model {model.id}: {e}")
            return Response(
                {"error": f"Failed to generate model analytics: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['get'])  
    def test_download(self, request, pk=None) -> Response:
        """Test endpoint to verify routing works"""
        logger.info(f"Test download action called for model ID: {pk}")
        return Response({"message": f"Test download for model {pk}"})

    @action(detail=True, methods=['get'])
    def download(self, request, pk=None) -> Response:
        """
        Download the trained model file.

        Args:
            request: The HTTP request
            pk: Primary key of the model

        Returns:
            FileResponse with the model file
        """
        logger.info(f"Download action called for model ID: {pk}")
        
        try:
            model = self.get_object()
            logger.info(f"Model found: {model.id} - {model.name}, Status: {model.status}")
        except Exception as e:
            logger.error(f"Error getting model object: {e}")
            return Response(
                {"error": "Model not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        if model.status not in ['complete', 'completed']:
            return Response(
                {"error": "Model has not been trained yet"},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not model.model_file or not model.model_file.name:
            logger.error(f"No model file path set. model.model_file: {model.model_file}, model.model_file.name: {getattr(model.model_file, 'name', 'None')}")
            return Response(
                {"error": "No model file available for download"},
                status=status.HTTP_404_NOT_FOUND
            )

        logger.info(f"Attempting to download file: '{model.model_file.name}'")

        try:
            from core.storage_utils import download_from_minio
            import io

            logger.info(f"Attempting to download model file: {model.model_file.name}")

            # Get the model file from MinIO
            model_data = download_from_minio(model.model_file.name)

            if not model_data:
                logger.error(f"Model file not found in MinIO: {model.model_file.name}")
                return Response(
                    {"error": "Model file not found in storage"},
                    status=status.HTTP_404_NOT_FOUND
                )

            # Create response with the file
            response = FileResponse(
                io.BytesIO(model_data),
                content_type='application/octet-stream'
            )
            # Determine file extension based on model type
            if model.model_type in ['cnn', 'neural_network']:
                file_extension = '.pt'  # PyTorch models
            else:
                file_extension = '.joblib'  # Sklearn models
                
            filename = f"{model.name.replace(' ', '_')}_{model.model_type}{file_extension}"
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            response['Content-Length'] = len(model_data)

            logger.info(f"Model file downloaded: {model.id} ({model.name})")
            return response

        except Exception as e:
            logger.error(f"Error downloading model {model.id}: {e}")
            return Response(
                {"error": f"Failed to download model file: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['post'])
    def reset(self, request, pk=None) -> Response:
        """
        Reset a trained model to its initial state.

        Clears all training data (accuracy, analytics, logs, model file)
        and resets status to pending, allowing the model to be retrained.

        Args:
            request: The HTTP request
            pk: Primary key of the model

        Returns:
            Response indicating reset status
        """
        from django.core.cache import cache
        from core.storage_utils import delete_from_minio

        model = self.get_object()

        if model.status == ModelStatus.TRAINING:
            return Response(
                {"error": "Cannot reset a model that is currently training"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Delete model file from MinIO if exists
            if model.model_file and model.model_file.name:
                try:
                    delete_from_minio(model.model_file.name)
                    logger.info(f"Deleted model file from storage: {model.model_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete model file: {e}")

            # Clear cached training logs
            cache_key = f"training_logs_{model.id}"
            cache.delete(cache_key)

            # Reset model fields (use empty string/dict for fields with NOT NULL constraints)
            model.status = ModelStatus.PENDING
            model.accuracy = None
            model.training_log = ''
            model.analytics_data = {}
            model.model_file = ''
            model.training_config = model.training_config or {}  # Keep config but clear it if None
            model.save(update_fields=['status', 'accuracy', 'training_log', 'analytics_data', 'model_file', 'training_config', 'updated_at'])

            # Reset training run if exists
            if hasattr(model, 'training_run') and model.training_run:
                model.training_run.history = []
                model.training_run.save(update_fields=['history'])

            logger.info(f"Model {model.id} ({model.name}) has been reset")

            return Response({
                "message": f"Model '{model.name}' has been reset successfully",
                "status": model.status
            })

        except Exception as e:
            logger.error(f"Error resetting model {model.id}: {e}")
            return Response(
                {"error": f"Failed to reset model: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['patch'])
    def update_dataset(self, request, pk=None) -> Response:
        """
        Update the dataset for a model.

        This allows changing the dataset before retraining.
        The model must be in pending or failed state.

        Args:
            request: The HTTP request with dataset_id
            pk: Primary key of the model

        Returns:
            Response indicating update status
        """
        model = self.get_object()

        if model.status == ModelStatus.TRAINING:
            return Response(
                {"error": "Cannot change dataset while model is training"},
                status=status.HTTP_400_BAD_REQUEST
            )

        dataset_id = request.data.get('dataset_id')
        target_column = request.data.get('target_column')

        if not dataset_id and not target_column:
            return Response(
                {"error": "dataset_id or target_column is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            update_fields = ['updated_at']

            # Update dataset if provided
            if dataset_id:
                new_dataset = Dataset.objects.get(id=dataset_id, deleted=False)
                model.dataset = new_dataset
                update_fields.append('dataset')

            # Update target column if provided
            if target_column:
                model.target_column = target_column
                update_fields.append('target_column')

            model.save(update_fields=update_fields)

            # Build response message
            messages = []
            response_data = {}
            if dataset_id:
                messages.append(f"Dataset updated to '{new_dataset.name}'")
                response_data["dataset_id"] = str(new_dataset.id)
                response_data["dataset_name"] = new_dataset.name
                logger.info(f"Model {model.id} dataset updated to {new_dataset.id}")
            if target_column:
                messages.append(f"Target column updated to '{target_column}'")
                response_data["target_column"] = target_column
                logger.info(f"Model {model.id} target column updated to {target_column}")

            response_data["message"] = ". ".join(messages)
            return Response(response_data)

        except Dataset.DoesNotExist:
            return Response(
                {"error": "Dataset not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Error updating dataset for model {model.id}: {e}")
            return Response(
                {"error": f"Failed to update dataset: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['patch'])
    def update_config(self, request, pk=None) -> Response:
        """
        Update training configuration for a model.

        Allows updating model parameters like max_iter, test_size, model_type, etc.
        The model must not be currently training.

        Args:
            request: The HTTP request with config parameters
            pk: Primary key of the model

        Returns:
            Response indicating update status
        """
        model = self.get_object()

        if model.status == ModelStatus.TRAINING:
            return Response(
                {"error": "Cannot update configuration while model is training"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            updated_fields = []

            # Update model_type if provided
            model_type = request.data.get('model_type')
            if model_type:
                model.model_type = model_type
                updated_fields.append('model_type')

            # Update target_column if provided
            target_column = request.data.get('target_column')
            if target_column:
                model.target_column = target_column
                updated_fields.append('target_column')

            # Update training_config if provided
            training_config = request.data.get('training_config')
            if training_config:
                # Merge with existing config
                existing_config = model.training_config or {}
                existing_config.update(training_config)
                model.training_config = existing_config
                updated_fields.append('training_config')

            # Individual config fields that can be updated directly
            config_fields = ['max_iter', 'test_size', 'n_estimators', 'max_depth',
                           'learning_rate', 'batch_size', 'epochs', 'hidden_layers']

            for field in config_fields:
                if field in request.data:
                    if not model.training_config:
                        model.training_config = {}
                    model.training_config[field] = request.data[field]
                    if 'training_config' not in updated_fields:
                        updated_fields.append('training_config')

            if not updated_fields:
                return Response(
                    {"error": "No valid fields provided for update"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            updated_fields.append('updated_at')
            model.save(update_fields=updated_fields)

            logger.info(f"Model {model.id} config updated: {updated_fields}")

            return Response({
                "message": f"Model configuration updated successfully",
                "model_id": str(model.id),
                "updated_fields": updated_fields,
                "training_config": model.training_config
            })

        except Exception as e:
            logger.error(f"Error updating config for model {model.id}: {e}")
            return Response(
                {"error": f"Failed to update configuration: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['post'])
    def compare(self, request) -> Response:
        """
        Compare multiple ML models side by side.

        Request body:
            model_ids: List of model IDs to compare

        Returns:
            Comparison data for all selected models
        """
        model_ids = request.data.get('model_ids', [])

        if not model_ids or len(model_ids) < 2:
            return Response(
                {"error": "At least 2 models are required for comparison"},
                status=status.HTTP_400_BAD_REQUEST
            )

        if len(model_ids) > 5:
            return Response(
                {"error": "Cannot compare more than 5 models at once"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            models = MLModel.objects.filter(id__in=model_ids)

            if models.count() != len(model_ids):
                return Response(
                    {"error": "Some models not found"},
                    status=status.HTTP_404_NOT_FOUND
                )

            comparison_data = []
            for model in models:
                # Parse analytics data
                analytics = model.analytics_data or {}

                data = {
                    'id': str(model.id),
                    'name': model.name,
                    'model_type': model.model_type,
                    'status': model.status,
                    'dataset_name': model.dataset.name if model.dataset else None,
                    'target_column': model.target_column,
                    'accuracy': model.accuracy,
                    'test_size': model.test_size,
                    'random_state': model.random_state,
                    'max_iter': model.max_iter,
                    'created_at': model.created_at.isoformat(),
                    'training_config': model.training_config or {},
                }

                # Add metrics if available
                if analytics.get('classification_report'):
                    report = analytics['classification_report']
                    data['precision'] = report.get('weighted avg', {}).get('precision')
                    data['recall'] = report.get('weighted avg', {}).get('recall')
                    data['f1_score'] = report.get('weighted avg', {}).get('f1-score')

                # Add confusion matrix if available
                if analytics.get('confusion_matrix'):
                    data['confusion_matrix'] = analytics['confusion_matrix']

                # Add feature importance if available
                if analytics.get('feature_importance'):
                    data['feature_importance'] = analytics['feature_importance'][:10]  # Top 10

                comparison_data.append(data)

            return Response({
                'models': comparison_data,
                'comparison_count': len(comparison_data)
            })

        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return Response(
                {"error": f"Failed to compare models: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class TrainingRunViewSet(viewsets.ReadOnlyModelViewSet):
    """
    Read-only ViewSet for training run information.
    
    Provides access to training run data filtered by user ownership.
    """
    queryset = TrainingRun.objects.all().order_by('id')
    serializer_class = TrainingRunSerializer
    authentication_classes = [JWTAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """
        Filter training runs based on model ownership.
        
        Returns:
            QuerySet of training runs for models owned by the current user
        """
        return TrainingRun.objects.filter(model__created_by=self.request.user).order_by('id')


# Standalone view for serving images (not part of ViewSet)
@api_view(['GET'])
@authentication_classes([JWTAuthentication, SessionAuthentication])
@permission_classes([IsAuthenticated])
def serve_image(request, dataset_id, image_path):
    """
    Standalone view to serve image files from datasets (MinIO storage).

    Args:
        request: HTTP request
        dataset_id: Dataset UUID
        image_path: Relative path to image within dataset

    Query params:
        thumbnail: if 'true', serve thumbnail version

    Returns:
        HttpResponse with image data
    """
    from pathlib import Path
    from urllib.parse import unquote
    from django.http import HttpResponse

    try:
        # Get the dataset
        dataset = Dataset.objects.get(id=dataset_id, deleted=False)

        # Check permissions
        if not request.user.is_authenticated:
            return Response({"error": "Authentication required"}, status=401)

        # URL decode the image path
        image_path = unquote(image_path)

        # Check if thumbnail is requested
        use_thumbnail = request.query_params.get('thumbnail', 'false').lower() == 'true'

        # Serve from MinIO
        from core.storage_utils import download_from_minio

        if use_thumbnail:
            minio_key = f"{dataset.minio_images_prefix}thumbnails/{Path(image_path).with_suffix('.jpg')}"
        else:
            minio_key = f"{dataset.minio_images_prefix}{image_path}"

        try:
            image_bytes = download_from_minio(minio_key)
        except Exception:
            # Fallback to original if thumbnail doesn't exist
            if use_thumbnail:
                minio_key = f"{dataset.minio_images_prefix}{image_path}"
                image_bytes = download_from_minio(minio_key)
            else:
                raise

        # Determine content type
        content_types = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.gif': 'image/gif',
            '.bmp': 'image/bmp', '.tiff': 'image/tiff',
            '.webp': 'image/webp'
        }
        content_type = content_types.get(Path(image_path).suffix.lower(), 'image/jpeg')

        response = HttpResponse(image_bytes, content_type=content_type)
        response['Content-Disposition'] = f'inline; filename="{Path(image_path).name}"'
        response['Cache-Control'] = 'public, max-age=3600'
        return response

    except Dataset.DoesNotExist:
        raise Http404("Dataset not found")
    except Exception as e:
        return Response({"error": str(e)}, status=500)


@api_view(['POST'])
@authentication_classes([JWTAuthentication, SessionAuthentication])
@permission_classes([IsAuthenticated])
def batch_serve_images(request, dataset_id):
    """
    Batch serve multiple images in a single request for better performance.

    Args:
        request: HTTP request with JSON body containing image paths
        dataset_id: Dataset UUID

    Request body:
        {
            "image_paths": ["path/to/image1.jpg", "path/to/image2.png"],
            "thumbnail": true/false (optional, default: false)
        }

    Returns:
        JSON response with base64-encoded images
    """
    import base64
    from pathlib import Path
    from urllib.parse import unquote

    try:
        # Get the dataset
        dataset = Dataset.objects.get(id=dataset_id, deleted=False)

        # Check permissions
        if not request.user.is_authenticated:
            return Response({"error": "Authentication required"}, status=401)

        # Get request data
        image_paths = request.data.get('image_paths', [])
        use_thumbnail = request.data.get('thumbnail', False)

        if not image_paths or not isinstance(image_paths, list):
            return Response(
                {"error": "image_paths must be a non-empty array"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Limit to 50 images per request
        if len(image_paths) > 50:
            return Response(
                {"error": "Maximum 50 images per request"},
                status=status.HTTP_400_BAD_REQUEST
            )

        results = []

        # Content type mapping
        content_types = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.gif': 'image/gif',
            '.bmp': 'image/bmp', '.tiff': 'image/tiff',
            '.webp': 'image/webp'
        }

        for img_path in image_paths:
            try:
                # URL decode the image path
                img_path = unquote(img_path)

                # Serve from MinIO
                from core.storage_utils import download_from_minio

                if use_thumbnail:
                    minio_key = f"{dataset.minio_images_prefix}thumbnails/{Path(img_path).with_suffix('.jpg')}"
                else:
                    minio_key = f"{dataset.minio_images_prefix}{img_path}"

                try:
                    image_data = download_from_minio(minio_key)
                except Exception:
                    # Fallback to original if thumbnail doesn't exist
                    if use_thumbnail:
                        minio_key = f"{dataset.minio_images_prefix}{img_path}"
                        image_data = download_from_minio(minio_key)
                    else:
                        raise

                encoded_data = base64.b64encode(image_data).decode('utf-8')
                content_type = content_types.get(Path(img_path).suffix.lower(), 'image/jpeg')

                results.append({
                    'path': img_path,
                    'data': f'data:{content_type};base64,{encoded_data}',
                    'content_type': content_type,
                    'size': len(image_data)
                })

            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                results.append({
                    'path': img_path,
                    'error': str(e),
                    'data': None
                })

        return Response({
            'dataset_id': dataset_id,
            'images': results,
            'total': len(results),
            'thumbnail': use_thumbnail
        })

    except Dataset.DoesNotExist:
        return Response(
            {"error": "Dataset not found"},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        logger.error(f"Error in batch_serve_images: {e}", exc_info=True)
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@authentication_classes([JWTAuthentication, SessionAuthentication])
@permission_classes([IsAuthenticated])
def get_gpu_info(request):
    """
    Get information about available CUDA GPUs.

    Returns:
        Response with GPU availability and device information
    """
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_info = {
            'cuda_available': cuda_available,
            'gpu_count': 0,
            'gpus': [],
            'cuda_version': None,
            'cudnn_version': None,
        }

        if cuda_available:
            gpu_info['gpu_count'] = torch.cuda.device_count()
            gpu_info['cuda_version'] = torch.version.cuda
            gpu_info['cudnn_version'] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info['gpus'].append({
                    'index': i,
                    'name': props.name,
                    'total_memory_gb': round(props.total_memory / (1024**3), 2),
                    'major': props.major,
                    'minor': props.minor,
                    'multi_processor_count': props.multi_processor_count,
                })

        return Response(gpu_info)

    except ImportError:
        return Response({
            'cuda_available': False,
            'gpu_count': 0,
            'gpus': [],
            'error': 'PyTorch not installed'
        })
    except Exception as e:
        logger.error(f"Error getting GPU info: {e}", exc_info=True)
        return Response({
            'cuda_available': False,
            'gpu_count': 0,
            'gpus': [],
            'error': str(e)
        })
