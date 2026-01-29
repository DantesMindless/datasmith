"""
Views for data segmentation and manual labeling operations.
"""
import pandas as pd
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from django.db import transaction
from django.shortcuts import get_object_or_404

from core.views import BaseAuthApiView
from rest_framework.authentication import SessionAuthentication
from rest_framework_simplejwt.authentication import JWTAuthentication
from app.models.main import Dataset, DataSegment, RowSegmentLabel
from app.models.choices import ModelType
from app.serializers import (
    DataSegmentSerializer,
    RowSegmentLabelSerializer,
    BulkRowSegmentLabelSerializer
)
from app.functions.segmentation import perform_clustering, determine_optimal_clusters


class DataSegmentViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing data segments.
    Supports CRUD operations and statistics.
    """
    queryset = DataSegment.objects.all()
    serializer_class = DataSegmentSerializer
    authentication_classes = [JWTAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Filter segments by dataset if specified"""
        queryset = DataSegment.objects.filter(dataset__created_by=self.request.user)
        dataset_id = self.request.query_params.get('dataset_id')

        if dataset_id:
            queryset = queryset.filter(dataset_id=dataset_id)

        # Order by creation date
        return queryset.order_by('-created_at')

    def perform_create(self, serializer):
        """Set created_by when creating a segment"""
        serializer.save(created_by=self.request.user)

    @action(detail=True, methods=['get'])
    def statistics(self, request, pk=None):
        """Get detailed statistics for a segment"""
        segment = self.get_object()
        dataset = segment.dataset

        # Get labeled rows
        labels = segment.row_labels.all()
        row_indices = [label.row_index for label in labels]

        # Load dataset to calculate statistics
        if not dataset.minio_csv_key:
            return Response(
                {"error": "Dataset has no CSV file"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            csv_obj = dataset.get_csv_file_object()
            df = pd.read_csv(csv_obj)

            # Filter to segmented rows
            if len(row_indices) > 0:
                segment_df = df.iloc[row_indices]

                # Calculate statistics
                stats = {
                    'total_rows': len(row_indices),
                    'percentage_of_dataset': (len(row_indices) / len(df)) * 100,
                    'numeric_stats': {},
                    'categorical_stats': {},
                }

                # Numeric column statistics
                numeric_cols = segment_df.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    stats['numeric_stats'][col] = {
                        'mean': float(segment_df[col].mean()),
                        'median': float(segment_df[col].median()),
                        'std': float(segment_df[col].std()),
                        'min': float(segment_df[col].min()),
                        'max': float(segment_df[col].max()),
                    }

                # Categorical column statistics
                categorical_cols = segment_df.select_dtypes(include=['object']).columns
                for col in list(categorical_cols)[:10]:  # Limit to first 10
                    value_counts = segment_df[col].value_counts().head(5)
                    stats['categorical_stats'][col] = {
                        str(k): int(v) for k, v in value_counts.items()
                    }

                return Response(stats)
            else:
                return Response({
                    'total_rows': 0,
                    'percentage_of_dataset': 0,
                    'numeric_stats': {},
                    'categorical_stats': {},
                })

        except Exception as e:
            return Response(
                {"error": f"Failed to calculate statistics: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['post'])
    def export(self, request, pk=None):
        """Export segment data to a new CSV file"""
        segment = self.get_object()
        dataset = segment.dataset

        if not dataset.minio_csv_key:
            return Response(
                {"error": "Dataset has no CSV file"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Get labeled rows
            labels = segment.row_labels.all()
            row_indices = [label.row_index for label in labels]

            if len(row_indices) == 0:
                return Response(
                    {"error": "Segment has no labeled rows"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Load dataset and filter
            csv_obj = dataset.get_csv_file_object()
            df = pd.read_csv(csv_obj)
            segment_df = df.iloc[row_indices]

            # Create new dataset
            from django.core.files.base import ContentFile
            import io

            csv_buffer = io.StringIO()
            segment_df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()

            # Upload CSV to MinIO
            from core.storage_utils import upload_to_minio
            import uuid
            temp_id = uuid.uuid4()
            filename = f"{segment.name.replace(' ', '_')}_export.csv"
            minio_key = f"datasets/{temp_id}/csv/{filename}"
            upload_to_minio(csv_content.encode('utf-8'), minio_key, content_type='text/csv')

            new_dataset = Dataset(
                name=f"{dataset.name} - {segment.name}",
                description=f"Exported segment: {segment.description or segment.name}",
                created_by=request.user,
                dataset_purpose=dataset.dataset_purpose,
                minio_csv_key=minio_key,
                row_count=len(row_indices),
                column_count=len(segment_df.columns),
                file_size=len(csv_content.encode('utf-8')),
            )
            new_dataset.save()

            return Response({
                'success': True,
                'dataset_id': str(new_dataset.id),
                'dataset_name': new_dataset.name,
                'rows_exported': len(row_indices),
            })

        except Exception as e:
            return Response(
                {"error": f"Export failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class RowSegmentLabelViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing row-to-segment label assignments.
    """
    queryset = RowSegmentLabel.objects.all()
    serializer_class = RowSegmentLabelSerializer
    authentication_classes = [JWTAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get_paginate_by(self):
        """Disable pagination when row_indices is specified"""
        if self.request.query_params.get('row_indices'):
            return None
        return super().get_paginate_by()

    @property
    def paginator(self):
        """Disable pagination when row_indices is specified"""
        if self.request.query_params.get('row_indices'):
            return None
        return super().paginator

    def get_queryset(self):
        """Filter labels by segment, dataset, or specific row indices"""
        queryset = RowSegmentLabel.objects.filter(segment__dataset__created_by=self.request.user)
        segment_id = self.request.query_params.get('segment_id')
        dataset_id = self.request.query_params.get('dataset_id')
        row_indices_param = self.request.query_params.get('row_indices')

        print(f"[SegmentLabels] Query params: segment_id={segment_id}, dataset_id={dataset_id}, row_indices={row_indices_param}")

        if segment_id:
            queryset = queryset.filter(segment_id=segment_id)
        elif dataset_id:
            queryset = queryset.filter(segment__dataset_id=dataset_id)

        # Filter by specific row indices if provided (for pagination efficiency)
        if row_indices_param:
            try:
                row_indices = [int(idx) for idx in row_indices_param.split(',')]
                queryset = queryset.filter(row_index__in=row_indices)
            except ValueError:
                pass  # Invalid indices, ignore

        final_count = queryset.count()
        print(f"[SegmentLabels] Returning {final_count} labels")

        return queryset.order_by('row_index')

    def perform_create(self, serializer):
        """Set assigned_by when creating a label"""
        serializer.save(assigned_by=self.request.user)

    @action(detail=False, methods=['post'])
    def bulk_assign(self, request):
        """Bulk assign multiple rows to a segment"""
        serializer = BulkRowSegmentLabelSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        segment_id = serializer.validated_data['segment_id']
        row_indices = serializer.validated_data['row_indices']
        assignment_method = serializer.validated_data['assignment_method']
        confidence = serializer.validated_data['confidence']
        notes = serializer.validated_data.get('notes', '')

        segment = get_object_or_404(DataSegment, id=segment_id)
        dataset = segment.dataset

        # Load dataset to get row data
        if not dataset.minio_csv_key:
            return Response(
                {"error": "Dataset has no CSV file"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            csv_obj = dataset.get_csv_file_object()
            df = pd.read_csv(csv_obj)

            # Validate row indices
            invalid_indices = [idx for idx in row_indices if idx >= len(df)]
            if invalid_indices:
                return Response(
                    {"error": f"Invalid row indices: {invalid_indices}"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            created_labels = []
            updated_labels = []
            moved_labels = []

            with transaction.atomic():
                for row_idx in row_indices:
                    row_data = df.iloc[row_idx].to_dict()

                    # Convert numpy types to Python types for JSON serialization
                    row_data = {
                        k: (v.item() if hasattr(v, 'item') else v)
                        for k, v in row_data.items()
                    }

                    # Check if this row is already assigned to a different segment
                    existing_labels = RowSegmentLabel.objects.filter(
                        segment__dataset=dataset,
                        row_index=row_idx
                    ).exclude(segment=segment)

                    if existing_labels.exists():
                        # Delete labels from other segments (move the row to this segment)
                        old_segments = list(existing_labels.values_list('segment__name', flat=True))
                        existing_labels.delete()
                        moved_labels.append({
                            'row_idx': row_idx,
                            'from_segments': old_segments
                        })

                    # Create or update label
                    label, created = RowSegmentLabel.objects.update_or_create(
                        segment=segment,
                        row_index=row_idx,
                        defaults={
                            'row_data': row_data,
                            'assigned_by': request.user,
                            'assignment_method': assignment_method,
                            'confidence': confidence,
                            'notes': notes,
                            'created_by': request.user,
                        }
                    )

                    if created:
                        created_labels.append(label)
                    else:
                        updated_labels.append(label)

                # Update row counts for all affected segments
                affected_segment_ids = set([segment.id])
                for move_info in moved_labels:
                    # Get segment IDs from moved labels
                    affected_segment_ids.update(
                        DataSegment.objects.filter(
                            dataset=dataset,
                            name__in=move_info['from_segments']
                        ).values_list('id', flat=True)
                    )

                # Update all affected segment row counts
                for seg_id in affected_segment_ids:
                    seg = DataSegment.objects.get(id=seg_id)
                    seg.row_count = seg.row_labels.count()
                    seg.save(update_fields=['row_count'])

            return Response({
                'success': True,
                'created': len(created_labels),
                'updated': len(updated_labels),
                'moved': len(moved_labels),
                'total': len(row_indices),
                'segment': DataSegmentSerializer(segment).data,
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response(
                {"error": f"Bulk assignment failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['post'])
    def bulk_delete(self, request):
        """Bulk delete labels for specific rows in a segment"""
        segment_id = request.data.get('segment_id')
        row_indices = request.data.get('row_indices', [])

        if not segment_id:
            return Response(
                {"error": "segment_id is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        segment = get_object_or_404(DataSegment, id=segment_id)

        with transaction.atomic():
            deleted_count, _ = RowSegmentLabel.objects.filter(
                segment=segment,
                row_index__in=row_indices
            ).delete()

            # Update segment row count
            segment.row_count = segment.row_labels.count()
            segment.save(update_fields=['row_count'])

        return Response({
            'success': True,
            'deleted': deleted_count,
            'segment': DataSegmentSerializer(segment).data,
        })


class DatasetSegmentationView(BaseAuthApiView):
    """
    View for getting segmentation overview for a dataset.
    """
    authentication_classes = [JWTAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, dataset_id):
        """Get all segments and label coverage for a dataset"""
        dataset = get_object_or_404(Dataset, id=dataset_id)
        segments = DataSegment.objects.filter(dataset=dataset)

        # Get all labels for the dataset
        all_labels = RowSegmentLabel.objects.filter(segment__dataset=dataset)

        # Calculate coverage
        total_rows = dataset.row_count or 0
        labeled_rows = all_labels.values('row_index').distinct().count()
        coverage_percentage = (labeled_rows / total_rows * 100) if total_rows > 0 else 0

        # Build response
        data = {
            'dataset_id': str(dataset.id),
            'dataset_name': dataset.name,
            'total_rows': total_rows,
            'labeled_rows': labeled_rows,
            'unlabeled_rows': total_rows - labeled_rows,
            'coverage_percentage': coverage_percentage,
            'segments': DataSegmentSerializer(segments, many=True).data,
            'segment_count': segments.count(),
        }

        return Response(data)


class AutoClusterView(APIView):
    """
    View for performing automatic clustering/segmentation on a dataset.
    """
    authentication_classes = [JWTAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request, dataset_id):
        """
        Perform automatic clustering and create segments from clusters.

        Request body:
        {
            "algorithm": "kmeans" | "dbscan" | "hierarchical" | "gaussian_mixture" | "mean_shift",
            "config": {
                "n_clusters": 3,  // for kmeans, hierarchical, gaussian_mixture
                "eps": 0.5,       // for dbscan
                "min_samples": 5, // for dbscan
                "linkage": "ward", // for hierarchical
                // ... other algorithm-specific params
            },
            "feature_columns": ["col1", "col2"],  // optional, uses all numeric if not specified
            "create_segments": true,  // whether to create DataSegment objects
            "segment_prefix": "Cluster"  // prefix for segment names
        }
        """
        dataset = get_object_or_404(Dataset, id=dataset_id)

        if not dataset.minio_csv_key:
            return Response(
                {"error": "Dataset has no CSV file"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Parse request data
        algorithm = request.data.get('algorithm', 'kmeans').lower()
        config = request.data.get('config', {})
        feature_columns = request.data.get('feature_columns')
        create_segments = request.data.get('create_segments', True)
        segment_prefix = request.data.get('segment_prefix', 'Cluster')

        # Map algorithm string to ModelType
        algorithm_map = {
            'kmeans': ModelType.KMEANS,
            'dbscan': ModelType.DBSCAN,
            'hierarchical': ModelType.HIERARCHICAL,
            'gaussian_mixture': ModelType.GAUSSIAN_MIXTURE,
            'mean_shift': ModelType.MEAN_SHIFT,
        }

        if algorithm not in algorithm_map:
            return Response(
                {"error": f"Unknown algorithm: {algorithm}. Supported: {list(algorithm_map.keys())}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        model_type = algorithm_map[algorithm]

        try:
            # Load dataset
            csv_obj = dataset.get_csv_file_object()
            df = pd.read_csv(csv_obj)

            # Perform clustering
            result = perform_clustering(df, model_type, config, feature_columns)

            print(f"[AutoCluster] Dataset has {len(df)} rows")
            print(f"[AutoCluster] Clustering produced {len(result['labels'])} labels")
            print(f"[AutoCluster] Unique cluster IDs: {sorted(set(result['labels']))}")

            # If create_segments is True, create DataSegment and RowSegmentLabel objects
            segments_created = []
            if create_segments:
                labels = result['labels']
                n_clusters = result['n_clusters']

                # Define colors for segments
                preset_colors = [
                    '#3b82f6', '#10b981', '#f59e0b', '#ef4444',
                    '#8b5cf6', '#ec4899', '#14b8a6', '#f97316',
                    '#6366f1', '#84cc16', '#06b6d4', '#d946ef',
                ]

                with transaction.atomic():
                    # Clear ALL existing labels for this dataset before re-clustering
                    # This ensures every row gets a fresh assignment
                    existing_segments = DataSegment.objects.filter(dataset=dataset)
                    RowSegmentLabel.objects.filter(segment__in=existing_segments).delete()

                    # Create segments for each cluster
                    cluster_ids = sorted(set(labels))
                    for cluster_id in cluster_ids:
                        if cluster_id == -1:
                            segment_name = f"{segment_prefix} - Noise/Outliers"
                            segment_color = '#9ca3af'  # gray for noise
                        else:
                            segment_name = f"{segment_prefix} {cluster_id + 1}"
                            segment_color = preset_colors[cluster_id % len(preset_colors)]

                        # Check if segment with this name already exists
                        segment, created = DataSegment.objects.get_or_create(
                            dataset=dataset,
                            name=segment_name,
                            defaults={
                                'description': f"Auto-generated segment using {algorithm.upper()} clustering",
                                'color': segment_color,
                                'created_by': request.user,
                            }
                        )

                        if not created:
                            # Update existing segment
                            segment.description = f"Auto-generated segment using {algorithm.upper()} clustering"
                            segment.color = segment_color
                            segment.save()

                        # Get row indices for this cluster
                        row_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]

                        # Create labels for all rows in this cluster
                        labels_to_create = []
                        for row_idx in row_indices:
                            row_data = df.iloc[row_idx].to_dict()
                            # Convert numpy types
                            row_data = {
                                k: (v.item() if hasattr(v, 'item') else v)
                                for k, v in row_data.items()
                            }

                            labels_to_create.append(RowSegmentLabel(
                                segment=segment,
                                row_index=row_idx,
                                row_data=row_data,
                                assigned_by=request.user,
                                assignment_method='ml',
                                confidence=result['metrics'].get('silhouette_score', 0.5) if result['metrics'] else 0.5,
                                created_by=request.user,
                            ))

                        # Bulk create labels
                        RowSegmentLabel.objects.bulk_create(labels_to_create)

                        # Update segment row count
                        segment.row_count = len(row_indices)
                        segment.save(update_fields=['row_count'])

                        segments_created.append({
                            'id': str(segment.id),
                            'name': segment.name,
                            'row_count': len(row_indices),
                            'color': segment.color,
                        })

                    # Log total labels created
                    total_labels = sum(s['row_count'] for s in segments_created)
                    print(f"[AutoCluster] Created {total_labels} labels across {len(segments_created)} segments")

            return Response({
                'success': True,
                'algorithm': algorithm,
                'n_clusters': result['n_clusters'],
                'n_noise_points': result.get('n_noise_points', 0),
                'metrics': result['metrics'],
                'cluster_stats': result['cluster_stats'],
                'segments_created': segments_created,
                'pca_coordinates': result.get('pca_coordinates'),
                'pca_variance_ratio': result.get('pca_variance_ratio'),
            })

        except ValueError as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response(
                {"error": f"Clustering failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class OptimalClustersView(APIView):
    """
    View for determining the optimal number of clusters for a dataset.
    """
    authentication_classes = [JWTAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request, dataset_id):
        """
        Determine optimal number of clusters using elbow or silhouette method.

        Request body:
        {
            "feature_columns": ["col1", "col2"],  // optional
            "max_clusters": 10,
            "method": "elbow" | "silhouette"
        }
        """
        dataset = get_object_or_404(Dataset, id=dataset_id)

        if not dataset.minio_csv_key:
            return Response(
                {"error": "Dataset has no CSV file"},
                status=status.HTTP_400_BAD_REQUEST
            )

        feature_columns = request.data.get('feature_columns')
        max_clusters = request.data.get('max_clusters', 10)
        method = request.data.get('method', 'elbow')

        try:
            csv_obj = dataset.get_csv_file_object()
            df = pd.read_csv(csv_obj)

            result = determine_optimal_clusters(
                df,
                feature_columns=feature_columns,
                max_clusters=max_clusters,
                method=method
            )

            return Response({
                'success': True,
                'optimal_k': result['optimal_k'],
                'scores': result['scores'],
                'method': result['method'],
            })

        except ValueError as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {"error": f"Analysis failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
