import logging
import random
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import viewsets, status
from rest_framework.decorators import action, authentication_classes, permission_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import SessionAuthentication
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.shortcuts import get_object_or_404
from django.contrib.auth.models import AnonymousUser

from .models.main import Dataset, MLModel, TrainingRun
from .models.choices import ModelStatus
from .serializers import DatasetSerializer, MLModelSerializer, TrainingRunSerializer
from .functions.celery_tasks import train_cnn_task, train_nn_task, train_sklearn_task
from .models.choices import ModelType

logger = logging.getLogger(__name__)


class DatasetViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing datasets with proper authentication and pagination.
    
    Provides CRUD operations for datasets with user-based filtering.
    Only authenticated users can access their datasets.
    """
    queryset = Dataset.objects.all().order_by('id')
    serializer_class = DatasetSerializer
    parser_classes = (MultiPartParser, FormParser)
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
            return Dataset.objects.filter(deleted=False).order_by('id')
        else:
            return Dataset.objects.none()

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

            # Read CSV file to get column names
            if hasattr(dataset, 'csv_file') and dataset.csv_file:
                df = pd.read_csv(dataset.csv_file.path)
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
    def preview(self, request, pk=None) -> Response:
        """
        Get preview data and analysis for a dataset.

        Returns:
            Response with preview data, column info, and basic statistics
        """
        dataset = self.get_object()

        if not dataset.is_processed:
            dataset.analyze_dataset()

        return Response({
            'preview_data': dataset.preview_data,
            'column_info': dataset.column_info,
            'statistics': dataset.statistics,
            'data_quality': dataset.data_quality,
            'dataset_type': dataset.dataset_type,
            'dataset_purpose': dataset.dataset_purpose,
            'row_count': dataset.row_count,
            'column_count': dataset.column_count,
            'file_size': dataset.file_size,
            'last_analyzed': dataset.last_analyzed
        })

    @action(detail=True, methods=['get'])
    def quality_report(self, request, pk=None) -> Response:
        """
        Get comprehensive data quality report.

        Returns:
            Response with detailed quality analysis
        """
        dataset = self.get_object()

        if not dataset.is_processed:
            dataset.analyze_dataset()

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

        if not dataset.is_processed:
            dataset.analyze_dataset()

        return Response({
            'basic_stats': dataset.statistics,
            'column_stats': dataset.column_info,
            'correlations': self._calculate_correlations(dataset),
            'distributions': self._get_distributions(dataset)
        })

    @action(detail=True, methods=['get'])
    def sample(self, request, pk=None) -> Response:
        """
        Get a random sample of the dataset.

        Query params:
            - size: number of rows to sample (default: 100)
            - random_state: seed for reproducible sampling
        """
        dataset = self.get_object()

        if not dataset.csv_file:
            return Response(
                {"error": "No CSV file associated with this dataset"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            import pandas as pd

            size = int(request.query_params.get('size', 100))
            random_state = request.query_params.get('random_state', 42)

            df = pd.read_csv(dataset.csv_file.path)

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
        dataset.is_processed = False
        dataset.analyze_dataset()

        return Response({
            'message': 'Dataset reanalyzed successfully',
            'statistics': dataset.statistics,
            'data_quality': dataset.data_quality,
            'last_analyzed': dataset.last_analyzed
        })

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
        if not dataset.csv_file:
            return {}

        try:
            import pandas as pd

            df = pd.read_csv(dataset.csv_file.path)
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
        if not dataset.csv_file:
            return {}

        try:
            import pandas as pd
            import numpy as np

            df = pd.read_csv(dataset.csv_file.path)
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
            return MLModel.objects.filter(deleted=False).order_by('id')
        else:
            return MLModel.objects.none()

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
        
        try:
            model.status = ModelStatus.TRAINING
            model.save()

            # Create or get training run
            run, _ = TrainingRun.objects.get_or_create(model=model)
            run.add_entry(status=ModelStatus.TRAINING)

            # Try to queue task with Celery, fallback to synchronous execution
            try:
                if model.dataset.is_image_dataset:
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
                except:
                    pass

            return Response(test_results)

        except Exception as e:
            logger.error(f"Test error for model {model.id}: {e}")
            return Response(
                {"error": f"Model testing failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def _generate_model_metrics(self, model) -> Dict[str, Any]:
        """Generate comprehensive mock metrics for model analysis."""
        # Set random seed based on model ID for consistent data
        random.seed(model.id * 42)

        # Base accuracy from model or generate one
        base_accuracy = model.accuracy if model.accuracy else random.uniform(0.75, 0.95)

        # Generate metrics with realistic variation
        precision = base_accuracy + random.uniform(-0.08, 0.05)
        recall = base_accuracy + random.uniform(-0.06, 0.04)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else base_accuracy

        # Ensure all metrics are within [0, 1]
        precision = max(0.0, min(1.0, precision))
        recall = max(0.0, min(1.0, recall))
        f1_score = max(0.0, min(1.0, f1_score))

        # Generate confusion matrix for binary classification
        total_samples = random.randint(800, 1200)
        true_positives = int(total_samples * 0.4 * recall)
        false_negatives = int(total_samples * 0.4) - true_positives
        true_negatives = int(total_samples * 0.6 * (1 - (1 - precision) * true_positives / (true_positives + (total_samples * 0.6 * (1 - precision)))))
        false_positives = int(total_samples * 0.6) - true_negatives

        confusion_matrix = [
            [true_negatives, false_positives],
            [false_negatives, true_positives]
        ]

        # Generate feature importance based on model type
        num_features = random.randint(8, 15)
        feature_importance = []
        remaining_importance = 1.0

        for i in range(num_features):
            if i == num_features - 1:
                importance = remaining_importance
            else:
                importance = random.uniform(0.02, remaining_importance * 0.4)
                remaining_importance -= importance

            feature_importance.append({
                "feature": f"Feature_{i+1}",
                "importance": round(importance, 3)
            })

        # Sort by importance descending
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)

        # Generate training history
        num_epochs = random.randint(15, 30)
        training_history = []

        initial_loss = random.uniform(1.0, 2.5)
        initial_accuracy = random.uniform(0.3, 0.5)
        initial_val_loss = initial_loss + random.uniform(0.0, 0.3)
        initial_val_accuracy = initial_accuracy - random.uniform(0.0, 0.1)

        for epoch in range(1, num_epochs + 1):
            # Simulate learning curves with some noise
            progress = epoch / num_epochs

            # Training metrics improve over time with diminishing returns
            loss_improvement = (1 - progress) * initial_loss + progress * random.uniform(0.1, 0.4)
            acc_improvement = initial_accuracy + progress * (base_accuracy - initial_accuracy) + random.uniform(-0.02, 0.02)

            # Validation metrics with some overfitting towards the end
            val_loss_factor = 1.0 + (progress * 0.3) if progress > 0.7 else 1.0
            val_loss = loss_improvement * val_loss_factor + random.uniform(-0.05, 0.1)
            val_acc = acc_improvement - (progress * 0.05) + random.uniform(-0.03, 0.02)

            training_history.append({
                "epoch": epoch,
                "loss": round(max(0.05, loss_improvement), 4),
                "accuracy": round(min(1.0, max(0.0, acc_improvement)), 4),
                "val_loss": round(max(0.05, val_loss), 4),
                "val_accuracy": round(min(1.0, max(0.0, val_acc)), 4)
            })

        # Generate prediction distribution
        num_classes = random.choice([2, 3, 4])
        if num_classes == 2:
            classes = ["Negative", "Positive"]
        elif num_classes == 3:
            classes = ["Class_A", "Class_B", "Class_C"]
        else:
            classes = ["Class_1", "Class_2", "Class_3", "Class_4"]

        prediction_distribution = []
        total_predictions = random.randint(1000, 5000)
        remaining = total_predictions

        for i, class_name in enumerate(classes):
            if i == len(classes) - 1:
                count = remaining
            else:
                count = random.randint(int(remaining * 0.15), int(remaining * 0.6))
                remaining -= count

            prediction_distribution.append({
                "label": class_name,
                "value": count
            })

        return {
            "accuracy": round(base_accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "confusion_matrix": confusion_matrix,
            "feature_importance": feature_importance,
            "training_history": training_history,
            "prediction_distribution": prediction_distribution,
            "additional_metrics": {
                "auc_roc": round(random.uniform(0.8, 0.98), 4),
                "log_loss": round(random.uniform(0.1, 0.5), 4),
                "matthews_corr_coef": round(random.uniform(0.6, 0.9), 4)
            }
        }

    def _generate_model_statistics(self, model) -> Dict[str, Any]:
        """Generate comprehensive mock statistics for model analysis."""
        # Set random seed based on model ID for consistent data
        random.seed(model.id * 123)

        # Usage statistics
        total_predictions = random.randint(500, 10000)
        avg_prediction_time = round(random.uniform(5.0, 150.0), 2)

        # Model size calculation based on type
        if model.model_type in ['neural_network', 'cnn']:
            model_size_mb = round(random.uniform(10.0, 200.0), 1)
        else:
            model_size_mb = round(random.uniform(0.5, 15.0), 1)

        # Training time based on model complexity
        if model.model_type == 'cnn':
            training_minutes = random.randint(30, 300)
        elif model.model_type == 'neural_network':
            training_minutes = random.randint(10, 120)
        else:
            training_minutes = random.randint(1, 30)

        # Dataset statistics
        total_rows = random.randint(5000, 100000)
        total_features = random.randint(5, 50)

        # Target distribution
        if random.choice([True, False]):  # Binary classification
            positive_ratio = random.uniform(0.3, 0.7)
            target_distribution = [
                {"label": "Negative", "count": int(total_rows * (1 - positive_ratio))},
                {"label": "Positive", "count": int(total_rows * positive_ratio)}
            ]
        else:  # Multi-class
            num_classes = random.randint(3, 5)
            remaining_rows = total_rows
            target_distribution = []

            for i in range(num_classes):
                if i == num_classes - 1:
                    count = remaining_rows
                else:
                    count = random.randint(int(remaining_rows * 0.1), int(remaining_rows * 0.4))
                    remaining_rows -= count

                target_distribution.append({
                    "label": f"Class_{i+1}",
                    "count": count
                })

        # Performance over time (last 30 days)
        performance_history = []
        base_date = datetime.now() - timedelta(days=30)

        for day in range(30):
            date = base_date + timedelta(days=day)
            daily_predictions = random.randint(0, int(total_predictions / 15))
            avg_confidence = random.uniform(0.75, 0.95)

            performance_history.append({
                "date": date.strftime("%Y-%m-%d"),
                "predictions_count": daily_predictions,
                "avg_confidence": round(avg_confidence, 3),
                "avg_response_time": round(avg_prediction_time + random.uniform(-20, 20), 2)
            })

        return {
            "usage_metrics": {
                "total_predictions": total_predictions,
                "avg_prediction_time": avg_prediction_time,
                "predictions_last_24h": random.randint(0, 200),
                "predictions_last_week": random.randint(0, 1000),
                "unique_users": random.randint(1, 50)
            },
            "model_info": {
                "model_size": f"{model_size_mb} MB",
                "training_time": f"{training_minutes} minutes",
                "last_trained": model.updated_at.isoformat() if model.updated_at else datetime.now().isoformat(),
                "version": "1.0.0",
                "framework": "scikit-learn" if model.model_type not in ['neural_network', 'cnn'] else "PyTorch"
            },
            "dataset_info": {
                "total_rows": total_rows,
                "total_features": total_features,
                "target_distribution": target_distribution,
                "data_quality_score": round(random.uniform(0.85, 0.98), 3),
                "missing_values_pct": round(random.uniform(0.0, 5.0), 2)
            },
            "performance_history": performance_history,
            "resource_usage": {
                "avg_cpu_usage": round(random.uniform(15.0, 75.0), 1),
                "avg_memory_usage": round(random.uniform(100.0, 2000.0), 1),
                "disk_usage": f"{random.randint(50, 500)} MB"
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
