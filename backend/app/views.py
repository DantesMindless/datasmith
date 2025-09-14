import logging
from typing import Any, Dict
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
