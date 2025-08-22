from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import viewsets, status
from rest_framework.decorators import action, authentication_classes, permission_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import AllowAny
from django.shortcuts import get_object_or_404

from .models.main import Dataset, MLModel, TrainingRun
from .models.choices import ModelStatus
from .serializers import DatasetSerializer, MLModelSerializer, TrainingRunSerializer
from .functions.celery_tasks import train_cnn_task, train_nn_task, train_sklearn_task
from .models.choices import ModelType



class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    parser_classes = (MultiPartParser, FormParser)

    def dispatch(self, request, *args, **kwargs):
        auth_header = request.META.get('HTTP_AUTHORIZATION', 'No auth header')
        print(f"Dataset dispatch - Method: {request.method}, Auth header: {auth_header}")
        return super().dispatch(request, *args, **kwargs)

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)

    def get_queryset(self):
        auth_header = self.request.META.get('HTTP_AUTHORIZATION', 'No auth header')
        print(f"Dataset get_queryset - User: {self.request.user}, Is authenticated: {self.request.user.is_authenticated}, Auth header: {auth_header}")
        if self.request.user.is_authenticated:
            # Temporarily show all datasets for testing
            return Dataset.objects.filter(deleted=False)
        else:
            return Dataset.objects.none()


class MLModelViewSet(viewsets.ModelViewSet):
    queryset = MLModel.objects.all()
    serializer_class = MLModelSerializer

    def dispatch(self, request, *args, **kwargs):
        auth_header = request.META.get('HTTP_AUTHORIZATION', 'No auth header')
        print(f"MLModel dispatch - Method: {request.method}, Auth header: {auth_header}")
        return super().dispatch(request, *args, **kwargs)

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)

    def get_queryset(self):
        auth_header = self.request.META.get('HTTP_AUTHORIZATION', 'No auth header')
        print(f"MLModel get_queryset - User: {self.request.user}, Is authenticated: {self.request.user.is_authenticated}, Auth header: {auth_header}")
        if self.request.user.is_authenticated:
            # Temporarily show all models for testing
            return MLModel.objects.filter(deleted=False)
        else:
            return MLModel.objects.none()

    @action(detail=True, methods=['post'])
    def train(self, request, pk=None):
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

            # Queue appropriate training task
            if model.dataset.is_image_dataset:
                train_cnn_task.delay(model.id)
            elif model.model_type == ModelType.NEURAL_NETWORK:
                train_nn_task.delay(model.id)
            else:
                train_sklearn_task.delay(model.id)

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

    @action(detail=True, methods=['get'])
    def training_status(self, request, pk=None):
        model = self.get_object()
        training_run = getattr(model, 'training_run', None)
        
        return Response({
            "status": model.status,
            "training_log": model.training_log,
            "history": training_run.history if training_run else []
        })


class TrainingRunViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = TrainingRun.objects.all()
    serializer_class = TrainingRunSerializer

    def get_queryset(self):
        return TrainingRun.objects.filter(model__created_by=self.request.user)
