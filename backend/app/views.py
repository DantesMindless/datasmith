from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser
from django.shortcuts import get_object_or_404

from .models.main import Dataset, MLModel, TrainingRun
from .models.choices import ModelStatus
from .serializers import DatasetSerializer, MLModelSerializer, TrainingRunSerializer
from .functions.celery_tasks import train_cnn_task, train_nn_task, train_sklearn_task
from .models.choices import ModelType


class HelloWorldAPIView(APIView):
    def get(self, request):
        return Response({"message": "Hello, World!"})


class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    parser_classes = (MultiPartParser, FormParser)

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)

    def get_queryset(self):
        return Dataset.objects.filter(created_by=self.request.user, deleted=False)


class MLModelViewSet(viewsets.ModelViewSet):
    queryset = MLModel.objects.all()
    serializer_class = MLModelSerializer

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)

    def get_queryset(self):
        return MLModel.objects.filter(created_by=self.request.user, deleted=False)

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
