from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import  DatasetViewSet, MLModelViewSet, TrainingRunViewSet

router = DefaultRouter()
router.register(r'datasets', DatasetViewSet)
router.register(r'models', MLModelViewSet)
router.register(r'training-runs', TrainingRunViewSet)

urlpatterns = [
    path("", include(router.urls)),
]
