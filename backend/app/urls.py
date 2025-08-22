from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import HelloWorldAPIView, DatasetViewSet, MLModelViewSet, TrainingRunViewSet

router = DefaultRouter()
router.register(r'datasets', DatasetViewSet)
router.register(r'models', MLModelViewSet)
router.register(r'training-runs', TrainingRunViewSet)

urlpatterns = [
    path("hello/", HelloWorldAPIView.as_view(), name="hello_world_api"),
    path("", include(router.urls)),
]
