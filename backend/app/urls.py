from django.urls import path, include, re_path
from rest_framework.routers import DefaultRouter

from .views import  DatasetViewSet, MLModelViewSet, TrainingRunViewSet, serve_image, batch_serve_images

router = DefaultRouter()
router.register(r'datasets', DatasetViewSet)
router.register(r'models', MLModelViewSet)
router.register(r'training-runs', TrainingRunViewSet)

urlpatterns = [
    # Image serving endpoints (must be before router to avoid conflicts)
    re_path(r'^datasets/(?P<dataset_id>[^/]+)/images/(?P<image_path>.+)$', serve_image, name='serve-image'),
    path('datasets/<str:dataset_id>/images/batch/', batch_serve_images, name='batch-serve-images'),
    # Router URLs
    path("", include(router.urls)),
]
