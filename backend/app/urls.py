from django.urls import path, include, re_path
from rest_framework.routers import DefaultRouter

from .views import  DatasetViewSet, MLModelViewSet, TrainingRunViewSet, serve_image, batch_serve_images, get_gpu_info
from .views_segmentation import (
    DataSegmentViewSet, RowSegmentLabelViewSet, DatasetSegmentationView,
    AutoClusterView, OptimalClustersView
)

router = DefaultRouter()
router.register(r'datasets', DatasetViewSet)
router.register(r'models', MLModelViewSet)
router.register(r'training-runs', TrainingRunViewSet)
router.register(r'segments', DataSegmentViewSet)
router.register(r'segment-labels', RowSegmentLabelViewSet)

urlpatterns = [
    # Image serving endpoints (must be before router to avoid conflicts)
    re_path(r'^datasets/(?P<dataset_id>[^/]+)/images/(?P<image_path>.+)$', serve_image, name='serve-image'),
    path('datasets/<str:dataset_id>/images/batch/', batch_serve_images, name='batch-serve-images'),
    # Segmentation endpoints
    path('datasets/<str:dataset_id>/segmentation/', DatasetSegmentationView.as_view(), name='dataset-segmentation'),
    path('datasets/<str:dataset_id>/auto-cluster/', AutoClusterView.as_view(), name='auto-cluster'),
    path('datasets/<str:dataset_id>/optimal-clusters/', OptimalClustersView.as_view(), name='optimal-clusters'),
    # System info endpoints
    path('system/gpu-info/', get_gpu_info, name='gpu-info'),
    # Router URLs
    path("", include(router.urls)),
]
