from django.urls import path
from .views import DataSourceView, DataSourceTestConnectionView

urlpatterns = [
    path("datasource/", DataSourceView.as_view(), name="datasource-list"),
    path(
        "datasource/detail/<uuid:id>/",
        DataSourceView.as_view(),
        name="datasource-detail",
    ),
    path(
        "datasource/test/<uuid:id>/",
        DataSourceTestConnectionView.as_view(),
        name="datasource-test",
    ),
]
