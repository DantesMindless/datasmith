from django.urls import path
from .views import (
    DataSourceView,
    DataSourceTestConnectionView,
    DataSourceTablesMetadataView,
)

urlpatterns = [
    path("datasource/", DataSourceView.as_view(), name="datasource-list"),
    path(
        "datasource/detail/<uuid:id>/",
        DataSourceView.as_view(),
        name="datasource-detail",
    ),
    path(
        "datasource/query/<uuid:id>/",
        DataSourceView.as_view(),
        name="datasource-query",
    ),
    path(
        "datasource/test/",
        DataSourceTestConnectionView.as_view(),
        name="datasource-test",
    ),
    path(
        "datasource/test/<uuid:id>/",
        DataSourceTestConnectionView.as_view(),
        name="datasource-test",
    ),
    path(
        "datasource-metadata/<uuid:id>/<str:table_name>/",
        DataSourceTablesMetadataView.as_view(),
        name="datasource-metadata",
    ),
    path(
        "datasource-metadata/update/<uuid:id>/",
        DataSourceTablesMetadataView.as_view(),
        name="datasource-metadata-update",
    ),
]
