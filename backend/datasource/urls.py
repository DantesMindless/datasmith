from django.urls import path
from .views import DataSourceView

urlpatterns = [
    path("datasource/", DataSourceView.as_view(), name="datasource"),
    path("datasource/detail/<int:id>", DataSourceView.as_view(), name="datasource"),
]
