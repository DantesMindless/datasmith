from core.models import BaseModel
from django.db import models


class DataCluster(BaseModel):
    name = models.CharField(max_length=255)
    description = models.TextField()
    user = models.ForeignKey("auth.User", on_delete=models.CASCADE)
    datasource = models.ForeignKey("DataSource", on_delete=models.CASCADE)

    class Meta:
        verbose_name = "Data Cluster"
        verbose_name_plural = "Data Clusters"


class DataSource(BaseModel):
    name = models.CharField(max_length=255)
    description = models.TextField()
    url = models.URLField()
    user = models.ForeignKey("auth.User", on_delete=models.CASCADE)
    credentioals = models.JSONField()

    class Meta:
        verbose_name = "Data Source"
        verbose_name_plural = "Data Sources"
