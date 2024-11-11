import logging
from functools import cached_property

from core.models import BaseModel
from django.core.exceptions import ValidationError
from django.db import models

from .constants.choices import DatasourceTypeChoices

logger = logging.getLogger(__name__)


class DataCluster(BaseModel):
    """
    DataCluster model represents a cluster of data associated with a user and a datasource.

    Attributes:
        name (CharField): The name of the data cluster, with a maximum length of 255 characters.
        description (TextField): A detailed description of the data cluster.
        user (ForeignKey): A reference to the user who owns the data cluster, linked to the auth.User model.
        datasource (ForeignKey): A reference to the datasource associated with the data cluster, linked to the DataSource model.

    Meta:
        verbose_name (str): The singular name for the model in the admin interface.
        verbose_name_plural (str): The plural name for the model in the admin interface.
    """

    name = models.CharField(max_length=255)
    description = models.TextField()
    user = models.ForeignKey("auth.User", on_delete=models.CASCADE)
    datasource = models.ForeignKey("DataSource", on_delete=models.CASCADE)

    class Meta:
        verbose_name = "Data Cluster"
        verbose_name_plural = "Data Clusters"


class DataSource(BaseModel):
    """
    DataSource model represents a data source configuration.

    Attributes:
        name (str): The name of the data source.
        type (str): The type of the data source, chosen from predefined choices.
        description (str): A detailed description of the data source.
        user (ForeignKey): A reference to the user who owns this data source.
        credentioals (JSONField): A JSON field to store credentials for accessing the data source.

    Meta:
        verbose_name (str): The human-readable name of the model.
        verbose_name_plural (str): The human-readable plural name of the model.
    """

    name = models.CharField(max_length=255)
    type = models.CharField(
        max_length=10,
        choices=DatasourceTypeChoices.choices,
        default=DatasourceTypeChoices.POSTGRES,
    )
    description = models.TextField()
    user = models.ForeignKey("auth.User", on_delete=models.CASCADE)
    credentials = models.JSONField()

    class Meta:
        verbose_name = "Data Source"
        verbose_name_plural = "Data Sources"

    @cached_property
    def adapter(self):
        """
        Returns the appropriate adapter class based on the datasource type.

        Returns:
            class: The adapter class corresponding to the datasource type.

        Raises:
            ValueError: If the datasource type is invalid.
        """
        return DatasourceTypeChoices.get_adapter(self.type)

    @cached_property
    def connection(self):
        """
        Establishes a connection using the adapter and credentials provided.

        Returns:
            object: A connection object created by the adapter.
        """
        return self.adapter(**self.credentials)

    def query(self, query, params=None) -> list:
        try:
            self.connection.connect()
            result = self.connection.query(query, params)
            self.connection.close()
            return result
        except Exception:
            logging.error("Failed to execute the query.", exc_info=True)

    def test_connection(self):
        """
        Tests the connection to the datasource.

        Returns:
            bool: True if the connection is successful, False otherwise.
        """
        return self.connection().test_conection()

    def save(self, *args, **kwargs) -> None:
        """
        Save the datasource instance after verifying the credentials and type.

        This method overrides the default save method to include validation
        of the datasource credentials and type before saving the instance.
        If the credentials and type are valid, the instance is saved using
        the superclass's save method. Otherwise, a ValidationError is raised.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ValidationError: If the credentials or datasource type are invalid.
        """
        if self.credentials and self.type:
            keys = self.credentials.keys()
            adapter = DatasourceTypeChoices.get_adapter(self.type)
            if adapter.verify_params(keys):
                super().save(*args, **kwargs)
        else:
            raise ValidationError("Invalid credentials or datasource type")
