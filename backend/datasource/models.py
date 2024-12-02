import logging
from functools import cached_property

from core.models import BaseModel
from django.core.exceptions import ValidationError
from django.db import models
from django.conf import settings
from .adapters.postgres import  PostgresConnection
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
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    datasource = models.ForeignKey("DataSource", on_delete=models.CASCADE)

    class Meta:
        verbose_name = "Data Cluster"
        verbose_name_plural = "Data Clusters"

class DatasourceTypeChoices(models.TextChoices):
    POSTGRES = 'postgres', 'PostgreSQL'
    MYSQL = 'mysql', 'MySQL'
    SQLITE = 'sqlite', 'SQLite'

    @staticmethod
    def get_adapter(db_type):
        if db_type == DatasourceTypeChoices.POSTGRES:
            return PostgresConnection
        # elif db_type == DatasourceTypeChoices.MYSQL:
        #     return MySQLAdapter
        # elif db_type == DatasourceTypeChoices.SQLITE:
        #     return SQLiteAdapter
        raise ValueError(f"Invalid datasource type: {db_type}")

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
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
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
        return DatasourceTypeChoices.get_adapter(self.type)(**self.credentials)

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
            self.adapter.connect()
            result = self.adapter.query(query, params)
            self.adapter.close()
            return result
        except Exception:
            logging.error("Query execution failed.", exc_info=True)
            return None

    def test_connection(self):
        """
        Tests the connection to the datasource.

        Returns:
            bool: True if the connection is successful, False otherwise.
        """
        return self.connection.test_conection()

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
            is_valid, message = adapter.verify_params(keys)
            if is_valid:
                super().save(*args, **kwargs)
            else:
                raise ValidationError(f"Missing required credentials: {message}")
        else:
            raise ValidationError("Invalid credentials or datasource type")

    def related_tables(self, table_name):
        query = """
        SELECT
            tc.table_name AS related_table,
            kcu.column_name AS related_column,
            ccu.table_name AS target_table,
            ccu.column_name AS target_column
        FROM
            information_schema.table_constraints AS tc
        JOIN
            information_schema.key_column_usage AS kcu
        ON
            tc.constraint_name = kcu.constraint_name
        AND
            tc.table_schema = kcu.table_schema
        JOIN
            information_schema.constraint_column_usage AS ccu
        ON
            ccu.constraint_name = tc.constraint_name
        AND
            ccu.table_schema = tc.table_schema
        WHERE
            tc.constraint_type = 'FOREIGN KEY'
        AND
            ccu.table_name = %s;
        """
        return self.query(query, [table_name])
