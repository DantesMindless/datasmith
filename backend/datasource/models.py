import logging
from functools import cached_property

from core.models import BaseModel
from django.core.exceptions import ValidationError
from django.db import models
from django.conf import settings
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


class DataSource(BaseModel):
    """
    DataSource model represents a data source configuration.

    Attributes:
        name (str): The name of the data source.
        type (str): The type of the data source, chosen from predefined choices.
        description (str): A detailed description of the data source.
        user (ForeignKey): A reference to the user who owns this data source.
        credentials (JSONField): A JSON field to store credentials for accessing the data source.
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
        Returns the appropriate adapter instance based on the datasource type.

        Returns:
            object: An instance of the adapter corresponding to the datasource type.
        """
        AdapterClass = DatasourceTypeChoices.get_adapter(self.type)
        return AdapterClass(
            host=self.credentials.get("host"),
            database=self.credentials.get("database"),
            user=self.credentials.get("user"),
            password=self.credentials.get("password"),
            port=self.credentials.get("port", 5432),
        )

    def test_connection(self):
        """
        Tests the connection to the datasource.

        Returns:
            bool: True if the connection is successful, False otherwise.
        """
        try:
            if self.adapter.connect():
                logger.info(f"Connection to DataSource '{self.name}' successful.")
                self.adapter.close()
                return True
            else:
                logger.error(f"Connection to DataSource '{self.name}' failed.")
                return False
        except Exception as e:
            logger.error(f"Error testing connection for DataSource '{self.name}': {e}", exc_info=True)
            return False

    def query(self, query: str, params=None):
        """
        Execute a query on the datasource.

        Args:
            query (str): The SQL query to execute.
            params (list, optional): Query parameters.

        Returns:
            list: Query results if successful, otherwise None.
        """
        try:
            self.adapter.connect()
            success, result, message = self.adapter.query(query, params)
            self.adapter.close()

            if success:
                return result
            else:
                logger.error(f"Query failed: {message}")
                return None
        except Exception as e:
            logger.error(f"Query execution error: {e}", exc_info=True)
            return None

    def save(self, *args, **kwargs):
        """
        Save the datasource instance after verifying the credentials and type.
        """
        try:
            if not self.credentials or not self.type:
                raise ValidationError("Invalid credentials or datasource type.")

            required_keys = {"host", "database", "user", "password"}
            if not required_keys.issubset(self.credentials.keys()):
                missing_keys = required_keys - self.credentials.keys()
                raise ValidationError(f"Missing required credentials: {missing_keys}")

            if not self.test_connection():
                raise ValidationError(f"Cannot connect to the DataSource: {self.name}")

            super().save(*args, **kwargs)
        except ValidationError as e:
            logger.error(f"Validation error while saving DataSource '{self.name}': {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while saving DataSource '{self.name}': {e}", exc_info=True)
            raise

    def related_tables(self, table_name: str):
        """
        Retrieve related tables for a given table using foreign key constraints.

        Args:
            table_name (str): The table name to find related tables for.

        Returns:
            list: A list of related tables and columns.
        """
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
            AND tc.table_schema = kcu.table_schema
        JOIN
            information_schema.constraint_column_usage AS ccu
        ON
            ccu.constraint_name = tc.constraint_name
            AND ccu.table_schema = tc.table_schema
        WHERE
            tc.constraint_type = 'FOREIGN KEY'
            AND ccu.table_name = %s;
        """
        return self.query(query, [table_name])
