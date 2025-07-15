import logging
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Union, Type

from datasource.adapters.mongo import MongoDBConnection
from datasource.adapters.mysql import MySQLConnection
from datasource.adapters.postgres import PostgresConnection
from datasource.adapters.redis_adapter import RedisConnection
from core.models import BaseModel
from django.core.exceptions import ValidationError
from django.db import models
from django.conf import settings
from .constants.choices import DatasourceTypeChoices

logger = logging.getLogger(__name__)


class DataCluster(BaseModel):
    """
    Represents a cluster of data associated with a user and a datasource.

    Attributes:
        name (str): The name of the data cluster.
        description (str): A detailed description of the data cluster.
        user (ForeignKey): Reference to the user who owns the data cluster.
        datasource (ForeignKey): Reference to the associated data source.
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
    Represents a data source configuration.

    Attributes:
        name (str): The name of the data source.
        type (str): The type of the data source (e.g., POSTGRES, MYSQL).
        description (str): A detailed description of the data source.
        user (ForeignKey): Reference to the user who owns this data source.
        credentials (Dict[str, Any]): Credentials for accessing the data source.
        metadata (Optional[Dict[str, Any]]): Cached metadata about the data source.
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
    metadata = models.JSONField(null=True, blank=True)

    class Meta:
        verbose_name = "Data Source"
        verbose_name_plural = "Data Sources"

    @cached_property
    def adapter(
        self,
    ) -> Optional[
        Union[
            Type[PostgresConnection],
            Type[MySQLConnection],
            Type[MongoDBConnection],
            Type[RedisConnection],
        ]
    ]:
        """]
        Retrieve the adapter class for the current datasource type.

        Returns:
            Any: The adapter class corresponding to the datasource type.

        Raises:
            ValueError: If the datasource type is invalid.
        """
        return DatasourceTypeChoices.get_adapter(self.type)

    @cached_property
    def connection(
        self,
    ) -> Union[PostgresConnection, MySQLConnection, MongoDBConnection]:
        """
        Establish a connection using the adapter and provided credentials.

        Returns:
            Any: A connection object created by the adapter.
        """
        return self.adapter(**self.credentials)

    def query(
        self, query: str, params: Optional[Tuple[Any]] = None
    ) -> Optional[Tuple[bool, Optional[List[Dict[str, Any]]], str]]:
        """
        Execute a query against the datasource.

        Args:
            query (str): The SQL query to execute.
            params (Optional[Union[Dict[str, Any], List[Any]]]): Query parameters.

        Returns:
            Tuple[bool, Optional[List[Dict[str, Any]]], str]: Query result or None if an error occurs.
        """
        try:
            self.connection.connect()
            result = self.connection.query(query, params)
            self.connection.close()
            return result
        except Exception as e:
            logger.error(f"Failed to execute the query: {e}", exc_info=True)
            return None

    def test_connection(self) -> bool:
        """
        Test the connection to the datasource.

        Returns:
            bool: True if the connection is successful, False otherwise.
        """
        return self.connection.test_connection()

    def save(self, *args: Any, **kwargs: Any) -> None:
        """
        Save the datasource instance after validating credentials and type.

        Raises:
            ValidationError: If the credentials or datasource type are invalid.
        """
        if self.credentials and self.type:
            adapter = DatasourceTypeChoices.get_adapter(self.type)
            if adapter is None:
                raise ValidationError("Invalid datasource type")
            is_valid, message = adapter.verify_params(self.credentials)
            if is_valid:
                super().save(*args, **kwargs)
            else:
                raise ValidationError(f"Missing required credentials: {message}")
        else:
            raise ValidationError("Invalid credentials or datasource type")

    def get_tables(
        self, schema: str = "public"
    ) -> Tuple[bool, Optional[List[Dict[str, Any]]], str]:
        """
        Retrieve tables within a specified schema.

        Args:
            schema (str): The schema to query (default: "public").

        Returns:
            Optional[List[Dict[str, Any]]]: List of tables or None if an error occurs.
        """
        self.connection.connect()
        data: Tuple[bool, Optional[List[Dict[str, Any]]], str] = (
            self.connection.get_tables(schema)
        )
        self.connection.close()
        return data

    def get_schemas(self) -> Tuple[bool, Optional[List[Dict[str, Any]]], str]:
        """
        Retrieve available schemas in the datasource.

        Returns:
            Tuple[bool, Optional[List[Dict[str, Any]]], str]: List of schema names or None if an error occurs.
        """
        self.connection.connect()
        data: Tuple[bool, Optional[List[Dict[str, Any]]], str] = (
            self.connection.get_schemas()
        )
        self.connection.close()
        return data

    def get_table_rows(self, query) -> Tuple[bool, Optional[List[Dict[str, Any]]], str]:
        """
        Retrieve rows from a specified table.

        Args:
            schema (str): The schema containing the table.
            table (str): The table to query.

        Returns:
            Tuple[bool, Optional[List[Dict[str, Any]]], str]: List of rows or None if an error occurs.
        """
        self.connection.connect()
        data: Tuple[bool, Optional[List[Dict[str, Any]]], str] = (
            self.connection.get_table_rows(query)
        )
        self.connection.close()
        return data

    def update_metadata(self, schema: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Update and save the metadata for the datasource.

        Returns:
            Optional[Dict[str, Any]]: The updated metadata or None if an error occurs.
        """
        self.connection.connect()
        if self.metadata is None:
            self.metadata = {}
        self.metadata[schema] = self.connection.get_metadata(schema)
        self.connection.close()
        self.save()
        metadata: Optional[Dict[str, Any]] = self.metadata
        return metadata
