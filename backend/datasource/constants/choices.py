from django.db import models

from ..adapters import (
    #     MongoDBAdapter,
    MySQLConnection,
    #     PandasMySQLQueryEngine,
    PostgresConnection,
)


class DatasourceTypeChoices(models.TextChoices):
    """
    DatasourceTypeChoices is an enumeration of different types of data sources.

    Attributes:
        POSTGRES (str): Represents a PostgreSQL datasource.
        MYSQL (str): Represents a MySQL datasource.
        MONGO (str): Represents a MongoDB datasource.
        PANDAS (str): Represents a Pandas datasource.

    Methods:
        get_adapter():
    """

    POSTGRES = "POSTGRES", "Postgres"
    MYSQL = "MYSQL", "MySQL"
    # MONGO = "MONGO", "MongoDB"
    # PANDAS = "PANDAS", "Pandas"

    def get_adapter(self):
        """
        Returns the appropriate adapter class based on the datasource type.

        Returns:
            class: The adapter class corresponding to the datasource type.

        Raises:
            ValueError: If the datasource type is invalid.
        """
        if self == DatasourceTypeChoices.POSTGRES:
            return PostgresConnection
        elif self == DatasourceTypeChoices.MYSQL:
            return MySQLConnection
        # elif self == DatasourceTypeChoices.MONGO:
        #     return MongoDBAdapter
        # elif self == DatasourceTypeChoices.PANDAS:
        # return PandasMySQLQueryEngine
        else:
            raise ValueError("Invalid datasource type")

    @classmethod
    def choices_list(cls):
        return [choice[0] for choice in cls.choices]
