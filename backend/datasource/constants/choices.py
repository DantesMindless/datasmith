from django.db import models
from typing import List, Optional, Type

from ..adapters import (
    # MongoDBAdapter,
    PostgresConnection,
)

import logging

logger = logging.getLogger(__name__)


class DatasourceTypeChoices(models.TextChoices):
    """
    DatasourceTypeChoices is an enumeration of different types of data sources.

    Attributes:
        POSTGRES (str): Represents a PostgreSQL datasource.
        MYSQL (str): Represents a MySQL datasource.

    Methods:
        get_adapter(): Returns the appropriate adapter class based on the datasource type.
        choices_list(): Returns a list of all available datasource types.
    """

    POSTGRES = "POSTGRES", "Postgres"
    MYSQL = "MYSQL", "MySQL"
    # MONGO = "MONGO", "MongoDB"
    # PANDAS = "PANDAS", "Pandas"

    def get_adapter(self) -> Optional[Type[PostgresConnection]]:
        """
        Returns the appropriate adapter class based on the datasource type.

        Returns:
            Optional[Type]: The adapter class corresponding to the datasource type,
                            or None if the type is invalid.
        """
        if self == DatasourceTypeChoices.POSTGRES:
            return PostgresConnection
        # elif self == DatasourceTypeChoices.MYSQL:
        #     return MySQLConnection
        # elif self == DatasourceTypeChoices.MONGO:
        #     return MongoDBAdapter
        # elif self == DatasourceTypeChoices.PANDAS:
        #     return PandasMySQLQueryEngine
        else:
            logger.error(f"Invalid datasource type: {self}")
            return None

    @classmethod
    def choices_list(cls) -> List[str]:
        """
        Returns a list of all available datasource types.

        Returns:
            List[str]: A list of datasource type identifiers.
        """
        return [choice[0] for choice in cls.choices]
