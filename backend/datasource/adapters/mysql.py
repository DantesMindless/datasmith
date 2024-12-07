import json
import logging
from typing import Optional, Any, Dict, List, Union

import mysql.connector
from mysql.connector import Error

from .mixins import VerifyInputsMixin

logger = logging.getLogger(__name__)


class MySQLConnection(VerifyInputsMixin):
    def __init__(
        self,
        host: str,
        database: str,
        user: str,
        password: str,
        port: int = 3306,
    ) -> None:
        """
        Initialize the MySQL connection.

        Args:
            host (str): The hostname of the MySQL server.
            database (str): The name of the database.
            user (str): The username for authentication.
            password (str): The password for authentication.
            port (int): The port number of the MySQL server (default: 3306).
        """
        self.host: str = host
        self.database: str = database
        self.user: str = user
        self.password: str = password
        self.port: int = port
        self.connection: Optional[mysql.connector.MySQLConnection] = None

    def connect(self) -> bool:
        """
        Establish a connection to the MySQL database.

        Returns:
            bool: True if the connection is successful, False otherwise.
        """
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port,
            )
            return True
        except Error:
            logger.error("Failed to connect to the database.", exc_info=True)
            self.connection = None
            return False

    def query(
        self, query: str, params: Optional[Union[Dict[str, Any], List[Any]]] = None
    ) -> Optional[str]:
        """
        Execute a query on the database and fetch results.

        Args:
            query (str): The SQL query to execute.
            params (Optional[Union[Dict[str, Any], List[Any]]]): Parameters for the query, if any.

        Returns:
            Optional[str]: JSON-encoded results of the query, or None if the query fails.
        """
        if self.connection is None:
            logger.error("No connection to the database.")
            return None

        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params)
            result: List[Dict[str, Any]] = cursor.fetchall()
            return json.dumps(result, default=str)
        except Error:
            logger.error("Failed to execute the query.", exc_info=True)
            return None
        finally:
            cursor.close()

    def close(self) -> None:
        """
        Close the connection to the database.
        """
        if self.connection:
            self.connection.close()
