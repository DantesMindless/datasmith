import json
import logging

import mysql.connector
from mysql.connector import Error

from .mixins import VerifyImputsMixin

logger = logging.getLogger(__name__)


class MySQLConnection(VerifyImputsMixin):
    def __init__(self, host, database, user, password, port=3306):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.connection = None

    def connect(self) -> bool:
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

    def query(self, query, params=None):
        if self.connection is None:
            logger.error("No connection to the database.")
            return None

        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params)
            result = cursor.fetchall()
            return json.dumps(result, default=str)
        except Error:
            logger.error("Failed to execute the query.", exc_info=True)
            return None
        finally:
            cursor.close()

    def close(self):
        if self.connection:
            self.connection.close()
