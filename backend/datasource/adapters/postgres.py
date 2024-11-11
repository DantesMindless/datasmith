import json
import logging

import psycopg2
from psycopg2.extras import RealDictCursor

from .mixins import VerifyImputsMixin

logger = logging.getLogger(__name__)


class PostgresConnection(VerifyImputsMixin):
    def __init__(self, host, database, user, password, port=5432):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.connection = None

    def connect(self) -> bool:
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port,
            )
            return True
        except psycopg2.Error:
            logger.error("Failed to connect to the database.", exc_info=True)
            self.connection = None
            return False

    def query(self, query, params=None):
        if self.connection is None:
            logger.error("No connection to the database.")
            return None

        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                result = cursor.fetchall()
                return json.dumps(result, default=str)
        except psycopg2.Error as e:
            logger.error("Failed to execute the query.", exc_info=True)
            return None

    def close(self):
        if self.connection:
            self.connection.close()
