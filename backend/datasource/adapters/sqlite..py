import logging
import sqlite3
from typing import Tuple

class SQLiteConnection:
    def __init__(self, database: str):
        self.database = database
        self.connection = None

    def connect(self) -> bool:
        try:
            self.connection = sqlite3.connect(self.database)
            return True
        except sqlite3.Error as e:
            logging.error(f"Failed to connect to SQLite database: {e}")
            self.connection = None
            return False

    def query(self, query: str, params: Tuple = None) -> Tuple[bool, list, str]:
        if not self.connection:
            return False, [], "No connection to the SQLite database."

        cursor = self.connection.cursor()
        try:
            cursor.execute(query, params or ())
            if cursor.description:
                result = cursor.fetchall()
                return True, result, "Query executed successfully."
            else:
                self.connection.commit()
                return True, [], "Query executed successfully without results."
        except sqlite3.Error as e:
            logging.error(f"Failed to execute SQLite query: {e}")
            return False, [], str(e)
        finally:
            cursor.close()

    def close(self):
        if self.connection:
            self.connection.close()
