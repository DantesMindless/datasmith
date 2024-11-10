import json

import mysql.connector
from mysql.connector import Error


class MySQLConnection:
    def __init__(self, host, database, user, password, port=3306):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port,
            )
        except Error as e:
            print(f"Error connecting to MySQL database: {e}")
            self.connection = None

    def query(self, query, params=None):
        if self.connection is None:
            print("No connection to the database.")
            return None

        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params)
            result = cursor.fetchall()
            return json.dumps(result, default=str)
        except Error as e:
            print(f"Error executing query: {e}")
            return None
        finally:
            cursor.close()

    def close(self):
        if self.connection:
            self.connection.close()


# Example usage:
if __name__ == "__main__":
    conn = MySQLConnection(
        host="your_host",
        database="your_database",
        user="your_user",
        password="your_password",
    )
    conn.connect()
    result = conn.query("SELECT * FROM your_table")
    print(result)
    conn.close()
