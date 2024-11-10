import json

import psycopg2
from psycopg2.extras import RealDictCursor


class PostgresConnection:
    def __init__(self, host, database, user, password, port=5432):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.connection = None

    def connect(self):
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port,
            )
        except psycopg2.Error as e:
            print(f"Error connecting to PostgreSQL database: {e}")
            self.connection = None

    def query(self, query, params=None):
        if self.connection is None:
            print("No connection to the database.")
            return None

        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                result = cursor.fetchall()
                return json.dumps(result, default=str)
        except psycopg2.Error as e:
            print(f"Error executing query: {e}")
            return None

    def close(self):
        if self.connection:
            self.connection.close()


# Example usage:
if __name__ == "__main__":
    conn = PostgresConnection(
        host="your_host",
        database="your_database",
        user="your_user",
        password="your_password",
    )
    conn.connect()
    result = conn.query("SELECT * FROM your_table")
    print(result)
    conn.close()
