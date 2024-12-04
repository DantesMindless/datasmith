import logging

import psycopg2
from typing import Tuple
from psycopg2.extras import RealDictCursor

from .mixins import VerifyImputsMixin
from cachetools import LFUCache

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

    def query(self, query, params=None) -> Tuple[bool, dict | None]:
        if self.connection is None:
            logger.error("No connection to the database.")
            return False, None, "No connection to the database."

        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute(query, params)

            if cursor.description:
                result = cursor.fetchall()
                return True, result, "No Data"
            else:
                self.connection.commit()
                return True, None, "Success"
        except Exception as e:
            logger.error("Failed to execute the query.", exc_info=True)
            return False, None, f"Failed to execute the query: {e}"
        finally:
            cursor.close()

    def close(self):
        if self.connection:
            self.connection.close()

    def get_tables(self):
        query = """
        SELECT * FROM information_schema.tables as t WHERE t.table_schema = 'public';
        """
        return self.query(query)[1]

    def related_tables(self, table_name):
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
        AND
            tc.table_schema = kcu.table_schema
        JOIN
            information_schema.constraint_column_usage AS ccu
        ON
            ccu.constraint_name = tc.constraint_name
        AND
            ccu.table_schema = tc.table_schema
        WHERE
            tc.constraint_type = 'FOREIGN KEY'
        AND
            ccu.table_name = %s;
        """
        return self.query(query, [table_name])

    def get_table_columns(self, table_name):
        query = f"""
                SELECT
                    column_name,
                    data_type,
                    character_maximum_length,
                    is_nullable,
                    udt_name AS enum_type,
                    CASE
                        WHEN data_type = 'USER-DEFINED' THEN
                            (SELECT array_agg(enumlabel)
                            FROM pg_enum
                            WHERE enumtypid = (
                                SELECT oid
                                FROM pg_type
                                WHERE typname = udt_name
                            ))
                        ELSE NULL
                    END AS enum_values
                FROM
                    information_schema.columns
                WHERE
                    table_name = '{table_name}';
                """
        return self.query(query)[1]

    def get_metadata(self):
        cache = LFUCache(maxsize=1024)
        tables = {}
        tables_relations = {}
        tables_list = self.get_tables()

        def get_relationships(table_name, scanned_tables=None):
            if data := cache.get(table_name):
                return data
            if scanned_tables is None:
                scanned_tables = set()
            table_relations = {}
            if relations := tables_relations.get(table_name):
                for related_table in relations:
                    table_relations.update(
                        {
                            related_table["related_table"]: tables[
                                related_table["related_table"]
                            ]
                        }
                    )
                    if sub_relations := get_relationships(
                        related_table["related_table"], scanned_tables
                    ):
                        table_relations[related_table["related_table"]][
                            "relations"
                        ].append(sub_relations)
                    scanned_tables.add(related_table["related_table"])
            cache[table_name] = table_relations
            return table_relations

        for table in tables_list:
            table_name = table["table_name"]
            tables.update(
                {
                    table_name: {
                        "fields": self.get_table_columns(table_name),
                        "relations": [],
                    }
                }
            )
            if relations := self.related_tables(table_name)[1]:
                tables_relations[table_name] = relations

        for table_name in tables:
            if tables_relations.get(table_name):
                tables[table_name]["relations"].append(get_relationships(table_name))

        self.close()
        return tables
