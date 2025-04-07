import logging
from typing import Optional, Tuple, List, Dict, Any, Set
from psycopg2.extras import RealDictCursor
import psycopg2
import re
from .mixins import VerifyInputsMixin, SerializerVerifyInputsMixin
from cachetools import LFUCache
from rest_framework import serializers

logger = logging.getLogger(__name__)


class PostgresConnectionSerializer(SerializerVerifyInputsMixin, serializers.Serializer):
    host = serializers.CharField(max_length=255)
    database = serializers.CharField(max_length=255)
    user = serializers.CharField(max_length=255)
    password = serializers.CharField(max_length=255, write_only=True)
    port = serializers.IntegerField(default=5432, min_value=1, max_value=65535)

    def validate_host(self, value: str) -> str:
        return super().validate_host(value)


class PostgresConnection(VerifyInputsMixin):
    serializer_class = PostgresConnectionSerializer

    def __init__(
        self,
        host: str,
        database: str,
        user: str,
        password: str,
        port: int = 5432,
    ) -> None:
        """
        Initialize a connection to the PostgreSQL database.

        Args:
            host (str): Database host.
            database (str): Database name.
            user (str): Database user.
            password (str): User password.
            port (int): Port number (default: 5432).
        """
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.connection: Optional[psycopg2.extensions.connection] = None

    def connect(self) -> bool:
        """
        Establish a connection to the PostgreSQL database.

        Returns:
            bool: True if the connection is successful, False otherwise.
        """
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port,
            )
            return True
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to the database: {e}", exc_info=True)
            self.connection = None
            return False

    def query(
        self, query: str, params: Optional[Tuple[Any]] = None
    ) -> Tuple[bool, Optional[List[Dict[str, Any]]], str]:
        """
        Execute a SQL query and return the result.

        Args:
            query (str): The SQL query to execute.
            params (Optional[Tuple[Any, ...]]): Query parameters.

        Returns:
            Tuple[bool, Optional[List[Dict[str, Any]]], str]:
                - Success status (bool),
                - Query result (List[Dict[str, Any]] or None),
                - Message (str).
        """
        if self.connection is None:
            logger.error("No connection to the database.")
            return False, None, "No connection to the database."

        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute(query, params)
            if cursor.description:
                result = cursor.fetchall()
                return True, result, "Success"
            else:
                self.connection.commit()
                return True, None, "Success"
        except psycopg2.Error as e:
            logger.error(f"Failed to execute the query: {e}", exc_info=True)
            return False, None, f"Failed to execute the query: {e}"
        finally:
            cursor.close()

    def close(self) -> None:
        """
        Close the database connection.
        """
        if self.connection:
            self.connection.close()

    def get_schemas(self) -> Tuple[bool, Optional[List[Dict[str, Any]]], str]:
        """
        Retrieve a list of schemas in the database.

        Returns:
            Tuple[bool, Optional[List[Dict[str, Any]]], str]:
                - Success status,
                - List of schema names or None,
                - Message.
        """
        query = """
                SELECT schema_name
                FROM information_schema.schemata;
                """
        success, data, message = self.query(query)
        if data:
            data = [schema["schema_name"] for schema in data]
        return success, data, message

    def get_tables(
        self, schema: str = "public"
    ) -> Tuple[bool, Optional[List[Dict[str, Any]]], str]:
        """
        Retrieve a list of tables within a schema.

        Args:
            schema (str): Schema name (default: 'public').

        Returns:
            Tuple[bool, Optional[List[Dict[str, Any]]], str]:
                - Success status,
                - List of tables or None,
                - Message.
        """
        query = f"""
        SELECT * FROM information_schema.tables as t WHERE t.table_schema = '{schema}';
        """
        return self.query(query)

    def related_tables(
        self, table_name: str
    ) -> Tuple[bool, Optional[List[Dict[str, Any]]], str]:
        """
        Retrieve related tables for a given table.

        Args:
            table_name (str): Table name.

        Returns:
            Tuple[bool, Optional[List[Dict[str, Any]]], str]:
                - Success status,
                - List of related tables or None,
                - Message.
        """
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
        success, data, message = self.query(query, (table_name,))
        if success and data:
            return True, data, "Related tables retrieved successfully."
        elif success:
            return False, None, f"No related tables found for table '{table_name}'."
        else:
            return False, None, message

    def get_tables_relations(self) -> Tuple[bool, Optional[List[Dict[str, Any]]], str]:
        """
        Retrieve related tables for all tables

        Returns:
            Tuple[bool, Optional[List[Dict[str, Any]]], str]:
                - Success status,
                - List of related tables or None,
                - Message.
        """
        query = """
    SELECT
    tc.table_name,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name
    FROM
        information_schema.table_constraints AS tc
    JOIN
        information_schema.key_column_usage AS kcu
    ON
        tc.constraint_name = kcu.constraint_name
    JOIN
        information_schema.constraint_column_usage AS ccu
    ON
        ccu.constraint_name = tc.constraint_name
    WHERE
        constraint_type = 'FOREIGN KEY';
        """

        success, data, message = self.query(query)
        if success and data:
            return True, data, "Related tables retrieved successfully."
        elif success:
            return False, None, "No related tables found"
        else:
            return False, None, message

    def get_table_columns(self, table_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve column information for a table.

        Args:
            table_name (str): Table name.

        Returns:
            Optional[List[Dict[str, Any]]]: List of columns or None.
        """
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

    def get_metadata(self, schema: str) -> Dict[str, Any]:
        """
        Retrieve metadata for all tables.

        Returns:
            Dict[str, Any]: Metadata for tables.
        """
        cache: LFUCache[str, Any] = LFUCache(maxsize=1024)
        tables: Dict[str, Any] = {}
        tables_relations: Dict[str, List[Dict[str, Any]]] = {}
        tables_list = self.get_tables(schema)[1]
        table_relations = []

        for table in tables_list:
            table_relations.append(self.related_tables(table["table_name"])[1])

        def get_relationships(
            table_name: str, scanned_tables: Optional[Set[str]] = None
        ) -> Dict[str, Any]:
            data: Optional[Dict[str, Any]] = None
            if data := cache.get(table_name, None):
                return data if isinstance(data, dict) else {}
            if scanned_tables is None:
                scanned_tables = set()
            table_relations: Dict[str, Any] = {}
            if relations := tables_relations.get(table_name):
                for related_table in relations:
                    table_relations.update(
                        {
                            related_table["related_table"]: tables[
                                related_table["related_table"]
                            ]
                        }
                    )
            cache[table_name] = table_relations
            return table_relations

        if tables_list:
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

    def get_table_rows(self, query) -> Tuple[bool, Optional[List[Dict[str, Any]]], str]:
        columns = query.get("activeColumns")
        schema = query.get("schema")
        table = query.get("table")
        page = query.get("page")
        per_page = query.get("perPage")
        offset = (page - 1) * per_page

        selects_string, joins_conditions_string = self.joins_engine(table, columns)
        if not columns:
            return "success", [], "No columns provided"
        else:
            columns = [column.split(".")[1] for column in columns if "." in column]
        query = f"SELECT {selects_string}, (SELECT COUNT(*) FROM {schema}.{table}) AS total_rows_number FROM {schema}.{table} {joins_conditions_string}LIMIT {per_page} OFFSET {offset};"
        return self.query(query)

    def joins_engine(self, table_name, columns) -> Tuple[str, str]:
        tables = set()
        scanned_columns = set()
        selects = []
        for column in columns:
            clean_column = re.sub(r"^.([A-z0-9])*\^-\^", "", column)
            if "." in column:
                column = clean_column.split(".")
                related_table_name = column[0]
                related_column_name = column[1]
                scanned_columns.add(related_column_name)
                tables.add(related_table_name)
                clean_column_value = (
                    clean_column
                    if related_column_name not in scanned_columns
                    else f"{clean_column} AS {related_table_name}_{related_column_name}"
                )
                selects.append(clean_column_value)
            else:
                tables.add(clean_column)

        if len(tables) > 0:
            relations = self.get_tables_relations()[1]
            joins_dict = PostgresConnection.generate_joins(table_name, relations)
            joins_list = PostgresConnection.concat_joins(
                joins_dict, tables, set([table_name])
            )
            joins_conditions_string = " ".join(joins_list) + " "
            selects_string = ", ".join(selects) + " "

            return selects_string, joins_conditions_string
        return "", ""

    def filters_engine(self, filters) -> Tuple[str, str]:
        return "PostgreSQL"

    @staticmethod
    def concat_joins(joins_dict, tables: list, joined_tables):
        joins = []
        for table in tables:
            if table not in joined_tables:
                if joins_string := joins_dict.get(table, {}):
                    joined_tables.add(table)
                    joins.append(joins_string.get("joiner", ""))
                    if related_joins := joins_string.get("related"):
                        joins += PostgresConnection.concat_joins(
                            related_joins, tables, joined_tables
                        )
                    break
        return joins

    @staticmethod
    def generate_joins(root_table, relations_metadata, joined=None):
        joins = {}
        joined = set([root_table]) if not joined else joined
        if relations_metadata:
            for rel in relations_metadata:
                joined_copy = set(joined) if joined else set()
                if rel["table_name"] == root_table:
                    joined_copy.add(root_table)
                    if rel["foreign_table_name"] not in joined:
                        joins.update(
                            {
                                rel["foreign_table_name"]: {
                                    "joiner": f"LEFT JOIN {rel["table_name"]} ON {rel["table_name"]}.{rel["column_name"]} = {rel["foreign_table_name"]}.{rel["foreign_column_name"]}",
                                    "related": PostgresConnection.generate_joins(
                                        rel["foreign_table_name"],
                                        relations_metadata,
                                        joined_copy,
                                    ),
                                }
                            }
                        )
                elif rel["foreign_table_name"] == root_table:
                    joined_copy.add(root_table)
                    if rel["table_name"] not in joined:
                        joins.update(
                            {
                                rel["table_name"]: {
                                    "joiner": f"LEFT JOIN {rel["table_name"]} ON {rel["foreign_table_name"]}.{rel["foreign_column_name"]} = {rel["table_name"]}.{rel["column_name"]}",
                                    "related": PostgresConnection.generate_joins(
                                        rel["table_name"], relations_metadata, joined_copy
                                    ),
                                }
                            }
                        )
        return joins
