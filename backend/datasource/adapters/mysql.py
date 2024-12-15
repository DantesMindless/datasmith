import logging
from typing import Optional, Any, Dict, List, Union, Tuple, Set
import mysql.connector
from mysql.connector import Error
import json

from .mixins import VerifyInputsMixin
from cachetools import LFUCache

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
    ) -> Tuple[bool, Optional[List[Dict[str, Any]]], str]:
        """
        Execute a query on the database and fetch results.

        Args:
            query (str): The SQL query to execute.
            params (Optional[Union[Dict[str, Any], List[Any]]]): Parameters for the query.

        Returns:
            Tuple[bool, Optional[List[Dict[str, Any]]], str]: Query success, result, and message.
        """
        if self.connection is None:
            logger.error("No connection to the database.")
            return False, None, "No connection to the database."

        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params)
            if cursor.with_rows:
                result = cursor.fetchall()
                return True, result, "Query executed successfully."
            self.connection.commit()
            return True, None, "Query executed successfully, no rows returned."
        except Error as e:
            logger.error(f"Failed to execute the query: {e}", exc_info=True)
            return False, None, f"Failed to execute the query: {e}"
        finally:
            cursor.close()

    def close(self) -> None:
        """
        Close the connection to the database.
        """
        if self.connection:
            self.connection.close()

    def get_schemas(self) -> Tuple[bool, Optional[List[str]], str]:
        """
        Retrieve a list of schemas (databases) in the MySQL server.

        Returns:
            Tuple[bool, Optional[List[str]], str]: Success status, list of schema names, and message.
        """
        query = "SELECT schema_name FROM information_schema.schemata;"
        try:
            result = self.query(query)
            if result:
                schemas = [row["schema_name"] for row in json.loads(result)]
                return True, schemas, "Schemas retrieved successfully."
            return False, None, "No schemas found."
        except Exception as e:
            return False, None, f"Error retrieving schemas: {str(e)}"

    def get_tables(self, schema: str = "public") -> Tuple[bool, Optional[List[Dict[str, Any]]], str]:
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
        query = """
        SELECT * 
        FROM information_schema.tables 
        WHERE table_schema = %s;
        """
        success, data, message = self.query(query, (schema,))
        if success and data:
            return True, data, "Tables retrieved successfully."
        elif success:
            return False, None, f"No tables found in schema '{schema}'."
        else:
            return False, None, message

    def related_tables(
            self, table_name: str
    ) -> Tuple[bool, Optional[List[Dict[str, Any]]], str]:
        """
        Retrieve related tables for the specified table.

        Args:
            table_name (str): The name of the table.

        Returns:
            Tuple[bool, Optional[List[Dict[str, Any]]], str]:
                - Success status,
                - List of related tables or None,
                - Message.
        """
        query = """
        SELECT
            kcu.table_name AS related_table,
            kcu.column_name AS related_column,
            ccu.table_name AS target_table,
            ccu.column_name AS target_column
        FROM
            information_schema.key_column_usage AS kcu
        JOIN
            information_schema.table_constraints AS tc
        ON
            kcu.constraint_name = tc.constraint_name
        AND
            kcu.table_schema = tc.table_schema
        JOIN
            information_schema.constraint_column_usage AS ccu
        ON
            ccu.constraint_name = tc.constraint_name
        AND
            ccu.table_schema = tc.table_schema
        WHERE
            tc.constraint_type = 'FOREIGN KEY'
        AND
            kcu.table_name = %s;
        """
        try:
            result = self.query(query, (table_name,))
            if result:
                data = json.loads(result)
                return True, data, "Related tables retrieved successfully."
            else:
                return False, None, f"No related tables found for table '{table_name}'."
        except Exception as e:
            return False, None, f"Error retrieving related tables: {str(e)}"

    def get_table_columns(self, table_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve a list of columns for the specified table.

        Args:
            table_name (str): The name of the table.

        Returns:
            Optional[List[Dict[str, Any]]]: A list of column details or None if the query fails.
        """
        query = """
        SELECT 
            column_name, 
            data_type, 
            character_maximum_length, 
            is_nullable 
        FROM 
            information_schema.columns 
        WHERE 
            table_name = %s;
        """
        result = self.query(query, (table_name,))
        if result:
            return json.loads(result)
        return None

    def get_metadata(self) -> Dict[str, Any]:
        """
        Retrieve metadata for all tables in the current schema.

        Returns:
            Dict[str, Any]: Metadata for tables.
        """
        cache: LFUCache[str, Any] = LFUCache(maxsize=1024)
        tables: Dict[str, Any] = {}
        tables_relations: Dict[str, List[Dict[str, Any]]] = {}
        tables_list = self.get_tables()

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
                    if sub_relations := get_relationships(
                            related_table["related_table"], scanned_tables
                    ):
                        table_relations[related_table["related_table"]][
                            "relations"
                        ].append(sub_relations)
                    scanned_tables.add(related_table["related_table"])
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

