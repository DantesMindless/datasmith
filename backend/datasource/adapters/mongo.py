# from .mixins import VerifyImputsMixin

# # from pymongo import MongoClient


# class MongoDBAdapter(VerifyImputsMixin):
#     # def __init__(self, uri, database_name):
#     #     self.client = MongoClient(uri)
#     #     self.database = self.client[database_name]

#     def find(self, collection_name, query):
#         collection = self.database[collection_name]
#         return collection.find(query)

#     def find_one(self, collection_name, query):
#         collection = self.database[collection_name]
#         return collection.find_one(query)

#     def insert_one(self, collection_name, document):
#         collection = self.database[collection_name]
#         return collection.insert_one(document)

#     def insert_many(self, collection_name, documents):
#         collection = self.database[collection_name]
#         return collection.insert_many(documents)

#     def update_one(self, collection_name, query, update):
#         collection = self.database[collection_name]
#         return collection.update_one(query, update)

#     def delete_one(self, collection_name, query):
#         collection = self.database[collection_name]
#         return collection.delete_one(query)

#     def close(self):
#         self.client.close()

import logging
from typing import Optional, Any
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from .mixins import VerifyInputsMixin, SerializerVerifyInputsMixin

from rest_framework import serializers

logger = logging.getLogger(__name__)


class MongoDBConnectionSerializer(
    VerifyInputsMixin, SerializerVerifyInputsMixin, serializers.Serializer
):
    host = serializers.CharField(max_length=255)
    database = serializers.CharField(max_length=255)
    user = serializers.CharField(max_length=255, required=False)
    password = serializers.CharField(max_length=255, write_only=True, required=False)
    port = serializers.IntegerField(default=27017, min_value=1, max_value=65535)

    def validate_host(self, value: str) -> str:
        return super().validate_host(value)


class MongoDBConnection(VerifyInputsMixin):
    serializer_class = MongoDBConnectionSerializer

    def __init__(
        self,
        host: str,
        database: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        port: int = 27017,
    ) -> None:
        """
        Initialize the MongoDB connection.

        Args:
            host (str): The hostname of the MongoDB server.
            database (str): The name of the database.
            user (Optional[str]): The username for authentication.
            password (Optional[str]): The password for authentication.
            port (int): The port number of the MongoDB server (default: 27017).
        """
        self.host: str = host
        self.database: str = database
        self.user: Optional[str] = user
        self.password: Optional[str] = password
        self.port: int = port
        self.client: Optional[MongoClient] = None
        self.db: Optional[Any] = None

    def connect(self) -> bool:
        """
        Establish a connection to the MongoDB database.

        Returns:
            bool: True if the connection is successful, False otherwise.
        """
        try:
            connection_string = "mongodb://"
            if self.user and self.password:
                connection_string += f"{self.user}:{self.password}@"
            connection_string += f"{self.host}:{self.port}"

            self.client = MongoClient(connection_string)
            self.db = self.client[self.database]
            # Verify connection by executing a simple command
            self.client.admin.command("ping")
            return True
        except ConnectionFailure:
            logger.error("Failed to connect to the database.", exc_info=True)
            self.client = None
            self.db = None
            return False

    # def query(
    #         self,
    #         collection: str,
    #         operation: str,
    #         filter_dict: Optional[Dict[str, Any]] = None,
    #         data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    #         options: Optional[Dict[str, Any]] = None
    # ) -> Tuple[bool, Optional[List[Dict[str, Any]]], str]:
    #     """
    #     Execute a query on the database and fetch results.
    #
    #     Args:
    #         collection (str): The name of the collection to query.
    #         operation (str): Type of operation ('find', 'insert', 'update', 'delete').
    #         filter_dict (Optional[Dict[str, Any]]): Filter criteria for the operation.
    #         data (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]): Data for insert/update operations.
    #         options (Optional[Dict[str, Any]]): Additional options for the operation.
    #
    #     Returns:
    #         Tuple[bool, Optional[List[Dict[str, Any]]], str]: Operation success, result, and message.
    #     """
    #     if self.db is None:
    #         logger.error("No connection to the database.")
    #         return False, None, "No connection to the database."
    #
    #     try:
    #         collection_obj = self.db[collection]
    #         result = None
    #
    #         if operation == 'find':
    #             cursor = collection_obj.find(filter_dict or {}, **(options or {}))
    #             result = list(cursor)
    #             return True, result, "Query executed successfully."
    #
    #         elif operation == 'insert':
    #             if isinstance(data, list):
    #                 result = collection_obj.insert_many(data)
    #                 return True, None, f"Inserted {len(result.inserted_ids)} documents."
    #             else:
    #                 result = collection_obj.insert_one(data)
    #                 return True, None, f"Inserted document with ID: {result.inserted_id}"
    #
    #         elif operation == 'update':
    #             if filter_dict and data:
    #                 result = collection_obj.update_many(filter_dict, {'$set': data})
    #                 return True, None, f"Updated {result.modified_count} documents."
    #
    #         elif operation == 'delete':
    #             if filter_dict:
    #                 result = collection_obj.delete_many(filter_dict)
    #                 return True, None, f"Deleted {result.deleted_count} documents."
    #
    #         return False, None, "Invalid operation specified."
    #
    #     except OperationFailure as e:
    #         logger.error(f"Failed to execute the operation: {e}", exc_info=True)
    #         return False, None, f"Failed to execute the operation: {e}"

    def close(self) -> None:
        """
        Close the connection to the database.
        """
        if self.client:
            self.client.close()

    # def get_databases(self) -> Tuple[bool, Optional[List[str]], str]:
    #     """
    #     Retrieve a list of databases in the MongoDB server.
    #
    #     Returns:
    #         Tuple[bool, Optional[List[str]], str]: Success status, list of database names, and message.
    #     """
    #     try:
    #         if self.client:
    #             databases = self.client.list_database_names()
    #             return True, databases, "Databases retrieved successfully."
    #         return False, None, "No connection to the server."
    #     except Exception as e:
    #         return False, None, f"Error retrieving databases: {str(e)}"
    #
    # def get_collections(self) -> Tuple[bool, Optional[List[str]], str]:
    #     """
    #     Retrieve a list of collections in the current database.
    #
    #     Returns:
    #         Tuple[bool, Optional[List[Dict[str, Any]]], str]: Success status, list of collections, and message.
    #     """
    #     try:
    #         if self.db:
    #             collections = self.db.list_collection_names()
    #             return True, collections, "Collections retrieved successfully."
    #         return False, None, "No database selected."
    #     except Exception as e:
    #         return False, None, f"Error retrieving collections: {str(e)}"
    #
    # def get_collection_metadata(self, collection_name: str) -> Optional[Dict[str, Any]]:
    #     """
    #     Retrieve metadata for a specific collection.
    #
    #     Args:
    #         collection_name (str): The name of the collection.
    #
    #     Returns:
    #         Optional[Dict[str, Any]]: Collection metadata or None if retrieval fails.
    #     """
    #     try:
    #         if self.db:
    #             # Get sample documents to infer schema
    #             sample_docs = list(self.db[collection_name].aggregate([
    #                 {'$sample': {'size': 100}},
    #                 {'$project': {'_id': 0}}
    #             ]))
    #
    #             # Analyze field types from sample documents
    #             field_types = {}
    #             for doc in sample_docs:
    #                 for field, value in doc.items():
    #                     if field not in field_types:
    #                         field_types[field] = set()
    #                     field_types[field].add(type(value).__name__)
    #
    #             # Get collection stats
    #             stats = self.db.command('collStats', collection_name)
    #
    #             return {
    #                 'name': collection_name,
    #                 'document_count': stats.get('count', 0),
    #                 'size': stats.get('size', 0),
    #                 'fields': {
    #                     field: list(types) for field, types in field_types.items()
    #                 },
    #                 'indexes': list(self.db[collection_name].list_indexes())
    #             }
    #         return None
    #     except Exception as e:
    #         logger.error(f"Error retrieving collection metadata: {e}", exc_info=True)
    #         return None
    #
    # def get_metadata(self) -> Dict[str, Any]:
    #     """
    #     Retrieve metadata for all collections in the current database.
    #
    #     Returns:
    #         Dict[str, Any]: Metadata for collections.
    #     """
    #     cache: LFUCache[str, Any] = LFUCache(maxsize=1024)
    #     metadata: Dict[str, Any] = {}
    #
    #     success, collections, _ = self.get_collections()
    #     if success and collections:
    #         for collection in collections:
    #             if cached_data := cache.get(collection):
    #                 metadata[collection] = cached_data
    #             else:
    #                 if collection_metadata := self.get_collection_metadata(collection):
    #                     metadata[collection] = collection_metadata
    #                     cache[collection] = collection_metadata
    #
    #     self.close()
    #     return metadata
