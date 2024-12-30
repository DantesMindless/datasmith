import logging
from typing import Optional, Any, Dict, List, Tuple
import redis
from redis import exceptions as redis_exceptions

from .mixins import VerifyInputsMixin, SerializerVerifyInputsMixin
from rest_framework import serializers

logger = logging.getLogger(__name__)


class RedisConnectionSerializer(
    VerifyInputsMixin, SerializerVerifyInputsMixin, serializers.Serializer
):
    host = serializers.CharField(max_length=255)
    database = serializers.CharField(default="0", max_length=255, required=False, allow_blank=True, allow_null=True)
    user = serializers.CharField(default="", max_length=255, required=False, allow_blank=True, allow_null=True)
    password = serializers.CharField(max_length=255, write_only=True, required=False, allow_null=True, allow_blank=True)
    port = serializers.IntegerField(default=6379, min_value=1, max_value=65535)

    def validate_host(self, value: str) -> str:
        return super().validate_host(value)

    def validate_database(self, value: str) -> str:
        """Convert database string to integer if possible"""
        if value == '':
            value = '0'
        try:
            int(value)
            return value
        except ValueError:
            raise serializers.ValidationError("Database must be a number for Redis")


class RedisConnection(VerifyInputsMixin):
    serializer_class = RedisConnectionSerializer

    def __init__(
            self,
            host: str,
            database: str,
            user: Optional[str] = None,
            password: Optional[str] = None,
            port: int = 6379,
    ) -> None:
        """
        Initialize the Redis connection.

        Args:
            host (str): The hostname of the Redis server.
            database (str): The database number as string (will be converted to int, could be empty string for database = 0).
            user (Optional[str]): The username for authentication (optional in Redis, could be empty string for user = 'default').
            password (Optional[str]): The password for authentication.
            port (int): The port number of the Redis server (default: 6379).
        """
        self.host: str = host
        if database == '':
            database = '0'
        self.db: int = int(database)
        if user == '':
            user = 'default'
        self.user: Optional[str] = user
        self.password: Optional[str] = password
        self.port: int = port
        self.connection: Optional[redis.Redis] = None

    def connect(self) -> bool:
        """
        Establish a connection to the Redis database.

        Returns:
            bool: True if the connection is successful, False otherwise.
        """
        try:
            connect_kwargs = {
                "host": self.host,
                "port": self.port,
                "db": self.db,
                "decode_responses": True
            }

            if self.password:
                connect_kwargs["password"] = self.password
            if self.user:
                connect_kwargs["username"] = self.user

            self.connection = redis.Redis(**connect_kwargs)
            # Test connection
            self.connection.ping()
            return True
        except redis_exceptions.RedisError:
            logger.error("Failed to connect to Redis.", exc_info=True)
            self.connection = None
            return False

    def query(
            self, key: str, operation: str = "GET"
    ) -> Tuple[bool, Optional[Any], str]:
        """
        Execute a Redis command.

        Args:
            key (str): The Redis key to operate on.
            operation (str): The Redis operation to perform (e.g., "GET", "SET", "HGETALL").

        Returns:
            Tuple[bool, Optional[Any], str]: Operation success, result, and message.
        """
        if self.connection is None:
            logger.error("No connection to Redis.")
            return False, None, "No connection to Redis."

        try:
            operation = operation.upper()
            if operation == "GET":
                result = self.connection.get(key)
            elif operation == "HGETALL":
                result = self.connection.hgetall(key)
            elif operation == "SMEMBERS":
                result = list(self.connection.smembers(key))
            elif operation == "ZRANGE":
                result = list(self.connection.zrange(key, 0, -1))
            elif operation == "LRANGE":
                result = self.connection.lrange(key, 0, -1)
            else:
                return False, None, f"Unsupported operation: {operation}"

            if result:
                return True, result, "Operation executed successfully."
            return True, None, "Operation executed successfully, no data returned."
        except redis_exceptions.RedisError as e:
            logger.error(f"Failed to execute the operation: {e}", exc_info=True)
            return False, None, f"Failed to execute the operation: {e}"

    def close(self) -> None:
        """
        Close the connection to Redis.
        """
        if self.connection:
            self.connection.close()

    def get_databases(self) -> Tuple[bool, Optional[List[int]], str]:
        """
        Retrieve a list of available Redis databases.

        Returns:
            Tuple[bool, Optional[List[int]], str]: Success status, list of database numbers, and message.
        """
        try:
            if self.connection:
                # Get database size for all databases (0-15 by default)
                databases = []
                for db_num in range(16):  # Redis default database range
                    temp_conn = redis.Redis(
                        host=self.host,
                        port=self.port,
                        db=db_num,
                        password=self.password,
                        decode_responses=True
                    )
                    if temp_conn.dbsize() > 0:
                        databases.append(db_num)
                    temp_conn.close()
                return True, databases, "Databases retrieved successfully."
            return False, None, "No connection to Redis."
        except redis_exceptions.RedisError as e:
            return False, None, f"Error retrieving databases: {str(e)}"

    def get_keys(
            self, pattern: str = "*"
    ) -> Tuple[bool, Optional[List[str]], str]:
        """
        Retrieve a list of keys matching the pattern.

        Args:
            pattern (str): Pattern to match keys (default: "*").

        Returns:
            Tuple[bool, Optional[List[str]], str]:
                - Success status,
                - List of keys or None,
                - Message.
        """
        try:
            if self.connection:
                keys = list(self.connection.keys(pattern))
                if keys:
                    return True, keys, "Keys retrieved successfully."
                return False, None, f"No keys found matching pattern '{pattern}'."
            return False, None, "No connection to Redis."
        except redis_exceptions.RedisError as e:
            return False, None, f"Error retrieving keys: {str(e)}"

    def get_key_type(self, key: str) -> Optional[str]:
        """
        Get the type of a Redis key.

        Args:
            key (str): The key to check.

        Returns:
            Optional[str]: The type of the key or None if the key doesn't exist.
        """
        try:
            if self.connection:
                return self.connection.type(key)
            return None
        except redis_exceptions.RedisError:
            return None

    def get_metadata(self) -> Dict[str, Any]:
        """
        Retrieve metadata about the Redis database.

        Returns:
            Dict[str, Any]: Metadata including key patterns, types, and sizes.
        """
        metadata: Dict[str, Any] = {
            "keys": {},
            "statistics": {}
        }

        if not self.connection:
            return metadata

        try:
            # Get general database statistics
            info = self.connection.info()
            metadata["statistics"] = {
                "total_keys": self.connection.dbsize(),
                "memory_used": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "uptime_days": info.get("uptime_in_days")
            }

            # Get key patterns and their types
            keys = self.connection.keys("*")
            for key in keys:
                key_type = self.get_key_type(key)
                if key_type:
                    if key_type not in metadata["keys"]:
                        metadata["keys"][key_type] = []

                    # Get additional information based on key type
                    key_info = {
                        "name": key,
                        "ttl": self.connection.ttl(key)
                    }

                    if key_type == "string":
                        key_info["length"] = len(self.connection.get(key) or "")
                    elif key_type == "list":
                        key_info["length"] = self.connection.llen(key)
                    elif key_type == "set":
                        key_info["length"] = self.connection.scard(key)
                    elif key_type == "hash":
                        key_info["length"] = self.connection.hlen(key)
                    elif key_type == "zset":
                        key_info["length"] = self.connection.zcard(key)

                    metadata["keys"][key_type].append(key_info)

        except redis_exceptions.RedisError as e:
            logger.error(f"Error retrieving metadata: {e}", exc_info=True)

        self.close()
        return metadata
