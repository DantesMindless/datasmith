import inspect
from typing import Tuple, Optional, Set, Any, Dict, Union
from rest_framework import serializers
import re


class SerializerVerifyInputsMixin:
    def validate_host(self, value: str) -> str:
        # Basic validation for a hostname or IP address
        hostname_regex = re.compile(r"^(?!-)[A-Z\d-]{1,63}(?<!-)$", re.IGNORECASE)
        ip_regex = re.compile(
            r"^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$"
        )
        docker_regex = re.compile(r"^host\.docker\.internal(:\d{1,5})?$")
        if not any(
            (
                hostname_regex.match(value),
                ip_regex.match(value),
                docker_regex.match(value),
            )
        ):
            raise serializers.ValidationError("Invalid hostname or IP address.")
        return value


class VerifyInputsMixin:
    serializer_class: serializers.Serializer = None

    @classmethod
    def verify_params(cls, params: Dict[str, Any]) -> Tuple[bool, Optional[Any]]:
        """
        Verifies the given parameters using the serializer class.

        Args:
            params (Dict[str, Any]): The parameters to be verified.

        Returns:
            Tuple[bool, Optional[Any]]: A tuple where the first element is a boolean indicating
            whether the parameters are valid, and the second element is None if valid, or the
            serializer errors if invalid.

        Raises:
            ValueError: If the serializer_class is not defined.
        """
        if cls.serializer_class is None:
            raise ValueError("serializer_class is not defined")
        serializer = cls.serializer_class(data=params)
        if serializer.is_valid():
            return True, None
        return False, serializer.errors

    @classmethod
    def get_initial_params(cls) -> Dict[str, Dict[str, Union[str, int]]]:
        """
        Retrieves the initial parameters for the adapter.

        Returns:
            Dict[str, Any]: A dictionary of initial parameters.
        """
        fields: Dict[str, Dict[str, Union[str, bool, int]]] = {}
        if cls.serializer_class is None:
            raise ValueError("serializer_class is not defined")
        serializer_fields = cls.serializer_class().get_fields()
        for field in serializer_fields:
            fields[field] = {
                "initial": serializer_fields[field].initial,
                "required": serializer_fields[field].required,
                "type": serializer_fields[field].__class__.__name__,
            }
        return fields

    def test_connection(self) -> bool:
        """
        Tests the connection by attempting to connect and then closing the connection.

        Returns:
            bool: True if connection is successful, False otherwise.
        """
        if self.connect():  # type: ignore
            self.close()  # type: ignore
            return True
        return False

    @classmethod
    def get_required_params(cls) -> Set[str]:
        """
        Retrieves the set of required parameters from the class's `__init__` method.

        Returns:
            Set[str]: A set of parameter names.
        """
        init_method = cls.__init__
        signature = inspect.signature(init_method)
        signature_params = set(signature.parameters.keys())
        signature_params.discard("self")
        return signature_params
