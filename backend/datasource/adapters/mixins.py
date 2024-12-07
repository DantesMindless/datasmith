import inspect
from typing import Tuple, List, Optional, Set


class VerifyInputsMixin:
    @classmethod
    def verify_params(cls, params: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Verifies if the provided params include all required parameters.

        Args:
            params (List[str]): List of parameter names to verify.

        Returns:
            Tuple[bool, Optional[str]]: A tuple where the first element is a boolean indicating success,
            and the second element is a string of missing parameters if not successful, or None if successful.
        """
        signature_params = cls.get_required_params()
        params_set = set(params)
        missing_params = ", ".join(signature_params - params_set)
        return (
            (True, None)
            if signature_params.issubset(params_set)
            else (False, missing_params)
        )

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
