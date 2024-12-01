import inspect
from typing import Tuple


class VerifyImputsMixin:
    @classmethod
    def verify_params(cls, params: list) -> Tuple[bool, str | None]:
        signature_params = cls.get_requred_params()
        params_set = set(params)
        missing_params = str(signature_params - params_set)
        return (
            (
                True,
                None,
            )
            if signature_params.issubset(params_set)
            else (
                False,
                missing_params,
            )
        )

    def test_conection(self) -> bool:
        if self.connect():
            self.close()
            return True
        else:
            return False

    @classmethod
    def get_requred_params(cls) -> set:
        init_method = cls.__init__
        signature = inspect.signature(init_method)
        signature_params = set(signature.parameters.keys())
        signature_params.remove("self")
        return set(signature_params)
