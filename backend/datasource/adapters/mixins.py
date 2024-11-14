import inspect


class VerifyImputsMixin:
    @classmethod
    def verify_params(cls, params: list) -> bool:
        init_method = cls.__init__
        signature = inspect.signature(init_method)
        signature_params = set(signature.parameters.keys())
        print(signature_params)
        print([param in signature_params for param in params])
        print([param for param in params])
        return all([param in signature_params for param in params])

    def test_conection(self) -> bool:
        if self.connect():
            self.close()
            return True
        else:
            return False
