import os
class Hash:
    def __init__(self, data: bytes):
        self._data = data
    @classmethod
    def new_unique(cls) -> "Hash":
        return cls(os.urandom(32))
    def __bytes__(self) -> bytes:
        return self._data
