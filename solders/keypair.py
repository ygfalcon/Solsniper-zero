import os
from .pubkey import Pubkey
class Keypair:
    def __init__(self):
        self._data = os.urandom(64)
    def to_bytes(self) -> bytes:
        return self._data
    @classmethod
    def from_bytes(cls, data: bytes) -> "Keypair":
        kp = cls.__new__(cls)
        kp._data = data
        return kp
    def pubkey(self) -> Pubkey:
        return Pubkey(self._data[:32])
