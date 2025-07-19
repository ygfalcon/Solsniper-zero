
from solders.pubkey import Pubkey

class PublicKey:
    def __init__(self, key: str | bytes):
        if isinstance(key, bytes):
            self._key = Pubkey.from_bytes(key)
        else:
            self._key = Pubkey.from_string(str(key))

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self._key)

    def __bytes__(self) -> bytes:  # pragma: no cover - trivial
        return bytes(self._key)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PublicKey):
            return self._key == other._key
        return False

__all__ = ["PublicKey"]
