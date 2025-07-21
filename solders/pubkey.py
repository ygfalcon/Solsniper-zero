class Pubkey:
    def __init__(self, data: bytes | None = None):
        self._data = data or b"\0" * 32
    @classmethod
    def from_bytes(cls, data: bytes) -> "Pubkey":
        return cls(data)
    @classmethod
    def from_string(cls, s: str) -> "Pubkey":
        return cls(s.encode()[:32].ljust(32, b"\0"))
    @classmethod
    def default(cls) -> "Pubkey":
        return cls()
    def __bytes__(self) -> bytes:
        return self._data
