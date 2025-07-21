class VersionedTransaction:
    def __init__(self, msg):
        self.msg = msg
    @classmethod
    def populate(cls, msg, signatures):
        return cls(msg)
    def __bytes__(self) -> bytes:
        return b"tx" + (self.msg if isinstance(self.msg, bytes) else b"")
