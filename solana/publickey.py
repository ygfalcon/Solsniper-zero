class PublicKey(str):
    """Minimal stand-in for solana-py's PublicKey"""
    def __new__(cls, value: str):
        return str.__new__(cls, value)
