class RpcTransactionLogsFilterMentions:
    def __init__(self, pubkey):
        self.pubkey = pubkey

class _DummyWS:
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        pass
    async def logs_subscribe(self, *args, **kwargs):
        pass
    async def recv(self):
        return []

aSyncWS = _DummyWS

async def connect(url: str):
    return _DummyWS()
