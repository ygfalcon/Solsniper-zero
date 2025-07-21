class AsyncClient:
    def __init__(self, url: str):
        self.url = url
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        pass
    async def get_fees(self):
        return {}
    async def send_raw_transaction(self, data, opts=None):
        class Resp:
            value = "sig"
        return Resp()
