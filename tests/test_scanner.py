from solhunter_zero import scanner
from solhunter_zero import scanner_common

data = {"data": [{"address": "abcbonk"}, {"address": "otherbonk"}]}

class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception('bad status')

    def json(self):
        return self._data

def test_scan_tokens_websocket(monkeypatch):
    async def fake_stream(url, *, suffix="bonk", include_pools=True):
        yield "webbonk"


    captured = {}

    def fake_get(url, headers=None, timeout=10):
        captured['headers'] = headers
        return FakeResponse(data)

    monkeypatch.setattr(scanner.requests, 'get', fake_get)
    monkeypatch.setattr(scanner, 'fetch_trending_tokens', lambda: ['trend'])
    monkeypatch.setattr(scanner, 'fetch_raydium_listings', lambda: ['ray'])
    monkeypatch.setattr(scanner, 'fetch_orca_listings', lambda: ['orca'])
    scanner_common.BIRDEYE_API_KEY = "test"
    scanner_common.HEADERS.clear()
    scanner_common.HEADERS["X-API-KEY"] = "test"

    tokens = scanner.scan_tokens()
    assert tokens == ['abcbonk', 'xyzBONK', 'trend', 'ray', 'orca']
    assert captured['headers'] == scanner.HEADERS




# codex/add-offline-option-to-solhunter_zero.main
def test_scan_tokens_offline(monkeypatch):
    called = {}

    def fake_get(*args, **kwargs):
        called['called'] = True
        return FakeResponse({}, 200)

    monkeypatch.setattr("solhunter_zero.websocket_scanner.stream_new_tokens", lambda *a, **k: (_ for _ in ()).throw(AssertionError('ws')))
    monkeypatch.setattr(scanner.requests, 'get', fake_get)
    monkeypatch.setattr(scanner, 'fetch_trending_tokens', lambda: (_ for _ in ()).throw(AssertionError('trending')))

    tokens = scanner.scan_tokens(offline=True)
    assert tokens == scanner.OFFLINE_TOKENS
    assert 'called' not in called


def test_scan_tokens_onchain(monkeypatch):
    captured = {}

    def fake_onchain(url):
        captured['url'] = url
        return ['tok']

    def fake_get(*args, **kwargs):
        raise AssertionError('should not call BirdEye')

    monkeypatch.setattr(scanner, 'scan_tokens_onchain', fake_onchain)
    monkeypatch.setattr(scanner.requests, 'get', fake_get)
    monkeypatch.setattr(scanner, 'fetch_trending_tokens', lambda: ['t2'])
    monkeypatch.setattr(scanner, 'fetch_raydium_listings', lambda: [])
    monkeypatch.setattr(scanner, 'fetch_orca_listings', lambda: [])

    scanner_common.SOLANA_RPC_URL = 'http://node'

    tokens = scanner.scan_tokens(method="onchain")
    assert tokens == ['tok', 't2']
    assert captured['url'] == 'http://node'


import asyncio
from solhunter_zero.async_scanner import scan_tokens_async as async_scan


def test_scan_tokens_async(monkeypatch):
    data = {"data": [{"address": "abcbonk"}, {"address": "otherbonk"}]}
    class FakeResp:
        status = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        async def json(self):
            return data
        def raise_for_status(self):
            pass

    class FakeSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        def get(self, url, headers=None, timeout=10):
            return FakeResp()

    monkeypatch.setattr("aiohttp.ClientSession", lambda: FakeSession())
    async def fake_trend():
        return ['trend']
    import solhunter_zero.async_scanner as async_scanner_mod
    monkeypatch.setattr(async_scanner_mod, 'fetch_trending_tokens_async', fake_trend)
    scanner_async_module = __import__('solhunter_zero.async_scanner', fromlist=[''])
    scanner_common.BIRDEYE_API_KEY = 'key'
    scanner_common.HEADERS.clear()
    scanner_common.HEADERS["X-API-KEY"] = "key"
    async def fr_func():
        return ['ray']
    async def fo_func():
        return ['orca']
    monkeypatch.setattr(async_scanner_mod, 'fetch_raydium_listings_async', fr_func)
    monkeypatch.setattr(async_scanner_mod, 'fetch_orca_listings_async', fo_func)
    tokens = asyncio.run(async_scan())
    assert tokens == ["abcbonk", "otherbonk", "trend", "ray", "orca"]


def test_scan_tokens_from_file(monkeypatch, tmp_path):
    path = tmp_path / "tokens.txt"
    path.write_text("tok1\n tok2\n\n#comment\n")

    def fake_get(*a, **k):
        raise AssertionError("should not call network")

    monkeypatch.setattr(scanner.requests, "get", fake_get)
    monkeypatch.setattr(scanner_common, "scan_tokens_onchain", lambda _: ["x"])  # should not be called
    monkeypatch.setattr(
        scanner,
        "fetch_trending_tokens",
        lambda: (_ for _ in ()).throw(AssertionError("trending")),
    )

    tokens = scanner.scan_tokens(token_file=str(path))
    assert tokens == ["tok1", "tok2"]


def test_scan_tokens_async_from_file(monkeypatch, tmp_path):
    path = tmp_path / "tokens.txt"
    path.write_text("a\nb\n")

    def fake_session():
        raise AssertionError("network should not be used")

    monkeypatch.setattr("aiohttp.ClientSession", fake_session)
    async def fail():
        raise AssertionError("trending")
    import solhunter_zero.async_scanner as async_scanner_mod
    monkeypatch.setattr(async_scanner_mod, "fetch_trending_tokens_async", fail)

    tokens = asyncio.run(async_scan(token_file=str(path)))
    assert tokens == ["a", "b"]


def test_scan_tokens_mempool(monkeypatch):
    async def fake_stream(url, *, suffix="bonk", include_pools=True):
        yield "memtok"

    monkeypatch.setattr(
        "solhunter_zero.mempool_scanner.stream_mempool_tokens", fake_stream
    )
    monkeypatch.setattr(scanner, "fetch_trending_tokens", lambda: [])
    monkeypatch.setattr(scanner, "fetch_raydium_listings", lambda: [])
    monkeypatch.setattr(scanner, "fetch_orca_listings", lambda: [])

    tokens = scanner.scan_tokens(method="mempool")
    assert tokens == ["memtok"]


def test_scan_tokens_async_mempool(monkeypatch):
    async def fake_stream(url, *, suffix=None, keywords=None, include_pools=True):
        yield "memtok"

    monkeypatch.setattr(
        "solhunter_zero.mempool_scanner.stream_mempool_tokens", fake_stream
    )
    async def fr():
        return []

    monkeypatch.setattr(scanner, "fetch_trending_tokens_async", fr)
    monkeypatch.setattr(scanner, "fetch_raydium_listings_async", fr)
    monkeypatch.setattr(scanner, "fetch_orca_listings_async", fr)

    result = asyncio.run(async_scan(method="mempool"))
    assert result == ["memtok"]
