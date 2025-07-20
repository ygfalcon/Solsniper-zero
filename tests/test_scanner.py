from solhunter_zero import scanner
from solhunter_zero import scanner_common

class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception('bad status')

    def json(self):
        return self._data

def test_scan_tokens_birdeye(monkeypatch):
    data = {
        'data': [
            {'address': 'abcbonk'},
            {'address': 'xyzBONK'},
            {'address': 'other'},
        ]
    }

    captured = {}

    def fake_get(url, headers=None, timeout=10):
        captured['headers'] = headers
        return FakeResponse(data)

    monkeypatch.setattr(scanner.requests, 'get', fake_get)
    scanner_common.BIRDEYE_API_KEY = "test"
    scanner_common.HEADERS.clear()
    scanner_common.HEADERS["X-API-KEY"] = "test"

    tokens = scanner.scan_tokens()
    assert tokens == ['abcbonk', 'xyzBONK']
    assert captured['headers'] == scanner.HEADERS



# codex/add-offline-option-to-solhunter_zero.main
def test_scan_tokens_offline(monkeypatch):
    called = {}

    def fake_get(*args, **kwargs):
        called['called'] = True
        return FakeResponse({}, 200)

    monkeypatch.setattr(scanner.requests, 'get', fake_get)

    tokens = scanner.scan_tokens(offline=True)
    assert tokens == scanner.OFFLINE_TOKENS
    assert 'called' not in called


def test_scan_tokens_onchain_when_no_key(monkeypatch):
    captured = {}

    def fake_onchain(url):
        captured['url'] = url
        return ['tok']

    def fake_get(*args, **kwargs):
        raise AssertionError('should not call BirdEye')

    monkeypatch.setattr(scanner_common, 'scan_tokens_onchain', fake_onchain)
    monkeypatch.setattr(scanner.requests, 'get', fake_get)

    scanner_common.BIRDEYE_API_KEY = None
    scanner_common.HEADERS.clear()
    scanner_common.SOLANA_RPC_URL = 'http://node'

    tokens = scanner.scan_tokens()
    assert tokens == ['tok']
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
    scanner_async_module = __import__('solhunter_zero.async_scanner', fromlist=[''])
    scanner_common.BIRDEYE_API_KEY = 'key'
    scanner_common.HEADERS.clear()
    scanner_common.HEADERS["X-API-KEY"] = "key"
    tokens = asyncio.run(async_scan())
    assert tokens == ["abcbonk", "otherbonk"]
