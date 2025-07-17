from solhunter_zero import scanner

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
    scanner.BIRDEYE_API_KEY = "test"
    scanner.HEADERS = {"X-API-KEY": "test"}

    tokens = scanner.scan_tokens()
    assert tokens == ['abcbonk', 'xyzBONK']
    assert captured['headers'] == scanner.HEADERS


def test_scan_tokens_onchain_fallback(monkeypatch):
    captured = {}

    def fake_scan(url):
        captured['url'] = url
        return ['onchain']

    monkeypatch.setattr(scanner, 'scan_tokens_onchain', fake_scan)
    scanner.BIRDEYE_API_KEY = None
    scanner.HEADERS = {}
    scanner.SOLANA_RPC_URL = 'http://node'

    tokens = scanner.scan_tokens()
    assert tokens == ['onchain']
    assert captured['url'] == 'http://node'
