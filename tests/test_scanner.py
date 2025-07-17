import types
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

def test_scan_tokens(monkeypatch):
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
    scanner.HEADERS = {"X-API-KEY": "test"}

    tokens = scanner.scan_tokens()
    assert tokens == ['abcbonk', 'xyzBONK']
    assert captured['headers'] == scanner.HEADERS


def test_scan_tokens_offline(monkeypatch):
    called = {}

    def fake_get(*args, **kwargs):
        called['called'] = True
        return FakeResponse({}, 200)

    monkeypatch.setattr(scanner.requests, 'get', fake_get)

    tokens = scanner.scan_tokens(offline=True)
    assert tokens == scanner.OFFLINE_TOKENS
    assert 'called' not in called
