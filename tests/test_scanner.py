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

    def fake_get(url, timeout=10):
        return FakeResponse(data)

    monkeypatch.setattr(scanner.requests, 'get', fake_get)

    tokens = scanner.scan_tokens()
    assert tokens == ['abcbonk', 'xyzBONK']
