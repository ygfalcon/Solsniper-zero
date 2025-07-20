
import asyncio
import requests
import aiohttp
from solhunter_zero import prices


class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

        self.text = "resp"

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError("bad", response=self)

    def json(self):
        return self._data


def test_fetch_token_prices(monkeypatch):
    data = {"data": {"tok": {"price": 2.0}, "bad": {"price": "x"}}}
    captured = {}

    def fake_get(url, timeout=10):
        captured["url"] = url
        return FakeResponse(data)

    monkeypatch.setattr(prices.requests, "get", fake_get)
    result = prices.fetch_token_prices(["tok", "bad"])
    assert result == {"tok": 2.0}
    assert "tok,bad" in captured["url"]


def test_fetch_token_prices_async(monkeypatch):
    data = {"data": {"tok": {"price": 1.5}}}
    captured = {}

    class FakeResp:
        def __init__(self, url):
            captured["url"] = url
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
        def get(self, url, timeout=10):
            return FakeResp(url)

    monkeypatch.setattr("aiohttp.ClientSession", lambda: FakeSession())
    result = asyncio.run(prices.fetch_token_prices_async(["tok"]))
    assert result == {"tok": 1.5}
    assert "tok" in captured["url"]


def test_fetch_token_prices_async_error(monkeypatch):
    """Return empty dict when aiohttp fails."""
    warnings = {}

    class FakeSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        def get(self, url, timeout=10):
            raise aiohttp.ClientError("boom")

    def fake_warning(msg, exc):
        warnings['msg'] = msg

    monkeypatch.setattr(prices.logger, "warning", fake_warning)
    monkeypatch.setattr("aiohttp.ClientSession", lambda: FakeSession())
    result = asyncio.run(prices.fetch_token_prices_async(["tok"]))
    assert result == {}
    assert 'msg' in warnings

