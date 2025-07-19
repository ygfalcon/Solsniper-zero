import types
from solhunter_zero.exchange import place_order
from solders.keypair import Keypair


class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception("bad status")

    def json(self):
        return self._data


def test_place_order_posts(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout=10):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse({"order_id": "1"})

    monkeypatch.setattr("solhunter_zero.exchange.requests.post", fake_post)
    result = place_order("tok", "buy", 1.0, 0.5, testnet=True)
    assert result == {"order_id": "1"}
    assert captured["json"]["token"] == "tok"
    assert "/v6/swap" in captured["url"]


def test_place_order_dry_run(caplog):
    result = place_order("tok", "buy", 1.0, 0.5, dry_run=True)
    assert result["dry_run"] is True


def test_place_order_with_keypair(monkeypatch):
    kp = Keypair()
    captured = {}

    def fake_post(url, json, timeout=10):
        captured["json"] = json
        return FakeResponse({"ok": True})

    monkeypatch.setattr("solhunter_zero.exchange.requests.post", fake_post)
    place_order("tok", "buy", 1.0, 0.5, keypair=kp)
    assert "signature" in captured["json"]

import asyncio
from solhunter_zero.exchange import place_order_async


def test_place_order_async_posts(monkeypatch):
    captured = {}

    class FakeResp:
        def __init__(self, url, payload):
            captured["url"] = url
            captured["json"] = payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def json(self):
            return {"order_id": "1"}

        def raise_for_status(self):
            pass

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def post(self, url, json, timeout=10):
            return FakeResp(url, json)

    monkeypatch.setattr("aiohttp.ClientSession", lambda: FakeSession())
    result = asyncio.run(place_order_async("tok", "buy", 1.0, 0.5, testnet=True))
    assert result == {"order_id": "1"}
    assert captured["json"]["token"] == "tok"
