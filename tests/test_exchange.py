import pytest
import requests
from solhunter_zero.exchange import place_order, OrderPlacementError
from solders.keypair import Keypair


class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code
        self.text = "response"

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError("bad status", response=self)

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


def test_place_order_http_error(monkeypatch):
    def fake_post(url, json, timeout=10):
        return FakeResponse({}, status_code=500)

    monkeypatch.setattr("solhunter_zero.exchange.requests.post", fake_post)
    with pytest.raises(OrderPlacementError):
        place_order("tok", "buy", 1.0, 0.5)


def test_place_order_network_error(monkeypatch):
    def fake_post(url, json, timeout=10):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr("solhunter_zero.exchange.requests.post", fake_post)
    with pytest.raises(OrderPlacementError):
        place_order("tok", "buy", 1.0, 0.5)
