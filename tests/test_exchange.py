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
    kp = Keypair()
    result = place_order("tok", "buy", 1.0, 0.5, testnet=True, keypair=kp)
    assert result == {"order_id": "1"}
    assert captured["json"]["token"] == "tok"
    assert captured["json"]["wallet"] == str(kp.pubkey())
    assert "/v6/swap" in captured["url"]


def test_place_order_dry_run(caplog):
    result = place_order("tok", "buy", 1.0, 0.5, dry_run=True)
    assert result["dry_run"] is True
