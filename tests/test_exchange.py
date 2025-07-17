import types
import base64
from solhunter_zero.exchange import place_order


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
        tx = base64.b64encode(b"tx").decode()
        return FakeResponse({"swapTransaction": tx})

    class FakeKeypair:
        public_key = "pub"

    class FakeTx:
        def sign(self, kp):
            captured["signed_with"] = kp

    class FakeClient:
        def __init__(self, url):
            captured["client_url"] = url

        def send_transaction(self, tx, kp):
            captured["sent"] = (tx, kp)
            return {"signature": "sig"}

    monkeypatch.setattr("solhunter_zero.exchange.requests.post", fake_post)
    monkeypatch.setattr("solhunter_zero.exchange.load_keypair", lambda: FakeKeypair())
    monkeypatch.setattr("solhunter_zero.exchange.Transaction.from_bytes", lambda b: FakeTx())
    monkeypatch.setattr("solhunter_zero.exchange.Client", FakeClient)

    result = place_order("tok", "buy", 1.0, 0.5, testnet=True)
    assert result == {"signature": "sig"}
    assert captured["json"]["token"] == "tok"
    assert "/v6/swap" in captured["url"]
    assert captured["sent"][0].__class__.__name__ == "FakeTx"


def test_place_order_dry_run(monkeypatch, caplog):
    called = {}

    class FakeClient:
        def __init__(self, url):
            called["client"] = True
        def send_transaction(self, tx, kp):
            called["sent"] = True

    monkeypatch.setattr("solhunter_zero.exchange.Client", FakeClient)
    monkeypatch.setattr(
        "solhunter_zero.exchange.load_keypair", lambda: types.SimpleNamespace(public_key="p")
    )
    result = place_order("tok", "buy", 1.0, 0.5, dry_run=True)
    assert result["dry_run"] is True
    assert "sent" not in called
