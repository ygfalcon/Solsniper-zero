from solhunter_zero import exchange
from solana.transaction import Transaction


class FakeClient:
    def __init__(self, url):
        self.url = url
        self.sent = None

    def send_transaction(self, tx, signer):
        self.sent = (tx, signer)
        return {"signature": "abc"}


def test_place_order_sends_transaction(monkeypatch):
    captured = {}

    def fake_client(url):
        client = FakeClient(url)
        captured["client"] = client
        return client

    monkeypatch.setattr(exchange, "Client", fake_client)

    result = exchange.place_order("tok", "buy", 1.0, 0.5, testnet=True)

    assert result == {"signature": "abc"}
    assert captured["client"].url == exchange.DEX_TESTNET_URL
    tx, signer = captured["client"].sent
    assert isinstance(tx, Transaction)
    assert signer is not None


def test_place_order_dry_run():
    result = exchange.place_order("tok", "buy", 1.0, 0.5, dry_run=True)
    assert result["dry_run"] is True
