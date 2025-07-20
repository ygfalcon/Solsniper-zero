import pytest

from solhunter_zero import onchain_metrics


class FakeClient:
    def __init__(self, url, data):
        self.url = url
        self._data = data

    def get_signatures_for_address(self, addr):
        return {"result": self._data.get(str(addr), [])}


def test_top_volume_tokens(monkeypatch):
    tokens = ["t1", "t2", "t3"]
    tx_data = {
        "t1": [{"amount": 1.0}, {"amount": 3.0}],
        "t2": [{"amount": 5.0}],
        "t3": [],
    }

    captured = {}

    def fake_scan(url):
        captured["url"] = url
        return tokens

    def fake_client(url):
        return FakeClient(url, tx_data)

    monkeypatch.setattr(onchain_metrics, "scan_tokens_onchain", fake_scan)
    monkeypatch.setattr(onchain_metrics, "Client", fake_client)
    monkeypatch.setattr(onchain_metrics, "PublicKey", lambda x: x)

    result = onchain_metrics.top_volume_tokens("http://node", limit=2)

    assert captured["url"] == "http://node"
    assert result == ["t2", "t1"]


class ErrorClient:
    def __init__(self, url):
        self.url = url

    def get_signatures_for_address(self, addr):
        raise Exception("boom")


def test_top_volume_tokens_error(monkeypatch):
    monkeypatch.setattr(onchain_metrics, "scan_tokens_onchain", lambda url: ["a"])
    monkeypatch.setattr(onchain_metrics, "Client", lambda url: ErrorClient(url))
    monkeypatch.setattr(onchain_metrics, "PublicKey", lambda x: x)

    result = onchain_metrics.top_volume_tokens("rpc", limit=1)
    assert result == ["a"]
