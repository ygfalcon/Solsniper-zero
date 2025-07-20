import logging
import requests
from solhunter_zero.prices import fetch_token_prices


class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError("bad status", response=self)

    def json(self):
        return {"data": self._data}


def test_fetch_token_prices_success(monkeypatch):
    def fake_get(url, timeout=10):
        assert "tok1,tok2" in url
        return FakeResponse({"tok1": {"price": 1}, "tok2": {"price": 2}})

    monkeypatch.setattr("solhunter_zero.prices.requests.get", fake_get)
    prices = fetch_token_prices(["tok1", "tok2"])
    assert prices == {"tok1": 1.0, "tok2": 2.0}


def test_fetch_token_prices_network_error(monkeypatch, caplog):
    def fake_get(url, timeout=10):
        raise requests.RequestException("network down")

    monkeypatch.setattr("solhunter_zero.prices.requests.get", fake_get)
    with caplog.at_level(logging.WARNING):
        prices = fetch_token_prices(["tok1"])
    assert prices == {}
    assert "Failed to fetch token prices" in caplog.text
