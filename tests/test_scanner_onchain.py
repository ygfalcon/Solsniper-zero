import pytest
from solhunter_zero import scanner_onchain
from solana.publickey import PublicKey


def setup_function(_):
    scanner_onchain.MEMPOOL_RATE_CACHE = scanner_onchain.TTLCache(
        maxsize=256, ttl=scanner_onchain.METRIC_CACHE_TTL
    )
    scanner_onchain.WHALE_ACTIVITY_CACHE = scanner_onchain.TTLCache(
        maxsize=256, ttl=scanner_onchain.METRIC_CACHE_TTL
    )
    scanner_onchain.AVG_SWAP_SIZE_CACHE = scanner_onchain.TTLCache(
        maxsize=256, ttl=scanner_onchain.METRIC_CACHE_TTL
    )

class FakeClient:
    def __init__(self, url):
        self.url = url
    def get_program_accounts(self, program_id, encoding="jsonParsed"):
        assert encoding == "jsonParsed"
        assert isinstance(program_id, PublicKey)
        return {
            "result": [
                {"account": {"data": {"parsed": {"info": {"name": "mybonk", "mint": "m1"}}}}},
                {"account": {"data": {"parsed": {"info": {"name": "other", "mint": "m2"}}}}},
            ]
        }

def test_scan_tokens_onchain(monkeypatch):
    captured = {}
    def fake_client(url):
        captured['url'] = url
        return FakeClient(url)
    monkeypatch.setattr(scanner_onchain, "Client", fake_client)
    tokens = scanner_onchain.scan_tokens_onchain("http://node")
    assert captured['url'] == "http://node"
    assert tokens == ["m1"]


def test_scan_tokens_onchain_requires_url():
    with pytest.raises(ValueError):
        scanner_onchain.scan_tokens_onchain("")


class FlakyClient:
    def __init__(self, url):
        self.url = url
        self.calls = 0

    def get_program_accounts(self, program_id, encoding="jsonParsed"):
        assert encoding == "jsonParsed"
        assert isinstance(program_id, PublicKey)
        self.calls += 1
        if self.calls < 3:
            raise Exception("rpc fail")
        return {
            "result": [
                {"account": {"data": {"parsed": {"info": {"name": "mybonk", "mint": "m1"}}}}}
            ]
        }


def test_scan_tokens_onchain_retries(monkeypatch):
    captured = {}

    def fake_client(url):
        client = FlakyClient(url)
        captured["client"] = client
        return client

    sleeps = []

    monkeypatch.setattr(scanner_onchain, "Client", fake_client)
    monkeypatch.setattr(scanner_onchain.time, "sleep", lambda t: sleeps.append(t))

    tokens = scanner_onchain.scan_tokens_onchain("http://node")

    assert tokens == ["m1"]
    assert captured["client"].calls == 3
    assert sleeps == [1, 2]



def test_mempool_tx_rate(monkeypatch):
    class Client:
        def __init__(self, url):
            self.url = url

        def get_signatures_for_address(self, addr, limit=20):
            return {"result": [{"blockTime": 1}, {"blockTime": 3}, {"blockTime": 4}]}

    monkeypatch.setattr(scanner_onchain, "Client", Client)
    monkeypatch.setattr(scanner_onchain, "PublicKey", lambda x: x)
    rate = scanner_onchain.fetch_mempool_tx_rate("tok", "http://node")
    assert rate == pytest.approx(3 / 3)


def test_whale_wallet_activity(monkeypatch):
    class Client:
        def __init__(self, url):
            self.url = url

        def get_token_largest_accounts(self, addr):
            return {"result": {"value": [{"uiAmount": 2000.0}, {"uiAmount": 50.0}]}}

    monkeypatch.setattr(scanner_onchain, "Client", Client)
    monkeypatch.setattr(scanner_onchain, "PublicKey", lambda x: x)
    activity = scanner_onchain.fetch_whale_wallet_activity(
        "tok", "http://node", threshold=1000.0
    )
    assert activity == pytest.approx(2000.0 / 2050.0)


def test_average_swap_size(monkeypatch):
    class Client:
        def __init__(self, url):
            self.url = url

        def get_signatures_for_address(self, addr, limit=20):
            return {"result": [{"amount": 2.0}, {"amount": 4.0}]}

    monkeypatch.setattr(scanner_onchain, "Client", Client)
    monkeypatch.setattr(scanner_onchain, "PublicKey", lambda x: x)
    size = scanner_onchain.fetch_average_swap_size("tok", "http://node")
    assert size == pytest.approx(3.0)


def test_mempool_tx_rate_cache(monkeypatch):
    calls = {"count": 0}

    class Client:
        def __init__(self, url):
            self.url = url

        def get_signatures_for_address(self, addr, limit=20):
            calls["count"] += 1
            return {"result": [{"blockTime": 1}, {"blockTime": 2}]}

    monkeypatch.setattr(scanner_onchain, "Client", Client)
    monkeypatch.setattr(scanner_onchain, "PublicKey", lambda x: x)

    scanner_onchain.fetch_mempool_tx_rate("tok", "http://node")
    scanner_onchain.fetch_mempool_tx_rate("tok", "http://node")

    assert calls["count"] == 1


def test_whale_wallet_activity_cache(monkeypatch):
    calls = {"count": 0}

    class Client:
        def __init__(self, url):
            self.url = url

        def get_token_largest_accounts(self, addr):
            calls["count"] += 1
            return {"result": {"value": [{"uiAmount": 2000.0}, {"uiAmount": 50.0}]}}

    monkeypatch.setattr(scanner_onchain, "Client", Client)
    monkeypatch.setattr(scanner_onchain, "PublicKey", lambda x: x)

    scanner_onchain.fetch_whale_wallet_activity("tok", "http://node")
    scanner_onchain.fetch_whale_wallet_activity("tok", "http://node")

    assert calls["count"] == 1


def test_average_swap_size_cache(monkeypatch):
    calls = {"count": 0}

    class Client:
        def __init__(self, url):
            self.url = url

        def get_signatures_for_address(self, addr, limit=20):
            calls["count"] += 1
            return {"result": [{"amount": 2.0}, {"amount": 4.0}]}

    monkeypatch.setattr(scanner_onchain, "Client", Client)
    monkeypatch.setattr(scanner_onchain, "PublicKey", lambda x: x)

    scanner_onchain.fetch_average_swap_size("tok", "http://node")
    scanner_onchain.fetch_average_swap_size("tok", "http://node")

    assert calls["count"] == 1

