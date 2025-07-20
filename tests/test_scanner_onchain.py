import pytest
from solhunter_zero import scanner_onchain
from solana.publickey import PublicKey

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


def test_scan_tokens_onchain_with_metrics(monkeypatch):
    class FakeClient:
        def __init__(self, url):
            self.url = url

        def get_program_accounts(self, program_id, encoding="jsonParsed"):
            return {
                "result": [
                    {
                        "account": {
                            "data": {"parsed": {"info": {"name": "mybonk", "mint": "m1"}}}
                        }
                    }
                ]
            }

    monkeypatch.setattr(scanner_onchain, "Client", lambda url: FakeClient(url))

    import solhunter_zero.onchain_metrics as om

    monkeypatch.setattr(om, "fetch_volume_onchain", lambda t, u: 1.0)
    monkeypatch.setattr(om, "fetch_liquidity_onchain", lambda t, u: 2.0)

    res = scanner_onchain.scan_tokens_onchain("http://node", return_metrics=True)

    assert res == [{"address": "m1", "volume": 1.0, "liquidity": 2.0}]
