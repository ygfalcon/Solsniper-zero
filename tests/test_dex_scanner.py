import asyncio
from solhunter_zero import dex_scanner, scanner, scanner_common


class FakeClient:
    def __init__(self, url):
        self.url = url

    def get_program_accounts(self, program_id, encoding="jsonParsed"):
        assert encoding == "jsonParsed"
        assert program_id == dex_scanner.DEX_PROGRAM_ID
        return {
            "result": [
                {"account": {"data": {"parsed": {"info": {"tokenA": {"mint": "abcbonk"}, "tokenB": {"mint": "x"}}}}}},
                {"account": {"data": {"parsed": {"info": {"tokenA": {"mint": "y"}, "tokenB": {"mint": "zzzBONK"}}}}}},
            ]
        }


def test_scan_new_pools(monkeypatch):
    captured = {}

    def fake_client(url):
        captured["url"] = url
        return FakeClient(url)

    monkeypatch.setattr(dex_scanner, "Client", fake_client)
    tokens = dex_scanner.scan_new_pools("http://node")
    assert captured["url"] == "http://node"
    assert tokens == ["abcbonk", "zzzBONK"]


def test_scanner_method_pools(monkeypatch):
    monkeypatch.setattr(dex_scanner, "scan_new_pools", lambda url: ["tokbonk"])
    monkeypatch.setattr(scanner.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("birdeye")))
    scanner_common.SOLANA_RPC_URL = "http://node"
    tokens = scanner.scan_tokens(method="pools")
    assert tokens == ["tokbonk"]


def test_scanner_async_method_pools(monkeypatch):
    monkeypatch.setattr(dex_scanner, "scan_new_pools", lambda url: ["tokbonk"])
    monkeypatch.setattr(scanner.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("birdeye")))
    scanner_common.SOLANA_RPC_URL = "http://node"
    result = asyncio.run(scanner.scan_tokens_async(method="pools"))
    assert result == ["tokbonk"]
