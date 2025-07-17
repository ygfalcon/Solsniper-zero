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
