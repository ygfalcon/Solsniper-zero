import json
import types
from solhunter_zero import scanner


class FakeTxResponse:
    def __init__(self, tx):
        self._tx = tx

    def to_json(self):
        return json.dumps({"result": {"transaction": self._tx}})


class FakeClient:
    def get_signatures_for_address(self, program, limit=100):
        return types.SimpleNamespace(
            value=[
                types.SimpleNamespace(signature="sig1"),
                types.SimpleNamespace(signature="sig2"),
                types.SimpleNamespace(signature="sig3"),
            ]
        )

    def get_transaction(self, sig, encoding="jsonParsed", max_supported_transaction_version=0):
        if sig == "sig1":
            tx = {
                "message": {
                    "instructions": [
                        {
                            "program": "spl-token",
                            "parsed": {"type": "initializeMint", "info": {"mint": "abcbonk"}},
                        }
                    ]
                }
            }
        elif sig == "sig2":
            tx = {
                "message": {
                    "instructions": [
                        {
                            "program": "spl-token",
                            "parsed": {"type": "initializeMint", "info": {"mint": "xyzBONK"}},
                        }
                    ]
                }
            }
        else:
            tx = {
                "message": {
                    "instructions": [
                        {
                            "program": "spl-token",
                            "parsed": {"type": "initializeMint", "info": {"mint": "other"}},
                        }
                    ]
                }
            }
        return FakeTxResponse(tx)


def test_scan_tokens(monkeypatch):
    monkeypatch.setattr(scanner, "Client", lambda url: FakeClient())
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)
    tokens = scanner.scan_tokens(limit=3)
    assert tokens == ["abcbonk", "xyzBONK"]
