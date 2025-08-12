import json
import sys
import types
import importlib.util
import importlib.machinery
import paper
import pytest
from solhunter_zero.event_bus import publish, subscribe

if importlib.util.find_spec("pydantic") is None:
    mod = types.ModuleType("pydantic")
    mod.__spec__ = importlib.machinery.ModuleSpec("pydantic", None)
    class BaseModel:  # minimal stand-in
        pass
    class AnyUrl(str):
        pass
    mod.BaseModel = BaseModel
    mod.AnyUrl = AnyUrl
    mod.ValidationError = Exception
    def _decorator(*a, **k):
        def wrap(f):
            return f
        return wrap
    mod.field_validator = _decorator
    mod.model_validator = _decorator
    mod.validator = _decorator
    mod.root_validator = _decorator
    sys.modules.setdefault("pydantic", mod)

class DummyStrategy:
    def __init__(self):
        self.events = []
        self._unsub = subscribe("price_update", self.events.append)

    def close(self):
        if self._unsub:
            self._unsub()


class FakePriceStreamManager:
    def __init__(self, streams, tokens):
        self.streams = streams
        self.tokens = tokens

    async def start(self):
        publish("price_update", {"venue": "orca", "token": "TOK", "price": 1.0})
        publish("price_update", {"venue": "raydium", "token": "TOK", "price": 1.3})

    async def stop(self):
        pass


def test_paper_price_stream(tmp_path, monkeypatch):
    sys.modules["solhunter_zero.price_stream_manager"] = types.SimpleNamespace(
        PriceStreamManager=FakePriceStreamManager
    )

    strategy = DummyStrategy()

    ticks = [{"timestamp": "2023-01-01", "price": 1.0}]
    data_path = tmp_path / "ticks.json"
    data_path.write_text(json.dumps(ticks))
    reports = tmp_path / "reports"
    monkeypatch.setenv("SOLHUNTER_PATCH_INVESTOR_DEMO", "1")

    paper.run([
        "--reports",
        str(reports),
        "--ticks",
        str(data_path),
        "--price-streams",
        "orca=ws://mock/orca,raydium=ws://mock/ray",
        "--tokens",
        "TOK",
    ])

    strategy.close()
    venues = {e.get("venue") for e in strategy.events}
    assert {"orca", "raydium"}.issubset(venues)
