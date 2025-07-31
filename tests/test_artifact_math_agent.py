import asyncio
import io
import json
import builtins
import pytest

artifact_math = pytest.importorskip("solhunter_zero.agents.artifact_math_agent")

ArtifactMathAgent = artifact_math.ArtifactMathAgent
decode_glyph_series = artifact_math.decode_glyph_series
pyramid_transform = artifact_math.pyramid_transform
from solhunter_zero.portfolio import Portfolio, Position

class DummyPortfolio(Portfolio):
    def __init__(self):
        super().__init__(path=None)
        self.balances = {}


def test_decode_glyph_series():
    mapping = {"A": 1, "B": -1}
    series = ["A", "B", "A"]
    assert decode_glyph_series(series, mapping) == [1, -1, 1]


def test_pyramid_transform():
    assert pyramid_transform([1, 2, 3]) == 8


def test_propose_trade_buy(monkeypatch):
    monkeypatch.setattr(artifact_math, "decode_glyph_series", lambda s, m: [1, 1])
    monkeypatch.setattr(artifact_math, "pyramid_transform", lambda scores: 1)
    monkeypatch.setattr(builtins, "open", lambda *a, **k: io.StringIO("{}"))
    monkeypatch.setattr(json, "load", lambda f: {"A": 1})
    agent = ArtifactMathAgent(threshold=0)
    pf = DummyPortfolio()
    actions = asyncio.run(agent.propose_trade("TOK", pf))
    assert actions and actions[0]["side"] == "buy"


def test_propose_trade_sell(monkeypatch):
    monkeypatch.setattr(artifact_math, "decode_glyph_series", lambda s, m: [-1])
    monkeypatch.setattr(artifact_math, "pyramid_transform", lambda scores: -1)
    monkeypatch.setattr(builtins, "open", lambda *a, **k: io.StringIO("{}"))
    monkeypatch.setattr(json, "load", lambda f: {"A": 1})
    agent = ArtifactMathAgent(threshold=0)
    pf = DummyPortfolio()
    pf.balances["TOK"] = Position("TOK", 1, 1.0, 1.0)
    actions = asyncio.run(agent.propose_trade("TOK", pf))
    assert actions and actions[0]["side"] == "sell"
